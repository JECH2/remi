import os
import pickle
from glob import glob

import chord_recognition
import numpy as np
import miditoolkit
import copy
import pretty_midi

# parameters for input
DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32+1, dtype=np.int)
DEFAULT_FRACTION = 32
DEFAULT_DURATION_BINS = np.arange(689, 88200, 689, dtype=int)
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]

# parameters for output
DEFAULT_RESOLUTION = 2756 # 1/8
QUANTIZE_RESOLUTION = 689 # 1/32
TICKS_PER_BEAT = 22050

mid_base = '../nesmdb_midi/'
nlm_base = '../nesmdb_nlm/'
#splits = ['train', 'valid', 'test']
splits = ['valid']
instrs = ['p1','p2','tr','no']
instr_dict = {'p1':1, 'p2':2, 'tr':3, 'no':4}
velocity = 60

def saveMuMIDI():
    for split in splits:
        mid_dir = mid_base + split
        for file in glob(mid_dir+'/*.mid'):
            # for all file, do and save output as pkl
            print(file)


# define "Item" for general storage
class Item(object):
    def __init__(self, name, start, end, velocity, pitch):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch

    def __repr__(self):
        return 'Item(name={}, start={}, end={}, velocity={}, pitch={})'.format(
            self.name, self.start, self.end, self.velocity, self.pitch)

# read notes and tempo changes from midi (assume there is only one track)
def read_items(file_path):
    midi_obj = miditoolkit.midi.parser.MidiFile(file_path)
    # note
    note_items = []
    for i in range(0, len(midi_obj.instruments)):
        notes = midi_obj.instruments[i].notes
        notes.sort(key=lambda x: (x.start, x.pitch))
        for note in notes:
            note_items.append(Item(
                name=midi_obj.instruments[i].name, 
                start=note.start, 
                end=note.end, 
                velocity=note.velocity, 
                pitch=note.pitch))
        note_items.sort(key=lambda x: x.start)

    # # key extraction
    # key_items = []
    # key_val = get_key(midi_obj)
    # key_items.append(Item(
    #     name='Key',
    #     start=0,
    #     end=None,
    #     velocity=None,
    #     pitch=int(key_val)))
    return note_items

# quantize items
def quantize_items(items, ticks=QUANTIZE_RESOLUTION):
    # grid
    grids = np.arange(0, items[-1].start, ticks, dtype=int)
    # process
    for item in items:
        index = np.argmin(abs(grids - item.start))
        shift = grids[index] - item.start
        item.start += shift
        item.end += shift
    return items      

# extract chord
def extract_chords(items):
    method = chord_recognition.MIDIChord()
    chords = method.extract(notes=items)
    output = []
    for chord in chords:
        output.append(Item(
            name='Chord',
            start=chord[0],
            end=chord[1],
            velocity=None,
            pitch=chord[2].split('/')[0]))
    return output

# group items for each 1/4 bak
def group_items(items, max_time, ticks_per_bar=TICKS_PER_BEAT*2):
    items.sort(key=lambda x: x.start)
    downbeats = np.arange(0, max_time+ticks_per_bar, ticks_per_bar)
    groups = []
    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
        p1 = []
        p2 = []
        tr = []
        no = []
        #insiders = dict()
        for item in items:
            if (item.start >= db1) and (item.start < db2):
                #if item.name not in insiders:
                #    insiders[item.name] = list()
                if item.name=='p1':
                    p1.append(item)
                elif item.name=='p2':
                    p2.append(item)
                elif item.name=='tr':
                    tr.append(item)
                else:
                    no.append(item)
                # insiders[item.name].append(item)
        overall = [db1] + p1 + p2 + tr + no + [db2]
        groups.append(overall)
    return groups

# define "Event" for event storage
class Event(object):
    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return 'Event(name={}, time={}, value={}, text={})'.format(
            self.name, self.time, self.value, self.text)

# item to event
def item2event(groups):
    events = []
    n_downbeat = 0
    for i in range(len(groups)):
        # print(groups[0][1:-1][0])
        # exit()
        event_lists = [item.name for item in groups[i][1:-1]]
        if ('p1' not in event_lists) and ('p2' not in event_lists) and ('tr' not in event_lists) and ('no' not in event_lists):
            continue
        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        events.append(Event(
            name='Bar',
            time=None, 
            value=None,
            text='{}'.format(n_downbeat)))
        prev_index = -1;
        prev_item_name = None
        for item in groups[i][1:-1]:
            # position
            flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
            index = np.argmin(abs(flags-item.start))
            if (index != prev_index):
                events.append(Event(
                    name='Position', 
                    time=item.start,
                    value='{}/{}'.format(index+1, DEFAULT_FRACTION),
                    text='{}'.format(item.start)))
                prev_index = index
            if item.name in instrs:
                if item.name is not prev_item_name:
                    events.append(Event(
                        name='Track',
                        time=None,
                        value=None,
                        text=item.name))
                    prev_item_name = item.name
                # velocity
                # velocity_index = np.searchsorted(
                #     DEFAULT_VELOCITY_BINS, 
                #     item.velocity, 
                #     side='right') - 1
                # events.append(Event(
                #     name='Note Velocity',
                #     time=item.start, 
                #     value=velocity_index,
                #     text='{}/{}'.format(item.velocity, DEFAULT_VELOCITY_BINS[velocity_index])))
                # pitch
                events.append(Event(
                    name='Note On',
                    time=item.start, 
                    value=item.pitch,
                    text='{}'.format(item.pitch)))
                # duration
                duration = item.end - item.start
                index = np.argmin(abs(DEFAULT_DURATION_BINS-duration))
                events.append(Event(
                    name='Note Duration',
                    time=item.start,
                    value=index,
                    text='{}/{}'.format(duration, DEFAULT_DURATION_BINS[index])))

    return events

def event2word(events):
    print(event)

#############################################################################################
# WRITE MIDI
#############################################################################################
def word_to_event(words, word2event):
    events = []
    for word in words:
        event_name, event_value = word2event.get(word).split('_')
        events.append(Event(event_name, None, event_value, None))
    return events

#def write_midi(words, word2event, output_path, prompt_path=None):
#    events = word_to_event(words, word2event)
def write_midi(events, output_path, prompt_path=None):
    # get downbeat and note (no time)
    temp_notes = []
    i = 0
    position = None
    track = None
    while(1):
        if i >= len(events):
            break 
        if events[i].name == 'Bar':
            temp_notes.append('Bar')
            i += 1
        elif events[i].name == 'Position':
            # start time and end time from position
            position = int(events[i].value.split('/')[0]) - 1
            i += 1
        elif(events[i].name == 'Track'):
            # instr name
            track = events[i].text
            i += 1
        elif (events[i].name == 'Note On' and events[i+1].name == 'Note Duration'):
            # # velocity
            # index = int(events[i+2].value)
            # velocity = int(DEFAULT_VELOCITY_BINS[index])
            # pitch
            pitch = int(events[i].value)
            # duration
            index = int(events[i+1].value)
            duration = DEFAULT_DURATION_BINS[index]
            # adding
            #print(track, pitch, duration)
            temp_notes.append([position, track, pitch, duration])
            i += 2
        # elif events[i].name == 'Position' and events[i+1].name == 'Chord':
        #     position = int(events[i].value.split('/')[0]) - 1
        #     temp_chords.append([position, events[i+1].value])
        # elif events[i].name == 'Position' and \
        #     events[i+1].name == 'Tempo Class' and \
        #     events[i+2].name == 'Tempo Value':
        #     position = int(events[i].value.split('/')[0]) - 1
        #     if events[i+1].value == 'slow':
        #         tempo = DEFAULT_TEMPO_INTERVALS[0].start + int(events[i+2].value)
        #     elif events[i+1].value == 'mid':
        #         tempo = DEFAULT_TEMPO_INTERVALS[1].start + int(events[i+2].value)
        #     elif events[i+1].value == 'fast':
        #         tempo = DEFAULT_TEMPO_INTERVALS[2].start + int(events[i+2].value)
        #     temp_tempos.append([position, tempo])
    # get specific time for notes
    ticks_per_beat = TICKS_PER_BEAT
    ticks_per_bar = TICKS_PER_BEAT * 2 # assume 2/4 NES
    p1 = []
    p2 = []
    tr = []
    no = []
    current_bar = 0
    for note in temp_notes:
        if note == 'Bar':
            current_bar += 1
        else:
            position, track, pitch, duration = note
            # position (start time)
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
            st = flags[position]
            # duration (end time)
            et = st + duration
            if track == "p1":
                p1.append(miditoolkit.Note(15, pitch, st, et))
            elif track == "p2":
                p2.append(miditoolkit.Note(15, pitch, st, et))
            elif track == "tr":
                tr.append(miditoolkit.Note(1, pitch, st, et))
            else:
                no.append(miditoolkit.Note(15, pitch, st, et))
            
    # # get specific time for chords
    # if len(temp_chords) > 0:
    #     chords = []
    #     current_bar = 0
    #     for chord in temp_chords:
    #         if chord == 'Bar':
    #             current_bar += 1
    #         else:
    #             position, value = chord
    #             # position (start time)
    #             current_bar_st = current_bar * ticks_per_bar
    #             current_bar_et = (current_bar + 1) * ticks_per_bar
    #             flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
    #             st = flags[position]
    #             chords.append([st, value])
    # get specific time for tempos
    # tempos = []
    # current_bar = 0
    # for tempo in temp_tempos:
    #     if tempo == 'Bar':
    #         current_bar += 1
    #     else:
    #         position, value = tempo
    #         # position (start time)
    #         current_bar_st = current_bar * ticks_per_bar
    #         current_bar_et = (current_bar + 1) * ticks_per_bar
    #         flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
    #         st = flags[position]
    #         tempos.append([int(st), value])
    # write
    if prompt_path:
        midi = miditoolkit.midi.parser.MidiFile(prompt_path)
        #
        last_time = DEFAULT_RESOLUTION * 4 * 4
        # note shift
        for note in notes:
            note.start += last_time
            note.end += last_time
        midi.instruments[0].notes.extend(notes)
        # # tempo changes
        # temp_tempos = []
        # for tempo in midi.tempo_changes:
        #     if tempo.time < DEFAULT_RESOLUTION*4*4:
        #         temp_tempos.append(tempo)
        #     else:
        #         break
        # for st, bpm in tempos:
        #     st += last_time
        #     temp_tempos.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        # midi.tempo_changes = temp_tempos
        # # write chord into marker
        # if len(temp_chords) > 0:
        #     for c in chords:
        #         midi.markers.append(
        #             miditoolkit.midi.containers.Marker(text=c[1], time=c[0]+last_time))
    else:
        midi = miditoolkit.midi.parser.MidiFile()
        midi.ticks_per_beat = TICKS_PER_BEAT
        # write instrument
        p1_Instrument = miditoolkit.midi.containers.Instrument(81, is_drum=False, name='p1') # program number for Lead 1 (square)
        p2_Instrument = miditoolkit.midi.containers.Instrument(82, is_drum=False, name='p2') # program number for Lead 2 (sawtooth)
        tr_Instrument = miditoolkit.midi.containers.Instrument(39, is_drum=False, name='tr') # program number for Synth Bass 1
        no_Instrument = miditoolkit.midi.containers.Instrument(122, is_drum=True, name='no') # program number for Breath Noise
        p1_Instrument.notes = p1
        p2_Instrument.notes = p2
        tr_Instrument.notes = tr
        no_Instrument.notes = no
        midi.instruments.append(p1_Instrument)
        midi.instruments.append(p2_Instrument)
        midi.instruments.append(tr_Instrument)
        midi.instruments.append(no_Instrument)

        # # write tempo
        # tempo_changes = []
        # for st, bpm in tempos:
        #     tempo_changes.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        # midi.tempo_changes = tempo_changes
        # # write chord into marker
        # if len(temp_chords) > 0:
        #     for c in chords:
        #         midi.markers.append(
        #             miditoolkit.midi.containers.Marker(text=c[1], time=c[0]))
    # write
    midi.dump(output_path)



if __name__ == '__main__':
    saveMuMIDI()
