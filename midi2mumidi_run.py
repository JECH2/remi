import miditoolkit
import utils
import midi2mumidi
from glob import glob
import numpy as np
import os
midilist = sorted(glob('../nesmdb_midi/valid/*.mid'))
txtlist = sorted(glob("../nesmdb_tx1/valid/*.txt"))
outdir = "./midi_out/"
split = ['train', 'valid', 'test']
# midi_obj = []
# for i in range(len(filelist)):
#     midi_obj.append(miditoolkit.midi.parser.MidiFile(filelist[i]))
#     print(filelist[i])

os.makedirs(outdir, exist_ok=True)

for i in range(0, len(midilist)):
    note_items = midi2mumidi.read_items(midilist[i])
    if len(note_items) == 0:
        continue
    note_items = midi2mumidi.quantize_items(note_items)

    items = note_items
    max_time = note_items[-1].end
    groups = midi2mumidi.group_items(items, max_time)

    # for g in groups:
    #     print(*g, sep='\n')
    #     print()

    events = midi2mumidi.item2event(groups)
    #print(*events[:100], sep='\n')
    print(f'midi file name : {midilist[i]} ')
    print(f'mumidi token number {len(events)}')

    filename = midilist[i].split('/')[-1]
    #output_path = outdir + filename
    #midi2mumidi.write_midi(events, output_path)
    # with open(txtlist[i], "r") as f:
    #     lines = f.readlines()
    #     print(f'txt file name : {txtlist[i]} ')
    #     print(len(lines))
    save_file = True
    if save_file:
        w_events = midi2mumidi.event2word(events)
        reconstructed_events = midi2mumidi.word_to_event(w_events)
        outfile = outdir + filename.replace('.mid','.txt')
        outfile2 = outdir + filename.replace('mid', '2.txt')
        with open(outfile, "w") as f:
            f.write(w_events)
        with open(outfile2, "w") as f:
            for reconstructed_event in reconstructed_events:
                f.write(str(reconstructed_event)+'\n')
