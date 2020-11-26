import miditoolkit
import utils
import midi2mumidi
from glob import glob
import numpy as np
import os
midilist = sorted(glob('../nesmdb_midi/valid/*.mid'))
txtlist = sorted(glob("../nesmdb_tx1/valid/*.txt"))
outdir = "./midi_out/"
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

    filename = midilist[i].split('/')[-1].replace(".txt",".mid")
    output_path = outdir + filename
    midi2mumidi.write_midi(events, output_path)
    # with open(txtlist[i], "r") as f:
    #     lines = f.readlines()
    #     print(f'txt file name : {txtlist[i]} ')
    #     print(len(lines))

