import numpy as np
DEFAULT_DURATION_BINS = np.arange(689, 88200, 689, dtype=int) # 1/32
DEFAULT_FRACTION = 32

vocab = []

# Add notes
ins_to_range = {
    'P1': [33, 108],
    'P2': [33, 108],
    'TR': [21, 108],
    'NO': [1, 16]
}

# BAR
vocab.append('BAR')

# POSITION
for i in range(DEFAULT_FRACTION):
    vocab.append(f'POS_{i+1}')

# TRACK
for ins in ['P1', 'P2', 'TR', 'NO']:
    #vocab.append(f'TRACK_{ins}')
    vocab.append(f'{ins}')

# DURATION
for dur in range(len(DEFAULT_DURATION_BINS)):
  vocab.append(f'DUR_{dur+1}')
  
# NOTE ON 
for ins in ['TR', 'NO']: # These include all note range
    lo, hi = ins_to_range[ins]
    for n in range(lo, hi + 1):
        vocab.append(f'NOTEON_{n}')

with open('vocab.txt', 'w') as f:
  f.write('\n'.join(vocab))
