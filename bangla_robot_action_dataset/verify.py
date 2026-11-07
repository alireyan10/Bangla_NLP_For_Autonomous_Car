import csv, re
from collections import Counter

rows = []
with open('preprocessed_output/data/preprocessed/train_preprocessed.csv', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))

seq_stop = [r for r in rows if r['Intent'] == 'SEQUENTIAL' and r['Speed Category'] == 'STOP']
print('=== SEQUENTIAL+STOP: Fix F2 check (sample 10) ===')
for r in seq_stop[:10]:
    print('ID=' + r['ID'])
    print('  Bangla   : ' + r['Bangla Command'][:65])
    print('  Pre      : ' + r['Preprocessed Command'])
    print('  AppSpeed : ' + r['Approach Speed'])
    print('  StepCount: ' + r['Step Count'])
    print()

print('=== Step Count distribution for ALL intents ===')
step_counts = Counter(int(r['Step Count']) for r in rows)
for k, v in sorted(step_counts.items()):
    print(f'  Step Count={k}: {v} rows')

print()
print('=== Step Count for SEQUENTIAL rows ===')
seq_rows = [r for r in rows if r['Intent'] == 'SEQUENTIAL']
seq_step_counts = Counter(int(r['Step Count']) for r in seq_rows)
for k, v in sorted(seq_step_counts.items()):
    print(f'  Step Count={k}: {v} rows')

print()
print('=== Verify NO [SPEED:STOP] + movement action in SEQUENTIAL+STOP ===')
movement_actions = {'[ACTION:MOVE_FORWARD]','[ACTION:MOVE_BACKWARD]',
                    '[ACTION:MOVE_TOWARD_TARGET]','[ACTION:MOVE_NEAR_TARGET]',
                    '[ACTION:MOVE_IN_FRONT_TARGET]','[ACTION:MOVE_PAST_OBJECT]',
                    '[ACTION:TURN_LEFT]','[ACTION:TURN_RIGHT]'}
bad = [r for r in seq_stop
       if '[SPEED:STOP]' in r['Preprocessed Command']
       and set(re.findall(r'\[ACTION:[A-Z_]+\]', r['Preprocessed Command'])) & movement_actions]
print('Rows with [SPEED:STOP] + movement action: ' + str(len(bad)) + '  (should be 0)')

print()
print('=== Approach Speed distribution for SEQUENTIAL+STOP rows ===')
asp_counter = Counter(r['Approach Speed'] for r in seq_stop)
for k, v in sorted(asp_counter.items()):
    print(f'  Approach Speed={k!r}: {v} rows')
