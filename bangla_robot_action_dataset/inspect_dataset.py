import json, csv

# Count lines in JSONL files
with open('train.jsonl', 'r', encoding='utf-8') as f:
    train_lines = sum(1 for _ in f)
with open('validation.jsonl', 'r', encoding='utf-8') as f:
    val_lines = sum(1 for _ in f)
with open('test.jsonl', 'r', encoding='utf-8') as f:
    test_lines = sum(1 for _ in f)

print(f'Train JSONL lines: {train_lines}')
print(f'Validation JSONL lines: {val_lines}')
print(f'Test JSONL lines: {test_lines}')

# Count CSV rows
with open('dataset.csv', 'r', encoding='utf-8') as f:
    csv_lines = sum(1 for _ in f) - 1  # subtract header
print(f'CSV rows: {csv_lines}')

# Get CSV columns
with open('dataset.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    headers = next(reader)
    print(f'CSV columns: {headers}')

# Sample train JSONL keys
with open('train.jsonl', 'r', encoding='utf-8') as f:
    sample = json.loads(f.readline())
    print(f'JSONL keys: {list(sample.keys())}')

# Get unique intents, targets, speeds from train
with open('train.jsonl', 'r', encoding='utf-8') as f:
    intents = set()
    targets = set()
    speeds = set()
    ids_train = []
    for line in f:
        row = json.loads(line)
        intents.add(row.get('Intent',''))
        targets.add(row.get('Target',''))
        speeds.add(row.get('Speed Category',''))
        ids_train.append(row.get('ID'))

print(f'Train unique intents: {sorted(intents)}')
print(f'Train unique targets: {sorted(targets)}')
print(f'Train unique speed categories: {sorted(speeds)}')
print(f'Train ID range: {min(ids_train)} to {max(ids_train)}')

# Check validation and test IDs
with open('validation.jsonl', 'r', encoding='utf-8') as f:
    ids_val = [json.loads(line)['ID'] for line in f]
with open('test.jsonl', 'r', encoding='utf-8') as f:
    ids_test = [json.loads(line)['ID'] for line in f]

print(f'Validation ID range: {min(ids_val)} to {max(ids_val)}')
print(f'Test ID range: {min(ids_test)} to {max(ids_test)}')

# Check for overlap
train_set = set(ids_train)
val_set = set(ids_val)
test_set = set(ids_test)
print(f'Train-Val overlap: {len(train_set & val_set)}')
print(f'Train-Test overlap: {len(train_set & test_set)}')
print(f'Val-Test overlap: {len(val_set & test_set)}')
print(f'Total unique IDs: {len(train_set | val_set | test_set)}')

# Print a few sample commands from each intent category
with open('train.jsonl', 'r', encoding='utf-8') as f:
    rows = [json.loads(line) for line in f]

print('\n--- SAMPLE COMMANDS BY INTENT ---')
seen = set()
for row in rows:
    intent = row['Intent']
    if intent not in seen:
        seen.add(intent)
        print(f"{intent}: {row['Bangla Command']!r}")
        if len(seen) >= 20:
            break
