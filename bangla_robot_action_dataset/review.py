import csv, json, re
from collections import defaultdict, Counter

rows = []
for fname, split in [
    ('preprocessed_output/data/preprocessed/train_preprocessed.csv', 'Train'),
    ('preprocessed_output/data/preprocessed/validation_preprocessed.csv', 'Validation'),
    ('preprocessed_output/data/preprocessed/test_preprocessed.csv', 'Test'),
]:
    with open(fname, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            r['_split'] = split
            rows.append(r)

print(f"Total rows: {len(rows)}")

# ── ISSUE 1: MOVE_SLOW vs MOVE_FAST both map to MOVE_FORWARD ────────────────
print("\n=== MOVE_SLOW vs MOVE_FAST vs MOVE_FORWARD action token comparison ===")
for intent in ['MOVE_SLOW', 'MOVE_FAST', 'MOVE_FORWARD']:
    irows = [r for r in rows if r['Intent'] == intent]
    patterns = Counter(r['Preprocessed Command'] for r in irows)
    print(f"\n{intent} (n={len(irows)}): {len(patterns)} distinct forms")
    for p, c in patterns.most_common(3):
        print(f"  [{c}x] {p}")
    # Show 2 sample commands
    for r in irows[:2]:
        print(f"  Example: {r['Bangla Command']}")

# ── ISSUE 2: Step Count == 1 for SEQUENTIAL — find ALL and show the bangla ──
print("\n\n=== ALL SEQUENTIAL rows with Step Count=1 ===")
seq_rows = [r for r in rows if r['Intent'] == 'SEQUENTIAL']
step1_seq = [r for r in seq_rows if int(r.get('Step Count','1')) == 1]
print(f"Total SEQUENTIAL: {len(seq_rows)}")
print(f"Step Count=1 (should be >= 2): {len(step1_seq)}")
for r in step1_seq[:20]:
    bangla = r['Bangla Command']
    # Check if it contains known sequential markers
    has_tarpar = 'তারপর' in bangla
    has_erpor = 'এরপর' in bangla
    has_ebong = 'এবং' in bangla
    print(f"  ID={r['ID']} markers=({has_tarpar},{has_erpor},{has_ebong}): {bangla[:80]}")

# ── ISSUE 3: Deeply inspect SEQUENTIAL action token assignment ───────────────
print("\n\n=== SEQUENTIAL: are action tokens semantically correct? ===")
# Check: when pre-tarpar clause has 'কাছে' → should be MOVE_NEAR_TARGET not MOVE_IN_FRONT_TARGET
wrong_near = [r for r in seq_rows
              if 'কাছে' in r['Bangla Command']
              and '[ACTION:MOVE_IN_FRONT_TARGET]' in r['Preprocessed Command']
              and '[ACTION:MOVE_NEAR_TARGET]' not in r['Preprocessed Command']]
print(f"\nSEQUENTIAL with 'কাছে' but MOVE_IN_FRONT_TARGET (should be MOVE_NEAR_TARGET): {len(wrong_near)}")
for r in wrong_near[:5]:
    print(f"  ID={r['ID']}: {r['Bangla Command'][:70]}")
    print(f"          => {r['Preprocessed Command']}")

wrong_past = [r for r in seq_rows
              if 'পাশ কাটিয়ে' in r['Bangla Command']
              and '[ACTION:MOVE_PAST_OBJECT]' not in r['Preprocessed Command']]
print(f"\nSEQUENTIAL with 'পাশ কাটিয়ে' but no MOVE_PAST_OBJECT: {len(wrong_past)}")
for r in wrong_past[:5]:
    print(f"  ID={r['ID']}: {r['Bangla Command'][:70]}")
    print(f"          => {r['Preprocessed Command']}")

# ── ISSUE 4: Same Intent + Same Target + Same Speed → same Preprocessed? ────
print("\n\n=== Same (Intent,Target,Speed) → multiple Preprocessed Command forms ===")
combo_to_pre = defaultdict(set)
for r in rows:
    key = (r['Intent'], r['Target'], r['Speed Category'])
    combo_to_pre[key].add(r['Preprocessed Command'])

conflicts = {k: v for k, v in combo_to_pre.items() if len(v) > 1}
print(f"Conflicting (Intent,Target,Speed) combos: {len(conflicts)}")
for key, pres in sorted(conflicts.items())[:15]:
    print(f"\n  {key}:")
    for p in sorted(pres):
        count = sum(1 for r in rows
                    if r['Intent']==key[0] and r['Target']==key[1] and r['Speed Category']==key[2]
                    and r['Preprocessed Command']==p)
        print(f"    [{count}x] {p}")

# ── ISSUE 5: MOVE_SLOW/MOVE_FAST: should speed token match intent? ───────────
print("\n\n=== MOVE_SLOW rows where Preprocessed Command contains [SPEED:FAST] ===")
bad_slow = [r for r in rows if r['Intent']=='MOVE_SLOW' and '[SPEED:FAST]' in r.get('Preprocessed Command','')]
print(f"MOVE_SLOW + [SPEED:FAST]: {len(bad_slow)}")
for r in bad_slow[:5]:
    print(f"  ID={r['ID']} SpeedCat={r['Speed Category']}: {r['Bangla Command']}")
    print(f"     => {r['Preprocessed Command']}")

print("\n=== MOVE_FAST rows where Preprocessed Command contains [SPEED:SLOW] ===")
bad_fast = [r for r in rows if r['Intent']=='MOVE_FAST' and '[SPEED:SLOW]' in r.get('Preprocessed Command','')]
print(f"MOVE_FAST + [SPEED:SLOW]: {len(bad_fast)}")
for r in bad_fast[:5]:
    print(f"  ID={r['ID']} SpeedCat={r['Speed Category']}: {r['Bangla Command']}")
    print(f"     => {r['Preprocessed Command']}")

# ── ISSUE 6: SEQUENTIAL — SPEED token from Speed Category vs first clause ───
print("\n\n=== SEQUENTIAL: does SPEED token match first-clause speed? ===")
# For SEQUENTIAL, the speed of the FIRST action matters.
# But we derive speed from Speed Category which is the overall category.
# Check if STOP appears in SEQUENTIAL commands (ID 209-212 seen above)
seq_stop = [r for r in seq_rows if '[SPEED:STOP]' in r.get('Preprocessed Command','')]
print(f"SEQUENTIAL rows with [SPEED:STOP]: {len(seq_stop)}")
for r in seq_stop[:5]:
    print(f"  ID={r['ID']} SpeedCat={r['Speed Category']}: {r['Bangla Command'][:70]}")
    print(f"     => {r['Preprocessed Command']}")

# ── ISSUE 7: How many distinct bangla command forms map to same preprocessed? ─
print("\n\n=== Distinct Bangla commands that share the same Preprocessed Command ===")
pre_to_bangla = defaultdict(set)
for r in rows:
    pre_to_bangla[r['Preprocessed Command']].add(r['Bangla Command'])

# Find preprocessed forms with most Bangla variants
top = sorted(pre_to_bangla.items(), key=lambda x: len(x[1]), reverse=True)[:10]
print("Top 10 preprocessed forms by number of distinct Bangla commands:")
for pre, banglas in top:
    print(f"  [{len(banglas)} variants] {pre}")
    for b in sorted(banglas)[:3]:
        print(f"    - {b}")

# ── ISSUE 8: Normalized Action column vs action tokens in Preprocessed ───────
print("\n\n=== Mismatch: Normalized Action vs Preprocessed Command action tokens ===")
mismatch = []
for r in rows:
    norm_action = r.get('Normalized Action','').strip()
    pre_cmd = r.get('Preprocessed Command','').strip()
    action_tokens_in_pre = ' '.join(re.findall(r'\[ACTION:[A-Z_]+\]', pre_cmd))
    if norm_action != action_tokens_in_pre:
        mismatch.append((r['ID'], norm_action, action_tokens_in_pre, r['Bangla Command'][:50]))

print(f"Rows where Normalized Action != actions in Preprocessed Command: {len(mismatch)}")
for id_, na, at, cmd in mismatch[:15]:
    print(f"  ID={id_}: NormAction={na!r}")
    print(f"           ActionTokens={at!r}")
    print(f"           Bangla: {cmd}")

# ── ISSUE 9: SEQUENTIAL with pre-clause having 'দিকে' → should be MOVE_TOWARD ─
print("\n\n=== SEQUENTIAL with 'দিকে' (toward) but NO MOVE_TOWARD_TARGET ===")
wrong_toward = [r for r in seq_rows
                if 'দিকে' in r['Bangla Command']
                and 'কাছে' not in r['Bangla Command']
                and 'পাশ কাটিয়ে' not in r['Bangla Command']
                and 'সামনে' not in r['Bangla Command'].split('তারপর')[0]
                and '[ACTION:MOVE_TOWARD_TARGET]' not in r['Preprocessed Command']
                and '[ACTION:MOVE_IN_FRONT_TARGET]' not in r['Preprocessed Command']]
print(f"Count: {len(wrong_toward)}")
for r in wrong_toward[:8]:
    print(f"  ID={r['ID']}: {r['Bangla Command'][:70]}")
    print(f"          => {r['Preprocessed Command']}")
