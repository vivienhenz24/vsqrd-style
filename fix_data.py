def clean(infile):
    with open(infile, encoding='utf-8') as f:
        lines = f.read().split('\n')
    cleaned = []
    skipped = 0
    for line in lines:
        if not line.strip():
            continue
        parts = line.split('|')
        if len(parts) != 3:
            skipped += 1
            continue
        path, phonemes, spk = parts
        if not path.endswith('.wav') or '\\' in path:
            skipped += 1
            continue
        phonemes = phonemes.replace('\r', ' ').replace('  ', ' ').strip()
        cleaned.append(f'{path}|{phonemes}|{spk}')
    with open(infile, 'w', encoding='utf-8') as f:
        f.write('\n'.join(cleaned))
    print(f'{infile}: {len(cleaned)} kept, {skipped} skipped')

clean('StyleTTS2/Data/tr_train.txt')
clean('StyleTTS2/Data/tr_val.txt')
