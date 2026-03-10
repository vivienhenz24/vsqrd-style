for f in ['StyleTTS2/Data/tr_train.txt', 'StyleTTS2/Data/tr_val.txt']:
    lines = [l for l in open(f).read().split('\n') if l.strip()]
    bad = sum(1 for l in lines if len(l.split('|')) != 3)
    cr = sum(1 for l in lines if '\r' in l)
    print(f, len(lines), 'bad:', bad, 'cr:', cr)
