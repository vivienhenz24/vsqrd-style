from pathlib import Path

import soundfile as sf


FILES = [
    Path("StyleTTS2/Data/tr_train.txt"),
    Path("StyleTTS2/Data/tr_val.txt"),
]


def iter_manifest_rows(manifest_path: Path):
    with manifest_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 3:
                yield line_no, None, f"bad column count: {len(parts)}"
                continue
            yield line_no, parts[0], None


def main():
    repo_root = Path.cwd()
    failures = 0

    for manifest in FILES:
        print(f"Checking {manifest} ...")
        for line_no, rel_path, parse_error in iter_manifest_rows(manifest):
            if parse_error:
                failures += 1
                print(f"  line {line_no}: {parse_error}")
                continue

            wav_path = (repo_root / rel_path).resolve()
            try:
                sf.info(str(wav_path))
            except Exception as exc:
                failures += 1
                print(f"  line {line_no}: {wav_path} -> {exc}")

    if failures:
        raise SystemExit(f"Found {failures} manifest/audio issues")

    print("All manifest entries are readable")


if __name__ == "__main__":
    main()
