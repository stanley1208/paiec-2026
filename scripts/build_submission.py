"""Package a submission folder into dist/<folder name>.zip, with files at the zip root.

    python scripts/build_submission.py submissions/constant

The archive is byte-reproducible on any OS: fixed timestamps, permissions and
creator system, entries sorted by archive name, and LF line endings for text files.
"""

import argparse
import fnmatch
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".txt", ".json", ".md"}
FIXED_TIME = (1980, 1, 1, 0, 0, 0)
# Never packed: an API key, a private zip that holds one, or files written while running.
SECRET_OR_RUNTIME = ("anthropic_api_key.txt", "*_private.zip", "rating_cache", "sdk_check_markers",
                     "api_check_markers")


def build(source):
    source = Path(source).resolve()
    output = ROOT / "dist" / f"{source.name}.zip"
    files = sorted((p for p in source.rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts),
                   key=lambda p: p.relative_to(source).as_posix())
    if "model.py" not in [p.relative_to(source).as_posix() for p in files]:
        raise SystemExit(f"{source / 'model.py'} is missing")
    found = sorted({p.relative_to(source).as_posix() for p in files
                    if any(fnmatch.fnmatch(part, pattern) for part in p.relative_to(source).parts
                           for pattern in SECRET_OR_RUNTIME)})
    if found:
        raise SystemExit(f"refusing to build: remove {', '.join(found)} from {source} first")
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            data = path.read_bytes()
            if path.suffix in TEXT_SUFFIXES:
                data = data.replace(b"\r\n", b"\n")
            info = zipfile.ZipInfo(path.relative_to(source).as_posix(), FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            info.create_system = 3  # Unix, so the hash does not depend on the build OS
            zf.writestr(info, data)
    return output, files


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="submission folder containing model.py")
    source = parser.parse_args().source.resolve()
    output, files = build(source)
    print(f"Wrote {output.relative_to(ROOT).as_posix()} ({output.stat().st_size} bytes)")
    for path in files:
        print(f"  {path.relative_to(source).as_posix()}")
    print(f"sha256 {hashlib.sha256(output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
