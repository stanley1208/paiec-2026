"""Write the public copy of this repository (committed files only) for release.

    python scripts/export_public.py [out]          # default: dist/public

Takes the files of the current commit (git archive HEAD), so uncommitted changes and
git-ignored files (data, private feedback, keys, local runs) never reach it, and leaves out:
- third_party/paiec_starting_kit: the official starting kit has no license; a README there tells
  users to download it from Codabench;
- internal notes on writing the report (the writing guide, fact sheet, outline, research notes,
  reference checks, style checker and plain-text drafts).
If out already holds a git repository, its .git folder is kept and everything else is replaced,
so the public history can continue across exports.
"""

import argparse
import io
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = ("third_party/paiec_starting_kit/",
           "docs/report/ai_tell_check.py", "docs/report/factsheet.json", "docs/report/outline_draft1.md",
           "docs/report/references_checked.json", "docs/report/research_notes.json", "docs/report/rewrite/",
           "docs/report/tex_to_draft.py", "docs/report/writing_guide.md")
KIT_README = """# Official starting kit (not included)

The harness imports the acquisition code of the official PAIEC starting kit, which ships without a
license, so it is not redistributed here. Download `starting_kit.zip` from the Codabench
competition page (competition 17828; build `5f7b7bf`; SHA-256
`a870970a7b9e82385929db37435aac29e3f0ee648cce7df143787b88ac4770f6`) and copy its `tools/` and
`sample_data/` folders into this folder, so that `tools/streaming_ingestion.py` sits next to this
file.
"""
KIT_IGNORE = "\n# Official starting kit, copied in by the user (no license to redistribute)\n" \
             "third_party/paiec_starting_kit/tools/\nthird_party/paiec_starting_kit/sample_data/\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, nargs="?", default=ROOT / "dist" / "public")
    args = parser.parse_args()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        print("note: uncommitted changes are not exported", file=sys.stderr)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for child in out.iterdir():
        if child.name != ".git":
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    archive = subprocess.run(["git", "archive", "--format=tar", "HEAD"], cwd=ROOT, capture_output=True,
                             check=True).stdout
    kept = 0
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            if member.isfile() and not member.name.startswith(EXCLUDE):
                tar.extract(member, out, filter="data")
                kept += 1
    kit = out / "third_party" / "paiec_starting_kit"
    kit.mkdir(parents=True, exist_ok=True)
    (kit / "README.md").write_text(KIT_README, encoding="utf-8", newline="\n")
    with (out / ".gitignore").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(KIT_IGNORE)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()
    print(f"wrote {kept} files of commit {head} to {out}")


if __name__ == "__main__":
    main()
