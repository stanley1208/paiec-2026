"""Download the pinned measurement-db tables into data/ (needs a Hugging Face login)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paiec import data  # noqa: E402


def main():
    manifest = data.download()
    print(f"Downloaded {len(manifest['files'])} files at revision {data.REVISION}")
    print(f"into {data.DATA_DIR}")


if __name__ == "__main__":
    main()
