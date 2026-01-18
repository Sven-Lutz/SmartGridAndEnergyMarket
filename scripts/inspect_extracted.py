# scripts/inspect_extracted.py
from __future__ import annotations

import sys

from src.cli import main as cli_main


if __name__ == "__main__":
    # Beispiel-Aufruf:
    #   python scripts/inspect_extracted.py --scope de_by --limit 5
    cli_args = ["inspect-extracted", *sys.argv[1:]]
    cli_main(cli_args)
