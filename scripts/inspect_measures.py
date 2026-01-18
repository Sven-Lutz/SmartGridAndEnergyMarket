#!/usr/bin/env python
from __future__ import annotations

import sys
from src.cli import main as cli_main

if __name__ == "__main__":
    # Beispiel:
    # python scripts/inspect_measures.py --scope de_by --samples 5
    cli_args = ["inspect-measures", *sys.argv[1:]]
    cli_main(cli_args)
