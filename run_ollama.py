#!/usr/bin/env python3
"""Compatibility wrapper for the new package CLI."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from llm_soc_nav.__main__ import main


if __name__ == "__main__":
    main()
