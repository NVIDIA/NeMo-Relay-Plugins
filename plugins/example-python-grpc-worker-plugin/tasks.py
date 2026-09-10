# SPDX-License-Identifier: Apache-2.0
"""Plugin-owned entrypoint; shared recipes implement this example's build contract."""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["REPO_DIR"])
from scripts.recipes import run_recipe

if __name__ == "__main__":
    run_recipe(Path(__file__).parent.name, sys.argv[1])
