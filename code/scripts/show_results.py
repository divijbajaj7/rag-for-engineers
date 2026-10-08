"""Print the running RAGAS scoreboard (from scripts/eval_results.json)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import print_scoreboard  # noqa: E402

if __name__ == "__main__":
    print_scoreboard()
