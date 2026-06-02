# /// script
# dependencies = ["lazy_loader"]
# ///

"""Benchmark llamabot import times for different usage patterns.

Run with: pixi run python scripts/benchmark_imports.py

With --ci: output JSON to benchmark-results.json for CI reporting.

Each benchmark is run in a subprocess to get a clean import
(no cached modules from previous benchmarks).
"""

import json
import subprocess
import sys
import textwrap


def bench(label: str, code: str) -> float:
    """Run *code* in a fresh subprocess and return the elapsed time."""
    script = textwrap.dedent(
        f"""\
        import time
        t = time.perf_counter()
        {code}
        elapsed = time.perf_counter() - t
        print(f"{{elapsed:.3f}}")
    """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  {label:50s} FAILED")
        print(f"    stderr: {result.stderr.strip()}")
        return -1.0
    elapsed = float(result.stdout.strip().splitlines()[-1])
    return elapsed


BENCHMARKS = [
    ("import llamabot", "import llamabot"),
    ("from llamabot import SimpleBot", "from llamabot import SimpleBot"),
    ("from llamabot import AgentBot", "from llamabot import AgentBot"),
    ("from llamabot import ToolBot", "from llamabot import ToolBot"),
    ("from llamabot import tool", "from llamabot import tool"),
    ("from llamabot import prompt", "from llamabot import prompt"),
    ("from llamabot import user, system", "from llamabot import user, system"),
    ("from llamabot import span", "from llamabot import span"),
    ("from llamabot import ChatMemory", "from llamabot import ChatMemory"),
    ("from llamabot import Experiment", "from llamabot import Experiment"),
    ("from llamabot import QueryBot", "from llamabot import QueryBot"),
    ("from llamabot import ImageBot", "from llamabot import ImageBot"),
    ("from llamabot import StructuredBot", "from llamabot import StructuredBot"),
    ("from llamabot.recorder import span", "from llamabot.recorder import span"),
    (
        "full (import everything)",
        "import llamabot; [getattr(llamabot, n) for n in llamabot.__all__]",
    ),
]


def main():
    ci_mode = "--ci" in sys.argv
    results = []

    print("=== llamabot Import Benchmarks ===")
    print(f"{'Import':50s} {'Time':>8s}")
    print("-" * 60)
    for label, code in BENCHMARKS:
        elapsed = bench(label, code)
        if elapsed >= 0:
            print(f"  {label:48s} {elapsed:>6.3f}s")
            results.append({"label": label, "time_s": round(elapsed, 3)})

    print()
    print("All benchmarks use fresh subprocesses (no module caching).")

    if ci_mode:
        with open("benchmark-results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nResults written to benchmark-results.json")


if __name__ == "__main__":
    main()
