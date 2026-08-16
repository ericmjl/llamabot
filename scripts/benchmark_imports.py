# /// script
# dependencies = ["lazy_loader"]
# ///

"""Benchmark llamabot import times for different usage patterns.

Run with: pixi run python scripts/benchmark_imports.py

With --ci: output JSON to benchmark-results.json for CI reporting.

Each benchmark is run in a subprocess to get a clean import
(no cached modules from previous benchmarks). Each import is sampled
3 times and the fastest sample is reported: CI runners are shared,
noisy hardware, and single samples wobble by hundreds of milliseconds.
The minimum is the stable estimate of the import cost itself.

A warmup subprocess (``python -c pass``) runs before the benchmarks so
that the first timed row does not pay one-time costs (interpreter
binary + stdlib cold reads) that have nothing to do with llamabot.
"""

import json
import subprocess
import sys
import textwrap


def bench(label: str, code: str, samples: int = 3) -> float:
    """Run *code* in fresh subprocesses and return the fastest elapsed time.

    :param label: Display label for the benchmark.
    :param code: Python code whose import is timed.
    :param samples: Number of subprocess samples to take; the minimum is kept.
    :return: Best elapsed time in seconds, or -1.0 if every sample failed.
    """
    script = textwrap.dedent(
        f"""\
        import time
        t = time.perf_counter()
        {code}
        elapsed = time.perf_counter() - t
        print(f"{{elapsed:.3f}}")
    """
    )
    best = -1.0
    for _ in range(samples):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  {label:50s} FAILED")
            print(f"    stderr: {result.stderr.strip()}")
            continue
        elapsed = float(result.stdout.strip().splitlines()[-1])
        if best < 0 or elapsed < best:
            best = elapsed
    return best


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


def warmup_interpreter() -> None:
    """Run one untimed subprocess so cold interpreter/stdlib reads do not land on row 1.

    Loading the Python binary and standard library from a cold file system
    costs hundreds of milliseconds and is identical for every Python
    program; charging it to the first benchmark row is measurement noise.
    """
    subprocess.run(
        [sys.executable, "-c", "pass"],
        capture_output=True,
        text=True,
    )


def main():
    ci_mode = "--ci" in sys.argv
    results = []

    warmup_interpreter()

    print("=== llamabot Import Benchmarks ===")
    print(f"{'Import':50s} {'Time':>8s}")
    print("-" * 60)
    for label, code in BENCHMARKS:
        elapsed = bench(label, code)
        if elapsed >= 0:
            print(f"  {label:48s} {elapsed:>6.3f}s")
            results.append({"label": label, "time_s": round(elapsed, 3)})

    print()
    print("All benchmarks use fresh subprocesses (no module caching);")
    print("each import is sampled 3 times and the fastest sample is reported.")

    if ci_mode:
        with open("benchmark-results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nResults written to benchmark-results.json")


if __name__ == "__main__":
    main()
