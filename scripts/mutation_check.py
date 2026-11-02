"""Mutation check: break the change-detection code on purpose, one bug at a time, and confirm the tests
notice each one. A test suite that passes no matter what proves nothing.

    python scripts/mutation_check.py      # from the repo root; restores every file afterwards
"""

import pathlib
import subprocess
import sys

B = chr(92)
MUTATIONS = [
    (
        "no number/negation cap",
        "src/answergate/runner.py",
        "        score = min(score, 0.5)",
        "        score = score",
    ),
    (
        "negation regex without word boundaries",
        "src/answergate/runner.py",
        'r"' + B + "b(not|no|never|none|cannot|can't|don't|doesn't|isn't|aren't|won't|without)" + B + 'b"',
        "r\"(not|no|never|none|cannot|can't|don't|doesn't|isn't|aren't|won't|without)\"",
    ),
    (
        "no normalization of 0 vs 0.0",
        "src/answergate/targets/parse.py",
        '        return f"{v:g}"',
        "        return str(v)",
    ),
    (
        "pipeline changes never computed",
        "src/answergate/baseline.py",
        "    v.pipeline_changes = pipeline_changes(current, baseline)",
        "    v.pipeline_changes = []",
    ),
    (
        "repeats ignored (always one ask)",
        "src/answergate/runner.py",
        "        if res.error is None and cfg.repeats > 1 and not case.expect_refusal:",
        "        if False:",
    ),
    (
        "old value's pipes not escaped",
        "src/answergate/report/__init__.py",
        "{_cell(c.before, 60) or",
        "{c.before or",
    ),
    (
        "new value's pipes not escaped",
        "src/answergate/report/__init__.py",
        "{_cell(c.after, 60) or",
        "{c.after or",
    ),
    (
        "no cap on how many changes are listed",
        "src/answergate/report/__init__.py",
        "listed = verdict.pipeline_changes[:10]",
        "listed = verdict.pipeline_changes",
    ),
    (
        "wrong reason when repeats lowered",
        "src/answergate/baseline.py",
        '"set repeats: 2 or more, as in the baseline" if name == "consistency" else "is the judge off?"',
        '"is the judge off?"',
    ),
    (
        "refusal questions asked repeatedly",
        "src/answergate/runner.py",
        "cfg.repeats > 1 and not case.expect_refusal:",
        "cfg.repeats > 1:",
    ),
    (
        "a failed repeat still scored",
        "src/answergate/runner.py",
        '                res.notes["consistency"] = f"Not scored: a repeat failed ({again.error})"[:300]\n                return',
        "                continue",
    ),
    (
        "PR comment drops the changes line",
        "src/answergate/report/__init__.py",
        '            f"**What changed in the pipeline:** {changed}" + (f" · and {more} more" if more else ""),',
        '            "",',
    ),
    (
        "CLI drops the changes",
        "src/answergate/cli.py",
        "    if verdict.pipeline_changes:\n        typer.secho",
        "    if False:\n        typer.secho",
    ),
    (
        "a broken run accepted as the baseline",
        "src/answergate/cli.py",
        "    if problems and not force:",
        "    if False:",
    ),
    (
        "settings varying between answers not flagged",
        "src/answergate/runner.py",
        'out[k] = value if len(counts) == 1 else f"{value} (varied: {len(counts)} values)"',
        "out[k] = value",
    ),
]
caught = 0
for name, path, a, b in MUTATIONS:
    p = pathlib.Path(path)
    orig = p.read_text(encoding="utf-8")
    assert a in orig, (name, a)
    p.write_text(orig.replace(a, b, 1), encoding="utf-8", newline="")
    try:
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "tests/test_changes_adversarial.py",
                "tests/test_changes.py",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
        )
        caught += bool(r.returncode)
        print(("CAUGHT  " if r.returncode else "MISSED  ") + f"{name}: {r.stdout.strip().splitlines()[-1]}")
    finally:
        p.write_text(orig, encoding="utf-8", newline="")
print(f"{caught}/{len(MUTATIONS)} deliberate bugs caught")
