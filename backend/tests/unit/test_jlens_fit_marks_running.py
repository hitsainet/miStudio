"""A fit must report `running` before its first prompt completes.

The progress row is opened as `queued`. Without an explicit mark, it only
flips when `on_progress` first fires — and that happens after a whole prompt's
Jacobians are accumulated across every fitted layer. Measured on gemma-4-12B
(2026-09-05): 3.4 minutes per prompt over 47 layers, so the panel read
"queued · 0%" for minutes while the GPU sat at 100% and 74 C.

jlens_acquire_tasks marks running in both of its entry points and finishes in
seconds; the fit task, which runs for hours, did not. This pins the asymmetry
shut.
"""

import ast
import pathlib

WORKERS = pathlib.Path(__file__).resolve().parents[2] / "src" / "workers"


def _calls_mark_running(path: pathlib.Path) -> bool:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "mark_running"):
            return True
    return False


def test_the_fit_task_marks_itself_running():
    """The load-bearing one: the longest job must not look queued."""
    assert _calls_mark_running(WORKERS / "jlens_fit_tasks.py"), (
        "jlens_fit_tasks does not call jlens_progress.mark_running, so a fit "
        "shows 'queued · 0%' until its first prompt finishes — minutes on a "
        "large model, and the whole run on one that stalls early."
    )


def test_mark_running_precedes_the_fit_call():
    """Marking AFTER fitter.fit() would be indistinguishable from not marking:
    fit() is the multi-hour call the mark exists to cover."""
    src = (WORKERS / "jlens_fit_tasks.py").read_text()
    mark = src.index("mark_running")
    fit_call = src.index("fitter.fit(")
    assert mark < fit_call, (
        "mark_running must be called BEFORE fitter.fit(); after it, the row "
        "stays 'queued' for the entire duration the mark is meant to cover"
    )


def test_every_long_running_jlens_task_marks_running():
    """Generalised, because fixing one representative and not the class is
    this repo's recorded anti-pattern."""
    long_running = ["jlens_fit_tasks.py", "jlens_acquire_tasks.py"]
    missing = [n for n in long_running
               if (WORKERS / n).exists() and not _calls_mark_running(WORKERS / n)]
    assert not missing, f"long-running jlens tasks not marking running: {missing}"
