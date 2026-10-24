"""Blend evaluation of a probe family (Feature 041 S2; FTDD §6.3, T1).

`candidates` (the vocabulary), `engine` (pure NumPy maths, no session, no file), `service` (reads, refusals,
admission, assembly) and `report_format` (the canonical bytes). Callers import the modules; this package
re-exports nothing, so a call into the engine is always visible at its call site.
"""
