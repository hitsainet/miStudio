"""One source of "now", and it is timezone-aware.

MIS-E2E-054. The deprecated naive-UTC call appeared at 37 sites. Its
deprecation in Python 3.12 was the smaller half of the problem.

Every one of this schema's 62 `DateTime` columns is declared
`DateTime(timezone=True)` — `timestamptz` in Postgres. The old call returned a
**naive** datetime, and Postgres interprets a naive value as being in the
session's timezone. On a server whose session TZ is not UTC, the stored instant
is silently shifted; nothing raises, and the row looks plausible. The same
mismatch bites in Python: a value read back out of one of those columns is
aware, so comparing it against a naive "now" raises `TypeError: can't compare
offset-naive and offset-aware datetimes`.

`utc_now()` returns an aware UTC datetime, which is what the columns were
declared to hold. Keeping it in one function means the next timezone decision
is made once rather than 37 times.
"""

from datetime import datetime, timezone


def utc_now() -> datetime:
    """The current instant, in UTC, timezone-aware."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """`utc_now()` as an ISO-8601 string ending in `Z`.

    The 37 sites included several hand-rolled `.isoformat() + "Z"`
    constructions. Those were wrong twice over: the value was naive, and the
    appended `Z` asserted a UTC offset the object did not carry, so the string
    claimed a precision the value did not have.
    """
    return utc_now().isoformat().replace("+00:00", "Z")
