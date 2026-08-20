"""Where the store-backed tests find an event store, and what happens when they cannot.

    SMOKE_STORE=/path/to/a/small/store python -m pytest

Without that variable the store-backed tests skip rather than fail, because the rest of the
suite is pure numpy and has to stay runnable on a laptop with no dataset. Run with ``-rs`` to
list the skips, so "no store" cannot be read as "everything passed".
"""

import os
from functools import lru_cache
from pathlib import Path

import pytest

from src.io.event_store import EventStore

#: A handful of events is enough: every test using it is about the scorer's behaviour on real
#: data rather than about a measurement.
_CONFIGURED = os.environ.get("SMOKE_STORE")
SMOKE_STORE = Path(_CONFIGURED) if _CONFIGURED else None


@lru_cache(maxsize=1)
def _open() -> EventStore:
    """Open the smoke store once for the whole session.

    No ``expect=``: these tests are about the scorer, and passing the config's expectations
    would report a config edit as a scorer regression. The contract check has its own test.
    """
    return EventStore(SMOKE_STORE)


def open_smoke_store() -> EventStore:
    """The smoke store, or a skip when SMOKE_STORE is unset or does not exist."""
    if SMOKE_STORE is None:
        pytest.skip("set SMOKE_STORE=/path/to/an/event/store to run the store-backed tests")
    if not SMOKE_STORE.exists():
        pytest.skip(f"no event store at {SMOKE_STORE} (from SMOKE_STORE)")
    return _open()


@pytest.fixture(scope="session")
def store() -> EventStore:
    """The smoke store as a fixture, for tests written to take one."""
    return open_smoke_store()


@pytest.fixture(scope="session")
def record(store: EventStore):
    """The first event of the smoke store."""
    return store[0]
