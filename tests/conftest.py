"""Where the tests find an event store, and what happens when they cannot.

Three test modules need a real store, and each used to name the same absolute path itself.
That was three places to edit and, more to the point, one hardcoded pileup condition: the
same tests are worth running against a pu200 smoke store, and there was no way to say so.

    SMOKE_STORE=/path/to/a/small/store python -m pytest tests

The default is the pu0 smoke store on the cluster. Without it the store-backed tests skip
rather than fail, because the rest of the suite is pure numpy and has to stay runnable on a
laptop with no dataset -- the same property that lets an assessor regenerate the figures. Run
with ``-rs`` to list the skips, so "no store" cannot be read as "everything passed".
"""

import os
from functools import lru_cache
from pathlib import Path

import pytest

from src.io.event_store import EventStore

DEFAULT_SMOKE_STORE = Path("/home/xucapfwi/eventstore_smoke/ttbar_pu0_20250_20255_v1")

#: A handful of events is enough. Every test that uses it is about the scorer's behaviour on
#: real data rather than about a measurement, so more events buy nothing and cost runtime.
SMOKE_STORE = Path(os.environ.get("SMOKE_STORE", DEFAULT_SMOKE_STORE))


@lru_cache(maxsize=1)
def _open() -> EventStore:
    """Open the smoke store once for the whole session.

    Cached because opening one validates its metadata and memory-maps every chunk, and the
    two dozen call sites all want the same store.

    No ``expect=`` on purpose. These tests are about the scorer, and passing the config's
    expectations would make them fail whenever the active dataset in `config/experiment.yaml`
    moved on from whatever the smoke store was dumped with -- reporting a config edit as a
    scorer regression. The contract check has its own test.
    """
    return EventStore(SMOKE_STORE)


def open_smoke_store() -> EventStore:
    """The smoke store, or a skip if it is not on this machine."""
    if not SMOKE_STORE.exists():
        pytest.skip(
            f"no event store at {SMOKE_STORE}; set SMOKE_STORE=/path/to/a/store to run the "
            f"store-backed tests"
        )
    return _open()


@pytest.fixture(scope="session")
def store() -> EventStore:
    """The smoke store as a fixture, for tests written to take one."""
    return open_smoke_store()


@pytest.fixture(scope="session")
def record(store: EventStore):
    """The first event of the smoke store."""
    return store[0]
