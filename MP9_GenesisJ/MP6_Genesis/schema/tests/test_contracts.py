import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import deal
import pytest

SCHEMA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCHEMA))
from contracts import VALIDATOR, VectorStore, admit_document, vdb_upsert  # noqa: E402


def fake_embed(text):
    """Deterministic stand-in for the real model: 384 numbers, unit length."""
    raw = [b - 127.5 for b in hashlib.sha256(text.encode()).digest() * 12]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


@pytest.fixture
def corpus():
    return json.loads((SCHEMA / "examples" / "corpus.json").read_text())


@pytest.fixture
def transcript():
    # MP5 transcripts have no licence; PS6 requires one.
    return {**json.loads((SCHEMA / "examples" / "transcripts.json").read_text()), "license_class": "public_domain"}


@pytest.fixture
def store():
    return VectorStore()


def test_admit_mp5_records(store, corpus, transcript):
    assert admit_document(store, corpus, fake_embed) >= 1
    assert admit_document(store, transcript, fake_embed) >= 1
    for rec in store.vectors.values():
        assert VALIDATOR.is_valid(rec)


def test_reject_bad_mp5_record(store, corpus):
    with pytest.raises(deal.PreContractError):
        admit_document(store, {**corpus, "ts_end": "2026-10-20T12:01:45"}, fake_embed)  # 45 s > 30 s


def test_reject_wrong_vector_size(store, corpus):
    with pytest.raises(deal.PreContractError):
        admit_document(store, corpus, lambda t: fake_embed(t)[:100])


def test_reject_nan_vector(store, corpus):
    admit_document(store, corpus, fake_embed)
    recs = copy.deepcopy(list(store.vectors.values()))
    recs[0]["embedding"]["vector"][0] = float("nan")
    with pytest.raises(deal.PreContractError):
        vdb_upsert(store, corpus, recs)


def test_reject_text_that_does_not_match_document(store, corpus):
    admit_document(store, corpus, fake_embed)
    recs = copy.deepcopy(list(store.vectors.values()))
    recs[0]["text"] += " extra"
    with pytest.raises(deal.PreContractError):
        vdb_upsert(store, corpus, recs)


def test_readmit_replaces_old_chunks(store, corpus):
    admit_document(store, corpus, fake_embed)
    edited = {**corpus, "text": "BTS has seven members."}
    admit_document(store, edited, fake_embed)
    assert [c["text"] for c in store.chunks.values()] == ["BTS has seven members."]


def test_invariant_catches_orphan(store, corpus):
    admit_document(store, corpus, fake_embed)
    with pytest.raises(deal.InvContractError):
        store.docs = {}          # chunks now point at a missing document


def test_invariant_catches_missing_vector(store, corpus):
    admit_document(store, corpus, fake_embed)
    with pytest.raises(deal.InvContractError):
        store.vectors = {}       # chunks without vectors
