import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import deal
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

ALLOWED_SOURCES = {"local_replay"}      # same allow-lists as MP5 contracts.py
ALLOWED_LICENSES = {"public_domain"}
DIM = 384                               # all-MiniLM-L6-v2
MAX_TOKENS = 256                        # model max_seq_length
CHUNKER_VERSION = "sentence-1.0"

HERE = Path(__file__).parent
_schemas = [json.loads((HERE / f).read_text()) for f in ("mp5_record.schema.json", "vdb_record.schema.json")]
VALIDATOR = Draft202012Validator(
    _schemas[1], registry=Registry().with_resources([(s["$id"], Resource.from_contents(s)) for s in _schemas]))


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def mp5_ok(r):
    """Same rules as MP5 check_events()."""
    try:
        duration = (datetime.fromisoformat(r["ts_end"]) - datetime.fromisoformat(r["ts_start"])).total_seconds()
        datetime.fromisoformat(r["ingest_ts"])
        return bool(r["record_id"].strip() and r["text"].strip()
                    and r["track"] in ("ASR", "VDB")
                    and r["source_id"] in ALLOWED_SOURCES
                    and r.get("license_class") in ALLOWED_LICENSES
                    and 0 < duration <= 30)
    except (KeyError, ValueError, AttributeError, TypeError):
        return False


def chunk_text(text, max_tokens=128):
    """Sentence-boundary chunking: pack whole sentences up to max_tokens words."""
    sentences = [m.span() for m in re.finditer(r"[^.!?]+[.!?]*", text) if text[m.start():m.end()].strip()]
    chunks, start, end, count = [], None, None, 0
    for s, e in sentences:
        n = len(text[s:e].split())
        if start is not None and count + n > max_tokens:
            chunks.append((start, end))
            start, count = None, 0
        if start is None:
            start = s + (len(text[s:e]) - len(text[s:e].lstrip()))
        end, count = e, count + n
    if start is not None:
        chunks.append((start, end))
    return chunks


# ------------------------------------------------------------------ invariants
def embedding_completeness(store):
    for cid, chunk in store.chunks.items():
        rec = store.vectors.get(cid)
        if rec is None:
            return False
        vec = rec["embedding"]["vector"]
        if len(vec) != DIM or not all(math.isfinite(x) for x in vec):
            return False
        if rec["provenance"]["content_sha256"] != sha(chunk["text"]):
            return False
    return True


def no_orphan_embeddings(store):
    return (all(cid in store.chunks for cid in store.vectors)
            and all(c["doc_id"] in store.docs for c in store.chunks.values()))


@deal.inv(embedding_completeness, message="EmbeddingCompleteness")
@deal.inv(no_orphan_embeddings, message="NoOrphanEmbeddings")
class VectorStore:
    def __init__(self):
        vars(self).update(docs={}, chunks={}, vectors={})   # one step, so the invariants never see a half-built store

    def ids_for(self, doc_id):
        return {cid for cid, c in self.chunks.items() if c["doc_id"] == doc_id}


# ------------------------------------------------------------------ contracts
def batch_ok(doc, records):
    for r in records:
        c = r["chunk"]
        if (r["doc_id"] != doc["record_id"]
                or r["text"] != doc["text"][c["char_start"]:c["char_end"]]
                or r["provenance"]["content_sha256"] != sha(r["text"])
                or r["provenance"]["doc_sha256"] != sha(doc["text"])):
            return False
    return True


# deal checks stacked contracts bottom-up
@deal.pre(lambda store, doc, records: all(
    len(r["embedding"]["vector"]) == DIM and all(math.isfinite(x) for x in r["embedding"]["vector"])
    for r in records), message="EmbeddingCompleteness: each record needs a finite 384-d vector")
@deal.pre(lambda store, doc, records: batch_ok(doc, records),
          message="provenance: records must match the document's text and hashes")
@deal.pre(lambda store, doc, records: all(VALIDATOR.is_valid(r) for r in records),
          message="every record must match vdb_record.schema.json")
@deal.pre(lambda store, doc, records: len(records) > 0, message="empty batch")
@deal.ensure(lambda store, doc, records, result: store.ids_for(doc["record_id"]) == {r["record_id"] for r in records},
             message="NoOrphanEmbeddings: old chunks of this document must be replaced, not left behind")
def vdb_upsert(store, doc, records):
    """Replace everything stored for `doc` with `records`."""
    for cid in store.ids_for(doc["record_id"]):     # drop old chunks first so they can't become orphans
        del store.chunks[cid], store.vectors[cid]
    store.docs[doc["record_id"]] = doc
    for r in records:
        store.chunks[r["record_id"]] = {"doc_id": doc["record_id"], "text": r["text"]}
        store.vectors[r["record_id"]] = r
    return len(records)


@deal.pre(lambda store, record, embed, model_id="test-model", model_revision="0": mp5_ok(record),
          message="record fails the MP5 admission rules")
@deal.ensure(lambda store, record, embed, model_id="test-model", model_revision="0", result=None:
             len(store.ids_for(record["record_id"])) == result,
             message="every chunk of the document must be stored")
def admit_document(store, record, embed, model_id="test-model", model_revision="0"):
    """Validate an MP5 record, chunk it, embed each chunk, and upsert it."""
    text = record["text"]
    records = []
    for i, (s, e) in enumerate(chunk_text(text)):
        chunk_txt = text[s:e]
        records.append({
            **{k: record[k] for k in ("track", "source_id", "ts_start", "ts_end", "ingest_ts", "license_class")},
            "record_id": f"{record['record_id']}#c{i}",
            "text": chunk_txt,
            "doc_id": record["record_id"],
            "chunk": {"strategy": "sentence_boundary", "index": i, "char_start": s, "char_end": e,
                      "token_count": len(chunk_txt.split())},
            "embedding": {"model_id": model_id, "model_revision": model_revision, "dim": DIM,
                          "vector": [float(x) for x in embed(chunk_txt)]},
            "provenance": {"doc_sha256": sha(text), "content_sha256": sha(chunk_txt),
                           "chunker_version": CHUNKER_VERSION,
                           "embedded_at": datetime.now(timezone.utc).isoformat()},
        })
    return vdb_upsert(store, record, records)
