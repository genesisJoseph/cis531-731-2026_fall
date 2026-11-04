"""MP7 classifiers: prototype, kNN over FAISS IndexFlatIP, and few-shot LLM behind one interface.

deal contracts on predict_proba check that probabilities are non-negative, rows sum to 1
within 1e-6, and there is one column per declared label; predict() may only return
declared labels. Everything runs on CPU.
"""
import copy
import json
import platform
import sys
import time
from pathlib import Path

import deal
import faiss
import numpy as np

from prompt_template import LABELS, LABEL_IDS, LETTERS, N_LABELS, SYSTEM, format_demo, format_options

ROOT = Path(__file__).resolve().parent


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _valid_proba(p):
    p = np.asarray(p)
    return (p.ndim == 2 and p.shape[1] == N_LABELS and np.isfinite(p).all()
            and (p >= 0).all() and np.allclose(p.sum(axis=1), 1.0, atol=1e-6, rtol=0))


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def resolve_revision(model_id, model, revision=None):
    """Exact Hugging Face commit hash of the loaded weights."""
    sha = getattr(model.config, "_commit_hash", None)
    if not sha and not Path(model_id).exists():
        try:
            from huggingface_hub import HfApi
            sha = HfApi().model_info(model_id, revision=revision).sha
        except Exception:  # noqa: BLE001  (offline)
            sha = None
    return sha or revision or "local"


# --------------------------------------------------------------------- embedder
class Embedder:
    """Mean-pooled, L2-normalised sentence embeddings (all-MiniLM-L6-v2 by default)."""

    def __init__(self, model_id="sentence-transformers/all-MiniLM-L6-v2", revision=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModel.from_pretrained(model_id, revision=revision).eval().to("cpu")
        self.model_id = model_id
        self.revision = resolve_revision(model_id, self.model, revision)

    def encode(self, texts, batch_size=32):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tok(list(texts[i:i + batch_size]), padding=True, truncation=True,
                               max_length=256, return_tensors="pt")
                h = self.model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                out.append(self.torch.nn.functional.normalize(v, dim=-1).numpy())
        return np.vstack(out).astype("float32")


# ------------------------------------------------------------ classifier API
class Classifier:
    """Unified interface: fit(exemplars) then predict_proba(texts) / predict(texts)."""
    name = "base"

    def fit(self, exemplars):
        raise NotImplementedError

    def _scores(self, texts):
        raise NotImplementedError

    @deal.post(_valid_proba, message="predict_proba: non-negative, rows sum to 1 (1e-6), one column per label")
    def predict_proba(self, texts):
        return self._scores(texts)

    @deal.post(lambda result: all(lbl in LABEL_IDS for lbl in result),
               message="predict: labels must come from the declared label space")
    def predict(self, texts):
        return [LABEL_IDS[i] for i in self.predict_proba(texts).argmax(axis=1)]


def label_prompts():
    """Text used for k = 0 (no exemplars): one description per label."""
    return [f"A song about {desc}." for desc in LABELS.values()]


class PrototypeClassifier(Classifier):
    name = "prototype"

    def __init__(self, embedder, temperature=20.0):
        self.emb, self.t = embedder, temperature

    def fit(self, exemplars):
        if exemplars:
            X = self.emb.encode([e["text"] for e in exemplars])
            y = np.array([LABEL_IDS.index(e["label"]) for e in exemplars])
            protos = np.vstack([X[y == i].mean(0) for i in range(N_LABELS)])
        else:  # k = 0: label descriptions act as prototypes
            protos = self.emb.encode(label_prompts())
        self.protos = protos / np.linalg.norm(protos, axis=1, keepdims=True)
        return self

    def _scores(self, texts):
        return softmax(self.t * (self.emb.encode(texts) @ self.protos.T))


class KNNClassifier(Classifier):
    """kNN over a FAISS IndexFlatIP exemplar index; similarity-weighted votes."""
    name = "knn"

    def __init__(self, embedder, n_neighbors=5, temperature=20.0, vectors=None):
        self.emb, self.n, self.t = embedder, n_neighbors, temperature
        self.vectors = vectors or {}    # record_id -> admitted vector (from the PS6 store)

    def fit(self, exemplars):
        if exemplars:
            X = np.vstack([self.vectors[e["record_id"]] if e["record_id"] in self.vectors
                           else self.emb.encode([e["text"]])[0] for e in exemplars]).astype("float32")
            self.y = np.array([LABEL_IDS.index(e["label"]) for e in exemplars])
        else:  # k = 0: one pseudo-exemplar per label description
            X, self.y = self.emb.encode(label_prompts()), np.arange(N_LABELS)
        self.index = faiss.IndexFlatIP(X.shape[1])
        self.index.add(np.ascontiguousarray(X))
        return self

    def _scores(self, texts):
        q = self.emb.encode(texts)
        k = min(self.n, self.index.ntotal)
        sims, idx = self.index.search(q, k)
        votes = np.zeros((len(texts), N_LABELS))
        w = np.exp(self.t * (sims - sims.max(axis=1, keepdims=True)))
        for r in range(len(texts)):
            np.add.at(votes[r], self.y[idx[r]], w[r])
        return votes / votes.sum(axis=1, keepdims=True)


class FewShotLLMClassifier(Classifier):
    """Fixed, class-balanced demonstrations; constrained decoding over the label letters.

    The model is run once per query. Only the next-token logits of the letters A..H are
    read (all other tokens are masked out), so the output can never be outside the label
    space. Optional contextual calibration divides by the prior on a content-free input.
    """
    name = "llm"

    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct", revision=None, calibrate=True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision).float().eval().to("cpu")
        self.model_id = model_id
        self.revision = resolve_revision(model_id, self.model, revision)
        self.letter_ids = []
        for L in LETTERS:
            ids = self.tok.encode(L, add_special_tokens=False)
            assert len(ids) == 1, f"verbalizer {L!r} is not a single token"
            self.letter_ids.append(ids[0])
        self.calibrate = calibrate
        self.forward_calls = 0
        self.prompt_tokens = 0

    def build_prompt(self, demos, query):
        """Return (prefix, suffix). The prefix (system + options + demos) is shared by all queries."""
        user_head = "Themes:\n" + format_options() + "\n\n" + "".join(format_demo(d["text"], d["label"]) for d in demos)
        marker = "␟"   # split point, never present in data
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_head + marker + f"Description: {query}\nTheme:"}]
        full = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prefix, suffix = full.split(marker)
        return prefix, suffix

    def fit(self, exemplars):
        self.demos = list(exemplars)       # order already permuted by the sweep (position-bias control)
        prefix, _ = self.build_prompt(self.demos, "")
        ids = self.tok(prefix, return_tensors="pt", add_special_tokens=False).input_ids
        with self.torch.no_grad():
            self.prefix_cache = self.model(ids, use_cache=True).past_key_values
        self.prefix_len = ids.shape[1]
        self.forward_calls += 1
        self.prompt_tokens += self.prefix_len
        self.prior = self._letter_probs(["N/A"])[0] if self.calibrate else np.ones(N_LABELS) / N_LABELS
        return self

    def _letter_probs(self, texts):
        out = []
        for text in texts:
            _, suffix = self.build_prompt(self.demos, text)
            ids = self.tok(suffix, return_tensors="pt", add_special_tokens=False).input_ids
            with self.torch.no_grad():
                logits = self.model(ids, past_key_values=copy.deepcopy(self.prefix_cache),
                                    use_cache=True).logits[0, -1]
            self.forward_calls += 1
            self.prompt_tokens += ids.shape[1]
            out.append(softmax(logits[self.letter_ids].double().numpy()))
        return np.vstack(out)

    def _scores(self, texts):
        p = self._letter_probs(texts) / self.prior        # contextual calibration (majority-label bias)
        return p / p.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- the sweep
def sample_demos(pool, k, seed):
    """k exemplars per label (class-balanced), then a seed-specific random order."""
    rng = np.random.default_rng(seed)
    chosen = []
    for lbl in LABEL_IDS:
        items = [e for e in pool if e["label"] == lbl]
        chosen += [items[i] for i in rng.choice(len(items), size=k, replace=False)]
    order = rng.permutation(len(chosen))           # demonstration-order permutation per seed
    return [chosen[i] for i in order]


def check_no_leakage(demos, eval_rows):
    eval_ids = {r["record_id"] for r in eval_rows}
    eval_texts = {r["text"].strip().lower() for r in eval_rows}
    assert not eval_ids & {d["record_id"] for d in demos}, "label leakage: eval item used as demo"
    assert not eval_texts & {d["text"].strip().lower() for d in demos}, "label leakage: duplicate text"


def run_sweep(classifiers, pool, eval_rows, ks=(0, 1, 3, 5, 8), seeds=(0, 1, 2, 3, 4), log=print):
    preds = []
    texts = [r["text"] for r in eval_rows]
    for clf in classifiers:
        for k in ks:
            for seed in seeds:
                demos = sample_demos(pool, k, seed)
                check_no_leakage(demos, eval_rows)
                t0 = time.time()
                if k == 0 and seed != seeds[0]:
                    proba = first_k0   # k = 0 has no demonstrations, so every seed is identical
                else:
                    proba = clf.fit(demos).predict_proba(texts)
                if k == 0:
                    first_k0 = proba
                for r, p in zip(eval_rows, proba):
                    preds.append({"classifier": clf.name, "k": k, "seed": seed, "record_id": r["record_id"],
                                  "gold": r["label"], "pred": LABEL_IDS[int(np.argmax(p))],
                                  "proba": {lbl: round(float(x), 6) for lbl, x in zip(LABEL_IDS, p)},
                                  "demo_ids": [d["record_id"] for d in demos]})
                log(f"{clf.name:9s} k={k} seed={seed}  {time.time() - t0:5.1f}s")
    return preds


def environment(models):
    import importlib.metadata as md
    pkgs = ["torch", "transformers", "faiss-cpu", "deal", "numpy", "jsonschema"]
    vers = {}
    for p in pkgs:
        try:
            vers[p] = md.version(p)
        except md.PackageNotFoundError:
            vers[p] = "not installed"
    try:
        import torch
        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": vers,
            "device": "cpu", "cuda_available_but_unused": cuda,
            "models": [{"role": r, "id": m.model_id, "revision": m.revision} for r, m in models]}
