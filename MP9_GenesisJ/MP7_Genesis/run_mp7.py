"""Run the full MP7 pipeline: admit exemplars, build classifiers, sweep, write outputs.

    python run_mp7.py            # real models (downloads all-MiniLM-L6-v2 and Qwen2.5-0.5B-Instruct)
Outputs: predictions.jsonl, results/summary.json, results/run_log.json, report.md
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
# PS6 contracts: reuse ../MP6_Genesis/schema from the PS6 branch
sys.path.insert(0, str(ROOT.parent / "MP6_Genesis" / "schema"))
import mp7lib as M  # noqa: E402

EMB_MODEL = os.getenv("MP7_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("MP7_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
KS = (0, 1, 3, 5, 8)
SEEDS = (0, 1, 2, 3, 4)


def admit_exemplars(pool, embedder):
    """Admit exemplar records through the PS6 contracts; return record_id -> admitted vector."""
    from contracts import VectorStore, admit_document
    store = VectorStore()
    for rec in pool:
        doc = {k: v for k, v in rec.items() if k != "label"}
        admit_document(store, doc, lambda t: embedder.encode([t])[0], model_id=embedder.model_id,
                       model_revision=embedder.revision)
    vecs = {}
    for r in store.vectors.values():
        vecs[r["doc_id"]] = np.asarray(r["embedding"]["vector"], dtype="float32")
    return vecs


def main(embedder=None, llm=None, ks=KS, seeds=SEEDS, B=1000, out_dir=ROOT, log=print):
    out_dir = Path(out_dir)
    (out_dir / "results").mkdir(exist_ok=True)
    pool = M.load_jsonl(ROOT / "data" / "exemplars.jsonl")
    eval_rows = M.load_jsonl(ROOT / "data" / "eval_mp7.jsonl")
    eval_ids = [r["record_id"] for r in eval_rows]
    t_start = time.time()

    embedder = embedder or M.Embedder(EMB_MODEL)
    vectors = admit_exemplars(pool, embedder)
    log(f"admitted {len(vectors)} exemplar chunks through the PS6 contracts")
    llm = llm or M.FewShotLLMClassifier(LLM_MODEL)
    classifiers = [M.PrototypeClassifier(embedder), M.KNNClassifier(embedder, vectors=vectors), llm]

    preds = M.run_sweep(classifiers, pool, eval_rows, ks=ks, seeds=seeds, log=log)
    with open(out_dir / "predictions.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")

    summary = M.summarize(preds, eval_ids, B=B)
    best = max(summary, key=lambda s: s["mean"])
    best_rows = [p for p in preds if p["classifier"] == best["classifier"] and p["k"] == best["k"]]
    per_class = M.per_class_pr([p["gold"] for p in best_rows], [p["pred"] for p in best_rows])
    run_log = {
        "environment": M.environment([("embedding", embedder), ("llm", llm)]),
        "quota": {"api_requests": 0, "api_tokens": 0, "note": "local CPU inference only; no hosted API used",
                  "llm_forward_calls": llm.forward_calls, "llm_prompt_tokens": llm.prompt_tokens},
        "wall_time_s": round(time.time() - t_start, 1),
        "ks": list(ks), "seeds": list(seeds), "bootstrap_B": B,
        "n_exemplars": len(pool), "n_eval": len(eval_rows),
    }
    (out_dir / "results" / "summary.json").write_text(json.dumps(
        {"summary": summary, "best": best, "per_class_best": per_class}, indent=2))
    (out_dir / "results" / "run_log.json").write_text(json.dumps(run_log, indent=2))
    import subprocess   # pinned environment: exact versions of every installed package
    lock = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (out_dir / "results" / "requirements-lock.txt").write_text(lock)
    write_report(out_dir, summary, best, per_class, run_log)
    log(f"done in {run_log['wall_time_s']}s; best = {best['classifier']} k={best['k']} macro-F1={best['mean']:.3f}")
    return summary, best, per_class, run_log


def write_report(out_dir, summary, best, per_class, run_log):
    tmpl = (ROOT / "report_template.md").read_text()
    names = {"prototype": "Embedding prototype", "knn": "kNN (FAISS IndexFlatIP)", "llm": "Few-shot LLM"}
    lines = ["| Classifier | k | Macro-F1 mean ± std (5 seeds) | 95% bootstrap CI |", "|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {names[s['classifier']]} | {s['k']} | {s['mean']:.3f} ± {s['std']:.3f} | "
                     f"[{s['ci_lo']:.3f}, {s['ci_hi']:.3f}] |")
    pc = ["| Label | Precision | Recall |", "|---|---|---|"]
    for lbl, (p, r) in per_class.items():
        pc.append(f"| {lbl} | {p:.2f} | {r:.2f} |")
    env = run_log["environment"]
    models = "\n".join(f"- {m['role']}: `{m['id']}` @ `{m['revision']}`" for m in env["models"])
    pk = ", ".join(f"{k} {v}" for k, v in env["packages"].items())
    report = (tmpl.replace("{{RESULTS_TABLE}}", "\n".join(lines))
                  .replace("{{BEST}}", f"{names[best['classifier']]}, k = {best['k']} "
                                       f"(macro-F1 {best['mean']:.3f} ± {best['std']:.3f})")
                  .replace("{{PER_CLASS}}", "\n".join(pc))
                  .replace("{{MODELS}}", models)
                  .replace("{{ENV}}", f"Python {env['python']} on {env['platform']}; {pk}")
                  .replace("{{QUOTA}}", f"{run_log['quota']['api_requests']} API requests, "
                                        f"{run_log['quota']['api_tokens']} API tokens (local inference only); "
                                        f"{run_log['quota']['llm_forward_calls']} LLM forward passes, "
                                        f"{run_log['quota']['llm_prompt_tokens']:,} prompt tokens processed")
                  .replace("{{WALL}}", f"{run_log['wall_time_s'] / 60:.1f} min"))
    (Path(out_dir) / "report.md").write_text(report)


if __name__ == "__main__":
    main()
