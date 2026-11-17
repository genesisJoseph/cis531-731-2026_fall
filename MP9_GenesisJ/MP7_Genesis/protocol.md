# MP 7 Experimental Protocol — Reconstructed from Submitted Implementation

**Status:** Retrospective documentation, NOT evidence of a pre-results protocol lock. The uploaded ZIP contains no `protocol.md` or Git history proving an earlier commit. Do not describe this as pre-registered.

## Task, track, and labels
CIS 731 VDB track: classify short song-theme descriptions using the `TropeLabel` extension in `ontology_mp7.json`, which points to the PS 6 ontology. Exactly one label per chunk. The eight IDs in `data/labels.json` are:

| ID | Meaning |
|---|---|
| `youth` | youth and growing up |
| `friendship` | friendship and loyalty |
| `romance` | love and romance |
| `self_love` | self-love and identity |
| `longing` | loss and longing |
| `fame` | fame and success |
| `social` | social commentary |
| `celebration` | celebration and fun |

## Data and evaluation source
`make_data.py` contains 20 original synthetic descriptions per class and writes the first 12 to `data/exemplars.jsonl` (96 total) and the remaining 8 to `data/eval_mp7.jsonl` (64 total). These are student-created examples, not instructor-provided `eval_mp7.jsonl` or authenticated BTS source chunks. The label space comes from the local MP 7 ontology extension. Keep evaluation texts and labels out of exemplars and LLM prompts. The code checks for shared IDs and exact normalized text overlap.

**INV-12 restriction:** All numbers from these files are *development-only* (`grading_eligible=False`). `evaluation.py` currently defines a local `compute_graded_metric` that returns `grading_eligible=True`; this is not the instructor's grading harness and must not be represented as satisfying INV-12. The instructor-provided set and harness were not in the uploaded files.

## Sweep and model configurations
- Methods: embedding prototype; similarity-weighted kNN with FAISS `IndexFlatIP` (up to five neighbors); fixed-demonstration, constrained-verbalizer LLM.
- `k = {0, 1, 3, 5, 8}` and seeds `{0, 1, 2, 3, 4}`. In this implementation, **k means examples PER CLASS**, not total examples: 0, 8, 24, 40, 64 demonstrations respectively. `k=0` uses label-description pseudo-exemplars for prototype/kNN, and the LLM uses no demonstrations.
- For each seed, select k examples from each of eight labels without replacement and randomly permute their order. The same selected demonstrations are used across classifiers for that (k, seed).
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`, mean-pooled, L2 normalized, CPU. Prototype uses mean class vectors and temperature-20 softmax; kNN uses inner product and similarity-weighted voting. LLM: `Qwen/Qwen2.5-0.5B-Instruct`, CPU, constrained next-token scoring of letter verbalizers A–H, followed by content-free-input calibration.

## Prompt and safeguards
System instruction, exactly as coded:

> You classify short descriptions of songs by their main theme. Answer with exactly one capital letter from the list of themes. Do not explain.

User prompt format: `Themes:` followed by `A) ...` through `H) ...` from `labels.json`; then each demonstration is `Description: <text>\nTheme: <letter>`; the query ends with `Description: <query>\nTheme:`. The tokenizer chat template wraps the system/user messages. The code reads next-token scores only for A–H, converts them to probabilities, and maps the highest-scoring letter back to its label ID. There is no free-form generated response parser; the restricted letter-token scores act as the output constraint.

Controls: (1) class-balanced demonstrations and content-free `N/A` calibration reduce majority-label bias; (2) seeded permutation changes demonstration order; (3) ID and exact-text-overlap assertions check against evaluation-example leakage. Exact-text checks do not rule out all semantic overlap.

## Metrics and confidence intervals
For each classifier and k, calculate macro-F1 separately for five seeds, then report the population mean and population standard deviation (`np.std`, ddof=0). For a 95% percentile bootstrap CI, perform **B=1,000** replicates with random generator seed 0; in each replicate sample the 64 evaluation IDs with replacement, compute macro-F1 for each seed, average the five values, and take the 2.5th and 97.5th percentiles. Compute per-class precision and recall for the best configuration, pooled across its five seed predictions. The existing `summarize()` calls the local stand-in graded function; treat outputs as advisory development metrics only.

## Reproducibility and provenance
Run command in `run_mp7.py`: `python run_mp7.py`. The notebook references `../MP6_Genesis/schema` for PS 6 contracts. Outputs in ZIP: `predictions.jsonl`, `results/summary.json`, `results/run_log.json`, and `results/requirements-lock.txt`. The uploaded notebook has **no executed outputs**, and the ZIP does not contain `report_template.md` even though `run_mp7.py` requires it to write `report.md`. The recorded execution was on macOS ARM CPU, not verified inside the pinned course DevContainer. The code records zero hosted API requests/tokens.
