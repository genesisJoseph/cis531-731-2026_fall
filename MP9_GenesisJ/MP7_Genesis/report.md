# MP 7 — Few-Shot Trope/Genre Classification
**CIS 731 | Implementation-based report | Development results only**

## 1. Data, labels, and experiment plan
I tested eight song-theme labels from my `ontology_mp7.json` extension of PS 6: `youth`, `friendship`, `romance`, `self_love`, `longing`, `fame`, `social`, and `celebration`. `make_data.py` generated **96 synthetic exemplars** (12 per label) and **64 separate synthetic evaluation examples** (8 per label). These are student-written descriptions, not instructor-provided evaluation data or verified BTS corpus chunks. My experiment uses **k = 0, 1, 3, 5, 8 examples per class**, and seeds **0, 1, 2, 3, 4**. This means the nonzero LLM prompt has 8, 24, 40, or 64 demonstrations. The 1,000-replicate 95% percentile bootstrap resamples evaluation IDs with replacement and averages macro-F1 across seeds.

**Protocol status:** `protocol.md` is reconstructed from the supplied code and run log. It was not present in the ZIP; I cannot verify it was committed before results. The required instructor evaluation set and official metric harness were also not provided. The local `compute_graded_metric` is only a stand-in and incorrectly sets `grading_eligible=True`; **all scores below are development-only (`grading_eligible=False`)**, not official graded results.

## 2. Classifiers, prompts, and safeguards
I compared (1) an **embedding prototype** using the mean normalized vector for each class, (2) **kNN** with FAISS `IndexFlatIP` and similarity-weighted voting over up to five neighbors, and (3) a **fixed-demonstration LLM** using `Qwen/Qwen2.5-0.5B-Instruct`. Embeddings use `sentence-transformers/all-MiniLM-L6-v2`. All three share a `Classifier` interface. `deal` postconditions check finite, nonnegative probability rows, sums within 1e-6, and predictions restricted to the eight labels.

The LLM system instruction is: *“You classify short descriptions of songs by their main theme. Answer with exactly one capital letter from the list of themes. Do not explain.”* The user prompt lists A–H with label descriptions, gives examples as `Description: ...` / `Theme: <letter>`, then ends with the query and `Theme:`. Instead of free-text generation, the implementation scores only the next-token letter verbalizers A–H. **Majority-label bias:** equal numbers of examples per class and `N/A` prior calibration. **Position bias:** seed-specific demonstration-order permutation. **Leakage:** assertions reject overlapping IDs and exact normalized text between demonstrations and evaluation records. These checks do not prove that semantically similar examples are absent.

## 3. Results — student-created development evaluation
The saved `predictions.jsonl` contains **4,800 predictions** (3 classifiers × 5 k values × 5 seeds × 64 records). The table shows mean macro-F1 ± standard deviation across five seeds and 95% bootstrap intervals; values come from `results/summary.json`.

| Classifier | k per class | Macro-F1 mean ± std | 95% bootstrap CI |
|---|---:|---:|---|
| prototype | 0 | 0.679 ± 0.000 | [0.542, 0.778] |
| prototype | 1 | 0.354 ± 0.064 | [0.286, 0.394] |
| prototype | 3 | 0.489 ± 0.061 | [0.393, 0.542] |
| prototype | 5 | 0.651 ± 0.078 | [0.547, 0.723] |
| prototype | 8 | 0.737 ± 0.030 | [0.633, 0.816] |
| knn | 0 | 0.679 ± 0.000 | [0.542, 0.778] |
| knn | 1 | 0.354 ± 0.064 | [0.286, 0.394] |
| knn | 3 | 0.446 ± 0.093 | [0.363, 0.498] |
| knn | 5 | 0.498 ± 0.063 | [0.403, 0.565] |
| knn | 8 | 0.548 ± 0.022 | [0.432, 0.632] |
| llm | 0 | 0.189 ± 0.000 | [0.098, 0.267] |
| llm | 1 | 0.042 ± 0.016 | [0.025, 0.055] |
| llm | 3 | 0.081 ± 0.039 | [0.056, 0.101] |
| llm | 5 | 0.104 ± 0.069 | [0.070, 0.133] |
| llm | 8 | 0.122 ± 0.062 | [0.079, 0.156] |

The highest recorded mean is **prototype, k=8: macro-F1 0.737 ± 0.030**, 95% CI **[0.633, 0.816]**. The few-shot LLM performs substantially worse on this local synthetic dataset. These results do not establish performance on the instructor's unseen evaluation set. At k=0, the code reuses the same predictions for all five seeds, explaining the zero standard deviation.

**Per-class precision / recall at the best configuration** (pooled over the five seed outputs):

| Label | Precision | Recall |
|---|---:|---:|
| youth | 0.944 | 0.850 |
| friendship | 0.692 | 0.675 |
| romance | 0.875 | 0.700 |
| self_love | 0.656 | 0.525 |
| longing | 0.702 | 0.825 |
| fame | 0.679 | 0.475 |
| social | 0.735 | 0.900 |
| celebration | 0.702 | 1.000 |

## 4. Reproducibility, limitations, and AI audit
The saved run log records **Python 3.12.1**, macOS ARM, CPU inference, **767.2 seconds** total, **0 hosted API requests / 0 API tokens**, **1,386 LLM forward calls**, and **44,201 processed prompt tokens**. Model revisions: embedding `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; LLM `7ae557604adf67be50417f59c2c2f167def9a775`. Exact installed package versions are in `results/requirements-lock.txt`. The supplied ZIP does not demonstrate a successful run in the course's pinned CPU DevContainer. The notebook cells are not executed in the uploaded copy. `run_mp7.py` also references a missing `report_template.md`, so its automatic report-writing step cannot be verified from this ZIP.

**GenAI:** AI assistance was used to draft this report and retrospective protocol from the submitted files. This disclosure does not replace the required `GenAI.pdf` with prompts, tollbooth inputs, and raw outputs. No unprovided test result, Git commit, or instructor evaluation was assumed.
