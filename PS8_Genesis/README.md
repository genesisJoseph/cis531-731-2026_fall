# PS8_Genesis: interim analysis (IEEE-CIS Fraud Detection)

- `interim/ps8_explore.ipynb`: run this. It produces the Q3 figures, the data card and the advisory baselines.
- `interim/ps8_lib.py`: the analysis code (data card, figures, UMAP sweep, baselines, version manifest).
- `interim/figures/`: the figures, created by the notebook.
- `interim/results.json`: every number used in the survey, created by the notebook.
- `data/`: the original Kaggle CSVs. They are not committed (see `.gitignore`).

```bash
python3 -m pip install -r requirements.txt
cd interim
python3 -m jupyter nbconvert --to notebook --execute --inplace ps8_explore.ipynb
```
