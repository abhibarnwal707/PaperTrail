# Running notebooks on the Mac

Everything runs from the project venv (`../.venv`, Python 3.15). XGBoost needs an OpenMP library that Homebrew would normally
provide; `scripts/run_local.sh` points it at the copy bundled in scikit-learn, so no Homebrew is needed.

## One-time: put the data in `data/local/` (git-ignored)
1. `transactions.parquet`: open https://www.kaggle.com/datasets/abhibarnwal707/papertrail-processed, click **Download**, unzip.
2. `HI-Small_Patterns.txt`: open https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml,
   download that single file from the Data tab.
3. Both files go directly in `PaperTrail/data/local/`.

## Run
```bash
cd ~/Documents/Projects/PaperTrail
PAPERTRAIL_QUICK=1 scripts/run_local.sh notebooks/03c_stability.py   # one seed: a fast first look
scripts/run_local.sh notebooks/03c_stability.py                      # full run, 3 seeds
```
Results are written to `reports/`; the console output is also saved to `logs/<name>.log` (git-ignored).
Optional DagsHub logging: `export DAGSHUB_USER_TOKEN=...` first (needs `pip install dagshub mlflow`); without it, MLflow logging is off.
