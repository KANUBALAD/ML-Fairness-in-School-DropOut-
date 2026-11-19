#!/usr/bin/env python3
"""
recompute_risk_metrics.py
=========================
Recomputes fairness + risk metrics for the AIRES paper with ONE consistent
convention, extending the original bulk_fairness_evaluation.py protocol:

CONVENTIONS (documented for the paper):
  * Sensitive attribute: S=1 = Male (privileged), S=0 = Female (unprivileged).
    Dataset-specific mappings (verified against the paper's EDA tables):
        brazil (SATDAP): Gender 0 = Male (64.8%), 1 = Female (35.2%)
        africa (SSSD)  : gender 'Male' / 'Female'
        india  (RDD)   : STUDENTGENDER 'M' / 'F'
    NOTE: the original pipeline encoded brazil as M-F but africa/india as
    F-M (see determine_privileged_group + FairnessMetrics.privileged_mask).
    Here DP/EO are ALWAYS Male - Female.
  * Label: y=1 = dropout (positive class), as in dataload.load_data.
  * DP  = P(y_hat=1|S=1) - P(y_hat=1|S=0)   (Male - Female)
  * EO  = TPR(S=1) - TPR(S=0)               (Male - Female)
  * FPR(S) = P(y_hat=1 | S, y=0)  -> false alarm / misdirected support risk
  * FNR(S) = P(y_hat=0 | S, y=1)  -> missed dropout / missed at-risk risk
  * Gaps  = Male - Female for FPR and FNR.
  * Both signed and absolute DP/EO are stored.
  * Protocol: StratifiedKFold(5, shuffle=True, random_state=42), exactly as
    the original bulk evaluation; per-fold metrics -> rows; run-level mean
    across folds is used for the one-sample t-tests (same as the paper).

USAGE:
  python recompute_risk_metrics.py --dataset brazil --outdir ../recompute_output
  (run one process per dataset in parallel; SSSD takes a few minutes)
"""

import os, sys, glob, re, argparse, time, warnings
import numpy as np
import pandas as pd
import yaml
from scipy import stats  # not used here, but kept for parity checks

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import src.dataload as dataload
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

# Verified gender mappings: column value -> male?
GENDER_MAP = {
    'brazil': {'0': True, '1': False},          # Gender column: 0=Male, 1=Female
    'africa': {'male': True, 'female': False},
    'india':  {'m': True, 'f': False},
}
YAML = {
    'brazil': 'yaml/brazil.yaml',
    'africa': 'yaml/africa.yaml',
    'india':  'yaml/india.yaml',
}
SYN_DIR = {
    # CORRECTED regeneration (Nov 2025): filenames mean Sens_X = X% MALE,
    # Label_Y = Y% DROPOUT; vary_sensitive fixes the label at the original
    # dropout prevalence; vary_label fixes the male share at the original
    # value. See src/regenerate_synthetic_corrected.py.
    # The older data lives in Synthetic_data/ and results/ and had inverted
    # label conditioning, ~50% labels in vary_sensitive files, and SSSD
    # target misses (see paper_dev tables/recomputed_notes.tex).
    'brazil': 'Synthetic_data_corrected/brazil',
    'africa': 'Synthetic_data_corrected/africa',
    'india':  'Synthetic_data_corrected/india',
}
SENS_COL = {'brazil': 'Gender', 'africa': 'gender', 'india': 'STUDENTGENDER'}

def load_with_sensitive(config, csv_path=None):
    """Load preprocessed X, y and binary sensitive S (1=Male) for a CSV."""
    original = config['datapath']
    if csv_path:
        config['datapath'] = csv_path
    try:
        X, y = dataload.load_data(config)
        path = csv_path if csv_path else original
        data = pd.read_csv(path, encoding='ISO-8859-1' if config['dataname'] == 'india' else None)
        sens = data[SENS_COL[config['dataname']]].astype(str).str.strip().str.lower()
        is_male = sens.map(GENDER_MAP[config['dataname']])
        if is_male.isna().any():
            raise ValueError(f"Unexpected gender values: {sens[is_male.isna()].unique()}")
        S = is_male.astype(int).values
        return X, np.asarray(y), S
    finally:
        config['datapath'] = original

def fold_metrics(y, yhat, S):
    """All metrics for one test fold. S: 1=Male, 0=Female; y: 1=dropout."""
    y, yhat, S = np.asarray(y), np.asarray(yhat), np.asarray(S)
    m, f = S == 1, S == 0
    def rate(cond, denom):
        return float(cond.sum() / denom.sum()) if denom.sum() > 0 else np.nan
    pr_m = rate(yhat[m] == 1, m)
    pr_f = rate(yhat[f] == 1, f)
    # TPR: P(yhat=1 | y=1, group) -- must intersect with the true label
    tpr_m = rate((yhat[m] == 1) & (y[m] == 1), y[m] == 1)
    tpr_f = rate((yhat[f] == 1) & (y[f] == 1), y[f] == 1)
    # FPR: P(yhat=1 | y=0, group)
    fpr_m = rate((yhat[m] == 1) & (y[m] == 0), y[m] == 0)
    fpr_f = rate((yhat[f] == 1) & (y[f] == 0), y[f] == 0)
    # FNR: P(yhat=0 | y=1, group)
    fnr_m = rate((yhat[m] == 0) & (y[m] == 1), y[m] == 1)
    fnr_f = rate((yhat[f] == 0) & (y[f] == 1), y[f] == 1)
    dp = pr_m - pr_f
    eo = (tpr_m - tpr_f) if not (np.isnan(tpr_m) or np.isnan(tpr_f)) else np.nan
    return {
        'acc': accuracy_score(y, yhat),
        'dp_signed': dp, 'dp_abs': abs(dp),
        'eo_signed': eo, 'eo_abs': abs(eo) if not np.isnan(eo) else np.nan,
        'tpr_m': tpr_m, 'tpr_f': tpr_f,
        'fpr_m': fpr_m, 'fpr_f': fpr_f,
        'fpr_gap': (fpr_m - fpr_f) if not (np.isnan(fpr_m) or np.isnan(fpr_f)) else np.nan,
        'fnr_m': fnr_m, 'fnr_f': fnr_f,
        'fnr_gap': (fnr_m - fnr_f) if not (np.isnan(fnr_m) or np.isnan(fnr_f)) else np.nan,
        'n_test': len(y), 'n_m': int(m.sum()), 'n_f': int(f.sum()),
    }

MODEL_FUNCS = {
    # NOTE: the original src/model.py does NOT set random_state for trees /
    # forests, so its DT/RF results were seed-nondeterministic. Here we fix
    # random_state=42 (same hyperparameters otherwise) so the recomputed
    # tables are fully reproducible.
    'logistic_regression': lambda X, y: LogisticRegression(max_iter=1000, random_state=42).fit(X, y),
    'decision_tree': lambda X, y: DecisionTreeClassifier(random_state=42).fit(X, y),
    'random_forest': lambda X, y: RandomForestClassifier(random_state=42).fit(X, y),
}

def evaluate(config, csv_path, dataset, run_type, experiment, sens_ratio, label_ratio, out_rows):
    X, y, S = load_with_sensitive(config, csv_path)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for model_name, fn in MODEL_FUNCS.items():
        folds_done = 0
        for fold, (tr, te) in enumerate(cv.split(X, y)):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y[tr], y[te]
            Ste = S[te]
            if len(np.unique(Ste)) < 2:
                continue
            mdl = fn(Xtr, ytr)
            yhat = mdl.predict(Xte)
            row = fold_metrics(yte, yhat, Ste)
            row.update(dict(dataset=dataset, model=model_name, fold=fold,
                            run_type=run_type, experiment=experiment,
                            sens_ratio=sens_ratio, label_ratio=label_ratio))
            out_rows.append(row)
            folds_done += 1
        print(f"    [{model_name}] {folds_done} folds done", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['brazil', 'africa', 'india'])
    ap.add_argument('--outdir', default='recompute_output')
    args = ap.parse_args()
    ds = args.dataset
    os.makedirs(args.outdir, exist_ok=True)
    config = yaml.safe_load(open(os.path.join('yaml', os.path.basename(YAML[ds]))))

    rows = []
    t0 = time.time()
    # 1) original baseline
    print(f"=== {ds}: original ===", flush=True)
    evaluate(config, None, ds, 'original', 'none', None, None, rows)
    # 2) synthetic experiments
    for exp in ['vary_label', 'vary_sensitive']:
        files = sorted(glob.glob(os.path.join(SYN_DIR[ds], exp, f'{ds}_Sens_*_Label_*.csv')))
        print(f"=== {ds}: {exp} ({len(files)} files) ===", flush=True)
        for f in files:
            base = os.path.basename(f)
            m = re.match(rf'{ds}_Sens_(\d+)_Label_(\d+)\.csv', base)
            sr, lr = int(m.group(1)) / 100, int(m.group(2)) / 100
            t1 = time.time()
            evaluate(config, f, ds, 'synthetic', exp, sr, lr, rows)
            print(f"  {base} done in {time.time()-t1:.1f}s (total {time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.outdir, f'{ds}_perfold.csv')
    df.to_csv(out_csv, index=False, float_format='%.6f')
    print(f"Saved {len(df)} fold rows to {out_csv}", flush=True)

if __name__ == '__main__':
    main()
