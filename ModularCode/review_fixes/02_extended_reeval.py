#!/usr/bin/env python3
"""
02_extended_reeval.py
=====================
METRICS-ONLY re-evaluation of the ALREADY GENERATED synthetic datasets
(Synthetic_data_corrected/...). No CTGAN is trained, no synthetic data is
created. Exactly the same protocol as src/recompute_risk_metrics.py
(StratifiedKFold(5, shuffle=True, random_state=42); LR/DT/RF with
random_state=42) so the DP/EO/accuracy columns reproduce recompute_output/
*_perfold.csv exactly.

ADDED metrics needed to answer Reviewer 3 (accuracy is not comparable
across different dropout prevalences):
  - balanced accuracy (overall and per gender group) = (TPR + TNR)/2
  - ROC AUC (overall)
  - majority-class test-set accuracy (the trivial baseline)
  - normalized accuracy (acc - majority) / (1 - majority)
  - per-group confusion counts (n_pos_m/f, n_neg_m/f)

Output: review_fixes/perfold_extended/{ds}_perfold_extended.csv
        review_fixes/perfold_extended/predictions/{ds}__{model}__{file}__{fold}.npz
"""
import os, sys, re, glob, argparse, time, warnings
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import src.dataload as dataload
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

GENDER_MAP = {
    'brazil': {'0': True, '1': False},
    'africa': {'male': True, 'female': False},
    'india': {'m': True, 'f': False},
}
YAML = {'brazil': 'yaml/brazil.yaml', 'africa': 'yaml/africa.yaml', 'india': 'yaml/india.yaml'}
SYN_DIR = {'brazil': 'Synthetic_data_corrected/brazil',
           'africa': 'Synthetic_data_corrected/africa',
           'india': 'Synthetic_data_corrected/india'}
SENS_COL = {'brazil': 'Gender', 'africa': 'gender', 'india': 'STUDENTGENDER'}

OUTDIR = os.path.join(os.path.dirname(__file__), 'perfold_extended')
PREDDIR = os.path.join(OUTDIR, 'predictions')
os.makedirs(PREDDIR, exist_ok=True)


def load_with_sensitive(config, csv_path=None):
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


def fold_metrics(y, yhat, yprob, S):
    y, yhat, S = np.asarray(y), np.asarray(yhat), np.asarray(S)
    m, f = S == 1, S == 0
    pos, neg = y == 1, y == 0

    def rate(cond, denom):
        return float(cond.sum() / denom.sum()) if denom.sum() > 0 else np.nan

    tpr_m = rate((yhat[m] == 1) & (y[m] == 1), y[m] == 1)
    tpr_f = rate((yhat[f] == 1) & (y[f] == 1), y[f] == 1)
    fpr_m = rate((yhat[m] == 1) & (y[m] == 0), y[m] == 0)
    fpr_f = rate((yhat[f] == 1) & (y[f] == 0), y[f] == 0)
    tpr_all = rate(yhat[pos] == 1, pos)
    tnr_all = rate(yhat[neg] == 0, neg)
    dp = rate(yhat[m] == 1, m) - rate(yhat[f] == 1, f)
    eo = (tpr_m - tpr_f) if not (np.isnan(tpr_m) or np.isnan(tpr_f)) else np.nan

    auc = np.nan
    if len(np.unique(y)) == 2 and yprob is not None:
        try:
            auc = roc_auc_score(y, yprob)
        except Exception:
            auc = np.nan
    bal_m = (tpr_m + (1 - fpr_m)) / 2 if not (np.isnan(tpr_m) or np.isnan(fpr_m)) else np.nan
    bal_f = (tpr_f + (1 - fpr_f)) / 2 if not (np.isnan(tpr_f) or np.isnan(fpr_f)) else np.nan
    maj = max(float(pos.mean()), 1 - float(pos.mean()))
    acc = accuracy_score(y, yhat)
    return {
        'acc': acc,
        'balacc': balanced_accuracy_score(y, yhat),
        'balacc_m': bal_m, 'balacc_f': bal_f,
        'auc': auc,
        'dp_signed': dp, 'dp_abs': abs(dp),
        'eo_signed': eo, 'eo_abs': abs(eo) if not np.isnan(eo) else np.nan,
        'tpr_m': tpr_m, 'tpr_f': tpr_f, 'fpr_m': fpr_m, 'fpr_f': fpr_f,
        'tpr_all': tpr_all, 'tnr_all': tnr_all,
        'majority_acc': maj,
        'norm_acc': (acc - maj) / (1 - maj) if (1 - maj) > 0 else np.nan,
        'n_test': len(y), 'n_pos': int(pos.sum()), 'n_neg': int(neg.sum()),
        'n_m': int(m.sum()), 'n_f': int(f.sum()),
        'n_pos_m': int((pos & m).sum()), 'n_pos_f': int((pos & f).sum()),
    }


MODEL_FUNCS = {
    'logistic_regression': lambda X, y: LogisticRegression(max_iter=1000, random_state=42).fit(X, y),
    'decision_tree': lambda X, y: DecisionTreeClassifier(random_state=42).fit(X, y),
    'random_forest': lambda X, y: RandomForestClassifier(random_state=42).fit(X, y),
}


def evaluate(config, csv_path, dataset, run_type, experiment, sens_ratio, label_ratio, out_rows):
    tag = os.path.basename(csv_path) if csv_path else 'original'
    X, y, S = load_with_sensitive(config, csv_path)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for model_name, fn in MODEL_FUNCS.items():
        for fold, (tr, te) in enumerate(cv.split(X, y)):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y[tr], y[te]
            Ste = S[te]
            if len(np.unique(Ste)) < 2:
                continue
            mdl = fn(Xtr, ytr)
            yhat = mdl.predict(Xte)
            yprob = None
            if hasattr(mdl, 'predict_proba'):
                try:
                    yprob = mdl.predict_proba(Xte)[:, 1]
                except Exception:
                    yprob = None
            row = fold_metrics(yte, yhat, yprob, Ste)
            row.update(dict(dataset=dataset, model=model_name, fold=fold,
                            run_type=run_type, experiment=experiment,
                            sens_ratio=sens_ratio, label_ratio=label_ratio,
                            file=tag))
            out_rows.append(row)
            pred_file = os.path.join(PREDDIR, f'{dataset}__{model_name}__{tag.replace(".csv","")}__f{fold}.npz')
            np.savez(pred_file, y=yte, yhat=yhat, yprob=yprob, S=Ste)
    print(f"    done {tag}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['brazil', 'africa', 'india'])
    args = ap.parse_args()
    ds = args.dataset
    config = yaml.safe_load(open(os.path.join('yaml', os.path.basename(YAML[ds]))))

    rows = []
    t0 = time.time()
    evaluate(config, None, ds, 'original', 'none', None, None, rows)
    for exp in ['vary_label', 'vary_sensitive']:
        files = sorted(glob.glob(os.path.join(SYN_DIR[ds], exp, f'{ds}_Sens_*_Label_*.csv')))
        print(f"=== {ds}: {exp} ({len(files)} files) ===", flush=True)
        for f in files:
            base = os.path.basename(f)
            m = re.match(rf'{ds}_Sens_(\d+)_Label_(\d+)\.csv', base)
            sr, lr = int(m.group(1)) / 100, int(m.group(2)) / 100
            evaluate(config, f, ds, 'synthetic', exp, sr, lr, rows)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUTDIR, f'{ds}_perfold_extended.csv')
    df.to_csv(out_csv, index=False, float_format='%.6f')
    print(f"Saved {len(df)} fold rows to {out_csv} (total {time.time()-t0:.1f}s)", flush=True)


if __name__ == '__main__':
    main()
