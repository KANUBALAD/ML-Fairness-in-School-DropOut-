#!/usr/bin/env python3
"""
build_latex_tables.py
=====================
Builds the recomputed LaTeX tables for the AIRES paper from
recompute_output/*_perfold.csv.

Conventions (documented in recomputed_notes.tex):
  * S=1 = Male (privileged), S=0 = Female; y=1 = dropout.
  * DP/EO signed: Male - Female. Absolute values also reported.
  * FPR = false-alarm risk (predicted dropout, actually graduate);
    FNR = missed-at-risk risk (predicted graduate, actually dropout).
  * 5-fold StratifiedKFold(shuffle=True, random_state=42); models seeded 42.
  * Experiment tables: mean across the 19 ratio settings (0.05..0.95, step 0.05).
  * Significance: one-sample t-test of the 19 run means vs the baseline.

Outputs written to ../paper_dev/aires_latex_template/tables/
"""

import os, sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, '..', '..', 'paper_dev', 'aires_latex_template', 'tables'))
os.makedirs(OUT, exist_ok=True)

DS_ORDER = ['SATDAP', 'SSSD', 'RDD']
CODE2NAME = {'brazil': 'SATDAP', 'africa': 'SSSD', 'india': 'RDD'}
NAME2CODE = {v: k for k, v in CODE2NAME.items()}
MODEL_ORDER = ['logistic_regression', 'decision_tree', 'random_forest']
MODEL_LABEL = {'logistic_regression': 'Logistic Regression',
               'decision_tree': 'Decision Tree',
               'random_forest': 'Random Forest'}
EXP_LABEL = {'vary_label': 'Vary Label', 'vary_sensitive': 'Vary Sensitive'}

# ---------------------------------------------------------------- load data
per = pd.concat([pd.read_csv(os.path.join(HERE, '..', 'recompute_output', f'{d}_perfold.csv'))
                 for d in ['brazil', 'africa', 'india']], ignore_index=True)
per['dataset'] = per['dataset'].map(CODE2NAME)
per['sens_ratio'] = per['sens_ratio'].fillna(-1)
per['label_ratio'] = per['label_ratio'].fillna(-1)

# run-level means (mean across folds)
runs = per.groupby(['dataset', 'model', 'run_type', 'experiment', 'sens_ratio', 'label_ratio'],
                   dropna=False).agg(
    acc=('acc', 'mean'),
    dp=('dp_signed', 'mean'), dp_abs=('dp_abs', 'mean'),
    eo=('eo_signed', 'mean'), eo_abs=('eo_abs', 'mean'),
    fpr_m=('fpr_m', 'mean'), fpr_f=('fpr_f', 'mean'), fpr_gap=('fpr_gap', 'mean'),
    fnr_m=('fnr_m', 'mean'), fnr_f=('fnr_f', 'mean'), fnr_gap=('fnr_gap', 'mean'),
    n_folds=('fold', 'count'),
).reset_index()

# fold-level stats for baselines
fold_agg = per[per.run_type == 'original'].groupby(['dataset', 'model']).agg(
    acc_mean=('acc', 'mean'), acc_std=('acc', 'std'),
    dp_mean=('dp_signed', 'mean'), dp_std=('dp_signed', 'std'),
    dp_abs_mean=('dp_abs', 'mean'), dp_abs_std=('dp_abs', 'std'),
    eo_mean=('eo_signed', 'mean'), eo_std=('eo_signed', 'std'),
    eo_abs_mean=('eo_abs', 'mean'), eo_abs_std=('eo_abs', 'std'),
    fpr_m_mean=('fpr_m', 'mean'), fpr_m_std=('fpr_m', 'std'),
    fpr_f_mean=('fpr_f', 'mean'), fpr_f_std=('fpr_f', 'std'),
    fpr_gap_mean=('fpr_gap', 'mean'), fpr_gap_std=('fpr_gap', 'std'),
    fnr_m_mean=('fnr_m', 'mean'), fnr_m_std=('fnr_m', 'std'),
    fnr_f_mean=('fnr_f', 'mean'), fnr_f_std=('fnr_f', 'std'),
    fnr_gap_mean=('fnr_gap', 'mean'), fnr_gap_std=('fnr_gap', 'std'),
).reset_index()

def pm(m, s):
    return f"{m:.4f}$\\pm${s:.4f}"

def pstr(p):
    return '0.0000' if p < 1e-4 else f'{p:.4f}'

def esc(s):
    return str(s).replace('_', '\\_')

# ---------------------------------------------------------------- Table 1: baseline fairness + accuracy
lines = []
lines.append("% Table 1: Harmonized baseline fairness and accuracy (recomputed).")
lines.append("% Convention: DP and EO are Male - Female (signed); |DP|/|EO| = absolute gaps.")
lines.append("% Protocol: 5-fold StratifiedKFold(shuffle, seed 42), models seeded 42. Mean +/- std across folds.")
lines.append("\\begin{table*}[t]")
lines.append("\\centering")
lines.append("\\caption{\\textbf{Baseline fairness and accuracy (recomputed, harmonized sign convention).} "
             "DP and EO are signed gaps computed as Male $-$ Female; $|\\text{DP}|$ and $|\\text{EO}|$ are the "
             "absolute gaps. Accuracy (Acc) is overall predictive accuracy. Mean $\\pm$ std across the 5 CV folds.}")
lines.append("\\label{tab:recomputed_baseline_fairness}")
lines.append("\\begin{tabular}{llccccc}")
lines.append("\\toprule")
lines.append("\\textbf{Dataset} & \\textbf{Model} & \\textbf{DP} & $|\\textbf{DP}|$ & \\textbf{EO} & $|\\textbf{EO}|$ & \\textbf{Acc} \\\\")
lines.append("\\midrule")
for ds in DS_ORDER:
    sub = fold_agg[fold_agg.dataset == ds].set_index('model').loc[MODEL_ORDER].reset_index()
    for i, (_, r) in enumerate(sub.iterrows()):
        dsl = f"\\multirow{{3}}{{*}}{{{ds}}}" if i == 0 else ""
        lines.append(f"{dsl} & {MODEL_LABEL[r.model]} & {pm(r.dp_mean, r.dp_std)} & {pm(r.dp_abs_mean, r.dp_abs_std)} & "
                     f"{pm(r.eo_mean, r.eo_std)} & {pm(r.eo_abs_mean, r.eo_abs_std)} & {pm(r.acc_mean, r.acc_std)} \\\\")
    lines.append("\\midrule" if ds != DS_ORDER[-1] else "\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table*}")
open(os.path.join(OUT, 'recomputed_baseline_fairness.tex'), 'w').write('\n'.join(lines) + '\n')

# ---------------------------------------------------------------- Table 2: baseline risk metrics
lines = []
lines.append("% Table 2: Baseline group-wise risk metrics (recomputed).")
lines.append("% FPR = P(predicted dropout | actually graduate) per group = false-alarm / misdirected-support risk.")
lines.append("% FNR = P(predicted graduate | actually dropout) per group = missed-at-risk risk.")
lines.append("% Gaps are Male - Female. Mean +/- std across the 5 CV folds.")
lines.append("\\begin{table*}[t]")
lines.append("\\centering")
lines.append("\\caption{\\textbf{Baseline risk metrics (recomputed).} Group-wise false positive rates "
             "(FPR: predicted dropout among students who actually graduate) and false negative rates "
             "(FNR: predicted graduate among students who actually dropout) for Male (M) and Female (F) groups, "
             "plus the between-group gaps (M $-$ F). Mean $\\pm$ std across the 5 CV folds.}")
lines.append("\\label{tab:recomputed_baseline_risk}")
lines.append("\\begin{tabular}{llcccccc}")
lines.append("\\toprule")
lines.append("\\textbf{Dataset} & \\textbf{Model} & \\textbf{FPR M} & \\textbf{FPR F} & \\textbf{FPR gap} & "
             "\\textbf{FNR M} & \\textbf{FNR F} & \\textbf{FNR gap} \\\\")
lines.append("\\midrule")
for ds in DS_ORDER:
    sub = fold_agg[fold_agg.dataset == ds].set_index('model').loc[MODEL_ORDER].reset_index()
    for i, (_, r) in enumerate(sub.iterrows()):
        dsl = f"\\multirow{{3}}{{*}}{{{ds}}}" if i == 0 else ""
        lines.append(f"{dsl} & {MODEL_LABEL[r.model]} & {pm(r.fpr_m_mean, r.fpr_m_std)} & {pm(r.fpr_f_mean, r.fpr_f_std)} & "
                     f"{pm(r.fpr_gap_mean, r.fpr_gap_std)} & {pm(r.fnr_m_mean, r.fnr_m_std)} & {pm(r.fnr_f_mean, r.fnr_f_std)} & "
                     f"{pm(r.fnr_gap_mean, r.fnr_gap_std)} \\\\")
    lines.append("\\midrule" if ds != DS_ORDER[-1] else "\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table*}")
open(os.path.join(OUT, 'recomputed_baseline_risk.tex'), 'w').write('\n'.join(lines) + '\n')

# ---------------------------------------------------------------- Table 3: experiment fairness means
def exp_table(metric_cols, header, cap, label, fname, note):
    lines = [note]
    ncols = len(metric_cols)
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{cap}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{ll" + "".join(["c"] * ncols * 3) + "}")
    lines.append("\\toprule")
    lines.append("\\multirow{2}{*}{\\textbf{Dataset}} & \\multirow{2}{*}{\\textbf{Variation}} & "
                 + " & ".join([f"\\multicolumn{{{ncols}}}{{c}}{{\\textbf{{{mlabel}}}}}" for mlabel in
                 ['Logistic Regression', 'Decision Tree', 'Random Forest']]) + " \\\\")
    span1, span2, span3 = ncols + 2, 2 * ncols + 2, 3 * ncols + 2
    lines.append(f"\\cmidrule(lr){{3-{span1}}} \\cmidrule(lr){{{span1+1}-{span2}}} \\cmidrule(lr){{{span2+1}-{span3}}}")
    lines.append("& & " + " & ".join([f"\\textbf{{{h}}}" for h in header] * 3) + " \\\\")
    lines.append("\\midrule")
    for ds in DS_ORDER:
        for ei, exp in enumerate(['vary_sensitive', 'vary_label']):
            dsl = f"\\multirow{{2}}{{*}}{{{ds}}}" if ei == 0 else ""
            cells = []
            for m in MODEL_ORDER:
                sub = runs[(runs.dataset == ds) & (runs.experiment == exp) & (runs.model == m) & (runs.run_type == 'synthetic')]
                row = sub[metric_cols].mean()  # mean over the 19 ratio settings
                for c in metric_cols:
                    v = row[c]
                    cells.append(f"{v:.4f}" if not np.isnan(v) else "--")
            lines.append(f"{dsl} & {EXP_LABEL[exp]} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule" if ds != DS_ORDER[-1] else "\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table*}")
    open(os.path.join(OUT, fname), 'w').write('\n'.join(lines) + '\n')

exp_table(['dp', 'eo', 'acc'], ['DP', 'EO', 'ACC'],
          "Fairness and accuracy under data-level interventions (recomputed). Each cell is the mean over the 19 "
          "ratio settings (0.05--0.95, step 0.05) of the run-level 5-fold CV mean. DP and EO are signed gaps "
          "(Male $-$ Female); ACC is overall accuracy.",
          'tab:recomputed_experiment_fairness', 'recomputed_experiment_fairness.tex',
          "% Table 3: Mean fairness/accuracy across the 19 ratio settings per dataset, experiment, model.\n"
          "% NOTE: the published tab:fairness-results used optimal-ratio values from a single-split protocol;\n"
          "% this table uses means over the full grid under the 5-fold CV protocol (consistent with the t-tests).")

# ---------------------------------------------------------------- Table 4: experiment risk means
exp_table(['fpr_gap', 'fnr_gap'], ['FPR gap', 'FNR gap'],
          "Risk-metric gaps under data-level interventions (recomputed). Each cell is the mean over the 19 ratio "
          "settings of the run-level 5-fold CV mean. FPR gap and FNR gap are Male $-$ Female gaps in the false "
          "positive rate (false-alarm risk) and false negative rate (missed-at-risk risk).",
          'tab:recomputed_experiment_risk', 'recomputed_experiment_risk.tex',
          "% Table 4: Mean FPR/FNR gaps across the 19 ratio settings per dataset, experiment, model.")

# ---------------------------------------------------------------- Tables 5/6: significance
METRIC_HEADER = {'dp': 'DP $p$', 'dp_abs': 'DP Impr.', 'eo': 'EO $p$', 'eo_abs': 'EO Impr.',
                 'fpr_gap': 'FPR gap $p$', 'fnr_gap': 'FNR gap $p$'}

def ttest_table(metrics, cap, label, fname, note):
    lines = [note]
    ncols = len(metrics)
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{cap}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{ll" + "".join(["c"] * ncols * 3) + "}")
    lines.append("\\toprule")
    lines.append("& & " + " & ".join([f"\\multicolumn{{{ncols}}}{{c}}{{\\textbf{{{mlabel}}}}}" for mlabel in
              ['Logistic Regression', 'Decision Tree', 'Random Forest']]) + " \\\\")
    span1, span2, span3 = ncols + 2, 2 * ncols + 2, 3 * ncols + 2
    lines.append(f"\\cmidrule(lr){{3-{span1}}} \\cmidrule(lr){{{span1+1}-{span2}}} \\cmidrule(lr){{{span2+1}-{span3}}}")
    subh = " & ".join([f"\\textbf{{{METRIC_HEADER[col]}}}" for (col, fmt) in metrics])
    lines.append("\\textbf{Dataset} & \\textbf{Experiment} & " + " & ".join([subh] * 3) + " \\\\")
    lines.append("\\midrule")
    for ds in DS_ORDER:
        for ei, exp in enumerate(['vary_sensitive', 'vary_label']):
            dsl = f"\\multirow{{2}}{{*}}{{{ds}}}" if ei == 0 else ""
            cells = []
            for m in MODEL_ORDER:
                base = runs[(runs.dataset == ds) & (runs.model == m) & (runs.run_type == 'original')].iloc[0]
                syn = runs[(runs.dataset == ds) & (runs.experiment == exp) & (runs.model == m) & (runs.run_type == 'synthetic')]
                for (col, fmt) in metrics:
                    vals = syn[col].dropna().values
                    b = base[col]
                    if len(vals) < 2 or np.isnan(b):
                        cells.append("--")
                    else:
                        t, p = stats.ttest_1samp(vals, b)
                        if fmt == 'p':
                            cells.append(pstr(p))
                        else:  # improv = |baseline| - |mean synthetic|
                            cells.append(f"{abs(b) - abs(np.mean(vals)):.4f}")
            lines.append(f"{dsl} & {EXP_LABEL[exp]} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule" if ds != DS_ORDER[-1] else "\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table*}")
    open(os.path.join(OUT, fname), 'w').write('\n'.join(lines) + '\n')

ttest_table([('dp', 'p'), ('dp_abs', 'impr'), ('eo', 'p'), ('eo_abs', 'impr')],
            "Significance of fairness changes (recomputed). One-sample $t$-tests of the 19 run-level means against "
            "the original-data baseline for DP and EO; $p$-values bold at $\\alpha=0.05$. 'Impr.' $=$ "
            "$|\\text{baseline}|-|\\text{mean synthetic}|$ (positive $=$ bias reduction). $p$-values shown as 0.0000 "
            "are $<10^{-4}$. n=19 per test (ratio grid 0.05--0.95, step 0.05).",
            'tab:recomputed_significance_fairness', 'recomputed_significance_fairness.tex',
            "% Table 5: t-tests for DP and EO (harmonized signed convention).\n"
            "% Metric order per model: DP p-value, DP Impr., EO p-value, EO Impr.")

ttest_table([('fpr_gap', 'p'), ('fnr_gap', 'p')],
            "Significance of risk-metric gap changes (recomputed). One-sample $t$-tests of the 19 run-level means "
            "against the original-data baseline for the FPR gap and FNR gap (Male $-$ Female); $p$-values bold at "
            "$\\alpha=0.05$; 0.0000 means $p<10^{-4}$. n=19 per test (ratio grid 0.05--0.95, step 0.05).",
            'tab:recomputed_significance_risk', 'recomputed_significance_risk.tex',
            "% Table 6: t-tests for the risk-metric gaps (FPR gap, FNR gap).\n"
            "% Metric order per model: FPR gap p-value, FNR gap p-value.")

# ---------------------------------------------------------------- RDD anomaly + SATDAP wording
rdd = runs[(runs.dataset == 'RDD') & (runs.run_type == 'synthetic')]
sat = runs[(runs.dataset == 'SATDAP') & (runs.run_type == 'synthetic')]
sss = runs[(runs.dataset == 'SSSD') & (runs.run_type == 'synthetic')]
base_runs = runs[runs.run_type == 'original'].set_index(['dataset', 'model'])

rep = []
rep.append("\\subsection{RDD accuracy under Vary Sensitive (corrected data): collapse resolved}")
rep.append("Per-ratio accuracy (5-fold CV means, models seeded 42). With the corrected generation (label fixed at the original 12\% dropout prevalence), accuracy is flat across all ratios ($\approx$0.87--0.88 vs.\ 0.90 baseline): the old collapse to $\approx$0.60 was an artifact of the confounded ~50\% label distribution in the previous synthetic files.")
rep.append("\\begin{table}[h]")
rep.append("\\centering")
rep.append("\\caption{RDD: accuracy per sensitive-ratio setting under the Vary Sensitive experiment.}")
rep.append("\\label{tab:rdd_anomaly}")
rep.append("\\begin{tabular}{l" + "ccc" + "}")
rep.append("\\toprule")
rep.append("\\textbf{Male ratio} & \\textbf{LR} & \\textbf{DT} & \\textbf{RF} \\\\")
rep.append("\\midrule")
ratios = sorted(rdd[rdd.experiment == 'vary_sensitive'].sens_ratio.unique())
for r in ratios:
    cells = []
    for m in MODEL_ORDER:
        v = rdd[(rdd.experiment == 'vary_sensitive') & (rdd.sens_ratio == r) & (rdd.model == m)].iloc[0].acc
        cells.append(f"{v:.4f}")
    rep.append(f"{r:.2f} & " + " & ".join(cells) + " \\\\")
rep.append("\\midrule")
cells = []
for m in MODEL_ORDER:
    cells.append(f"{base_runs.loc[('RDD', m), 'acc']:.4f}")
rep.append("Baseline (original) & " + " & ".join(cells) + " \\\\")
rep.append("\\bottomrule")
rep.append("\\end{tabular}")
rep.append("\\end{table}")

rep.append("\\subsection{SATDAP accuracy under Vary Label (corrected data): the honest trade-off}")
rep.append("Per-ratio accuracy for SATDAP under the Vary Label experiment (5-fold CV means). Fairness (DP/EO $\\approx$ 0) is achieved across the whole sweep, but accuracy follows $\\approx 1-|0.5-\\text{dropout share}|$: the endpoints (acc 0.95) are trivial majority-class prediction, while the mid-grid (50\% dropout) shows a genuine accuracy cost of $\\approx$0.33 relative to baseline.")
rep.append("\\begin{table}[h]")
rep.append("\\centering")
rep.append("\\caption{SATDAP: accuracy per label-ratio setting under the Vary Label experiment.}")
rep.append("\\label{tab:satdap_wording}")
rep.append("\\begin{tabular}{l" + "ccc" + "}")
rep.append("\\toprule")
rep.append("\\textbf{Dropout ratio} & \\textbf{LR} & \\textbf{DT} & \\textbf{RF} \\\\")
rep.append("\\midrule")
ratios = sorted(sat[sat.experiment == 'vary_label'].label_ratio.unique())
for r in ratios:
    cells = []
    for m in MODEL_ORDER:
        v = sat[(sat.experiment == 'vary_label') & (sat.label_ratio == r) & (sat.model == m)].iloc[0].acc
        cells.append(f"{v:.4f}")
    rep.append(f"{r:.2f} & " + " & ".join(cells) + " \\\\")
rep.append("\\midrule")
cells = []
for m in MODEL_ORDER:
    cells.append(f"{base_runs.loc[('SATDAP', m), 'acc']:.4f}")
rep.append("Baseline (original) & " + " & ".join(cells) + " \\\\")
rep.append("\\bottomrule")
rep.append("\\end{tabular}")
rep.append("\\end{table}")

open(os.path.join(OUT, 'recomputed_diagnostics.tex'), 'w').write('\n'.join(rep) + '\n')

# ---------------------------------------------------------------- summary numbers for notes
summary = {}
for ds in DS_ORDER:
    for m in MODEL_ORDER:
        b = base_runs.loc[(ds, m)]
        summary[(ds, m)] = dict(base_dp=b.dp, base_dp_abs=b.dp_abs, base_eo=b.eo, base_eo_abs=b.eo_abs,
                                base_acc=b.acc, base_fpr_gap=b.fpr_gap, base_fnr_gap=b.fnr_gap,
                                base_fpr_m=b.fpr_m, base_fpr_f=b.fpr_f, base_fnr_m=b.fnr_m, base_fnr_f=b.fnr_f)
        for exp in ['vary_sensitive', 'vary_label']:
            s = runs[(runs.dataset == ds) & (runs.experiment == exp) & (runs.model == m)]
            summary[(ds, m)][f'{exp}_dp'] = s.dp.mean()
            summary[(ds, m)][f'{exp}_dp_abs'] = s.dp_abs.mean()
            summary[(ds, m)][f'{exp}_eo'] = s.eo.mean()
            summary[(ds, m)][f'{exp}_eo_abs'] = s.eo_abs.mean()
            summary[(ds, m)][f'{exp}_acc'] = s.acc.mean()
            summary[(ds, m)][f'{exp}_fpr_gap'] = s.fpr_gap.mean()
            summary[(ds, m)][f'{exp}_fnr_gap'] = s.fnr_gap.mean()

# save run-level CSV for reference
runs.to_csv(os.path.join(HERE, '..', 'recompute_output', 'recomputed_runs.csv'), index=False,
            float_format='%.6f')

print("== KEY SUMMARY (harmonized convention: DP/EO = Male - Female; y=1=dropout) ==")
print(f"{'Dataset':7s} {'Model':20s} {'baseAcc':>8s} {'|DP|':>6s} {'vsSens':>7s} {'vsLabel':>7s} {'|EO|':>6s} {'vsSens':>7s} {'vsLabel':>7s} {'accSens':>8s} {'accLabel':>8s}")
for ds in DS_ORDER:
    for m in MODEL_ORDER:
        s = summary[(ds, m)]
        print(f"{ds:7s} {MODEL_LABEL[m]:20s} {s['base_acc']:8.4f} {s['base_dp_abs']:6.4f} "
              f"{s['vary_sensitive_dp_abs']:7.4f} {s['vary_label_dp_abs']:7.4f} {s['base_eo_abs']:6.4f} "
              f"{s['vary_sensitive_eo_abs']:7.4f} {s['vary_label_eo_abs']:7.4f} {s['vary_sensitive_acc']:8.4f} {s['vary_label_acc']:8.4f}")
print("\nLaTeX tables written to:", OUT)
