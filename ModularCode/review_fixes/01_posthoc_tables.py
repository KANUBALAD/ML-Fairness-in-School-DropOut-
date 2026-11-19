#!/usr/bin/env python3
"""
01_posthoc_tables.py
====================
Reviewer-rebuttal tables computed ENTIRELY by post-processing of existing
artifacts (no model retraining, no synthetic data generation):

  - recompute_output/{brazil,africa,india}_perfold.csv  (per-fold DP/EO/acc)
  - Synthetic_data_corrected/{brazil,africa,india}/...   (saved synthetic CSVs)
  - data/*.csv                                          (original datasets)

Produces, in review_fixes/csv and review_fixes/tex:

  01_joint_audit.csv                 -> evidence for Reviewer 1
  02_matched_marginals.csv           -> evidence for Reviewers 1 & 2
  03_satdap_accuracy_decomposition.csv -> evidence for Reviewer 2
  05_sssd_eo_intervention.csv        -> evidence for Reviewer 4
  06_significance_recomputed.csv     -> replacement for tab:full_significance
  tex/joint_tables.tex               -> corrected appendix joint tables
  tex/matched_marginals_table.tex
  tex/sssd_eo_table.tex
  tex/significance_table.tex

Conventions (same as src/recompute_risk_metrics.py):
  S=1 Male (privileged), y=1 dropout. DP = P(yhat=1|M) - P(yhat=1|F).
  EO = TPR(M) - TPR(F). All values are 5-fold-CV means unless noted.
"""
import os, re, glob
import numpy as np
import pandas as pd
from scipy import stats

OUT_CSV = os.path.join(os.path.dirname(__file__), 'csv')
OUT_TEX = os.path.join(os.path.dirname(__file__), 'tex')
os.makedirs(OUT_CSV, exist_ok=True)
os.makedirs(OUT_TEX, exist_ok=True)

DS_NAME = {'brazil': 'SATDAP', 'africa': 'SSSD', 'india': 'RDD'}
ORIG_SENS = {'brazil': 0.648, 'africa': 0.441, 'india': 0.348}   # true P(Male)
ORIG_LABEL = {'brazil': 0.321, 'africa': 0.091, 'india': 0.120}   # true P(dropout)
GRID_SENS = {'brazil': 0.65, 'africa': 0.45, 'india': 0.35}       # nearest sweep grid point
GRID_LABEL = {'brazil': [0.30, 0.35], 'africa': [0.10], 'india': [0.10, 0.15]}  # nearest label grid points
GENDER_MAP = {'brazil': {'0': 'M', '1': 'F'}, 'africa': {'male': 'M', 'female': 'F'},
              'india': {'m': 'M', 'f': 'F'}}
LABEL_COL = {'brazil': 'Target', 'africa': 'dropout', 'india': 'STUDENT_DROPOUT_STATUS'}
SENS_COL = {'brazil': 'Gender', 'africa': 'gender', 'india': 'STUDENTGENDER'}
ORIG_PATH = {'brazil': 'data/brazil_data.csv', 'africa': 'data/Secondary_school_dropout_dataset.csv',
             'india': 'data/asia_dropout.csv'}

def is_dropout(v, ds):
    v = str(v).strip()
    if ds == 'brazil':
        return 1.0 if v == 'Dropout' else 0.0
    if ds == 'africa':
        return 1.0 if v.lower() in ('1', '1.0', 'yes', 'y', 'true') else 0.0
    return 1.0 if v.lower() in ('y', 'yes', '1', '1.0', 'dropout', 'drop out') else 0.0

def load_csv(ds, path):
    return pd.read_csv(path, encoding='ISO-8859-1' if ds == 'india' else None)

def joint_stats(ds, path):
    d = load_csv(ds, path)
    s = d[SENS_COL[ds]].astype(str).str.strip().str.lower()
    y = d[LABEL_COL[ds]].apply(lambda v: is_dropout(v, ds))
    m = s.map(GENDER_MAP[ds]) == 'M'
    p_m = m.mean()
    p_drop = y.mean()
    p_dm = y[m].mean() if m.sum() else np.nan
    p_df = y[~m].mean() if (~m).sum() else np.nan
    return p_m, p_drop, p_dm, p_df, (p_dm - p_df)

# ----------------------------------------------------------------------------
# Artifact 01: joint-distribution audit (generator forces A perp Y)
# ----------------------------------------------------------------------------
rows = []
for ds in ['brazil', 'africa', 'india']:
    rows.append(dict(dataset=DS_NAME[ds], file='ORIGINAL', p_male=ORIG_SENS[ds],
                     p_dropout=ORIG_LABEL[ds],
                     p_drop_male=joint_stats(ds, ORIG_PATH[ds])[2],
                     p_drop_female=joint_stats(ds, ORIG_PATH[ds])[3],
                     cond_gap=joint_stats(ds, ORIG_PATH[ds])[4]))
    # label sweep: sens fixed at original, 3 label ratios
    for lr in [10, 50, 90]:
        f = glob.glob(f'Synthetic_data_corrected/{ds}/vary_label/{ds}_Sens_*_Label_{lr}.csv')
        if f:
            pm, pd_, pdm, pdf, gap = joint_stats(ds, f[0])
            rows.append(dict(dataset=DS_NAME[ds], file=f[0].split('/')[-1], p_male=pm,
                             p_dropout=pd_, p_drop_male=pdm, p_drop_female=pdf, cond_gap=gap))
    # sensitive sweep: label fixed at original, 3 sens ratios
    for sr in [10, 50, 90]:
        f = glob.glob(f'Synthetic_data_corrected/{ds}/vary_sensitive/{ds}_Sens_{sr}_Label_*.csv')
        if f:
            pm, pd_, pdm, pdf, gap = joint_stats(ds, f[0])
            rows.append(dict(dataset=DS_NAME[ds], file=f[0].split('/')[-1], p_male=pm,
                             p_dropout=pd_, p_drop_male=pdm, p_drop_female=pdf, cond_gap=gap))
audit = pd.DataFrame(rows)
audit.to_csv(f'{OUT_CSV}/01_joint_audit.csv', index=False, float_format='%.6f')
print(audit.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print()

# ----------------------------------------------------------------------------
# Load per-fold results
# ----------------------------------------------------------------------------
PERFOLD = {ds: pd.read_csv(f'recompute_output/{ds}_perfold.csv') for ds in ['brazil', 'africa', 'india']}
for ds in PERFOLD:
    PERFOLD[ds]['sens_ratio'] = PERFOLD[ds]['sens_ratio'].fillna(-1)
    PERFOLD[ds]['label_ratio'] = PERFOLD[ds]['label_ratio'].fillna(-1)

def run_mean_std(df, exp=None, sens=None, lab=None):
    d = df.copy()
    if exp is not None:
        d = d[(d.run_type == 'synthetic') & (d.experiment == exp)]
    else:
        d = d[d.run_type == 'original']
    if sens is not None: d = d[np.isclose(d.sens_ratio, sens)]
    if lab is not None: d = d[np.isclose(d.label_ratio, lab)]
    g = d.groupby('model').agg(
        acc_m=('acc', 'mean'), acc_s=('acc', 'std'),
        dp_m=('dp_signed', 'mean'), dp_s=('dp_signed', 'std'),
        eo_m=('eo_signed', 'mean'), eo_s=('eo_signed', 'std'))
    return g

def fmt(m, s):
    return f'{m:.4f} $\\pm$ {s:.4f}'

# ----------------------------------------------------------------------------
# Artifact 02: matched marginals - baseline vs synthetic at ORIGINAL marginals
# ----------------------------------------------------------------------------
mm_rows = []
for ds in ['brazil', 'africa', 'india']:
    df = PERFOLD[ds]
    base = run_mean_std(df[df.run_type == 'original'])
    sens_at_orig = run_mean_std(df, exp='vary_sensitive', sens=GRID_SENS[ds])
    for lr in GRID_LABEL[ds]:
        lab_at_orig = run_mean_std(df, exp='vary_label', lab=lr)
        for m in ['logistic_regression', 'decision_tree', 'random_forest']:
            mm_rows.append(dict(
                dataset=DS_NAME[ds], model=m.replace('_', ' '),
                orig_sens=ORIG_SENS[ds], orig_label=ORIG_LABEL[ds],
                baseline_acc=base.loc[m, 'acc_m'], baseline_dp=base.loc[m, 'dp_m'],
                baseline_eo=base.loc[m, 'eo_m'],
                sens_pt=GRID_SENS[ds], sens_acc=sens_at_orig.loc[m, 'acc_m'],
                sens_dp=sens_at_orig.loc[m, 'dp_m'], sens_eo=sens_at_orig.loc[m, 'eo_m'],
                label_pt=lr, label_acc=lab_at_orig.loc[m, 'acc_m'],
                label_dp=lab_at_orig.loc[m, 'dp_m'], label_eo=lab_at_orig.loc[m, 'eo_m'],
                acc_drop_sens=base.loc[m, 'acc_m'] - sens_at_orig.loc[m, 'acc_m'],
                acc_drop_label=base.loc[m, 'acc_m'] - lab_at_orig.loc[m, 'acc_m']))
matched = pd.DataFrame(mm_rows)
matched.to_csv(f'{OUT_CSV}/02_matched_marginals.csv', index=False, float_format='%.6f')
print(matched.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print()

# ----------------------------------------------------------------------------
# Artifact 03: SATDAP accuracy decomposition (single cause for the drop)
# ----------------------------------------------------------------------------
df = PERFOLD['brazil']
base = run_mean_std(df)
decomp = []
gs = run_mean_std(df, exp='vary_sensitive', sens=0.65)
for m in ['logistic_regression', 'decision_tree', 'random_forest']:
    b_acc = base.loc[m, 'acc_m']
    decomp.append(dict(
        model=m.replace('_', ' '), setting=f'baseline (orig data, dropout={ORIG_LABEL["brazil"]:.2f})',
        dropout_rate=ORIG_LABEL['brazil'], majority_class_acc=max(ORIG_LABEL['brazil'], 1 - ORIG_LABEL['brazil']),
        acc=b_acc, acc_vs_baseline=0.0))
    decomp.append(dict(
        model=m.replace('_', ' '),
        setting='gender sweep @ original male share 0.65 (label held 0.32)',
        dropout_rate=0.32, majority_class_acc=0.68,
        acc=gs.loc[m, 'acc_m'], acc_vs_baseline=gs.loc[m, 'acc_m'] - b_acc))
    for lr in [0.30, 0.35, 0.50]:
        ls = run_mean_std(df, exp='vary_label', lab=lr)
        decomp.append(dict(
            model=m.replace('_', ' '),
            setting=f'label sweep @ dropout {lr:.2f} (male share held 0.65)',
            dropout_rate=lr, majority_class_acc=max(lr, 1 - lr),
            acc=ls.loc[m, 'acc_m'], acc_vs_baseline=ls.loc[m, 'acc_m'] - b_acc))
dec = pd.DataFrame(decomp)
dec.to_csv(f'{OUT_CSV}/03_satdap_accuracy_decomposition.csv', index=False, float_format='%.6f')
print(dec.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print()

# ----------------------------------------------------------------------------
# Artifact 05: SSSD EO - label vs gender rebalancing (verdict flip evidence)
# ----------------------------------------------------------------------------
df = PERFOLD['africa']
base = run_mean_std(df)
sssd = []
for m in ['logistic_regression', 'decision_tree', 'random_forest']:
    b_eo = base.loc[m, 'eo_m']
    for exp in ['vary_sensitive', 'vary_label']:
        sub = df[(df.run_type == 'synthetic') & (df.experiment == exp) & (df.model == m)]
        run = sub.groupby(['sens_ratio', 'label_ratio']).eo_signed.mean()  # 19 run means
        eo_abs = run.abs()
        sssd.append(dict(
            model=m.replace('_', ' '), experiment=exp.replace('_', ' '),
            baseline_eo=b_eo, baseline_eo_abs=abs(b_eo),
            eo_min=run.min(), eo_max=run.max(), eo_abs_min=eo_abs.min(),
            eo_abs_median=eo_abs.median(), eo_abs_max=eo_abs.max(),
            eo_abs_mean=eo_abs.mean(),
            improvement_abs=abs(b_eo) - eo_abs.mean(),
            frac_runs_below_003=(eo_abs < 0.03).mean(),
            frac_runs_below_baseline=(eo_abs < abs(b_eo)).mean()))
ss = pd.DataFrame(sssd)
ss.to_csv(f'{OUT_CSV}/05_sssd_eo_intervention.csv', index=False, float_format='%.6f')
print(ss.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print()

# ----------------------------------------------------------------------------
# Artifact 06: significance table recomputed from corrected per-fold data
# ----------------------------------------------------------------------------
sig_rows = []
for ds in ['brazil', 'africa', 'india']:
    df = PERFOLD[ds]
    base = df[df.run_type == 'original'].groupby('model')[['dp_signed', 'eo_signed']].mean()
    for exp in ['vary_sensitive', 'vary_label']:
        sub = df[(df.run_type == 'synthetic') & (df.experiment == exp)]
        for m in ['logistic_regression', 'decision_tree', 'random_forest']:
            for metric in ['dp_signed', 'eo_signed']:
                runs = sub[sub.model == m].groupby(['sens_ratio', 'label_ratio'])[metric].mean()
                b = base.loc[m, metric]
                t, p = stats.ttest_1samp(runs.values, b)
                sig_rows.append(dict(
                    dataset=DS_NAME[ds], experiment=exp.replace('_', ' ').title(), model=m.replace('_', ' '),
                    metric=metric.replace('_signed', '').upper(), n=len(runs),
                    baseline=float(b), mean_synthetic=float(runs.mean()),
                    sd_synthetic=float(runs.std(ddof=1)), t=float(t), p_value=float(p),
                    improvement=abs(b) - abs(float(runs.mean())),
                    significant=bool(p < 0.05)))
sig = pd.DataFrame(sig_rows)
sig.to_csv(f'{OUT_CSV}/06_significance_recomputed.csv', index=False, float_format='%.6f')
print(sig.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print()

# ----------------------------------------------------------------------------
# LaTeX tables
# ----------------------------------------------------------------------------
def latex_joint_tables():
    out = []
    for ds in ['brazil', 'africa', 'india']:
        name = DS_NAME[ds]
        orig = joint_stats(ds, ORIG_PATH[ds])
        # label sweep joint percentages
        lab_rows = []
        for lr in [0.10, 0.50, 0.90]:
            f = glob.glob(f'Synthetic_data_corrected/{ds}/vary_label/{ds}_Sens_*_Label_{int(lr*100)}.csv')[0]
            d = load_csv(ds, f)
            s = d[SENS_COL[ds]].astype(str).str.strip().str.lower()
            y = d[LABEL_COL[ds]].apply(lambda v: is_dropout(v, ds))
            m = s.map(GENDER_MAP[ds]) == 'M'
            lab_rows.append([
                100 * (m & (y == 1)).mean(), 100 * (m & (y == 0)).mean(),
                100 * (~m & (y == 1)).mean(), 100 * (~m & (y == 0)).mean()])
        sens_rows = []
        for sr in [0.10, 0.50, 0.90]:
            f = glob.glob(f'Synthetic_data_corrected/{ds}/vary_sensitive/{ds}_Sens_{int(sr*100)}_Label_*.csv')[0]
            d = load_csv(ds, f)
            s = d[SENS_COL[ds]].astype(str).str.strip().str.lower()
            y = d[LABEL_COL[ds]].apply(lambda v: is_dropout(v, ds))
            m = s.map(GENDER_MAP[ds]) == 'M'
            sens_rows.append([
                100 * (m & (y == 1)).mean(), 100 * (m & (y == 0)).mean(),
                100 * (~m & (y == 1)).mean(), 100 * (~m & (y == 0)).mean()])
        om = 100 * ORIG_SENS[ds] * orig[2]; onm = 100 * ORIG_SENS[ds] * (1 - orig[2])
        of = 100 * (1 - ORIG_SENS[ds]) * orig[3]; onf = 100 * (1 - ORIG_SENS[ds]) * (1 - orig[3])
        out.append(f"""
\\begin{{table}}[ht]
\\centering\\footnotesize
\\caption{{{name}: joint gender $\\times$ dropout structure, original vs. synthetic
(\\% of all rows; synthetic files from \\texttt{{Synthetic\\_data\\_corrected}}).}}
\\begin{{tabular}}{{lrrrr}}
\\toprule
Cell (Gender, Dropout) & Orig. & Label 0.10 & Label 0.50 & Label 0.90 \\\\
\\midrule
Male, Dropout    & {om:.2f} & {lab_rows[0][0]:.2f} & {lab_rows[1][0]:.2f} & {lab_rows[2][0]:.2f} \\\\
Male, No dropout & {onm:.2f} & {lab_rows[0][1]:.2f} & {lab_rows[1][1]:.2f} & {lab_rows[2][1]:.2f} \\\\
Female, Dropout    & {of:.2f} & {lab_rows[0][2]:.2f} & {lab_rows[1][2]:.2f} & {lab_rows[2][2]:.2f} \\\\
Female, No dropout & {onf:.2f} & {lab_rows[0][3]:.2f} & {lab_rows[1][3]:.2f} & {lab_rows[2][3]:.2f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{table}}[ht]
\\centering\\footnotesize
\\caption{{{name}: joint gender $\\times$ dropout structure under gender rebalancing
(\\% of all rows; dropout rate held at its original value).}}
\\begin{{tabular}}{{lrrrr}}
\\toprule
Cell (Gender, Dropout) & Orig. & Sens 0.10 & Sens 0.50 & Sens 0.90 \\\\
\\midrule
Male, Dropout    & {om:.2f} & {sens_rows[0][0]:.2f} & {sens_rows[1][0]:.2f} & {sens_rows[2][0]:.2f} \\\\
Male, No dropout & {onm:.2f} & {sens_rows[0][1]:.2f} & {sens_rows[1][1]:.2f} & {sens_rows[2][1]:.2f} \\\\
Female, Dropout    & {of:.2f} & {sens_rows[0][2]:.2f} & {sens_rows[1][2]:.2f} & {sens_rows[2][2]:.2f} \\\\
Female, No dropout & {onf:.2f} & {sens_rows[0][3]:.2f} & {sens_rows[1][3]:.2f} & {sens_rows[2][3]:.2f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")
    return '\n'.join(out)

with open(f'{OUT_TEX}/joint_tables.tex', 'w') as fh:
    fh.write(latex_joint_tables())
print('wrote tex/joint_tables.tex')

# matched marginals LaTeX (SATDAP focus)
mmt = matched[matched.dataset == 'SATDAP']
lines = []
for m in ['logistic regression', 'decision tree', 'random forest']:
    r0 = mmt[mmt.model == m].iloc[0]
    r30 = mmt[(mmt.model == m) & (mmt.label_pt == 0.30)].iloc[0]
    r35 = mmt[(mmt.model == m) & (mmt.label_pt == 0.35)].iloc[0]
    lines.append(
        f"{m:20s} & {r0.baseline_dp:+.4f} & {r0.baseline_acc:.4f} & "
        f"{r0.sens_dp:+.4f} & {r0.sens_acc:.4f} & "
        f"{r30.label_dp:+.4f} & {r30.label_acc:.4f} & "
        f"{r35.label_dp:+.4f} & {r35.label_acc:.4f} \\\\")
with open(f'{OUT_TEX}/matched_marginals_table.tex', 'w') as fh:
    fh.write("""\\begin{table}[ht]
\\centering\\footnotesize
\\caption{SATDAP: fairness and accuracy on real data vs. CTGAN synthetic data
\\emph{at the original marginals} (no rebalancing). The DP gap disappears in the
synthetic data even when the male share and dropout rate are unchanged, because
the generator forces gender $\\perp$ dropout. The accuracy drop ($\\approx$18--24 pts)
is the same in both sweeps: it is the cost of replacing the real data with CTGAN
output, not the cost of rebalancing.}
\\label{tab:matched_marginals}
\\begin{tabular}{l c c c c c c c c}
\\toprule
Model & \\multicolumn{2}{c}{Original data} &
\\multicolumn{2}{c}{Gender sweep @ 0.65} &
\\multicolumn{2}{c}{Label sweep @ 0.30} &
\\multicolumn{2}{c}{Label sweep @ 0.35} \\\\
 & DP & Acc & DP & Acc & DP & Acc & DP & Acc \\\\
\\midrule
""" + '\n'.join(lines) + """\\bottomrule
\\end{tabular}
\\end{table}
""")
print('wrote tex/matched_marginals_table.tex')

# SSSD EO table LaTeX
slines = []
for m in ['logistic regression', 'decision tree', 'random forest']:
    for exp in ['vary sensitive', 'vary label']:
        r = ss[(ss.model == m) & (ss.experiment == exp)].iloc[0]
        slines.append(
            f"{m:20s} & {exp:14s} & {r.baseline_eo:+.4f} & {r.eo_abs_min:.4f} & "
            f"{r.eo_abs_median:.4f} & {r.eo_abs_max:.4f} & {r.eo_abs_mean:.4f} & "
            f"{r.frac_runs_below_003*100:.0f}\\% \\\\")
with open(f'{OUT_TEX}/sssd_eo_table.tex', 'w') as fh:
    fh.write("""\\begin{table}[ht]
\\centering\\footnotesize
\\caption{SSSD: Equality-of-Opportunity gap (|EO|, 5-fold CV means over the 19-point
grid) under gender rebalancing vs. label rebalancing. Label rebalancing moves the
EO gap below 0.03 in most grid settings; gender rebalancing leaves |EO| up to
$\\approx$0.22. The RQ3/RQ2 claim that gender rebalancing is preferable for SSSD
is therefore not supported by the data; label rebalancing is the better EO
reducer, consistent with RQ4 and the Discussion.}
\\label{tab:sssd_eo}
\\begin{tabular}{l l c c c c c c}
\\toprule
Model & Sweep & Base |EO| & min |EO| & median |EO| & max |EO| & mean |EO| & \\% runs $<$0.03 \\\\
\\midrule
""" + '\n'.join(slines) + """\\bottomrule
\\end{tabular}
\\end{table}
""")
print('wrote tex/sssd_eo_table.tex')

# significance LaTeX table (same layout as paper's tab:full_significance)
sigp = sig.pivot_table(index=['dataset', 'experiment'], columns=['model', 'metric'],
                       values=['p_value', 'improvement'])
with open(f'{OUT_TEX}/significance_table.tex', 'w') as fh:
    fh.write("% Re-computed from corrected synthetic data (Synthetic_data_corrected),\n")
    fh.write("% Male-Female convention throughout; one-sample t-test, n=19 grid points.\n")
    fh.write(sig.to_latex(index=False, float_format=lambda x: f'{x:.4f}',
                          caption='Recomputed significance of fairness changes (p-values and improvement '
                                  '$|\\mathrm{baseline}|-|\\mathrm{mean\\ synthetic}|$), corrected data.',
                          label='tab:significance_corrected'))
print('wrote tex/significance_table.tex')
print('\nDONE. CSV files in', OUT_CSV, '; LaTeX in', OUT_TEX)
