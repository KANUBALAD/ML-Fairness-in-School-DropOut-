#!/usr/bin/env python3
"""
03_prevalence_tables.py
=======================
Builds prevalence-aware performance tables and updated figures from the
metrics-only re-evaluation (perfold_extended/*_perfold_extended.csv).

Tables (review_fixes/csv):
  04_prevalence_aware_performance.csv  acc / majority / balacc / AUC per grid point
  07_recast_accuracy_costs.csv         paper's accuracy-cost claims recast with
                                       balanced accuracy & AUC (Reviewer 3)

Figures (review_fixes/figures):
  cross_dataset_vary_label_4panels.pdf/.png    DP, EO, Acc, Balanced Acc
  cross_dataset_vary_sensitive_4panels.pdf/.png
  satdap_label_sweep_metrics.pdf               SATDAP: acc vs majority vs balacc vs AUC
  sssd_eo_two_sweeps.pdf                       SSSD EO: label vs gender sweep
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(HERE, 'csv')
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

DS_NAME = {'brazil': 'SATDAP', 'africa': 'SSSD', 'india': 'RDD'}
ORIG_MALE = {'SATDAP': 0.648, 'SSSD': 0.441, 'RDD': 0.348}
GRID_MALE = {'SATDAP': 0.65, 'SSSD': 0.45, 'RDD': 0.35}   # nearest sweep grid point
ORIG_DROP = {'SATDAP': 0.321, 'SSSD': 0.091, 'RDD': 0.120}
COLORS = {'SATDAP': '#d62728', 'SSSD': '#1f77b4', 'RDD': '#2ca02c'}
MARKERS = {'SATDAP': 'o', 'SSSD': 's', 'RDD': '^'}

# ----------------------------------------------------------------- load ----
frames = {}
for ds in ['brazil', 'africa', 'india']:
    f = os.path.join(HERE, 'perfold_extended', f'{ds}_perfold_extended.csv')
    frames[ds] = pd.read_csv(f)
    frames[ds]['dataset'] = frames[ds]['dataset'].map(DS_NAME)

allf = pd.concat(frames.values(), ignore_index=True)
allf['sens_ratio'] = allf['sens_ratio'].fillna(-1)
allf['label_ratio'] = allf['label_ratio'].fillna(-1)
# run-level: mean over folds
runs = allf.groupby(['dataset', 'model', 'run_type', 'experiment', 'sens_ratio', 'label_ratio'])[
    ['acc', 'balacc', 'auc', 'dp_signed', 'eo_signed', 'majority_acc', 'norm_acc']].mean().reset_index()
runs.to_csv(os.path.join(CSV_DIR, 'recomputed_runs_extended.csv'), index=False, float_format='%.6f')

# --------------------------------------------------- Artifact 04 ------------
rows = []
for ds in ['SATDAP', 'SSSD', 'RDD']:
    sub = runs[runs.dataset == ds]
    base = sub[sub.run_type == 'original'].groupby('model').mean(numeric_only=True)
    for exp, xcol, orig in [('vary_label', 'label_ratio', ORIG_DROP[ds]),
                            ('vary_sensitive', 'sens_ratio', ORIG_MALE[ds])]:
        g = sub[(sub.run_type == 'synthetic') & (sub.experiment == exp)]
        for x in sorted(g[xcol].dropna().unique()):
            for m in ['logistic_regression', 'decision_tree', 'random_forest']:
                r = g[(g[xcol] == x) & (g.model == m)].iloc[0]
                rows.append(dict(
                    dataset=ds, experiment=exp, x_axis=xcol, ratio=x,
                    model=m.replace('_', ' '),
                    acc=r.acc, majority_acc=r.majority_acc, norm_acc=r.norm_acc,
                    balacc=r.balacc, auc=r.auc,
                    acc_minus_majority=r.acc - r.majority_acc))
prev = pd.DataFrame(rows)
prev.to_csv(f'{CSV_DIR}/04_prevalence_aware_performance.csv', index=False, float_format='%.6f')

# ------------------------------------------------ Artifact 07 ----------------
# Recast the "21/27/3-point accuracy costs" with prevalence-invariant metrics.
rows7 = []
for ds, dso in [('SATDAP', 'brazil'), ('RDD', 'india'), ('SSSD', 'africa')]:
    sub = runs[runs.dataset == ds]
    base = sub[sub.run_type == 'original'].groupby('model')[['acc', 'balacc', 'auc']].mean()
    for m in ['logistic_regression', 'decision_tree', 'random_forest']:
        b = base.loc[m]
        # gender sweep at nearest grid point to the original male share
        gs = sub[(sub.run_type == 'synthetic') & (sub.experiment == 'vary_sensitive')
                 & (np.isclose(sub.sens_ratio, GRID_MALE[ds]))].set_index('model').loc[m]
        # label sweep nearest original dropout
        lr_orig = ORIG_DROP[ds]
        grid = sorted(sub[sub.experiment == 'vary_label'].label_ratio.dropna().unique())
        lr_near = min(grid, key=lambda g: abs(g - lr_orig))
        ls = sub[(sub.run_type == 'synthetic') & (sub.experiment == 'vary_label')
                 & (np.isclose(sub.label_ratio, lr_near))].set_index('model').loc[m]
        # mid-grid 50%
        lm = sub[(sub.run_type == 'synthetic') & (sub.experiment == 'vary_label')
                 & (np.isclose(sub.label_ratio, 0.50))].set_index('model').loc[m]
        rows7.append(dict(
            dataset=ds, model=m.replace('_', ' '),
            baseline_acc=b.acc, baseline_balacc=b.balacc, baseline_auc=b.auc,
            gen_sweep_at_orig_male_acc=gs.acc, gen_sweep_at_orig_male_balacc=gs.balacc,
            gen_sweep_at_orig_male_auc=gs.auc,
            label_sweep_near_orig_dropout=lr_near, label_sweep_near_orig_acc=ls.acc,
            label_sweep_near_orig_balacc=ls.balacc, label_sweep_near_orig_auc=ls.auc,
            label_sweep_50_acc=lm.acc, label_sweep_50_balacc=lm.balacc, label_sweep_50_auc=lm.auc,
            acc_cost_at_matched_marginals=float(b.acc - min(gs.acc, ls.acc)),
            balacc_cost_at_matched_marginals=float(b.balacc - min(gs.balacc, ls.balacc)),
            auc_cost_at_matched_marginals=float(b.auc - min(gs.auc, ls.auc))))
rec = pd.DataFrame(rows7)
rec.to_csv(f'{CSV_DIR}/07_recast_accuracy_costs.csv', index=False, float_format='%.6f')
print(rec.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print()

# ------------------------------------------------------------ Figures --------
def base_values(ds):
    b = runs[(runs.dataset == ds) & (runs.run_type == 'original')]
    return b.groupby('model')[['dp_signed', 'eo_signed', 'acc', 'balacc', 'auc']].mean().mean()

def make_4panel(experiment, xlabel, stem, xcol):
    fig, axes = plt.subplots(1, 4, figsize=(18.5, 4.0))
    for ax, (k, name) in zip(axes, [('dp_signed', 'Demographic Parity (DP)'),
                                    ('eo_signed', 'Equality of Opportunity (EO)'),
                                    ('acc', 'Accuracy'),
                                    ('balacc', 'Balanced Accuracy')]):
        if k in ('dp_signed', 'eo_signed'):
            ax.axhline(0, color='0.6', linestyle='--', linewidth=1, zorder=1)
        for ds in ['SATDAP', 'SSSD', 'RDD']:
            sub = runs[(runs.dataset == ds) & (runs.run_type == 'synthetic')
                       & (runs.experiment == experiment)]
            agg = sub.groupby(xcol)[k].mean()
            ax.plot(agg.index, agg.values, color=COLORS[ds], marker=MARKERS[ds],
                    markersize=4.5, linewidth=1.8, label=ds, zorder=3)
            x0 = ORIG_MALE[ds] if experiment == 'vary_sensitive' else ORIG_DROP[ds]
            b = base_values(ds)
            v = agg.loc[x0] if x0 in agg.index else b[k]
            ax.scatter([x0], [v], marker='*', s=260, color='black', zorder=5,
                       edgecolor='white', linewidth=0.6)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(name, fontsize=10)
        ax.grid(alpha=0.25)
        if k == 'acc':
            ax.set_ylim(0.45, 1.0)
    axes[0].legend(fontsize=8, loc='best')
    fig.suptitle(f'{stem.replace("_", " ").title()} — model-averaged, 5-fold CV '
                 f'(stars = original data at its own marginal; corrected synthetic data)',
                 fontsize=10)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'{stem}.{ext}'), dpi=160)
    plt.close(fig)

make_4panel('vary_label', 'Dropout share (label ratio)', 'cross_dataset_vary_label_4panels', 'label_ratio')
make_4panel('vary_sensitive', 'Male share (sensitive ratio)', 'cross_dataset_vary_sensitive_4panels', 'sens_ratio')

# SATDAP label sweep: accuracy vs prevalence-invariant metrics
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.0))
sub = runs[(runs.dataset == 'SATDAP') & (runs.run_type == 'synthetic')
           & (runs.experiment == 'vary_label')]
agg = sub.groupby('label_ratio')[['acc', 'majority_acc', 'balacc', 'auc']].mean()
axes[0].plot(agg.index, agg.majority_acc, 'k--', linewidth=1.5, label='Majority-class baseline')
axes[0].plot(agg.index, agg.acc, color='#d62728', marker='o', markersize=4.5, label='Model accuracy')
axes[0].set_xlabel('Dropout share'); axes[0].set_ylabel('Accuracy'); axes[0].set_title('Accuracy vs trivial baseline')
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)
axes[1].plot(agg.index, agg.balacc, color='#1f77b4', marker='s', markersize=4.5, label='Balanced accuracy')
axes[1].axhline(0.5, color='0.6', linestyle='--', linewidth=1)
axes[1].set_xlabel('Dropout share'); axes[1].set_ylabel('Balanced accuracy')
axes[1].set_title('Prevalence-invariant performance'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)
axes[2].plot(agg.index, agg.auc, color='#2ca02c', marker='^', markersize=4.5, label='ROC AUC')
axes[2].axhline(0.5, color='0.6', linestyle='--', linewidth=1)
axes[2].set_xlabel('Dropout share'); axes[2].set_ylabel('AUC')
axes[2].set_title('Ranking performance'); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.25)
fig.suptitle('SATDAP label sweep (model-averaged, 5-fold CV): the U-shaped accuracy curve is '
             'almost entirely the majority-class effect', fontsize=10)
fig.tight_layout()
for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(FIG_DIR, f'satdap_label_sweep_metrics.{ext}'), dpi=160)
plt.close(fig)

# SSSD EO: label vs gender sweep
fig, ax = plt.subplots(figsize=(7.5, 4.0))
for exp, xcol, lab, col, mk in [('vary_label', 'label_ratio', 'Label rebalancing', '#1f77b4', 's'),
                                ('vary_sensitive', 'sens_ratio', 'Gender rebalancing', '#d62728', 'o')]:
    sub = runs[(runs.dataset == 'SSSD') & (runs.run_type == 'synthetic') & (runs.experiment == exp)]
    agg = sub.groupby(xcol).eo_signed.mean()
    ax.plot(agg.index, agg.abs() if False else agg, color=col, marker=mk, markersize=5, linewidth=1.8, label=lab)
ax.axhline(0, color='0.6', linestyle='--', linewidth=1)
b = base_values('SSSD')
ax.scatter([ORIG_DROP['SSSD']], [b['eo_signed']], marker='*', s=260, color='black', zorder=5,
           edgecolor='white', linewidth=0.6, label='Baseline (orig. data)')
ax.set_xlabel('Ratio varied'); ax.set_ylabel('EO (male $-$ female TPR)')
ax.set_title('SSSD: label rebalancing drives EO to $\\approx$0; gender rebalancing does not')
ax.legend(fontsize=8); ax.grid(alpha=0.25)
fig.tight_layout()
for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(FIG_DIR, f'sssd_eo_two_sweeps.{ext}'), dpi=160)
plt.close(fig)

print('wrote', os.listdir(FIG_DIR))
print('DONE')
