#!/usr/bin/env python3
"""
make_cross_dataset_figures.py
=============================
Generates the two headline cross-dataset figures for the AIRES paper from the
CORRECTED synthetic data (ModularCode/recompute_output/recomputed_runs.csv):

  * cross_dataset_comparison_vary_sensitive.pdf/.png
        x-axis = male share (Sens_X = X% male; dropout fixed at the original
        prevalence). Panels: DP, EO, Accuracy.
  * cross_dataset_comparison_vary_label.pdf/.png
        x-axis = dropout share (Label_Y = Y% dropout; male share fixed at the
        original value). Panels: DP, EO, Accuracy. The degenerate endpoints
        (dropout <= 0.20 and >= 0.80, where models output constant
        predictions) are shaded.

Style follows the published figures: one line per dataset (SATDAP red,
SSSD blue, RDD green), values averaged over the three models, baseline
(original-data) values marked with black stars at the original marginal,
horizontal reference line at 0 for DP/EO.

Outputs are written to
paper_dev/aires_latex_template/figures/ (overwrites the old versions;
old ones are backed up in figures_previous/).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'recompute_output', 'recomputed_runs.csv')
OUT = os.path.normpath(os.path.join(HERE, '..', '..', 'paper_dev',
                                    'aires_latex_template', 'figures'))

DATASETS = {'SATDAP': ('#d62728', 'o'),   # red
            'SSSD':   ('#1f77b4', 's'),   # blue
            'RDD':    ('#2ca02c', '^')}   # green
MODELS = ['logistic_regression', 'decision_tree', 'random_forest']
# original marginal values (corrected grid keeps these fixed in each experiment)
ORIG_MALE = {'SATDAP': 0.648, 'SSSD': 0.441, 'RDD': 0.348}
ORIG_DROP = {'SATDAP': 0.321, 'SSSD': 0.091, 'RDD': 0.120}

runs = pd.read_csv(RUNS)
base = runs[runs.run_type == 'original'].groupby('dataset')[['dp', 'eo', 'acc']].mean()

def panel(ax, x, y, ds, color, marker):
    ax.plot(x, y, color=color, marker=marker, markersize=4.5, linewidth=1.8,
            label=ds, zorder=3)

def baselines(ax, ds, x0, metrics):
    for k, name in metrics:
        v = base.loc[ds, k]
        ax.scatter([x0], [v], marker='*', s=260, color='black', zorder=5,
                   edgecolor='white', linewidth=0.6)

def make_figure(experiment, xlabel, out_stem, shade_ends=False):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    for ax, (k, name, ylim) in zip(axes, [
            ('dp', 'Demographic Parity (DP)', None),
            ('eo', 'Equality of Opportunity (EO)', None),
            ('acc', 'Accuracy', (0.5, 1.0))]):
        if shade_ends:
            ax.axvspan(0.05, 0.20, color='0.88', zorder=0)
            ax.axvspan(0.80, 0.95, color='0.88', zorder=0)
        if k in ('dp', 'eo'):
            ax.axhline(0, color='0.6', linestyle='--', linewidth=1, zorder=1)
        for ds, (color, marker) in DATASETS.items():
            sub = runs[(runs.dataset == ds) & (runs.run_type == 'synthetic')
                       & (runs.experiment == experiment)]
            agg = sub.groupby('sens_ratio' if experiment == 'vary_sensitive' else 'label_ratio')[k].mean()
            panel(ax, agg.index, agg.values, ds, color, marker)
            x0 = ORIG_MALE[ds] if experiment == 'vary_sensitive' else ORIG_DROP[ds]
            baselines(ax, ds, x0, [(k, name)])
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(name, fontsize=10)
        ax.grid(alpha=0.25)
        if ylim:
            ax.set_ylim(*ylim)
        ax.tick_params(labelsize=9)
    if shade_ends:
        axes[2].text(0.125, 0.975, 'trivial\n(constant\nprediction)', ha='center', va='top',
                     fontsize=7.5, color='0.35')
        axes[2].text(0.875, 0.975, 'trivial\n(constant\nprediction)', ha='center', va='top',
                     fontsize=7.5, color='0.35')
    handles, labels = axes[0].get_legend_handles_labels()
    star = plt.Line2D([], [], marker='*', linestyle='None', color='black', markersize=12,
                      label='Baseline (original data)')
    fig.legend(handles=handles + [star], labels=labels + ['Baseline (original data)'],
               loc='upper center', ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUT, f'{out_stem}.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_stem}.pdf/.png")

if __name__ == '__main__':
    make_figure('vary_sensitive',
                'Male share (sensitive-attribute ratio)',
                'cross_dataset_comparison_vary_sensitive', shade_ends=False)
    make_figure('vary_label',
                'Dropout share (label ratio)',
                'cross_dataset_comparison_vary_label', shade_ends=True)
