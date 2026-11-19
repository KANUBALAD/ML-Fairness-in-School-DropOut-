# Rebuttal plan: reviewer concerns, evidence, and new artifacts

> Scope of this package: **no experiment re-runs**. No CTGAN is retrained and no
> synthetic data is regenerated. Everything below is (a) post-processing of the
> saved per-fold results in `recompute_output/*_perfold.csv` and the saved
> synthetic files in `Synthetic_data_corrected/`, or (b) a metrics-only
> re-evaluation of those already-generated files with the *same* protocol
> (`StratifiedKFold(5, shuffle=True, random_state=42)`, LR/DT/RF with
> `random_state=42`) extended with prevalence-invariant metrics (balanced
> accuracy, ROC AUC). The re-evaluation was verified to reproduce the existing
> `recompute_output` numbers exactly (max diff = 0.0).
> No paper text has been changed; each fix below says what text change it enables.

Conventions used throughout: S=1 = Male (privileged), y=1 = dropout,
DP = P(ŷ=1|M) − P(ŷ=1|F), EO = TPR(M) − TPR(F), all 5-fold CV.

---

## Reviewer 1 — fairness gains are built into the generator

**What the reviewer says.** `build_target_specs` sets every (gender, dropout)
cell count to `N × P(gender) × P(dropout)`, so gender ⊥ dropout in every
synthetic dataset. The paper's own sweeps show DP ≈ 0 already at the original
marginals (dropout 0.32; male share 0.65), so "rebalancing reduces disparity"
cannot be separated from "the generator removed the correlation". C1 and C2
depend on that separation.

**Verified against the saved data (yes).**

| | P(drop|M) | P(drop|F) | conditional gap |
|---|---|---|---|
| SATDAP original | 0.251 | 0.451 | **−0.199** |
| SATDAP synthetic (any grid point) | = label ratio | = label ratio | ≤ 0.0002 |
| SSSD original | 0.078 | 0.102 | −0.024 |
| RDD original | 0.159 | 0.099 | +0.060 |

Baseline DP (SATDAP) is −0.201 (LR), −0.177 (DT), −0.187 (RF). At the
original male share 0.65 (gender sweep) DP is +0.001/+0.010/+0.006; at dropout
0.30 (label sweep) DP is −0.002/+0.001/+0.002. The reviewer is right.

**New artifacts.**
- `csv/01_joint_audit.csv` — original vs synthetic joint/gender-conditional
  dropout rates for every dataset and representative grid points.
- `csv/02_matched_marginals.csv` — baseline vs synthetic **at the original
  marginals** (no rebalancing), per model: DP/EO/accuracy.
- `tex/matched_marginals_table.tex` — publishable version.
- `tex/joint_tables.tex` — corrected appendix joint tables (the current
  appendix tables contain stale numbers such as 47.49 % that no longer match
  the corrected data).

**How to fix the text (proposed, not applied).** Instead of fighting the
criticism, *convert it into the finding*:
- State explicitly (Methods, Algorithm 1) that the generator enforces
  A ⊥ Y, i.e. cell counts are the product of the target marginals.
- Re-frame C1/C2: "the observed DP disparity in SATDAP is attributable to the
  gender–dropout association (conditional gap −0.199), not to the marginal
  male share or dropout rate alone: in synthetic data with identical marginals
  but no association, DP collapses to ≈ 0 without any rebalancing".
- Delete/replace every sentence that attributes the DP improvement to
  "label balancing" or "sensitive rebalancing" *per se*; attribute it to the
  removal of the association, and note that real-world rebalancing (which
  preserves or alters the association) is a different operation.
- C2 (intervention effectiveness depends on bias profiles) must be re-derived
  from what remains measurable: the *association gap* (P(drop|M) − P(drop|F))
  predicts the DP sign/magnitude, and the label sweep is then a stress test of
  the no-association regime rather than an intervention.
- The "diagnostic framework" (C3) becomes: measure the conditional gap and the
  marginal imbalance separately; the conditional gap is the part that fairness
  metrics respond to.

---

## Reviewer 2 — the SATDAP accuracy loss has two different explanations

**What the reviewer says.** Gender sweep drops SATDAP accuracy to ≈0.68 with
the dropout rate unchanged → paper blames generator signal loss. Label sweep
mid-grid gives ≈0.67 → paper calls it the "≈21-point cost of label
rebalancing". Same drop, two stories.

**Verified (yes, with corrected numbers).** SATDAP baseline acc = 0.876 / 0.809
/ 0.874 (LR/DT/RF). Gender sweep at original male share 0.65: 0.677 / 0.568 /
0.676. Label sweep at dropout 0.30 (nearest grid point to 0.32): 0.699 / 0.588
/ 0.698; at 0.35: 0.645 / 0.551 / 0.640. Both sweeps lose ≈18–24 points at
matched marginals; the drop is the *same* phenomenon — replacing real data
with CTGAN output — and must carry the same name everywhere.

**New artifacts.**
- `csv/03_satdap_accuracy_decomposition.csv` — baseline, both sweeps at
  matched marginals, and mid-grid, with the majority-class baseline at each
  dropout rate.
- `csv/07_recast_accuracy_costs.csv` — the accuracy "costs" recast with
  balanced accuracy and AUC (needs the metrics-only re-evaluation; see
  Reviewer 3).
- `csv/recomputed_runs_extended.csv` — run-level table with acc, balacc, AUC,
  DP, EO for all datasets/experiments/ratios/models.

**How to fix the text (proposed, not applied).**
- Use one term for the drop in both sweeps: "CTGAN synthetic-replacement cost"
  (or "generator fidelity gap"), quantified once (≈18–24 pts at matched
  marginals for SATDAP; balanced-accuracy/AUC version in `07_...csv`).
- Remove "the ≈21-point cost of label rebalancing" from Abstract/Intro/RQ3 and
  replace with the decomposition: (i) synthetic-replacement cost at matched
  marginals, (ii) prevalence change thereafter (Reviewer 3).

---

## Reviewer 3 — accuracy is compared across different dropout rates

**What the reviewer says.** Majority-class guessing scores 0.88 at 12 %
dropout and 0.50 at 50 %; the "21/27/3-point" costs mostly reflect prevalence,
not the safeguard. Balanced accuracy or AUC is needed.

**Verified (yes).** SATDAP label sweep: model accuracy tracks the
majority-class baseline almost exactly across the grid (0.950 at 5 % dropout →
0.500 at 50 %; LR/RF literally equal the majority rate at every endpoint).
RDD's synthetic data has AUC ≈ 0.51–0.54 (no ranking signal) vs 0.77–0.80 on
the real data. Any accuracy delta across grid points is prevalence change.

**New artifacts (metrics-only re-evaluation of saved files).**
- `perfold_extended/{brazil,africa,india}_perfold_extended.csv` — per-fold
  acc, balanced acc (overall + per group), ROC AUC, majority baseline,
  normalized accuracy, confusion counts. Verified to reproduce the original
  DP/EO/acc values exactly.
- `csv/04_prevalence_aware_performance.csv` — per ratio × model × dataset:
  acc, majority, acc − majority, balanced acc, AUC.
- `csv/07_recast_accuracy_costs.csv` — e.g. SATDAP LR: baseline acc/balacc/AUC
  0.876/0.837/0.921 → synthetic at matched marginals ≈0.68/0.50/0.57 → label
  0.50 ≈0.54/0.54/0.57. Across the whole label grid SATDAP's balanced accuracy
  stays ≈0.50 and AUC ≈0.53 — the U-shaped accuracy curve is entirely the
  majority-class effect, not a cost of rebalancing (RDD behaves the same:
  AUC ≈0.51–0.55 on synthetic vs 0.77–0.80 on real data; SSSD retains signal:
  AUC ≈0.93–0.95).
- `figures/satdap_label_sweep_metrics.pdf` — the U-shaped accuracy curve next
  to the majority baseline, balanced accuracy, and AUC.
- `figures/cross_dataset_vary_{label,sensitive}_4panels.pdf` — the paper's
  headline figures with a 4th panel (balanced accuracy).
- `tex/prevalence_satdap_table.tex`, `tex/recast_costs_table.tex` — the two
  publishable tables for the accuracy reframing.

**How to fix the text (proposed, not applied).**
- Replace every cross-prevalence accuracy comparison with balanced accuracy
  and/or AUC, or report accuracy only at fixed prevalence; state the
  majority-class baseline whenever accuracy is shown.
- Recompute "accuracy cost" claims using balanced accuracy/AUC at matched
  prevalence (use `07_recast_accuracy_costs.csv`).

---

## Reviewer 4 — RQ3 verdict contradicts the paper's own tables

**What the reviewer says.** RQ3 (and RQ2) says gender rebalancing is
preferable for SSSD; the tables show label rebalancing cuts SSSD's EO gap to
≤ 0.03 while gender rebalancing leaves it up to 0.12; RQ4 and the Discussion
agree with the tables.

**Verified (yes, corrected data).** SSSD baseline EO = +0.241 (LR), −0.326
(DT), −0.325 (RF). Over the 19-point grid (5-fold CV run means):
- Label rebalancing: |EO| min 0.001–0.002, median 0.022–0.025, max
  0.031–0.084; 79–95 % of grid points have |EO| < 0.03.
- Gender rebalancing: |EO| min 0.004–0.016, median 0.055–0.113, max
  0.181–0.223; only 11–32 % of grid points are below 0.03.
Label rebalancing is the better EO reducer — matching RQ4/Discussion, not the
RQ3/RQ2 verdict. (The recomputed significance table also shows Vary Label
improves SSSD EO by 0.225–0.309 vs 0.119–0.280 for Vary Sensitive.)

**New artifacts.**
- `csv/05_sssd_eo_intervention.csv` — per model/sweep: baseline |EO|, min /
  median / max / mean |EO|, fraction of grid points below 0.03.
- `tex/sssd_eo_table.tex` — publishable version.
- `csv/06_significance_recomputed.csv` + `tex/significance_table.tex` — the
  significance table recomputed on the corrected data (single Male−Female
  convention), replacing `tab:full_significance`.
- `figures/sssd_eo_two_sweeps.pdf` — EO trajectory for both sweeps.

**How to fix the text (proposed, not applied).**
- Flip the RQ3 verdict for SSSD: "for SSSD, label rebalancing reduces the EO
  gap to ≤ 0.03 across most of the grid, whereas gender rebalancing leaves
  |EO| up to ≈ 0.22; label rebalancing is therefore the preferable EO
  intervention for SSSD as well".
- RQ2's "High EO Bias → Sensitive Attribute Intervention" bullet must be
  replaced: the SSSD evidence shows label rebalancing is the EO-reducer. A
  defensible replacement rule: high DP + large conditional dropout gap → the
  gap is the lever (label-side); high EO with sparse group–label cells → label
  rebalancing still wins on EO.
- Align the Abstract, C1, and Discussion claims with these numbers.

---

## Cross-cutting changes implied by all four reviews

1. Methods: document that synthetic cell counts are product-of-marginals
   (A ⊥ Y enforced), and add the majority-class baseline + balanced
   accuracy/AUC to the evaluation protocol (they are already in the
   re-evaluation CSVs, so no new runs are needed).
2. Appendix: replace the stale joint tables with `tex/joint_tables.tex`.
3. Figures: swap in the 4-panel figures (balanced accuracy added) and the
   SSSD EO figure.
4. Contributions 1–3 re-stated around the association gap (see Reviewer 1).

## How the numbers were produced (reproducibility)

- `review_fixes/01_posthoc_tables.py` — post-processing only.
- `review_fixes/02_extended_reeval.py` — metrics-only re-evaluation of the
  saved `Synthetic_data_corrected/` files (same models, same CV seeds; adds
  balanced accuracy/AUC; saves per-fold predictions under
  `perfold_extended/predictions/`).
- `review_fixes/03_prevalence_tables.py` — tables 04/07 and figures.
- All outputs in `review_fixes/csv`, `review_fixes/tex`,
  `review_fixes/figures`.
