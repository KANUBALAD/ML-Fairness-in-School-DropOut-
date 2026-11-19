#!/usr/bin/env python3
"""
regenerate_synthetic_corrected.py
=================================
Regenerates the synthetic experiment data with CORRECTED semantics.

Problems found in the original generation (Synthetic_data/ + results/):
  [B1] --label_ratio was applied to the *most frequent* label value, which is
        the NEGATIVE class for all three datasets ("Graduate", "No",
        "NOT DROPOUT").  So the vary-label sweep was INVERTED: filename ratio
        X produced (1 - X) dropout share.
  [B2] --sensitive_ratio was applied to the *second-most-frequent* gender
        value, which is Male for africa/india but Female for brazil, so the
        gender sweep direction was inconsistent across datasets.
  [B3] The vary-sensitive files used label_ratio = 0.50 instead of the
        original dropout prevalence (paper claims "label fixed at original").
  [B4] SATDAP synthetic files never contained "Enrolled" (mode collapse of
        the 3-class target under binary conditioning).

Corrected semantics (this script):
  * Filename  Sens_X_Label_Y  means:  X% MALE (privileged),  Y% DROPOUT.
  * vary_label:      male share fixed at the original value,
                     dropout share swept 0.05 .. 0.95 (step 0.05).
  * vary_sensitive:  male share swept 0.05 .. 0.95 (step 0.05),
                     dropout share fixed at the original value.
  * One CTGAN is trained per dataset; all 38 target settings are sampled from
    it with SDV conditional sampling; the conditioned columns are overwritten
    with the requested values afterwards (exact marginals up to rounding).
  * SATDAP's negative class is split between "Graduate" and "Enrolled"
    according to their original proportions (fixes B4).

USAGE:
  python regenerate_synthetic_corrected.py --dataset brazil
  python regenerate_synthetic_corrected.py --dataset africa --epochs 100
  (run one process per dataset; africa takes the longest)
"""

import os, sys, argparse, time, warnings
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sdv.single_table import CTGANSynthesizer
from sdv.sampling import Condition
from sdv.metadata import SingleTableMetadata
warnings.filterwarnings('ignore')

# Verified value maps (see recompute_risk_metrics.py and the paper's EDA)
SPEC = {
    'brazil': dict(sens_col='Gender', male='0', female='1',
                   label_col='Target', pos='Dropout',
                   neg_vals=['Graduate', 'Enrolled']),      # 3-class target
    'africa': dict(sens_col='gender', male='Male', female='Female',
                   label_col='dropout', pos='Yes',
                   neg_vals=['No']),
    'india':  dict(sens_col='STUDENTGENDER', male='M', female='F',
                   label_col='STUDENT_DROPOUT_STATUS', pos='DROPOUT',
                   neg_vals=['NOT DROPOUT']),
}
RATIO_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)  # 0.05..0.95, 19 steps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['brazil', 'africa', 'india'])
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=500)  # must be a multiple of the pac (10)
    ap.add_argument('--outdir', default='Synthetic_data_corrected')
    ap.add_argument('--verbose', action='store_true', default=False)
    args = ap.parse_args()

    ds = args.dataset
    spec = SPEC[ds]
    cfg = yaml.safe_load(open(os.path.join('yaml', f'{ds}.yaml')))
    data = pd.read_csv(cfg['datapath'], encoding='ISO-8859-1' if ds == 'india' else None)
    n = len(data)

    # ---- value coercion: keep original dtypes so conditioning matches CTGAN
    sens_col, label_col = spec['sens_col'], spec['label_col']
    for v in [spec['male'], spec['female']]:
        assert (data[sens_col].astype(str).astype(str) == str(v)).any() or \
               (data[sens_col].astype(str) == v).any(), f"value {v} not found in {sens_col}"
    def coerce(vals):
        out = []
        for v in vals:
            if str(v) in set(data[sens_col].astype(str)):
                out.append(data[sens_col][data[sens_col].astype(str) == str(v)].iloc[0])
            else:
                out.append(v)
        return out
    male_v, female_v = coerce([spec['male'], spec['female']])
    pos_v = data[label_col][data[label_col].astype(str) == str(spec['pos'])].iloc[0]
    neg_vs = [data[label_col][data[label_col].astype(str) == str(x)].iloc[0] for x in spec['neg_vals']]
    # share of each negative value within the negatives (for SATDAP split)
    neg_frac = np.array([(data[label_col].astype(str) == str(x)).sum() for x in spec['neg_vals']], float)
    neg_frac = neg_frac / neg_frac.sum()

    male_share0 = float((data[sens_col].astype(str) == str(male_v)).mean())
    drop_share0 = float((data[label_col].astype(str) == str(pos_v)).mean())
    print(f"[{ds}] n={n} male_share={male_share0:.4f} dropout_share={drop_share0:.4f}", flush=True)
    print(f"[{ds}] sens values: male={male_v!r} female={female_v!r} | label: pos={pos_v!r} negs={neg_vs}", flush=True)

    # ---- train one CTGAN per dataset
    t0 = time.time()
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(data)
    model = CTGANSynthesizer(metadata=meta, epochs=args.epochs,
                             batch_size=args.batch_size, verbose=args.verbose)
    print(f"[{ds}] training CTGAN ({args.epochs} epochs)...", flush=True)
    model.fit(data)
    print(f"[{ds}] CTGAN trained in {time.time()-t0:.0f}s", flush=True)

    # ---- target settings
    settings = []  # (experiment, male_share, drop_share)
    for r in RATIO_GRID:
        settings.append(('vary_label', male_share0, r))
    for r in RATIO_GRID:
        settings.append(('vary_sensitive', r, drop_share0))

    audit_rows = []
    for exp, m_share, d_share in settings:
        # group counts
        cnt = dict(
            m_pos=int(round(n * m_share * d_share)),
            m_neg=int(round(n * m_share * (1 - d_share))),
            f_pos=int(round(n * (1 - m_share) * d_share)),
            f_neg=int(round(n * (1 - m_share) * (1 - d_share))))
        # conditions: split negative-class rows across the negative values
        conds = []
        for key, s_val, l_val in [('m_pos', male_v, pos_v), ('f_pos', female_v, pos_v)]:
            if cnt[key] > 0:
                conds.append(Condition({sens_col: s_val, label_col: l_val}, num_rows=cnt[key]))
        for key, s_val in [('m_neg', male_v), ('f_neg', female_v)]:
            for i, neg_v in enumerate(neg_vs):
                c = int(round(cnt[key] * neg_frac[i]))
                if c > 0:
                    conds.append(Condition({sens_col: s_val, label_col: neg_v}, num_rows=c))

        t1 = time.time()
        try:
            syn = model.sample_from_conditions(conds)
        except Exception as e:
            print(f"  [{ds}] sample_conditions failed ({e}); falling back to per-condition sampling", flush=True)
            parts = []
            for c in conds:
                num = c.get_num_rows()
                cond = c.get_column_values()
                parts.append(model.sample_from_conditions([Condition(cond, num_rows=num)]))
            syn = pd.concat(parts, ignore_index=True)

        # enforce exact marginals on the conditioned columns
        out = syn.copy()
        n_pos, n_neg, n_m = 0, 0, 0
        pos_rows = (out[label_col].astype(str) == str(pos_v)).values
        m_rows = (out[sens_col].astype(str) == str(male_v)).values
        # we trust the counts came out per condition; just re-assert via direct overwrite per group
        parts = []
        g = out.groupby([(out[sens_col].astype(str) == str(male_v)), (out[label_col].astype(str) == str(pos_v))],
                        sort=False)
        # simpler: rebuild row-by-row target assignment from conditions
        built = []
        idx = 0
        for c in conds:
            num = min(c.get_num_rows(), len(out) - idx)
            block = out.iloc[idx:idx + num].copy()
            cols = c.get_column_values()
            block[sens_col] = cols[sens_col]
            block[label_col] = cols[label_col]
            built.append(block)
            idx += num
        out = pd.concat(built, ignore_index=True)
        out = out.sample(frac=1, random_state=42).reset_index(drop=True)

        pct_m = int(round(100 * m_share))
        pct_d = int(round(100 * d_share))
        fname = f"{ds}_Sens_{pct_m}_Label_{pct_d}.csv"
        odir = os.path.join(args.outdir, ds, exp)
        os.makedirs(odir, exist_ok=True)
        out.to_csv(os.path.join(odir, fname), index=False)

        ach_m = float((out[sens_col].astype(str) == str(male_v)).mean())
        ach_d = float((out[label_col].astype(str) == str(pos_v)).mean())
        audit_rows.append(dict(dataset=ds, experiment=exp, filename=fname,
                               requested_male=m_share, achieved_male=round(ach_m, 4),
                               requested_dropout=d_share, achieved_dropout=round(ach_d, 4),
                               n_rows=len(out)))
        print(f"  [{ds}] {fname}: n={len(out)} male={ach_m:.3f} (req {m_share:.2f}) "
              f"dropout={ach_d:.3f} (req {d_share:.2f})  [{time.time()-t1:.0f}s]", flush=True)

    audit = pd.DataFrame(audit_rows)
    apath = os.path.join(args.outdir, ds, 'generation_audit.csv')
    audit.to_csv(apath, index=False)
    print(f"[{ds}] audit saved to {apath}", flush=True)

if __name__ == '__main__':
    main()
