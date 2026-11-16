#!/usr/bin/env python3
"""
Generic CTGAN based synthetic data generator
===========================================

usage:
    python CTGAN_synthetic_generator.py  yaml/brazil.yaml \
           --balanced                                   \
           --total_rows   10000                         \
           --outdir       ./synthetic_files

    python CTGAN_synthetic_generator.py  yaml/africa.yaml \
           --sensitive_ratio 0.30 --label_ratio 0.45

Key cli flags
-------------
--balanced                 Force the four (priv/unpriv × pos/neg) groups
                           to contain the same number of rows.
--sensitive_ratio  <float> Desired share of the *un-privileged* sensitive
                           value  (0.10 → 10 %   … 0.95 → 95 %).
--label_ratio      <float> Desired share of the *positive* label value.
--total_rows       <int>   Synthetic rows to create
                           (defaults to size of original data).
--sensitive_col / --label_col   Manually specify columns
--privileged_val  / --unprivileged_val  Override sensitive values
--positive_val    / --negative_val      Override label values
--epochs / --batch_size  SDV-CTGAN training parameters
--outdir                   Where to store the .csv files
--keep_original            If given, we append original rows to the
                           synthetic ones (“augmented” file).
"""
import argparse, yaml, sys, json, math, warnings
from pathlib import Path
import numpy  as np
import pandas as pd
from typing import Dict, List
from dynamic_schema_extractor import DynamicSchemaExtractor

# -----------------------------------------------------------------------------
# SDV import with graceful fallback
# -----------------------------------------------------------------------------
try:
    from sdv.single_table import CTGANSynthesizer
    from sdv.sampling      import Condition
    from sdv.metadata      import SingleTableMetadata
    print("✓  Using SDV ≥ 1.3 single-table API")
except ImportError:                        # old SDV (<1.3)
    from sdv.tabular import CTGAN as CTGANSynthesizer   # type: ignore
    Condition = None                       # noqa
    SingleTableMetadata = None             # noqa
    print("⚠  Falling back to legacy sdv.tabular. Conditional sampling limited")

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------#
# Helper functions                                                             #
# -----------------------------------------------------------------------------#
def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

def detect_columns(df: pd.DataFrame,
                   extractor: DynamicSchemaExtractor,
                   override_sensitive: str = None,
                   override_label: str     = None) -> Dict:
    """Return {'sens_col': …, 'label_col': …, 'sens_vals': (priv, unpriv),
               'label_vals': (positive, negative)}"""
    schema = extractor.extract_comprehensive_schema(df)
    sens_col = override_sensitive or (
        schema["relationships"]["sensitive_column_candidates"][0]
        if schema["relationships"]["sensitive_column_candidates"] else None)
    label_col = override_label or (
        schema["relationships"]["label_column_candidates"][0]
        if schema["relationships"]["label_column_candidates"] else None)

    if sens_col is None or label_col is None:
        raise ValueError("Could not automatically detect sensitive / label "
                         "columns.  Please supply --sensitive_col / --label_col")

    # Pick the most frequent value as “privileged / negative”
    sens_counts  = df[sens_col].value_counts()
    label_counts = df[label_col].value_counts()

    sens_vals  = tuple(sens_counts.index[:2])   # (priv, unpriv)
    label_vals = tuple(label_counts.index[:2])  # (positive, negative)

    return dict(sens_col=sens_col, label_col=label_col,
                sens_vals=sens_vals, label_vals=label_vals)

def build_target_specs(total_rows: int,
                       balanced: bool,
                       sens_ratio: float,
                       label_ratio: float,
                       sens_vals: tuple, label_vals: tuple) -> List[Dict]:
    """
    Return a list of dicts
       [{'cond': {sens_col: val, label_col: val}, 'n': rows_to_generate}, …]
    """

    # helper to compute counts
    def _count(sr: float, lr: float, priv: bool, pos: bool) -> int:
        p_s = 1.0 - sr if priv else sr
        p_l = lr if pos else 1.0 - lr
        return int(round(total_rows * p_s * p_l))

    if balanced:
        group_rows = int(round(total_rows / 4))
        counts = [group_rows] * 4
    else:
        counts = [
            _count(sens_ratio, label_ratio, priv=True , pos=True ),  # priv/pos
            _count(sens_ratio, label_ratio, priv=True , pos=False),  # priv/neg
            _count(sens_ratio, label_ratio, priv=False, pos=True ),  # unpriv/pos
            _count(sens_ratio, label_ratio, priv=False, pos=False)   # unpriv/neg
        ]

    targets = []
    for n, (s_val, l_val) in zip(counts,
                                 [(sens_vals[0], label_vals[0]),
                                  (sens_vals[0], label_vals[1]),
                                  (sens_vals[1], label_vals[0]),
                                  (sens_vals[1], label_vals[1])]):
        if n <= 0:                                                    # skip empty
            continue
        targets.append({"cond": { }, "n": n})
        targets[-1]["cond"]["__sens_val__"]  = s_val
        targets[-1]["cond"]["__label_val__"] = l_val
    return targets

def sample_with_condition(model: CTGANSynthesizer,
                          cond_dict: Dict,
                          n_rows: int) -> pd.DataFrame:
    """Try conditional sampling first.  If not supported fall back to
       rejection sampling."""
    if Condition is not None:
        try:
            cond = Condition(
                {k: v for k, v in cond_dict.items() if not k.startswith("__")},
                num_rows=n_rows)
            return model.sample_conditions([cond])
        except Exception:
            pass

    # ---- Manual rejection sampling fallback ---------------------------------
    collected = []
    tries, max_tries = 0, 10
    while len(collected) < n_rows and tries < max_tries:
        tries += 1
        batch = model.sample(num_rows=max(1000, n_rows))
        mask = True
        for k, v in cond_dict.items():
            if k.startswith("__"):        # skip meta keys
                continue
            mask &= (batch[k] == v)
        collected.append(batch[mask])

    out = pd.concat(collected).reset_index(drop=True)
    return out.head(n_rows)

# -----------------------------------------------------------------------------#
# Core generator                                                               #
# -----------------------------------------------------------------------------#
def generate_synthetic(df: pd.DataFrame,
                       sens_col: str, label_col: str,
                       targets: List[Dict],
                       epochs: int = 100, batch_size: int = 500,
                       verbose: bool = True) -> pd.DataFrame:

    # ------------- metadata ---------------------------------------------------
    if SingleTableMetadata:                         # SDV ≥1.3
        meta = SingleTableMetadata()
        meta.detect_from_dataframe(df)
        model = CTGANSynthesizer(metadata=meta,
                                 epochs=epochs, batch_size=batch_size,
                                 verbose=verbose)
    else:                                           # legacy
        model = CTGANSynthesizer(epochs=epochs,
                                 batch_size=batch_size,
                                 verbose=verbose)

    print(f"Training CTGAN for {epochs} epochs on {len(df)} rows …")
    model.fit(df)
    print("✓  CTGAN training finished")

    # ------------- generation -------------------------------------------------
    synthetic_parts = []
    for spec in targets:
        n  = spec["n"]
        cd = {sens_col: spec["cond"]["__sens_val__"],
              label_col: spec["cond"]["__label_val__"]}
        print(f"    Sampling {n:>6} rows  where "
              f"{sens_col}={cd[sens_col]} & {label_col}={cd[label_col]}")
        part = sample_with_condition(model, cd, n)
        # enforce correctness (robust against sampling errors)
        part[sens_col]  = cd[sens_col]
        part[label_col] = cd[label_col]
        synthetic_parts.append(part)

    synthetic = pd.concat(synthetic_parts).sample(frac=1).reset_index(drop=True)
    return synthetic

# -----------------------------------------------------------------------------#
# CLI                                                                          #
# -----------------------------------------------------------------------------#
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generic CTGAN synthetic data generator")
    parser.add_argument("config", type=Path, help="yaml/<dataset>.yaml file")

    # generation profile -------------------------------------------------------
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--balanced", action="store_true",
                       help="create fully balanced 4-way dataset")
    group.add_argument("--sensitive_ratio", type=float, default=None,
                       help="share of un-privileged records (0.1 … 0.95)")
    parser.add_argument("--label_ratio", type=float, default=None,
                        help="share of positive label records")

    parser.add_argument("--total_rows",  type=int, default=None,
                        help="rows to synthesise (default = len(original))")
    # column overrides ---------------------------------------------------------
    parser.add_argument("--sensitive_col",      type=str)
    parser.add_argument("--label_col",          type=str)
    parser.add_argument("--privileged_val",     type=str)
    parser.add_argument("--unprivileged_val",   type=str)
    parser.add_argument("--positive_val",       type=str)
    parser.add_argument("--negative_val",       type=str)

    # CTGAN hyper-params -------------------------------------------------------
    parser.add_argument("--epochs",      type=int, default=100)
    parser.add_argument("--batch_size",  type=int, default=512)

    # misc ---------------------------------------------------------------------
    parser.add_argument("--outdir",        type=Path, default=Path("./synthetic"))
    parser.add_argument("--keep_original", action="store_true",
                        help="append original rows (augmented file)")
    args = parser.parse_args()

    # ------------------------------- set-up -----------------------------------
    cfg  = load_yaml(args.config)
    data = pd.read_csv(cfg["datapath"])
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # NEW: Use YAML column specifications if available
    yaml_sensitive = cfg.get("sensitive_column")
    yaml_label = cfg.get("label_column")
    
    print(f"Dataset: {cfg['dataname']}")
    if yaml_sensitive:
        print(f"YAML specifies sensitive column: {yaml_sensitive}")
    if yaml_label:
        print(f"YAML specifies label column: {yaml_label}")

    extractor = DynamicSchemaExtractor()
    col_info  = detect_columns(data, extractor,
                               override_sensitive=args.sensitive_col or yaml_sensitive,
                               override_label=args.label_col or yaml_label)

    sens_col = col_info["sens_col"]
    label_col= col_info["label_col"]
    sens_vals= col_info["sens_vals"]      # (priv, unpriv)
    label_vals=col_info["label_vals"]     # (pos , neg)

    print(f"✓ Using sensitive column: {sens_col}")
    print(f"✓ Using label column: {label_col}")
    print(f"✓ Sensitive values: {sens_vals} (privileged, unprivileged)")
    print(f"✓ Label values: {label_vals} (positive, negative)")

    # allow cli value overrides -----------------------------------------------
    if args.privileged_val   is not None: sens_vals  = (args.privileged_val, sens_vals[1])
    if args.unprivileged_val is not None: sens_vals  = (sens_vals[0], args.unprivileged_val)
    if args.positive_val     is not None: label_vals = (args.positive_val , label_vals[1])
    if args.negative_val     is not None: label_vals = (label_vals[0], args.negative_val)

    total_rows = (args.total_rows or len(data))
    sensitive_ratio = (args.sensitive_ratio if args.sensitive_ratio is not None else
                       0.5)                       # default 50 %
    label_ratio     = (args.label_ratio     if args.label_ratio     is not None else
                       0.5)

    targets = build_target_specs(total_rows,
                                 balanced=args.balanced,
                                 sens_ratio=sensitive_ratio,
                                 label_ratio=label_ratio,
                                 sens_vals=sens_vals,
                                 label_vals=label_vals)

    # -------------------------- do the generation -----------------------------
    synthetic = generate_synthetic(data,
                                   sens_col, label_col,
                                   targets,
                                   epochs=args.epochs,
                                   batch_size=args.batch_size)

    # ------------------------------ write files -------------------------------
    perc_sens  = int(round(100 * sensitive_ratio)) if not args.balanced else 25
    perc_label = int(round(100 * label_ratio    )) if not args.balanced else 50
    stem       = f"{cfg['dataname']}_Sens_{perc_sens}_Label_{perc_label}"
    syn_path   = outdir / f"{stem}.csv"
    synthetic.to_csv(syn_path, index=False)
    print(f"\n✓  Synthetic file written to {syn_path}")

    if args.keep_original:
        aug_path = outdir / f"{stem}_plusOriginal.csv"
        pd.concat([data, synthetic]).to_csv(aug_path, index=False)
        print(f"✓  Augmented file (original + synthetic) → {aug_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)