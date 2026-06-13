"""One-time data prep: mirror the v1-legacy statpy databases into v2 format.

Walks every v1 custom-JSON statpy DB under SRC_ROOT (files whose name contains
``.sample``), converts each with ``statpy.database.io.load_v1_json``, and writes
the v2 pickle to the mirrored path under DST_ROOT with a ``.db`` name. Raw data
in the tree (``.npy`` fields/gauge configs, Grid checkpoints, scripts) is not a
statpy database and is skipped.

Each converted entry is validated against the values stored in the v1 file: the
re-derived ``mean`` (all tags) and ``jks`` (a sample of cfgs) must reproduce the
v1 leave-one-out jackknife. Run under the statpy worktree venv:

    cd ../libs/statpy-v1migrate && uv run python <abs path>/scripts/build_v2_data.py
"""
import json
import os

import numpy as np

from statpy.database.io import load_v1_json, decode_v1_ndarray

SRC_ROOT = os.path.expanduser("~/workspace/data/RBC-UKQCD/v1-legacy-format")
DST_ROOT = os.path.expanduser("~/workspace/data/RBC-UKQCD/v2-format")

RTOL = 1e-10


def out_name(fname):
    """Map a v1 db filename to its v2 ``.db`` name (blinded-ness stays in the dir)."""
    stem = fname.replace(".sample.blinded", "").replace(".sample", "")
    return stem + ".db"


def find_v1_dbs():
    """Relative paths of all v1 statpy DB files under SRC_ROOT."""
    out = []
    for dp, _, fns in os.walk(SRC_ROOT):
        for fn in fns:
            if ".sample" in fn:
                out.append(os.path.relpath(os.path.join(dp, fn), SRC_ROOT))
    return sorted(out)


def validate(src, db):
    """Compare derived mean/jks to the v1 file, where the v1 file cached them.

    Some v1 leaves store ``mean``/``jks`` as null (only ``sample`` is kept); for
    those there is nothing to compare against -- the recompute is the same
    deterministic path covered by the leaves that did cache values. Returns the
    worst relative errors plus how many tags/cfgs were actually cross-checked.
    """
    with open(src) as f:
        raw = json.load(f)
    max_mean_err = max_jks_err = 0.0
    n_mean = n_jks = 0
    for tag, wrapped in raw.items():
        leaf = wrapped["__leaf__"]
        entry = db.database[tag]
        if leaf.get("mean") is not None:
            max_mean_err = max(max_mean_err, _relerr(entry.mean, decode_v1_ndarray(leaf["mean"])))
            n_mean += 1
        jks = leaf.get("jks")
        if isinstance(jks, dict):
            idx = {c: i for i, c in enumerate(entry.cfgs)}
            for c in list(jks)[:8]:
                if jks[c] is None:
                    continue
                max_jks_err = max(max_jks_err, _relerr(entry.jks[idx[c]], decode_v1_ndarray(jks[c])))
                n_jks += 1
    return max_mean_err, max_jks_err, n_mean, n_jks


def _relerr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    denom = np.maximum(np.abs(b), 1e-300)
    return float(np.max(np.abs(a - b) / denom))


def main():
    rels = find_v1_dbs()
    print(f"Found {len(rels)} v1 statpy DB files under {SRC_ROOT}\n")
    failures = []
    for i, rel in enumerate(rels, 1):
        src = os.path.join(SRC_ROOT, rel)
        dst_rel = os.path.join(os.path.dirname(rel), out_name(os.path.basename(rel)))
        dst = os.path.join(DST_ROOT, dst_rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            db = load_v1_json(src, silent=True)
            mean_err, jks_err, n_mean, n_jks = validate(src, db)
            db.save(dst)
            ok = mean_err < RTOL and jks_err < RTOL
            status = "PASS" if ok else "FAIL"
            if not ok:
                failures.append((rel, mean_err, jks_err))
            print(f"[{i:2d}/{len(rels)}] {status}  {rel}  ->  {dst_rel}  "
                  f"(n={len(db.database)}, mean_relerr={mean_err:.1e}, jks_relerr={jks_err:.1e}, "
                  f"checked={n_mean}/{len(db.database)} means, {n_jks} jks)")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures.append((rel, "ERROR", repr(exc)))
            print(f"[{i:2d}/{len(rels)}] ERROR {rel}: {exc!r}")

    print(f"\n{'='*60}")
    print(f"Converted {len(rels) - len(failures)}/{len(rels)} files into {DST_ROOT}")
    if failures:
        print(f"{len(failures)} FAILURES:")
        for rel, a, b in failures:
            print(f"  {rel}: {a} {b}")
        return 1
    print("All files migrated and validated against their v1 mean/jks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
