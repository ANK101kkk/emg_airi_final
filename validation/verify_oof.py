"""Score held-day predictions against official AIRI targets, with NumPy alone."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import csv
import json
from pathlib import Path

import numpy as np

JOINTS = [f"{finger}_{joint}" for finger in ("index", "middle", "little", "ring", "thumb")
          for joint in ("mcp_theta", "mcp_fi", "pip", "dip")]
FINGERS = ("Index", "Middle", "Little", "Ring", "Thumb")
SOURCES = ("spectral/predictions", "knn/manifold_hmm", "prior/ordinal")
WEIGHTS = np.array([.375, .375, .25], dtype=np.float32)


def aggregate_sessions(records):
    sessions = defaultdict(list)
    for record in records:
        sessions[record["session_id"]].append(record["wmae_rad"])
    return float(np.mean([np.mean(scores) for scores in sessions.values()]))


def decode(frames, size, spec):
    gain, lead, lag = spec
    if lag:
        previous = np.maximum(0, np.arange(len(frames)) - 1)
        frames = frames + lag * (frames[previous] - frames)
    index = np.arange(size)
    k = index // 25
    elapsed = np.maximum(1, np.minimum(k, 2)).astype(np.float32) * 25
    multiplier = ((gain * (index % 25) + lead) / elapsed).astype(np.float32)
    change = frames[k] - frames[np.maximum(0, k - 2)]
    return frames[k] + np.clip(multiplier[:, None] * change, -.12, .12)


def weighted_mae(errors, action):
    finger_error = errors.reshape(-1, 5, 4).mean(axis=(0, 2))
    if action in ("Fist", "Open", "Close"):
        return float(finger_error.mean())
    active = FINGERS.index(action)
    return float(.5 * (finger_error[active] + np.delete(finger_error, active).mean()))


def summarize(records):
    samples = sum(x["n_samples"] for x in records)
    sum_field = lambda name: sum(x[name] for x in records)
    total = sum_field("sum_y2") - sum_field("sum_y") ** 2 / samples
    pos, neg = sum_field("n_pos"), sum_field("n_neg")
    eligible = (pos >= 100) & (neg >= 100)
    tpr = sum_field("n_pos_correct")[eligible] / pos[eligible]
    tnr = sum_field("n_neg_correct")[eligible] / neg[eligible]
    return dict(recordings=len(records), sessions=len({x["session_id"] for x in records}),
                days=len({x["day_id"] for x in records}), samples=samples, angle_values=20*samples,
                wmae_rad=aggregate_sessions(records), mae_rad=float(sum_field("sum_abs").sum()/(20*samples)),
                r2_uniform_mean=float(np.mean(1-sum_field("sum_sq")/total)),
                within10deg_pct=100*sum_field("n_within10")/(20*samples),
                sign_balanced_pct=50*float(np.mean(tpr+tnr)), sign_eligible_joints=int(eligible.sum()),
                sign_eligible_names=[name for name, mask in zip(JOINTS, eligible) if mask])


def replay(args):
    train = args.data if (args.data/"metadata.csv").is_file() else args.data/"train"
    with (train/"metadata.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    meta = {r["recording_id"]: r for r in rows}
    days = sorted({r["day_id"] for r in rows})
    if len(meta) != len(rows) or len(days) != 7:
        raise ValueError("Expected unique recording IDs and seven held-out days")
    def target(rid):
        return np.load(train/"targets"/(rid+".npy"), mmap_mode="r", allow_pickle=False)
    ranges = {}
    for rid in meta:
        y = target(rid)[1000:]
        ranges[rid] = (y.min(axis=0), y.max(axis=0))
    def limits(ids):
        group = [ranges[rid] for rid in ids]
        return np.stack((np.min([v[0] for v in group], axis=0), np.max([v[1] for v in group], axis=0)))
    all_bounds = limits(meta)
    configs = json.loads(args.spec.read_text())
    if args.model_config is not None:
        production = json.loads(args.model_config.read_text())
        np.testing.assert_array_equal(all_bounds, production["bounds"])
        np.testing.assert_array_equal(WEIGHTS, production["recipe"]["weights"])
        if production["recipe"]["gain"] != configs["fixed"][0]:
            raise ValueError("Production gain differs from the verified fixed recipe")
    data, folds, saved_frames = defaultdict(list), [], {}
    for fold, day in enumerate(days):
        ids = [rid for rid,r in meta.items() if r["day_id"] == day]
        training_ids = [rid for rid,r in meta.items() if r["day_id"] != day]
        bounds = limits(training_ids)
        if frozenset(meta[r]["session_id"] for r in ids) & frozenset(meta[r]["session_id"] for r in training_ids):
            raise ValueError("Train/validation session overlap")
        folds.append(dict(fold=fold, held_day=day, train_days=sorted(set(days)-{day}),
                          training_recordings=len(training_ids), held_recordings=len(ids),
                          days_fully_disjoint=True, lo=bounds[0].tolist(), hi=bounds[1].tolist(),
                          bound_entries_different_from_all=int(np.count_nonzero(bounds != all_bounds)),
                          held_truth_entries_below_train_min=0, held_truth_entries_above_train_max=0))
        with ExitStack() as stack:
            if args.branches:
                banks = [stack.enter_context(np.load(args.branches/directory/(str(fold)+".npz"))) for directory in SOURCES]
                if any(set(bank.files) != set(ids) for bank in banks):
                    raise ValueError(f"Saved prediction IDs do not equal the held-out IDs in fold {fold}")
            else:
                combined = stack.enter_context(np.load(args.frames, allow_pickle=False))
                if set(combined.files) != set(meta):
                    raise ValueError("Frame archive and public recordings differ")
            for rid in ids:
                r = meta[rid]
                y = target(rid)
                if args.branches:
                    frames = sum(bank[rid] * w for bank,w in zip(banks, WEIGHTS))
                    if args.save_frames:
                        saved_frames[rid] = frames
                else:
                    frames = combined[rid]
                if frames.shape != ((len(y)+24)//25, 20) or not np.isfinite(frames).all():
                    raise ValueError(f"Invalid mixed frames in {rid}")
                obs = np.asarray(y[1000:], dtype=np.float64)
                folds[-1]["held_truth_entries_below_train_min"] += int((obs < bounds[0]).sum())
                folds[-1]["held_truth_entries_above_train_max"] += int((obs > bounds[1]).sum())
                for protocol,spec in (("fixed",configs["fixed"]),("nested_filter",configs["nested"][fold])):
                    pred = np.clip(decode(frames, len(y), spec)[1000:], *bounds)
                    err = pred - obs
                    ab = np.abs(err)
                    pos,neg = obs > .05, obs < -.05
                    data[protocol].append(dict(fold=fold, recording_id=rid, day_id=day,
                                         session_id=r["session_id"], action=r["action"], n_samples=len(obs),
                                         wmae_rad=weighted_mae(ab,r["action"]), sum_abs=ab.sum(0),
                                         sum_sq=np.square(err).sum(0), sum_y=obs.sum(0),
                                         sum_y2=np.square(obs).sum(0), n_within10=int((ab<=np.deg2rad(10.)).sum()),
                                         n_pos=pos.sum(0), n_neg=neg.sum(0), n_pos_correct=(pos & (pred>0)).sum(0),
                                         n_neg_correct=(neg & (pred<0)).sum(0)))
    if args.save_frames:
        if not saved_frames:
            raise ValueError("Saving mixed frames requires the three original branch directories")
        np.savez_compressed(args.save_frames, **saved_frames)
    report = {mode:summarize(records) for mode,records in data.items()}
    for mode,metrics in report.items():
        np.testing.assert_allclose(metrics["wmae_rad"], configs["reference"][mode], rtol=0, atol=1e-9)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, value in (("metrics",report), ("fold_bounds",folds)):
        (args.out/(name+".json")).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    for prefix, records, columns in (
        ("by_record", [dict(protocol=mode, **record) for mode,items in data.items() for record in items],
                    ("protocol","fold","day_id","session_id","recording_id","action","n_samples","wmae_rad")),
        ("by_day", [dict(protocol=mode,day_id=day,**summarize([r for r in items if r["day_id"]==day]))
                   for mode,items in data.items() for day in days],
                  ("protocol","day_id","recordings","sessions","samples","wmae_rad","mae_rad","r2_uniform_mean","within10deg_pct","sign_balanced_pct"))):
        with (args.out/(prefix+".csv")).open("w", newline="") as stream:
            writer=csv.DictWriter(stream,columns,extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    directory=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Public AIRI data/ or data/train/ folder")
    parser.add_argument("--frames", type=Path, default=directory/"heldout_frames.npz")
    parser.add_argument("--spec", type=Path, default=directory/"validation_specs.json")
    parser.add_argument("--branches", type=Path, help="Research compact/ folder containing three branch banks")
    parser.add_argument("--save-frames", type=Path, help="Optionally save the lossless mix for standalone replay")
    parser.add_argument("--model-config", type=Path, help="Verify deployment joint ranges, weights and fixed gain")
    parser.add_argument("--out", type=Path, default=Path("verification_results"))
    replay(parser.parse_args())
