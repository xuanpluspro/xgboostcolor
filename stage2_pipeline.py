import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import color
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


DEFAULT_TRAIN_CSV_CANDIDATES = [
    Path(__file__).resolve().parent / "数据集.csv",
    Path(__file__).resolve().parent.parent / "植被分割" / "数据集.csv",
]


def get_features_simple(pixels):
    if pixels.dtype != np.uint8:
        pixels = pixels.astype(np.uint8)

    pixels_float = pixels.astype(np.float32)
    r = pixels_float[:, 0]
    g = pixels_float[:, 1]
    b = pixels_float[:, 2]

    pixels_normalized = pixels_float / 255.0
    hsv = color.rgb2hsv(pixels_normalized.reshape(-1, 1, 3)).reshape(-1, 3)
    lab = color.rgb2lab(pixels_normalized.reshape(-1, 1, 3)).reshape(-1, 3)

    h = hsv[:, 0] * 360.0
    s = hsv[:, 1] * 100.0
    a = lab[:, 1]
    bb = lab[:, 2]
    gray = np.mean(pixels_float, axis=1)

    c = 1.0 - r / 255.0
    m_raw = 1.0 - g / 255.0
    y_raw = 1.0 - b / 255.0
    k = np.minimum.reduce([c, m_raw, y_raw])
    denom = 1.0 - k + 1e-8
    m = ((m_raw - k) / denom) * 100.0
    ye = ((y_raw - k) / denom) * 100.0

    pil_line = Image.fromarray(pixels.reshape(-1, 1, 3), mode="RGB")
    ycbcr = np.array(pil_line.convert("YCbCr"), dtype=np.float32).reshape(-1, 3)
    cb = ycbcr[:, 1]
    cr = ycbcr[:, 2]

    i = 0.596 * r - 0.275 * g - 0.321 * b
    q = 0.212 * r - 0.523 * g + 0.311 * b

    features = np.column_stack([r, g, b, h, s, a, bb, gray, m, ye, cb, cr, i, q])
    return np.nan_to_num(features)


def fix_xgboost_model(model):
    new_attrs = [
        "grow_policy",
        "max_bin",
        "eval_metric",
        "callbacks",
        "early_stopping_rounds",
        "max_cat_to_onehot",
        "max_leaves",
        "sampling_method",
        "enable_categorical",
        "feature_types",
        "max_cat_threshold",
        "predictor",
        "tree_method",
        "device",
    ]

    for attr in new_attrs:
        if not hasattr(model, attr):
            setattr(model, attr, None)

    if hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            for attr in new_attrs:
                if not hasattr(booster, attr):
                    setattr(booster, attr, None)
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal standalone stage-2 pipeline for green vs. senescent vegetation."
    )
    parser.add_argument("--task", choices=["train", "predict"], default="train")
    parser.add_argument("--csv", type=Path, help="Training or prediction CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Output directory for models and reports.",
    )
    parser.add_argument("--model", choices=["xgboost", "svm", "both"], default="both")
    parser.add_argument("--model-path", type=Path, help="Model path used by --task predict.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("stage2_predictions.csv"),
        help="Prediction CSV written by --task predict.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--green-label", type=int, default=1)
    parser.add_argument("--senescent-label", type=int, default=0)
    parser.add_argument("--ignore-labels", type=str, default="2")
    parser.add_argument("--rep-train", default="TRAIN")
    parser.add_argument("--rep-test", default="TEST")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=0)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-gamma", type=float, default=1e-3)
    parser.add_argument("--xgb-n-estimators", type=int, default=300)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.9)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.9)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def resolve_local_path(path_value: Path):
    if path_value.is_absolute():
        return path_value
    return Path(__file__).resolve().parent / path_value


def get_default_train_csv():
    for candidate in DEFAULT_TRAIN_CSV_CANDIDATES:
        if candidate.exists():
            return candidate
    candidate_text = "\n".join(str(path) for path in DEFAULT_TRAIN_CSV_CANDIDATES)
    raise FileNotFoundError(
        "没有找到默认训练数据集。请把 数据集.csv 放到以下任一位置，或手动传 --csv：\n"
        f"{candidate_text}"
    )


def read_csv_rows(csv_path: Path):
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with csv_path.open("r", encoding=encoding, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise FileNotFoundError(csv_path)


def require_columns(rows, required_columns):
    if not rows:
        raise ValueError("CSV is empty.")
    missing = set(required_columns).difference(rows[0].keys())
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")


def parse_ignore_labels(raw_value):
    if not raw_value.strip():
        return set()
    return {int(part.strip()) for part in raw_value.split(",") if part.strip()}


def normalize_rows(rows, green_label, senescent_label, ignore_labels):
    filtered_rows = []
    ignored_counter = Counter()
    used_counter = Counter()
    allowed = {green_label, senescent_label}

    for row in rows:
        label = int(float(row["Y"]))
        if label in ignore_labels or label not in allowed:
            ignored_counter[label] += 1
            continue

        normalized = dict(row)
        normalized["Y_raw"] = label
        normalized["Y_binary"] = 1 if label == green_label else 0
        filtered_rows.append(normalized)
        used_counter[label] += 1

    return filtered_rows, used_counter, ignored_counter


def maybe_cap_per_class(rows, max_per_class, seed):
    if max_per_class <= 0:
        return rows

    rng = np.random.default_rng(seed)
    by_class = {0: [], 1: []}
    for row in rows:
        by_class[row["Y_binary"]].append(row)

    capped = []
    for class_rows in by_class.values():
        if len(class_rows) > max_per_class:
            indices = rng.choice(len(class_rows), size=max_per_class, replace=False)
            capped.extend(class_rows[i] for i in indices)
        else:
            capped.extend(class_rows)

    rng.shuffle(capped)
    return capped


def split_rows(rows, rep_train, rep_test, test_size, seed):
    if rows and "REP" in rows[0]:
        train_rows = [row for row in rows if str(row["REP"]).upper() == rep_train.upper()]
        test_rows = [row for row in rows if str(row["REP"]).upper() == rep_test.upper()]
        if train_rows and test_rows:
            return train_rows, test_rows, "rep"

    labels = [row["Y_binary"] for row in rows]
    train_rows, test_rows = train_test_split(
        rows,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    return train_rows, test_rows, "random"


def build_rgb_matrix(rows):
    return np.array([[int(row["R"]), int(row["G"]), int(row["B"])] for row in rows], dtype=np.uint8)


def build_feature_matrix(rows):
    rgb = build_rgb_matrix(rows)
    y = np.array([int(row["Y_binary"]) for row in rows], dtype=np.uint8)
    features = get_features_simple(rgb)
    return rgb, features, y


def evaluate_model(model, features, labels):
    predictions = np.asarray(model.predict(features), dtype=np.uint8)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[:, 1]
    elif hasattr(model, "decision_function"):
        probabilities = model.decision_function(features)
    else:
        probabilities = None

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_green": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "recall_green": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "f1_green": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "precision_senescent": float(precision_score(labels, predictions, pos_label=0, zero_division=0)),
        "recall_senescent": float(recall_score(labels, predictions, pos_label=0, zero_division=0)),
        "f1_senescent": float(f1_score(labels, predictions, pos_label=0, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }
    if probabilities is not None:
        metrics["roc_auc_green"] = float(roc_auc_score(labels, probabilities))

    return predictions, probabilities, metrics


def make_xgboost(args, y_train):
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model = XGBClassifier(
        n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    return model, sample_weight


def make_svm(args):
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    C=args.svm_c,
                    gamma=args.svm_gamma,
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced",
                    random_state=args.seed,
                ),
            ),
        ]
    )


def save_model(model, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(model, f)


def save_predictions_csv(rows, predictions, probabilities, output_path):
    fieldnames = list(rows[0].keys()) + ["pred_binary", "pred_text", "prob_green"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, pred, prob in zip(rows, predictions, probabilities):
            record = dict(row)
            record["pred_binary"] = int(pred)
            record["pred_text"] = "green" if int(pred) == 1 else "senescent"
            record["prob_green"] = "" if prob is None else round(float(prob), 6)
            writer.writerow(record)


def load_stage2_model(model_path):
    with model_path.open("rb") as f:
        model = pickle.load(f)

    if hasattr(model, "get_booster") or type(model).__module__.startswith("xgboost"):
        fix_xgboost_model(model)

    return model


def predict_labels(model, features, threshold):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[:, 1]
        predictions = (probabilities >= threshold).astype(np.uint8)
    else:
        predictions = np.asarray(model.predict(features), dtype=np.uint8)
        probabilities = [None] * len(predictions)
    return predictions, probabilities


def run_train(args):
    csv_path = args.csv.resolve() if args.csv else get_default_train_csv().resolve()
    output_dir = resolve_local_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = read_csv_rows(csv_path)
    require_columns(raw_rows, {"R", "G", "B", "Y"})

    ignore_labels = parse_ignore_labels(args.ignore_labels)
    filtered_rows, used_counter, ignored_counter = normalize_rows(
        raw_rows,
        green_label=args.green_label,
        senescent_label=args.senescent_label,
        ignore_labels=ignore_labels,
    )
    filtered_rows = maybe_cap_per_class(filtered_rows, args.max_per_class, args.seed)
    if not filtered_rows:
        raise ValueError("No usable rows remain after label filtering.")

    train_rows, test_rows, split_mode = split_rows(
        filtered_rows,
        rep_train=args.rep_train,
        rep_test=args.rep_test,
        test_size=args.test_size,
        seed=args.seed,
    )

    _, train_features, y_train = build_feature_matrix(train_rows)
    _, test_features, y_test = build_feature_matrix(test_rows)

    summary = {
        "source_csv": str(csv_path),
        "green_label_raw": args.green_label,
        "senescent_label_raw": args.senescent_label,
        "ignored_labels": sorted(ignore_labels),
        "raw_row_count": len(raw_rows),
        "used_label_counts_raw": dict(used_counter),
        "ignored_label_counts_raw": dict(ignored_counter),
        "train_row_count": len(train_rows),
        "test_row_count": len(test_rows),
        "split_mode": split_mode,
        "feature_shape_train": list(train_features.shape),
        "feature_shape_test": list(test_features.shape),
        "binary_train_counts": dict(Counter(int(x) for x in y_train)),
        "binary_test_counts": dict(Counter(int(x) for x in y_test)),
    }

    trained_models = []
    if args.model in ("xgboost", "both"):
        trained_models.append("xgboost")
    if args.model in ("svm", "both"):
        trained_models.append("svm")

    all_reports = {"task": "train", "dataset_summary": summary, "models": {}}

    for model_name in trained_models:
        if model_name == "xgboost":
            model, sample_weight = make_xgboost(args, y_train)
            model.fit(train_features, y_train, sample_weight=sample_weight)
            model_path = output_dir / "xgboost_stage2.pkl"
        else:
            model = make_svm(args)
            model.fit(train_features, y_train)
            model_path = output_dir / "svm_stage2.pkl"

        predictions, probabilities, metrics = evaluate_model(model, test_features, y_test)
        save_model(model, model_path)
        save_predictions_csv(
            test_rows,
            predictions,
            probabilities if probabilities is not None else [None] * len(predictions),
            output_dir / f"{model_name}_test_predictions.csv",
        )
        all_reports["models"][model_name] = {
            "model_path": str(model_path),
            "metrics": metrics,
        }

    report_path = output_dir / "training_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)

    print(json.dumps(all_reports, ensure_ascii=False, indent=2))


def run_predict(args):
    if args.csv is None:
        raise ValueError("--task predict 时必须提供 --csv。")
    csv_path = args.csv.resolve()
    if args.model_path is None:
        raise ValueError("--model-path is required when --task predict.")

    model_path = resolve_local_path(args.model_path)
    output_csv = resolve_local_path(args.output_csv)

    rows = read_csv_rows(csv_path)
    require_columns(rows, {"R", "G", "B"})

    rgb = build_rgb_matrix(rows)
    features = get_features_simple(rgb)
    model = load_stage2_model(model_path)
    predictions, probabilities = predict_labels(model, features, args.threshold)
    save_predictions_csv(rows, predictions, probabilities, output_csv)

    summary = {
        "task": "predict",
        "source_csv": str(csv_path),
        "model_path": str(model_path),
        "row_count": len(rows),
        "threshold": args.threshold,
        "output_csv": str(output_csv),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    if args.task == "predict":
        run_predict(args)
        return
    run_train(args)


if __name__ == "__main__":
    main()
