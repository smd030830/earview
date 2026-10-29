"""Local YAMNet embedding transfer learning; candidate export, never app deployment."""
import argparse
import hashlib
import json
from pathlib import Path
import wave

from dataset import load_manifest


def metrics(labels, scores, threshold):
    positives = sum(labels)
    negatives = len(labels) - positives
    return {
        "recall": sum(y == 1 and s >= threshold for y, s in zip(labels, scores)) / positives,
        "false_positive_rate": sum(y == 0 and s >= threshold for y, s in zip(labels, scores)) / negatives,
        "positive_clips": positives,
        "negative_clips": negatives,
    }


def choose_threshold(labels, scores, max_fpr):
    thresholds = sorted(set([0.0, 1.000001] + scores))
    allowed = [(t, metrics(labels, scores, t)) for t in thresholds]
    allowed = [(t, m) for t, m in allowed if m["false_positive_rate"] <= max_fpr]
    return max(allowed, key=lambda item: (item[1]["recall"], -item[1]["false_positive_rate"], item[0]))[0]


def train(args, rows):
    output = Path(args.output)
    if output.exists():
        raise ValueError("Output directory already exists; choose a new output")
    # These packages belong in a separate, user-provided ML environment.
    import numpy as np
    import tensorflow as tf

    tf.keras.utils.set_random_seed(42)
    yamnet_path = Path(args.yamnet).resolve()
    yamnet = tf.saved_model.load(str(yamnet_path))
    clips = []
    for row in rows:
        with wave.open(str(row["resolved_path"]), "rb") as audio:
            waveform = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        scores, embeddings, _ = yamnet(tf.constant(waveform))
        if (len(embeddings.shape) != 2 or embeddings.shape[1] != 1024 or
                len(scores.shape) != 2 or scores.shape[1] != 521 or
                not np.isfinite(embeddings.numpy()).all() or not np.isfinite(scores.numpy()).all()):
            raise ValueError("Unexpected YAMNet output contract")
        clips.append({**row, "features": embeddings.numpy(), "baseline": float(np.max(scores.numpy()[:, [349, 350]]))})

    head = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    head.compile(optimizer="adam", loss="binary_crossentropy")

    def frames(split):
        selected = [c for c in clips if c["split"] == split]
        features = np.concatenate([c["features"] for c in selected])
        labels = np.concatenate([np.full(len(c["features"]), c["label"] == "doorbell", dtype=np.float32) for c in selected])
        return features, labels

    x, y = frames("train")
    vx, vy = frames("val")
    head.fit(x, y, validation_data=(vx, vy), epochs=args.epochs, batch_size=32,
             callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])

    def predictions(split):
        selected = [c for c in clips if c["split"] == split]
        labels = [int(c["label"] == "doorbell") for c in selected]
        predicted = [float(np.max(head(c["features"], training=False).numpy())) for c in selected]
        if not all(np.isfinite(s) and 0 <= s <= 1 for s in predicted):
            raise ValueError("Invalid candidate scores")
        return selected, labels, predicted

    _, labels, scores = predictions("val")
    threshold = choose_threshold(labels, scores, args.max_fpr)
    report = {"threshold": threshold, "seed": 42, "epochs_limit": args.epochs,
              "validation_max_fpr": args.max_fpr,
              "contract": "float32 [N,1024] YAMNet embeddings -> float32 [N,1] doorbell score",
              "aggregation": "maximum score over all frames in each clip",
              "clips": [{k: str(v) for k, v in r.items() if k != "resolved_path"} for r in rows]}
    for split in ("val", "test"):
        selected, labels, scores = predictions(split)
        report[split] = {
            "candidate": metrics(labels, scores, threshold),
            "baseline": metrics(labels, [c["baseline"] for c in selected], 0.5),
        }
        report[split]["subtypes"] = {}
        for subtype in sorted({c["subtype"] for c in selected if c["label"] == "doorbell"}):
            indexes = [i for i, c in enumerate(selected) if c["label"] == "doorbell" and c["subtype"] == subtype]
            report[split]["subtypes"][subtype] = {
                "clips": len(indexes),
                "candidate_recall": sum(scores[i] >= threshold for i in indexes) / len(indexes),
                "baseline_recall": sum(selected[i]["baseline"] >= 0.5 for i in indexes) / len(indexes),
            }
    baseline, candidate = report["test"]["baseline"], report["test"]["candidate"]
    report["offline_gate_passed"] = (candidate["recall"] > baseline["recall"] and
                                     candidate["false_positive_rate"] <= baseline["false_positive_rate"] and
                                     candidate["false_positive_rate"] <= args.max_fpr)
    converter = tf.lite.TFLiteConverter.from_keras_model(head)
    model = converter.convert()
    runtime = tf.lite.Interpreter(model_content=model)
    runtime.allocate_tensors()
    input_info, output_info = runtime.get_input_details()[0], runtime.get_output_details()[0]
    max_error = 0.0
    for clip in clips:
        if clip["split"] != "test":
            continue
        for feature in clip["features"]:
            sample = feature.reshape(1, 1024).astype(np.float32)
            runtime.set_tensor(input_info["index"], sample)
            runtime.invoke()
            actual = float(runtime.get_tensor(output_info["index"])[0, 0])
            expected = float(head(sample, training=False).numpy()[0, 0])
            if not np.isfinite(actual) or not np.isfinite(expected):
                raise ValueError("Nonfinite output during TFLite parity check")
            max_error = max(max_error, abs(actual - expected))
    report["tflite_max_absolute_error"] = max_error
    if max_error > 1e-4:
        raise ValueError("TFLite parity failed; candidate was not exported")
    report["head_sha256"] = hashlib.sha256(model).hexdigest()
    report["yamnet_files"] = {
        str(p.relative_to(yamnet_path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(yamnet_path.rglob("*")) if p.is_file()
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "doorbell_head_candidate.tflite").write_bytes(model)
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Candidate exported. Offline gate passed:", report["offline_gate_passed"])
    print("This is an embedding head, not a replacement for app/assets/yamnet.tflite.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "train"])
    parser.add_argument("manifest")
    parser.add_argument("--yamnet", help="Local original YAMNet SavedModel exposing scores, embeddings, spectrogram")
    parser.add_argument("--output", default="doorbell-candidate")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-fpr", type=float, default=0.05)
    args = parser.parse_args()
    if args.epochs < 1 or not 0 <= args.max_fpr <= 1:
        parser.error("epochs must be positive and max-fpr must be between 0 and 1")
    rows = load_manifest(args.manifest)
    if args.command == "validate":
        print(f"Validated {len(rows)} clips; no duplicate audio or cross-split source IDs.")
    elif not args.yamnet:
        parser.error("train requires --yamnet")
    else:
        train(args, rows)


if __name__ == "__main__":
    main()
