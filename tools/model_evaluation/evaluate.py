"""BL-001/006: offline benchmark of the exact bundled models; no microphone access."""
import argparse
import csv
import hashlib
import io
import json
import math
import platform
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LABELS = ['FIRE_ALARM', 'SIREN', 'GLASS_BREAKING', 'DOORBELL', 'DOOR_KNOCK', 'CAR_HORN']
CATEGORIES = {'siren': 1, 'glass_breaking': 2, 'door_wood_knock': 4, 'car_horn': 5}
GROUPS = [[393, 394], [390, 391, 316, 317, 318, 319], [437], [349, 350], [353], [302]]
THRESHOLDS = np.array([.5, .5, .5, .5, .5, .3], dtype=np.float32)
HOP = 7800
WINDOW = 31200


def fetch(url, destination):
    if destination.exists():
        return destination.read_bytes()
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            return data
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def load_audio(path):
    rate, samples = wavfile.read(path)
    if samples.dtype != np.int16 or samples.ndim != 1:
        raise ValueError('Expected mono signed PCM16 ESC-50 audio')
    samples = samples.astype(np.float32) / 32768.0
    divisor = math.gcd(rate, 16000)
    return resample_poly(samples, 16000 // divisor, rate // divisor).astype(np.float32)


def windows(samples):
    # SlidingAudioWindow emits only after a full context at a 7800-sample boundary.
    for end in range(WINDOW, len(samples) + 1, HOP):
        yield samples[end - WINDOW:end]


def summary(records, model):
    result = {}
    for index, label in enumerate(LABELS):
        positive = [r for r in records if CATEGORIES.get(r['category']) == index]
        # ESC labels do not annotate overlapping events: these are nominal negatives.
        negative = [r for r in records if CATEGORIES.get(r['category']) != index]
        tp = sum(r['peak_scores'][model][index] >= float(THRESHOLDS[index]) for r in positive)
        fp = sum(r['peak_scores'][model][index] >= float(THRESHOLDS[index]) for r in negative)
        result[label] = dict(threshold=float(THRESHOLDS[index]), positives=len(positive),
                             detected=tp, recall=tp / len(positive) if positive else None,
                             nominal_negatives=len(negative), false_positive_clips=fp,
                             nominal_clip_fpr=fp / len(negative) if negative else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', help='Pinned 40-character ESC-50 Git revision')
    parser.add_argument('--download-only', action='store_true')
    args = parser.parse_args()
    cache = HERE / '.cache'
    cache.mkdir(exist_ok=True)
    revision = args.revision
    if revision is None:
        with urllib.request.urlopen('https://api.github.com/repos/karolpiczak/ESC-50/commits/master', timeout=60) as response:
            revision = json.load(response)['sha']
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Expected full lowercase Git SHA')
    cache = cache / revision
    base = f'https://raw.githubusercontent.com/karolpiczak/ESC-50/{revision}/'
    metadata = fetch(base + 'meta/esc50.csv', cache / 'esc50.csv')
    fetch(base + 'LICENSE', cache / 'LICENSE')
    rows = list(csv.DictReader(io.StringIO(metadata.decode())))
    # All positives for the four available targets; two clips per background class.
    selected = []
    for category in sorted({r['category'] for r in rows}):
        clips = sorted((r for r in rows if r['category'] == category), key=lambda r: r['filename'])
        selected.extend(clips if category in CATEGORIES else clips[:2])
    def download(row):
        fetch(base + 'audio/' + row['filename'], cache / row['filename'])
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(download, selected))
    print(f'Cached {len(selected)} clips at revision {revision}', flush=True)
    if args.download_only:
        return
    import tensorflow as tf
    paths = {'yamnet': ROOT / 'app/src/main/assets/yamnet.tflite',
             'efficientat': ROOT / 'app/src/main/assets/ensemble/efficientat_mn04_as.tflite',
             'panns': ROOT / 'app/src/main/assets/ensemble/panns_mobilenetv2.tflite'}
    models = {}
    for name, path in paths.items():
        interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=2)
        interpreter.allocate_tensors()
        models[name] = interpreter
    records = []
    started = time.monotonic()
    for number, row in enumerate(selected, 1):
        path = cache / row['filename']
        samples = load_audio(path)
        peaks = {name: np.zeros(len(LABELS), dtype=np.float32) for name in [*models, 'ensemble']}
        count = 0
        for window in windows(samples):
            scores = []
            for name, interpreter in models.items():
                inp = interpreter.get_input_details()[0]
                length = int(np.prod(inp['shape']))
                interpreter.set_tensor(inp['index'], window[-length:].reshape(inp['shape']))
                interpreter.invoke()
                value = interpreter.get_tensor(interpreter.get_output_details()[0]['index']).reshape(-1)
                if name == 'yamnet':
                    value = np.array([max(value[group]) for group in GROUPS])
                if value.shape != (len(LABELS),) or not np.all(np.isfinite(value)) or not np.all((value >= 0) & (value <= 1)):
                    raise ValueError(f'Invalid {name} model scores')
                scores.append(value)
                peaks[name] = np.maximum(peaks[name], value)
            # Average aligned frames, never average clip maxima.
            peaks['ensemble'] = np.maximum(peaks['ensemble'], np.mean(scores, axis=0))
            count += 1
        if not count:
            raise ValueError('Clip has no full ensemble window')
        records.append(dict(filename=row['filename'], category=row['category'], fold=row['fold'],
                            src_file=row['src_file'], sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            frames=count, peak_scores={k: v.tolist() for k, v in peaks.items()}))
        if number % 10 == 0:
            print(f'{number}/{len(selected)} clips, {time.monotonic()-started:.1f}s', flush=True)
    report = dict(dataset='ESC-50', dataset_revision=revision, runtime='PC TensorFlow Lite',
                  tensorflow=tf.__version__, numpy=np.__version__, platform=platform.platform(),
                  output_order=LABELS, sample_rate=16000, context_samples=WINDOW, hop_samples=HOP,
                  protocol='clip detected if any complete aligned window reaches threshold; no padding; no tuning',
                  limits=['No fire alarm or doorbell positives', 'Nominal negatives may contain overlapping sounds',
                          'Not real-device accuracy, latency, or false alarms per hour',
                          'Exploratory benchmark, not independent release validation or calibration'],
                  model_hashes={k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()},
                  clips=len(records), elapsed_seconds=time.monotonic()-started,
                  results={k: summary(records, k) for k in [*models, 'ensemble']}, records=records)
    output = HERE / 'report.json'
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['results'], indent=2), flush=True)
    print(f'Report: {output}', flush=True)


if __name__ == '__main__':
    main()
