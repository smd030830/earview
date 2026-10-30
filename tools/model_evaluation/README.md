# Bundled model evaluation (BL-001, BL-003, BL-006)

Offline PC benchmark using the exact three Android TFLite assets. No microphone
is accessed. App behavior, dependencies, and thresholds are unchanged.

Run with a separate Python environment containing TensorFlow, NumPy, and SciPy:

```powershell
.venv-ml/Scripts/python.exe -m unittest discover -s tools/model_evaluation
.venv-ml/Scripts/python.exe tools/model_evaluation/evaluate.py --revision 33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6
```

The first run downloads public evaluation fixtures from the
[official ESC-50 repository](https://github.com/karolpiczak/ESC-50).
Source audio and its upstream license are cached locally and excluded from git;
respect the upstream noncommercial license and individual source attribution.
Metadata includes original source IDs. Results include each clip's source ID,
filename, fold, SHA-256, and per-model clip maximum scores.

All 40 clips each of siren, glass breaking, door/wood knock and car horn are selected, plus the first
two filenames in each remaining category (92 background clips). This is a
deterministic exploratory sample, not an unbiased release validation set.
Fire alarm and doorbell have no positive examples here: their recall is null.
Other category labels do not guarantee absence of overlapping target sounds;
negative results are nominal clip false-positive rates, not false alarms/hour.

PCM16 mono audio is scaled by 32768, resampled to 16 kHz using SciPy polyphase
filtering without volume normalization. All models use the application's
31,200-sample context and 7,800-sample hop; YAMNet receives the newest 15,600
samples. Only full windows are scored, with no padding or incomplete final hop.
This matches shared application cadence rather than giving standalone YAMNet
extra early windows. A positive clip is detected if any evaluated window meets
0.5 (car horn: 0.3). Ensemble scores are averaged per aligned window before
computing the clip maximum. No thresholds are selected using this dataset.

`report.json` captures the model hashes, runtime, protocol and per-clip results.
`report-five-sounds.json` preserves the historical baby-crying benchmark for the
previous five-output assets; it does not describe current model behavior.
This PC test does not measure acoustic playback through a phone microphone,
Android runtime behavior, notification latency, battery, or calibrated accuracy.
These recordings are not asserted to be independent of upstream pretraining.
