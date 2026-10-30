# Pretrained ensemble export

This build tool converts the official **EfficientAT mn04_as** and **PANNs
MobileNetV2** AudioSet checkpoints into standalone float32 LiteRT models. No
model training is performed. The app has no PyTorch, Python, or network dependency.

## Contract

- Input: float32 `[31200]`, 1.95 seconds of mono PCM at 16 kHz, range `[-1, 1]`.
- Output: float32 `[1, 6]` grouped sigmoid scores in the order
  `FIRE_ALARM, SIREN, GLASS_BREAKING, DOORBELL, DOOR_KNOCK, CAR_HORN`.
- Grouping uses the maximum of the matching AudioSet labels, matching the app's
  YAMNet sound rules. These are model scores, **not calibrated probabilities**.
- Frontends are included: torchaudio sinc resampling 16 to 32 kHz, original
  model-specific windowing, reflect padding, STFT, mel filters, log conversion,
  and normalization. Upsampling does not recover frequencies above 8 kHz.
- Float32 networks use TFLite builtin operations only; no Select TF ops.

## Reproduce

Use a Python virtual environment with TensorFlow, PyTorch, torchvision,
torchaudio, torchlibrosa, NumPy, and requests. Dependencies are build-only.
Exact versions used for the checked-in artifacts are recorded in the assets'
`provenance.json`. PyTorch, torchvision, and torchaudio imports succeeded and
both conversions passed reference parity with that combination.

```powershell
.venv-ml\Scripts\python.exe -m pip install -r tools/ensemble_export/requirements.txt
.venv-ml\Scripts\python.exe tools/ensemble_export/fetch_sources.py
.venv-ml\Scripts\python.exe tools/ensemble_export/export_models.py
```

`fetch_sources.py` pins official code revisions and verifies checkpoint hashes.
`export_models.py` verifies the checkpoints again before loading with
`weights_only=True`. Downloaded code, checkpoints, temporary files, and logs are
ignored by the local `.gitignore`.

The translator maps the official eval-mode PyTorch layers into NHWC TensorFlow
operations and preserves convolution padding, pooling, normalization epsilon,
squeeze-excitation, and activations. PANNs' official torchlibrosa forward is
used for reference validation; temporary librosa initialization helpers only
allocate tensors which strict checkpoint loading then replaces completely.

The exporter compares both TensorFlow and LiteRT output against official
PyTorch inference on deterministic synthetic silence, noise, multitone, and
chirp waveforms. It writes an asset only after all cases pass (`atol=0.0002`,
`rtol=0.002`), then records errors and SHA-256 hashes in
`app/src/main/assets/ensemble/validation.json`.

This verifies conversion parity, not danger-sound detection accuracy, score
calibration, battery use, or physical-device latency. Short, bandlimited input
differs from the original AudioSet evaluation regime. Validate those separately
on representative labeled audio and physical Android/iOS devices.

## Provenance and attribution

### Six-target output update (2026-10-30)

The currently bundled six-output assets were produced with
`retarget_outputs.py`, rather than rerunning PyTorch conversion. Both old graphs
already contain all 527 native sigmoid scores. This tool changes only the final
grouping: Baby cry is replaced by Shatter (443), Knock (359) is added, and the
other four groups, frontend, and pretrained parameters remain unchanged.
Indices refer to the pinned upstream AudioSet label list, not YAMNet indices.

```powershell
.venv-ml/Scripts/python.exe tools/ensemble_export/retarget_outputs.py
```

The tool needs the exact previous five-output assets (hashes are checked) in
`tools/ensemble_export/.cache/five-sounds/`. During the initial update it copies
them from the asset directory. For a fresh checkout, retrieve those historical
assets from the preceding Git revision into that cache. The full exporter above
also supports the six-target grouping directly, but its byte hashes may differ.

Current `validation.json` checks six-output parity against the original full
527-score TFLite graph and verifies the unchanged four scores on 218 inputs
(four synthetic, 214 cached external fixtures). This is distinct from original
PyTorch conversion parity, preserved in `validation-five-sounds.json`.
Neither parity check establishes detection accuracy or Android compatibility.
The runtime used for retargeting is recorded in `provenance.json`.

EfficientAT code and weights:

- Florian Schmid et al., EfficientAT, MIT license.
- Source: https://github.com/fschmid56/EfficientAT
- Pinned revision: `a425fdce92572e602a1d5634799bd9f1f2efa806`
- Checkpoint: https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn04_as_mAP_432.pt
- Checkpoint SHA-256: `899a8c6217063f6941793102a2780b3dd788ab11fce468d9cf5f0577c453321c`

PANNs code:

- Qiuqiang Kong, MIT license.
- Source: https://github.com/qiuqiangkong/audioset_tagging_cnn
- Pinned revision: `d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4`

PANNs pretrained weights:

- Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D.
  Plumbley, *PANNs: Large-Scale Pretrained Audio Neural Networks for Audio
  Pattern Recognition (Pretrained Models)*, Zenodo record 3987831.
- Source: https://zenodo.org/records/3987831
- License: https://creativecommons.org/licenses/by/4.0/
- Checkpoint: `MobileNetV2_mAP=0.383.pth`, 20,834,417 bytes.
- Publisher MD5: `838421a9262cce976aa247946d6e2983`.
- Checkpoint SHA-256: `22437af9eb5ed1854b5ad285b4a17d5fc566f1a3b23df40db5541f5705ca340f`.

Derived assets modify deployment format and add the waveform frontend and
six grouped outputs. Original training weights are retained. Upstream MIT
notices and pretrained-weight attribution accompany the assets.
