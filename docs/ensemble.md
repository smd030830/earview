# Three-model detection experiment (BL-001, BL-003, BL-005, BL-006)

The target ensemble is YAMNet + EfficientAT mn04_as + PANNs MobileNetV2.
All inference stays on-device. Microphone audio is processed in memory only.

`EarViewApplication` loads all three bundled models through `loadEnsemble`.
A missing or incompatible model fails detection startup; it never silently
switches to a different number of models. Already opened models are released.
The existing foreground service, permissions and two screens are preserved.

Export instructions and upstream attribution are in `tools/ensemble_export/README.md`.
Bundled `assets/ensemble/validation.json` records conversion parity and asset hashes;
`provenance.json` records source checkpoints and build versions.

## Shared contract

- Input: mono float PCM, 16,000 Hz, values in [-1, 1].
- Output: six independent scores in this exact order: FIRE_ALARM, SIREN,
  GLASS_BREAKING, DOORBELL, DOOR_KNOCK, CAR_HORN. Multiple sounds may exceed their thresholds.
- Map each model's labels by name, never by another model's numerical indices.
- Combine related labels using their maximum score, matching the existing YAMNet
  grouping. Apply sigmoid to logits where required; never softmax across sounds.
- Align input windows at the newest sample. Different window lengths are allowed.
- Keep the shortest model's half-window decision cadence (YAMNet: 7,800 samples,
  487.5 ms), rather than slowing decisions to half the longest model's window.
- Take the arithmetic mean of the three mapped confidences for each sound.

## Experimental limits

Scores in [0, 1] are not necessarily calibrated probabilities. This initial
experiment uses bounded confidences without claiming statistical calibration.
The additional checkpoints expect 32 kHz audio. Their exported wrappers resample
the shared 16 kHz microphone input to 32 kHz; this does not recover frequencies
above 8 kHz. Evaluate these deployed wrappers, not just their original checkpoints.
Existing thresholds (horn 0.3, others 0.5) are provisional and were measured only
for YAMNet; they are not validated for this ensemble.

Before release, use a calibration split to fit per-model/per-sound calibration
and select ensemble thresholds, then a separate held-out evaluation split to
compare recall, false alarms per hour, and end-to-end latency against YAMNet.
Include fire alarm, glass breaking, doorbell and door knock, overlapping sounds, distant sounds, silence,
speech, music and household background sounds. Do not collect microphone audio
through the app. Use separately supplied evaluation fixtures.

Verify the PRD's 3-second notification target and battery target on real devices.
Model conversion parity alone does not establish detection accuracy.

## Historical verification on 2026-10-29 (previous five-sound model)

- 88 JVM unit tests passed, including averaging, alias grouping, window alignment,
  resource cleanup, invalid scores and hashes of the deployed model assets.
- Debug application and instrumented-test APKs built successfully.
- `EnsembleModelsTest` passed on the Medium_Phone Android emulator using the
  application's LiteRT 1.4.2 runtime. It opens all three actual assets and invokes
  the ensemble twice with synthetic silence, checking five finite bounded scores.
- Original PyTorch versus exported LiteRT maximum absolute errors across four
  deterministic synthetic inputs: EfficientAT 8.44e-9, PANNs 3.44e-7.
- Physical-device detection accuracy, calibration, battery and latency are pending.
- Kotlin style checks and Android lint were attempted but could not complete:
  compiler/IntelliJ/Groovy checker dependencies were missing from the local cache
  and their network downloads did not complete. `git diff --check` passed.
  Gradle's connected-test launcher also lacked its UTP dependency; the compiled
  instrumented test was installed and executed directly with `adb am instrument`.

## Six-sound target update on 2026-10-30

The approved targets remove baby crying and add glass breaking (`Shatter`) and
door knock (`Knock`). Generic `Glass` or `Engine knocking` scores do not trigger
these new targets. YAMNet uses its matching class indices; the other two bundled
models must export the same six-score order. New thresholds remain provisional
at 0.5, and neither class has a validated real-device detection rate.
Historical five-sound evaluation results must not be presented as six-sound
validation. The UI, TalkBack labels, vibration patterns and overflow handling
follow the six-sound contract; at most three names remain displayed.

The bundled EfficientAT/PANNs files now output all six scores. Their final label
grouping was retargeted without changing pretrained weights or preprocessing;
218 inputs per model passed comparison against original full AudioSet outputs,
including checks that the retained four sound scores were unchanged.
`validation-five-sounds.json` preserves the original conversion evidence.

PC evaluation with the actual assets on 252 ESC-50 clips (40 positives per
available target) produced the following recall at the provisional thresholds:

| Model | Glass breaking | Door knock |
|---|---:|---:|
| YAMNet | 11/40 (27.5%) | 14/40 (35%) |
| EfficientAT | 0/40 (0%) | 29/40 (72.5%) |
| PANNs | 3/40 (7.5%) | 31/40 (77.5%) |
| Three-model average | 4/40 (10%) | 23/40 (57.5%) |

The low glass-breaking recall is a known limitation of the current deployed
window/score/threshold combination. No calibration or threshold tuning was done
on this evaluation set. These results do not establish real-device accuracy.
The nominal negative clips yielded zero glass/knock detections at 0.5, which
does not establish zero false alarms in use. Source labels do not rule out
overlapping sounds. Complete scores, model hashes and conditions are recorded
in `tools/model_evaluation/report.json`; the prior five-sound benchmark is kept
in `report-five-sounds.json`.

Verification after the six-target change:

- 92 JVM unit tests passed, including exact new label mapping, ignoring isolated
  baby-cry/generic-glass/engine-knock scores, six-way overflow, vibration and return.
- 18 Android instrumented tests passed on Medium_Phone (API 37), including actual
  bundled LiteRT inference returning six finite scores and new UI/TalkBack labels.
- Debug APK build, Android lint and ktlint passed. Lint retains an API-level warning
  in the pre-existing foreground service; it was not introduced by this change.
- Four evaluation-runner tests passed; model hashes/order and 252 report records
  were checked. `git diff --check` passed.
- Real-device microphone accuracy, calibration, notification latency and battery
  remain unverified. New thresholds are provisional, not release-validated.

## Future iOS

iOS implementation remains outside this Android change. Preserve model versions,
preprocessing constants, label mapping, output order and score fusion rules as
the portable contract. Validate the same supplied fixtures on both platforms;
do not assume bit-identical output across inference delegates.
