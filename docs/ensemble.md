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
- Output: five independent scores in this exact order: FIRE_ALARM, SIREN,
  BABY_CRYING, DOORBELL, CAR_HORN. Multiple sounds may exceed their thresholds.
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
Include fire alarm and doorbell, overlapping sounds, distant sounds, silence,
speech, music and household background sounds. Do not collect microphone audio
through the app. Use separately supplied evaluation fixtures.

Verify the PRD's 3-second notification target and battery target on real devices.
Model conversion parity alone does not establish detection accuracy.

## Verification on 2026-10-29

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

## Future iOS

iOS implementation remains outside this Android change. Preserve model versions,
preprocessing constants, label mapping, output order and score fusion rules as
the portable contract. Validate the same supplied fixtures on both platforms;
do not assume bit-identical output across inference delegates.
