package com.example.earview.detection

import android.content.Context

/** Opens all three bundled models atomically; a failed load never silently changes the ensemble. */
internal fun loadEnsemble(context: Context): EnsembleClassifier {
  // Validate every coefficient before opening native model resources.
  val calibrations = SoundModel.entries.associateWith { SoundScoreCalibrations.forModel(it) }
  val opened = mutableListOf<DangerScoreClassifier>()
  try {
    opened += CalibratedSoundClassifier(YamnetClassifier.load(context), calibrations.getValue(SoundModel.YAMNET))
    opened += CalibratedSoundClassifier(
      TfliteDangerClassifier.load(context, "ensemble/efficientat_mn04_as.tflite"),
      calibrations.getValue(SoundModel.EFFICIENTAT),
    )
    opened += CalibratedSoundClassifier(
      TfliteDangerClassifier.load(context, "ensemble/panns_mobilenetv2.tflite"),
      calibrations.getValue(SoundModel.PANNS),
    )
    return EnsembleClassifier(opened.toList())
  } catch (e: Exception) {
    opened.asReversed().forEach { model ->
      try {
        model.close()
      } catch (closeFailure: Exception) {
        e.addSuppressed(closeFailure)
      }
    }
    throw e
  }
}
