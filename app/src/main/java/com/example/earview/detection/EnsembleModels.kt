package com.example.earview.detection

import android.content.Context

/** Opens all three bundled models atomically; a failed load never silently changes the ensemble. */
internal fun loadEnsemble(context: Context): EnsembleClassifier {
  val opened = mutableListOf<DangerScoreClassifier>()
  try {
    opened += YamnetClassifier.load(context)
    opened += TfliteDangerClassifier.load(context, "ensemble/efficientat_mn04_as.tflite")
    opened += TfliteDangerClassifier.load(context, "ensemble/panns_mobilenetv2.tflite")
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
