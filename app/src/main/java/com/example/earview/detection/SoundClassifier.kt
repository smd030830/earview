package com.example.earview.detection

/** Scores a window of audio against the YAMNet classes. */
interface SoundClassifier : DangerScoreClassifier {
  /** Number of samples [classify] needs. */
  override val windowSize: Int

  /** Returns one score per YAMNet class for [window], which holds [windowSize] samples. */
  fun classify(window: FloatArray): FloatArray

  override fun dangerScores(window: FloatArray): FloatArray {
    val scores = classify(window)
    require(scores.size == YAMNET_CLASS_COUNT)
    return FloatArray(DangerSound.entries.size) { index ->
      DangerSound.entries[index].rule.classIndices.maxOf { scores[it] }
    }
  }
}
