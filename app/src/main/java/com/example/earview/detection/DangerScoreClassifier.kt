package com.example.earview.detection

/** Platform-independent contract: mono 16 kHz PCM in memory, scores in [DangerSound] order. */
interface DangerScoreClassifier : AutoCloseable {
  val windowSize: Int

  /** Microphone samples between decisions; longer ensemble windows must not slow this cadence. */
  val hopSize: Int
    get() = (windowSize / 2).coerceAtLeast(1)

  /** Scores are bounded confidences, not calibrated probabilities. */
  fun dangerScores(window: FloatArray): FloatArray
}
