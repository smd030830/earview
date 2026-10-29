package com.example.earview.detection

/**
 * Equal-weight score averaging (BL-005/006). No audio or inference results are persisted.
 * Scores are not calibrated probabilities; thresholds need ensemble-specific validation.
 */
class EnsembleClassifier(private val models: List<DangerScoreClassifier>) : DangerScoreClassifier {
  init {
    require(models.size == 3) { "The ensemble requires exactly three models" }
    require(models.all { it.windowSize > 0 })
  }

  override val windowSize: Int = models.maxOf { it.windowSize }
  override val hopSize: Int = models.minOf { it.hopSize }

  override fun dangerScores(window: FloatArray): FloatArray {
    require(window.size == windowSize)
    val total = FloatArray(DangerSound.entries.size)
    models.forEach { model ->
      // Align the end of every model's input with the newest microphone sample.
      val input = if (model.windowSize == window.size) window else window.copyOfRange(window.size - model.windowSize, window.size)
      val scores = model.dangerScores(input)
      require(scores.size == total.size) { "Every model must score all five danger sounds" }
      scores.forEachIndexed { index, score ->
        require(score.isFinite() && score in 0f..1f) { "Invalid model confidence" }
        total[index] += score / models.size
      }
    }
    return total
  }

  override fun close() {
    var failure: Exception? = null
    models.asReversed().forEach { model ->
      try {
        model.close()
      } catch (e: Exception) {
        val previous = failure
        if (previous == null) failure = e else previous.addSuppressed(e)
      }
    }
    failure?.let { throw it }
  }
}
