package com.example.earview.detection

/** Identifies the source model independently of its position in the ensemble. */
internal enum class SoundModel {
  YAMNET,
  EFFICIENTAT,
  PANNS,
}

/** Internal score conversion: (raw * scale + offset), limited to the shared 0..1 range. */
internal data class ScoreCorrection(val scale: Float = 1f, val offset: Float = 0f) {
  init {
    require(scale.isFinite() && scale > 0f) { "Score scale must be finite and positive" }
    require(offset.isFinite()) { "Score offset must be finite" }
  }

  fun apply(raw: Float): Float {
    require(raw.isFinite() && raw in 0f..1f) { "Invalid raw model confidence" }
    // Double arithmetic prevents finite coefficients from overflowing before limiting the result.
    return (raw.toDouble() * scale + offset).coerceIn(0.0, 1.0).toFloat()
  }
}

/** All six sounds must be configured explicitly; absent entries never silently skip conversion. */
internal class ModelScoreCalibration(corrections: Map<DangerSound, ScoreCorrection>) {
  private val corrections = corrections.toMap()

  init {
    require(this.corrections.keys == DangerSound.entries.toSet()) { "Configure every danger sound" }
  }

  fun apply(rawScores: FloatArray): FloatArray {
    require(rawScores.size == DangerSound.entries.size) { "Expected six raw sound scores" }
    return FloatArray(rawScores.size) { index -> corrections.getValue(DangerSound.entries[index]).apply(rawScores[index]) }
  }
}

/**
 * Single configuration point for the 18 model/sound corrections (BL-003, BL-006).
 *
 * Identity values are intentional: no validated fitted coefficients exist yet. Example numbers
 * belong in tests, not production. These conversions do not establish calibrated probabilities.
 */
internal object SoundScoreCalibrations {
  private val byModel = mapOf(
    SoundModel.YAMNET to ModelScoreCalibration(
      mapOf(
        DangerSound.FIRE_ALARM to ScoreCorrection(),
        DangerSound.SIREN to ScoreCorrection(),
        DangerSound.GLASS_BREAKING to ScoreCorrection(),
        DangerSound.DOORBELL to ScoreCorrection(),
        DangerSound.DOOR_KNOCK to ScoreCorrection(),
        DangerSound.CAR_HORN to ScoreCorrection(),
      ),
    ),
    SoundModel.EFFICIENTAT to ModelScoreCalibration(
      mapOf(
        DangerSound.FIRE_ALARM to ScoreCorrection(),
        DangerSound.SIREN to ScoreCorrection(),
        DangerSound.GLASS_BREAKING to ScoreCorrection(),
        DangerSound.DOORBELL to ScoreCorrection(),
        DangerSound.DOOR_KNOCK to ScoreCorrection(),
        DangerSound.CAR_HORN to ScoreCorrection(),
      ),
    ),
    SoundModel.PANNS to ModelScoreCalibration(
      mapOf(
        DangerSound.FIRE_ALARM to ScoreCorrection(),
        DangerSound.SIREN to ScoreCorrection(),
        DangerSound.GLASS_BREAKING to ScoreCorrection(),
        DangerSound.DOORBELL to ScoreCorrection(),
        DangerSound.DOOR_KNOCK to ScoreCorrection(),
        DangerSound.CAR_HORN to ScoreCorrection(),
      ),
    ),
  )

  fun forModel(model: SoundModel): ModelScoreCalibration = byModel.getValue(model)
}

/** Converts every model score before the ensemble sees it; audio and scores stay in memory. */
internal class CalibratedSoundClassifier(
  private val source: DangerScoreClassifier,
  private val calibration: ModelScoreCalibration,
) : DangerScoreClassifier {
  override val windowSize: Int get() = source.windowSize
  override val hopSize: Int get() = source.hopSize

  override fun dangerScores(window: FloatArray): FloatArray = calibration.apply(source.dangerScores(window))

  override fun close() = source.close()
}
