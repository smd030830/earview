package com.example.earview.detection

import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class ScoreCalibrationTest {
  private class Model(private val scores: FloatArray) : DangerScoreClassifier {
    override val windowSize = 4
    override val hopSize = 2
    var closed = false

    override fun dangerScores(window: FloatArray) = scores

    override fun close() {
      closed = true
    }
  }

  private fun calibration(sound: DangerSound, correction: ScoreCorrection) = ModelScoreCalibration(
    DangerSound.entries.associateWith { if (it == sound) correction else ScoreCorrection() },
  )

  private fun scores(sound: DangerSound, score: Float) = FloatArray(DangerSound.entries.size).also { it[sound.ordinal] = score }

  @Test
  fun example_convertsEachModelBeforeAveraging() {
    val sound = DangerSound.DOOR_KNOCK
    val sources = listOf(Model(scores(sound, 0.3f)), Model(scores(sound, 0.5f)), Model(scores(sound, 0.6f)))
    val corrections = listOf(ScoreCorrection(scale = 5f / 3f), ScoreCorrection(scale = 0.8f), ScoreCorrection())
    val models = sources.zip(corrections) { model, correction -> CalibratedSoundClassifier(model, calibration(sound, correction)) }

    assertArrayEquals(floatArrayOf(0.5f, 0.4f, 0.6f), models.map { it.dangerScores(FloatArray(4))[sound.ordinal] }.toFloatArray(), 0.00001f)
    val ensemble = EnsembleClassifier(models)
    assertEquals(0.5f, ensemble.dangerScores(FloatArray(4))[sound.ordinal], 0.00001f)
    // The underlying model outputs are not overwritten by the correction.
    assertEquals(0.3f, sources[0].dangerScores(FloatArray(4))[sound.ordinal], 0f)
  }

  @Test
  fun detectorUsesTheCorrectedAverageForItsThreshold() = runTest {
    val sound = DangerSound.DOOR_KNOCK
    val models = listOf(0.3f, 0.5f, 0.6f).map { raw ->
      CalibratedSoundClassifier(Model(scores(sound, raw)), calibration(sound, ScoreCorrection(offset = 0.1f)))
    }
    val source = object : AudioSource {
      override fun chunks(chunkSize: Int) = flowOf(FloatArray(chunkSize), FloatArray(chunkSize))
    }
    val detector = YamnetSoundDetector(source, { EnsembleClassifier(models) }, StandardTestDispatcher(testScheduler))

    // Raw average .4667 would miss the .5 threshold; corrected average .5667 must trigger.
    assertEquals(listOf(sound), detector.detect().toList())
  }

  @Test
  fun correctionsAreIndependentPerSound() {
    val correction = calibration(DangerSound.GLASS_BREAKING, ScoreCorrection(scale = 2f, offset = -0.1f))
    val raw = FloatArray(DangerSound.entries.size) { 0.4f }
    val actual = correction.apply(raw)
    DangerSound.entries.forEach { sound ->
      assertEquals(if (sound == DangerSound.GLASS_BREAKING) 0.7f else 0.4f, actual[sound.ordinal], 0.00001f)
    }
  }

  @Test
  fun limitsOnlyValidCorrectedScoresToZeroAndOne() {
    assertEquals(1f, ScoreCorrection(scale = 3f).apply(0.8f), 0f)
    assertEquals(0f, ScoreCorrection(offset = -0.5f).apply(0.2f), 0f)
    assertEquals(1f, ScoreCorrection(Float.MAX_VALUE, Float.MAX_VALUE).apply(1f), 0f)
  }

  @Test
  fun invalidRawScoresAreRejectedBeforeCorrectionCanHideThem() {
    listOf(Float.NaN, Float.POSITIVE_INFINITY, -0.1f, 1.1f).forEach { raw ->
      assertTrue(runCatching { ScoreCorrection(scale = 0.1f).apply(raw) }.exceptionOrNull() is IllegalArgumentException)
    }
  }

  @Test
  fun invalidCoefficientsAreRejected() {
    listOf(0f, -1f, Float.NaN, Float.POSITIVE_INFINITY).forEach { scale ->
      assertTrue(runCatching { ScoreCorrection(scale = scale) }.exceptionOrNull() is IllegalArgumentException)
    }
    assertTrue(runCatching { ScoreCorrection(offset = Float.NaN) }.exceptionOrNull() is IllegalArgumentException)
    assertTrue(runCatching { ScoreCorrection(offset = Float.NEGATIVE_INFINITY) }.exceptionOrNull() is IllegalArgumentException)
  }

  @Test
  fun incompleteConfigurationAndWrongScoreCountAreRejected() {
    assertTrue(runCatching { ModelScoreCalibration(emptyMap()) }.exceptionOrNull() is IllegalArgumentException)
    val correction = calibration(DangerSound.FIRE_ALARM, ScoreCorrection())
    assertTrue(runCatching { correction.apply(FloatArray(5)) }.exceptionOrNull() is IllegalArgumentException)
  }

  @Test
  fun productionDefaultsPreserveScoresForEveryModelUntilFittedValuesExist() {
    val raw = floatArrayOf(0f, 0.1f, 0.3f, 0.5f, 0.8f, 1f)
    SoundModel.entries.forEach { model ->
      assertArrayEquals(raw, SoundScoreCalibrations.forModel(model).apply(raw), 0f)
    }
  }

  @Test
  fun wrapperPreservesAudioTimingAndClosesItsSource() {
    val source = Model(FloatArray(DangerSound.entries.size))
    val wrapper = CalibratedSoundClassifier(source, SoundScoreCalibrations.forModel(SoundModel.YAMNET))
    assertEquals(source.windowSize, wrapper.windowSize)
    assertEquals(source.hopSize, wrapper.hopSize)
    wrapper.close()
    assertTrue(source.closed)
  }
}
