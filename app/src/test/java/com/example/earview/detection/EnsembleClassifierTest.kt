package com.example.earview.detection

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class EnsembleClassifierTest {
  private class Model(override val windowSize: Int, val scores: FloatArray) : DangerScoreClassifier {
    var input = floatArrayOf()
    var closed = false

    override fun dangerScores(window: FloatArray): FloatArray {
      input = window.copyOf()
      return scores
    }

    override fun close() {
      closed = true
    }
  }

  @Test
  fun averagesEachSoundIndependently_andAlignsNewestSamples() {
    val a = Model(4, floatArrayOf(0.9f, 0f, 0f, 0.9f, 0f))
    val b = Model(2, floatArrayOf(0.6f, 0f, 0f, 0.6f, 0f))
    val c = Model(3, floatArrayOf(0.3f, 0f, 0f, 0.3f, 0f))
    val ensemble = EnsembleClassifier(listOf(a, b, c))
    assertEquals(4, ensemble.windowSize)
    assertEquals(1, ensemble.hopSize)
    assertArrayEquals(floatArrayOf(0.6f, 0f, 0f, 0.6f, 0f), ensemble.dangerScores(floatArrayOf(1f, 2f, 3f, 4f)), 0.00001f)
    assertArrayEquals(floatArrayOf(3f, 4f), b.input, 0f)
    assertArrayEquals(floatArrayOf(2f, 3f, 4f), c.input, 0f)
    ensemble.close()
    assertTrue(listOf(a, b, c).all { it.closed })
  }

  @Test(expected = IllegalArgumentException::class)
  fun rejectsMissingModels() {
    EnsembleClassifier(emptyList())
  }

  @Test(expected = IllegalArgumentException::class)
  fun rejectsInvalidConfidenceInsteadOfSilentlyMissingDanger() {
    val model = Model(2, FloatArray(5) { Float.NaN })
    EnsembleClassifier(List(3) { model }).dangerScores(FloatArray(2))
  }

  @Test(expected = IllegalArgumentException::class)
  fun rejectsMissingSoundScores() {
    val model = Model(2, FloatArray(4))
    EnsembleClassifier(List(3) { model }).dangerScores(FloatArray(2))
  }

  @Test
  fun yamnetAdapterGroupsRelatedClassesBeforeAveraging() {
    val model = object : SoundClassifier {
      override val windowSize = 2

      override fun classify(window: FloatArray): FloatArray = FloatArray(YAMNET_CLASS_COUNT).also { scores ->
        DangerSound.entries.forEach { sound ->
          sound.rule.classIndices.forEachIndexed { index, label -> scores[label] = (index + 1) / 10f }
        }
      }

      override fun close() = Unit
    }
    val expected = FloatArray(5) { DangerSound.entries[it].rule.classIndices.size / 10f }
    assertArrayEquals(expected, model.dangerScores(FloatArray(2)), 0f)
  }

  @Test
  fun releasesOtherModelsWhenOneCloseFails() {
    val a = Model(2, FloatArray(5))
    val b = Model(2, FloatArray(5))
    val broken = object : DangerScoreClassifier {
      override val windowSize = 2

      override fun dangerScores(window: FloatArray) = FloatArray(5)

      override fun close(): Unit = error("close failed")
    }
    val ensemble = EnsembleClassifier(listOf(a, b, broken))
    runCatching { ensemble.close() }
    assertTrue(a.closed && b.closed)
  }
}
