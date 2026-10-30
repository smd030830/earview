package com.example.earview.detection

import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Executes the bundled models with the Android runtime, rather than substituting fakes. */
@RunWith(AndroidJUnit4::class)
class EnsembleModelsTest {
  @Test
  fun bundledModelsAcceptAudioAndReturnSixFiniteScores() {
    loadEnsemble(ApplicationProvider.getApplicationContext()).use { ensemble ->
      assertEquals(31_200, ensemble.windowSize)
      repeat(2) {
        val scores = ensemble.dangerScores(FloatArray(ensemble.windowSize))
        assertEquals(DangerSound.entries.size, scores.size)
        assertTrue(scores.all { it.isFinite() && it in 0f..1f })
      }
    }
  }
}
