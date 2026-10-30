package com.example.earview.detection

import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.TestCoroutineScheduler
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class YamnetSoundDetectorTest {
  private val chunks = MutableSharedFlow<FloatArray>()
  private val source = FakeAudioSource(chunks)
  private val classifier = FakeClassifier(windowSize = 4)

  private fun detectorOn(scheduler: kotlinx.coroutines.test.TestCoroutineScheduler) = YamnetSoundDetector(source, ::openClassifier, UnconfinedTestDispatcher(scheduler))

  private fun openClassifier() = classifier.also { it.open = true }

  private fun chunk(size: Int = 2) = FloatArray(size)

  private fun sceneWith(vararg sounds: DangerSound): FloatArray = FloatArray(YAMNET_CLASS_COUNT).also { s -> sounds.forEach { s[it.rule.classIndices.first()] = 1f } }

  @Test
  fun readsHalfAWindowAtATime() = runTest {
    val detector = detectorOn(testScheduler)
    backgroundScope.launch { detector.detect().collect {} }
    runCurrent()

    assertEquals(2, source.requestedChunkSize)
  }

  @Test
  fun classifiesOnlyOnceTheWindowIsFull() = runTest {
    val detector = detectorOn(testScheduler)
    backgroundScope.launch { detector.detect().collect {} }
    runCurrent()

    chunks.emit(chunk())
    assertEquals(0, classifier.windows.size)

    chunks.emit(chunk())
    assertEquals(1, classifier.windows.size)
    chunks.emit(chunk())
    assertEquals(2, classifier.windows.size)
  }

  @Test
  fun emitsTheSoundsTheClassifierHears_mostDangerousFirst() = runTest {
    val detector = detectorOn(testScheduler)
    val heard = mutableListOf<DangerSound>()
    backgroundScope.launch { detector.detect().toList(heard) }
    runCurrent()
    classifier.next = sceneWith(DangerSound.CAR_HORN, DangerSound.FIRE_ALARM)

    chunks.emit(chunk())
    chunks.emit(chunk())
    runCurrent()

    assertEquals(listOf(DangerSound.FIRE_ALARM, DangerSound.CAR_HORN), heard)
  }

  @Test
  fun quietAudio_emitsNothing() = runTest {
    val detector = detectorOn(testScheduler)
    val heard = mutableListOf<DangerSound>()
    backgroundScope.launch { detector.detect().toList(heard) }
    runCurrent()

    repeat(4) { chunks.emit(chunk()) }

    assertTrue(heard.isEmpty())
  }

  @Test
  fun emitsGlassBreakingAndDoorKnockFromTheMappedScores() = runTest {
    val heard = mutableListOf<DangerSound>()
    backgroundScope.launch { detectorOn(testScheduler).detect().toList(heard) }
    runCurrent()
    classifier.next = sceneWith(DangerSound.DOOR_KNOCK, DangerSound.GLASS_BREAKING)

    repeat(2) { chunks.emit(chunk()) }
    runCurrent()

    assertEquals(listOf(DangerSound.GLASS_BREAKING, DangerSound.DOOR_KNOCK), heard)
  }

  @Test
  fun classifierSeesTheNewestAudio_oldestFirst() = runTest {
    val detector = detectorOn(testScheduler)
    backgroundScope.launch { detector.detect().collect {} }
    runCurrent()

    chunks.emit(floatArrayOf(1f, 2f))
    chunks.emit(floatArrayOf(3f, 4f))
    chunks.emit(floatArrayOf(5f, 6f))

    assertEquals(listOf(listOf(1f, 2f, 3f, 4f), listOf(3f, 4f, 5f, 6f)), classifier.windows)
  }

  @Test
  fun modelAndMicrophoneAreOnlyOpenWhileCollecting() = runTest {
    val detector = detectorOn(testScheduler)
    assertFalse(source.open)
    assertFalse(classifier.open)

    val job = backgroundScope.launch { detector.detect().collect {} }
    runCurrent()
    assertTrue(source.open)
    assertTrue(classifier.open)

    job.cancel()
    runCurrent()
    assertFalse(source.open)
    assertFalse(classifier.open)
  }

  @Test
  fun microphoneFailure_failsTheFlow_andReleasesTheModel() = runTest {
    val failing = FakeAudioSource(flow { throw IllegalStateException("microphone lost") })
    val detector = YamnetSoundDetector(failing, ::openClassifier, UnconfinedTestDispatcher(testScheduler))
    var failure: Throwable? = null

    backgroundScope.launch { runCatching { detector.detect().collect {} }.onFailure { failure = it } }
    runCurrent()

    assertEquals("microphone lost", failure?.message)
    assertFalse(classifier.open)
  }

  @Test
  fun modelLoadFailure_failsTheFlow() = runTest {
    val detector = YamnetSoundDetector(source, { error("model missing") }, UnconfinedTestDispatcher(testScheduler))
    var failure: Throwable? = null

    backgroundScope.launch { runCatching { detector.detect().collect {} }.onFailure { failure = it } }
    runCurrent()

    assertEquals("model missing", failure?.message)
    assertFalse(source.open)
  }

  @Test
  fun microphoneEnding_completesTheFlow() = runTest {
    val ending = FakeAudioSource(flow { emit(chunk()) })
    val detector = YamnetSoundDetector(ending, ::openClassifier, UnconfinedTestDispatcher(testScheduler))
    var completed = false

    backgroundScope.launch {
      detector.detect().collect {}
      completed = true
    }
    runCurrent()

    assertTrue(completed)
  }

  private class FakeAudioSource(private val chunks: Flow<FloatArray>) : AudioSource {
    var open = false
    var requestedChunkSize = 0

    override fun chunks(chunkSize: Int): Flow<FloatArray> {
      requestedChunkSize = chunkSize
      return flow {
        open = true
        try {
          chunks.collect { emit(it) }
        } finally {
          open = false
        }
      }
    }
  }

  private class FakeClassifier(override val windowSize: Int) : SoundClassifier {
    var open = false
    var next = FloatArray(YAMNET_CLASS_COUNT)
    val windows = mutableListOf<List<Float>>()

    override fun classify(window: FloatArray): FloatArray {
      windows += window.toList()
      return next
    }

    override fun close() {
      open = false
    }
  }
}
