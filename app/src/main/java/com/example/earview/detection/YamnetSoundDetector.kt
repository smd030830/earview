package com.example.earview.detection

import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn

/**
 * On-device [SoundDetector]: microphone chunks go through a sliding window into the classifier and
 * the scores are turned into danger sounds (BL-005, BL-006).
 *
 * Audio lives only in memory, in arrays that are overwritten as new audio arrives (BL-007).
 * The microphone and the models are opened when collection starts and released when it ends.
 */
class YamnetSoundDetector(
  private val source: AudioSource,
  private val classifierFactory: () -> DangerScoreClassifier,
  private val dispatcher: CoroutineDispatcher = Dispatchers.IO,
) : SoundDetector {
  override fun detect(): Flow<DangerSound> = flow {
    classifierFactory().use { classifier ->
      val window = SlidingAudioWindow(classifier.windowSize)
      // Keep the shortest model's cadence even when other models need a longer context.
      val hop = classifier.hopSize
      source.chunks(hop).collect { chunk ->
        val audio = window.append(chunk) ?: return@collect
        val scores = classifier.dangerScores(audio)
        require(scores.size == DangerSound.entries.size)
        DangerSound.entries.filter { scores[it.ordinal] >= it.rule.threshold }.forEach { emit(it) }
      }
    }
  }.flowOn(dispatcher)
}
