package com.example.earview.detection

import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class DetectionControllerTest {
  private val detector = FakeSoundDetector()
  private val vibrator = FakeAlertVibrator()

  private fun TestScope.createController() = DetectionController(detector, vibrator, backgroundScope).also { runCurrent() }

  private fun TestScope.start(controller: DetectionController) {
    controller.start()
    runCurrent()
  }

  private suspend fun TestScope.detect(vararg sounds: DangerSound) {
    sounds.forEach { detector.emit(it) }
    runCurrent()
  }

  private fun TestScope.advance(millis: Long) {
    advanceTimeBy(millis)
    runCurrent()
  }

  private fun detected(vararg sounds: DangerSound, overflow: Int = 0) = DetectionState.Detected(sounds.toList(), overflow)

  // Off / Idle

  @Test
  fun initialState_isOff_andNotListening() = runTest {
    val controller = createController()

    assertEquals(DetectionState.Off, controller.state.value)
    assertFalse(detector.isListening)
  }

  @Test
  fun start_movesToIdle_andStartsListening() = runTest {
    val controller = createController()

    start(controller)

    assertEquals(DetectionState.Idle, controller.state.value)
    assertTrue(detector.isListening)
  }

  @Test
  fun stopFromIdle_movesToOff_andStopsListening() = runTest {
    val controller = createController()
    start(controller)

    controller.stop()
    runCurrent()

    assertEquals(DetectionState.Off, controller.state.value)
    assertFalse(detector.isListening)
  }

  @Test
  fun startAfterStop_returnsToIdle() = runTest {
    val controller = createController()
    start(controller)
    controller.stop()
    runCurrent()

    start(controller)

    assertEquals(DetectionState.Idle, controller.state.value)
    assertTrue(detector.isListening)
  }

  // Detected

  @Test
  fun detection_movesIdleToDetected() = runTest {
    val controller = createController()
    start(controller)

    detect(DangerSound.FIRE_ALARM)

    assertEquals(detected(DangerSound.FIRE_ALARM), controller.state.value)
  }

  @Test
  fun glassBreakingAndDoorKnock_alertVibrateAndReturnToIdle() = runTest {
    val controller = createController()
    start(controller)

    detect(DangerSound.DOOR_KNOCK, DangerSound.GLASS_BREAKING)

    assertEquals(detected(DangerSound.GLASS_BREAKING, DangerSound.DOOR_KNOCK), controller.state.value)
    assertEquals(listOf(DangerSound.DOOR_KNOCK, DangerSound.GLASS_BREAKING), vibrator.vibrations)

    advance(ALERT_AUTO_RETURN_MILLIS)
    assertEquals(DetectionState.Idle, controller.state.value)
  }

  @Test
  fun stopWhileDetected_movesToOff_andClearsSounds() = runTest {
    val controller = createController()
    start(controller)
    detect(DangerSound.SIREN)

    controller.stop()
    runCurrent()

    assertEquals(DetectionState.Off, controller.state.value)
    assertFalse(detector.isListening)

    start(controller)
    assertEquals(DetectionState.Idle, controller.state.value)
  }

  @Test
  fun pendingTimers_afterStop_doNotChangeState() = runTest {
    val controller = createController()
    start(controller)
    detect(DangerSound.SIREN)
    controller.stop()
    runCurrent()

    advance(ALERT_AUTO_RETURN_MILLIS)

    assertEquals(DetectionState.Off, controller.state.value)
  }

  // Auto-return after 10s

  @Test
  fun detected_returnsToIdle_after10Seconds() = runTest {
    val controller = createController()
    start(controller)
    detect(DangerSound.FIRE_ALARM)

    advance(ALERT_AUTO_RETURN_MILLIS - 1)
    assertTrue(controller.state.value is DetectionState.Detected)

    advance(1)
    assertEquals(DetectionState.Idle, controller.state.value)
    assertTrue(detector.isListening)
  }

  @Test
  fun newDetection_restartsAutoReturnTimer() = runTest {
    val controller = createController()
    start(controller)
    detect(DangerSound.FIRE_ALARM)
    advance(6_000)

    detect(DangerSound.FIRE_ALARM)
    advance(ALERT_AUTO_RETURN_MILLIS - 1)
    assertTrue(controller.state.value is DetectionState.Detected)

    advance(1)
    assertEquals(DetectionState.Idle, controller.state.value)
  }

  // Tap to return

  @Test
  fun tapWhileDetected_returnsToIdleImmediately() = runTest {
    val controller = createController()
    start(controller)
    detect(DangerSound.FIRE_ALARM)

    controller.dismissAlert()

    assertEquals(DetectionState.Idle, controller.state.value)
    assertTrue(detector.isListening)
  }

  @Test
  fun tapWhileDetected_cancelsPendingTimers() = runTest {
    val controller = createController()
    start(controller)
    detect(DangerSound.FIRE_ALARM)
    controller.dismissAlert()

    advance(ALERT_AUTO_RETURN_MILLIS)

    assertEquals(DetectionState.Idle, controller.state.value)
  }

  @Test
  fun detectionAfterTap_movesToDetectedAgain_withoutOldSounds() = runTest {
    val controller = createController()
    start(controller)
    detect(DangerSound.FIRE_ALARM)
    controller.dismissAlert()

    detect(DangerSound.DOORBELL)

    assertEquals(detected(DangerSound.DOORBELL), controller.state.value)
  }

  @Test
  fun tapWhileIdle_isIgnored() = runTest {
    val controller = createController()
    start(controller)

    controller.dismissAlert()

    assertEquals(DetectionState.Idle, controller.state.value)
  }

  @Test
  fun tapWhileOff_isIgnored() = runTest {
    val controller = createController()

    controller.dismissAlert()

    assertEquals(DetectionState.Off, controller.state.value)
  }

  // Multiple sounds: order, max 3, 3s removal

  @Test
  fun simultaneousSounds_areOrderedByDanger() = runTest {
    val controller = createController()
    start(controller)

    detect(DangerSound.CAR_HORN, DangerSound.SIREN)

    assertEquals(detected(DangerSound.SIREN, DangerSound.CAR_HORN), controller.state.value)
  }

  @Test
  fun moreThanThreeSounds_showsThreeAndOverflowCount() = runTest {
    val controller = createController()
    start(controller)

    detect(*DangerSound.entries.reversed().toTypedArray())

    assertEquals(
      detected(DangerSound.FIRE_ALARM, DangerSound.SIREN, DangerSound.GLASS_BREAKING, overflow = 3),
      controller.state.value,
    )
  }

  @Test
  fun exactlyThreeSounds_hasNoOverflow() = runTest {
    val controller = createController()
    start(controller)

    detect(DangerSound.FIRE_ALARM, DangerSound.SIREN, DangerSound.DOORBELL)

    assertEquals(0, (controller.state.value as DetectionState.Detected).overflowCount)
  }

  @Test
  fun sound_isRemovedFromList_3SecondsAfterLastDetection() = runTest {
    val controller = createController()
    start(controller)
    detect(DangerSound.SIREN)
    advance(1_000)
    detect(DangerSound.CAR_HORN)

    advance(SOUND_EXPIRY_MILLIS - 1_000)

    assertEquals(detected(DangerSound.CAR_HORN), controller.state.value)
  }

  @Test
  fun sound_justBeforeExpiry_isStillListed() = runTest {
    val controller = createController()
    start(controller)
    detect(DangerSound.SIREN, DangerSound.CAR_HORN)

    advance(SOUND_EXPIRY_MILLIS - 1)

    assertEquals(detected(DangerSound.SIREN, DangerSound.CAR_HORN), controller.state.value)
  }

  @Test
  fun redetectedSound_restartsItsExpiryTimer() = runTest {
    val controller = createController()
    start(controller)
    detect(DangerSound.SIREN)
    advance(2_000)
    detect(DangerSound.SIREN)

    advance(2_000)

    assertEquals(detected(DangerSound.SIREN), controller.state.value)
  }

  @Test
  fun overflowCount_shrinksWhenSoundsExpire() = runTest {
    val controller = createController()
    start(controller)
    detect(DangerSound.FIRE_ALARM, DangerSound.SIREN, DangerSound.GLASS_BREAKING, DangerSound.DOORBELL)
    assertEquals(1, (controller.state.value as DetectionState.Detected).overflowCount)
    advance(2_000)
    detect(DangerSound.DOORBELL)

    advance(1_000)

    assertEquals(detected(DangerSound.DOORBELL), controller.state.value)
  }

  @Test
  fun allSoundsExpired_staysDetectedWithEmptyList_untilAutoReturn() = runTest {
    val controller = createController()
    start(controller)
    detect(DangerSound.FIRE_ALARM)

    advance(SOUND_EXPIRY_MILLIS)
    assertEquals(detected(), controller.state.value)

    advance(ALERT_AUTO_RETURN_MILLIS - SOUND_EXPIRY_MILLIS)
    assertEquals(DetectionState.Idle, controller.state.value)
  }

  // Vibration (FR-02)

  @Test
  fun firstDetection_vibratesWithThatSoundsPattern() = runTest {
    val controller = createController()
    start(controller)

    detect(DangerSound.DOORBELL)

    assertEquals(listOf(DangerSound.DOORBELL), vibrator.vibrations)
  }

  @Test
  fun idle_doesNotVibrate() = runTest {
    val controller = createController()
    start(controller)

    assertTrue(vibrator.vibrations.isEmpty())
  }

  @Test
  fun sameSoundKeepsGoing_vibratesOnlyOnce() = runTest {
    val controller = createController()
    start(controller)

    detect(DangerSound.FIRE_ALARM)
    advance(1_000)
    detect(DangerSound.FIRE_ALARM)
    advance(1_000)
    detect(DangerSound.FIRE_ALARM)

    assertEquals(listOf(DangerSound.FIRE_ALARM), vibrator.vibrations)
  }

  @Test
  fun soundDetectedAgainAfterExpiry_vibratesAgain() = runTest {
    val controller = createController()
    start(controller)
    detect(DangerSound.CAR_HORN)

    advance(SOUND_EXPIRY_MILLIS)
    detect(DangerSound.CAR_HORN)

    assertEquals(listOf(DangerSound.CAR_HORN, DangerSound.CAR_HORN), vibrator.vibrations)
  }

  @Test
  fun anotherSoundWhileAlerting_vibratesWithItsOwnPattern() = runTest {
    val controller = createController()
    start(controller)

    detect(DangerSound.CAR_HORN)
    detect(DangerSound.SIREN)

    assertEquals(listOf(DangerSound.CAR_HORN, DangerSound.SIREN), vibrator.vibrations)
  }

  @Test
  fun tapToReturn_cancelsVibration_andDetectionAfterwardsVibratesAgain() = runTest {
    val controller = createController()
    start(controller)
    detect(DangerSound.FIRE_ALARM)
    val cancelsBefore = vibrator.cancelCount

    controller.dismissAlert()
    assertEquals(cancelsBefore + 1, vibrator.cancelCount)

    detect(DangerSound.FIRE_ALARM)
    assertEquals(listOf(DangerSound.FIRE_ALARM, DangerSound.FIRE_ALARM), vibrator.vibrations)
  }

  @Test
  fun stop_cancelsVibration() = runTest {
    val controller = createController()
    start(controller)
    detect(DangerSound.FIRE_ALARM)
    val cancelsBefore = vibrator.cancelCount

    controller.stop()

    assertTrue(vibrator.cancelCount > cancelsBefore)
  }

  @Test
  fun soundAfterStop_doesNotVibrate() = runTest {
    val controller = createController()
    start(controller)
    controller.stop()
    runCurrent()

    detect(DangerSound.FIRE_ALARM)

    assertTrue(vibrator.vibrations.isEmpty())
  }

  // Stop events (FR-10)

  @Test
  fun userStop_emitsUserStopEvent() = runTest {
    val controller = createController()
    val events = mutableListOf<StopReason>()
    backgroundScope.launch(UnconfinedTestDispatcher(testScheduler)) { controller.stopEvents.toList(events) }
    start(controller)

    controller.stop()
    runCurrent()

    assertEquals(listOf(StopReason.User), events)
  }

  @Test
  fun stopWhileNotRunning_emitsNothing() = runTest {
    val controller = createController()
    val events = mutableListOf<StopReason>()
    backgroundScope.launch(UnconfinedTestDispatcher(testScheduler)) { controller.stopEvents.toList(events) }

    controller.stop()
    runCurrent()

    assertTrue(events.isEmpty())
  }

  @Test
  fun detectorFailure_emitsUnexpectedStopEvent() = runTest {
    val failingDetector = SoundDetector { flow { throw IllegalStateException("microphone lost") } }
    val controller = DetectionController(failingDetector, vibrator, backgroundScope)
    val events = mutableListOf<StopReason>()
    backgroundScope.launch(UnconfinedTestDispatcher(testScheduler)) { controller.stopEvents.toList(events) }

    start(controller)

    assertEquals(listOf(StopReason.Unexpected), events)
  }

  @Test
  fun alertOnUnexpectedStop_notifiesOnlyForUnexpectedStops() = runTest {
    val notifier = RecordingNotifier()
    val events = MutableSharedFlow<StopReason>()
    backgroundScope.launch(UnconfinedTestDispatcher(testScheduler)) { events.alertOnUnexpectedStop(notifier) }

    events.emit(StopReason.User)
    assertEquals(0, notifier.shown)

    events.emit(StopReason.Unexpected)
    assertEquals(1, notifier.shown)
  }

  @Test
  fun isRunning_followsStartAndStop() = runTest {
    val controller = createController()
    assertFalse(controller.isRunning)

    start(controller)
    assertTrue(controller.isRunning)

    controller.stop()
    assertFalse(controller.isRunning)
  }

  // Detector ending on its own

  @Test
  fun detectorFailure_movesToOff() = runTest {
    val failingDetector = SoundDetector { flow { throw IllegalStateException("microphone lost") } }
    val controller = DetectionController(failingDetector, vibrator, backgroundScope)

    start(controller)

    assertEquals(DetectionState.Off, controller.state.value)
  }

  @Test
  fun detectorCompletion_movesToOff_andCanBeRestarted() = runTest {
    var completes = true
    val detector = SoundDetector { if (completes) flow { } else flow<DangerSound> { awaitCancellation() } }
    val controller = DetectionController(detector, vibrator, backgroundScope)
    start(controller)
    assertEquals(DetectionState.Off, controller.state.value)

    completes = false
    start(controller)

    assertEquals(DetectionState.Idle, controller.state.value)
  }
}

/** Test double that lets tests emit detections and observe whether anything is listening. */
private class FakeSoundDetector : SoundDetector {
  private val events = MutableSharedFlow<DangerSound>()

  val isListening: Boolean
    get() = events.subscriptionCount.value > 0

  override fun detect(): Flow<DangerSound> = events

  suspend fun emit(sound: DangerSound) = events.emit(sound)
}

private class RecordingNotifier : StoppedNotifier {
  var shown = 0
  var cleared = 0

  override fun showStopped() {
    shown++
  }

  override fun clearStopped() {
    cleared++
  }
}
