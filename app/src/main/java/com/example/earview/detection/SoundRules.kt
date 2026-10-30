package com.example.earview.detection

/** Number of classes YAMNet scores. */
internal const val YAMNET_CLASS_COUNT = 521

/**
 * Which YAMNet classes stand for a [DangerSound] and how sure the model must be.
 *
 * A sound counts as heard when the best score among [classIndices] reaches [threshold].
 */
internal class SoundRule(val classIndices: IntArray, val threshold: Float)

// YAMNet class indexes (see yamnet_label_list.txt; SoundRulesTest checks every name below).
private const val SHATTER = 437 // "Shatter"
private const val KNOCK = 353 // "Knock"
private const val DOORBELL = 349 // "Doorbell"
private const val DING_DONG = 350 // "Ding-dong"
private const val EMERGENCY_VEHICLE = 316 // "Emergency vehicle"
private const val POLICE_CAR = 317 // "Police car (siren)"
private const val AMBULANCE = 318 // "Ambulance (siren)"
private const val FIRE_ENGINE = 319 // "Fire engine, fire truck (siren)"
private const val VEHICLE_HORN = 302 // "Vehicle horn, car horn, honking"
private const val SIREN = 390 // "Siren"
private const val CIVIL_DEFENSE_SIREN = 391 // "Civil defense siren"
private const val SMOKE_DETECTOR = 393 // "Smoke detector, smoke alarm"
private const val FIRE_ALARM = 394 // "Fire alarm"

/*
 * Thresholds are internal constants, not user settings. They are provisional starting values.
 *
 * Measured with the bundled model on ESC-50 clips (5 s each, scored per 0.975 s window; best
 * window per clip), in recall on the matching category and false alarms on 672 clips of other
 * categories:
 *  - siren: 97% of 40 clips at 0.5
 *  - car horn: 48% at 0.5 and 57% at 0.3 of 40 clips, with 2 and 3 false alarms; many of its
 *    clips are short or distant, so it gets the lower value
 *  - fire alarm and doorbell: ESC-50 has no such clips, so these are NOT measured; 0.5 is a guess
 *  - glass breaking and door knock: 0.5 is a provisional value, not a measured threshold
 * These historical figures are for YAMNet alone, not the current three-model ensemble.
 * Clips per category are far too few to estimate false alarms per hour, so every value still needs
 * tuning on real devices (PRD: at most 3 false alarms per hour).
 */
private val FIRE_ALARM_RULE = SoundRule(intArrayOf(SMOKE_DETECTOR, FIRE_ALARM), threshold = 0.5f)
private val SIREN_RULE =
  SoundRule(
    intArrayOf(SIREN, CIVIL_DEFENSE_SIREN, EMERGENCY_VEHICLE, POLICE_CAR, AMBULANCE, FIRE_ENGINE),
    threshold = 0.5f,
  )
private val GLASS_BREAKING_RULE = SoundRule(intArrayOf(SHATTER), threshold = 0.5f)
private val DOORBELL_RULE = SoundRule(intArrayOf(DOORBELL, DING_DONG), threshold = 0.5f)
private val DOOR_KNOCK_RULE = SoundRule(intArrayOf(KNOCK), threshold = 0.5f)
private val CAR_HORN_RULE = SoundRule(intArrayOf(VEHICLE_HORN), threshold = 0.3f)

internal val DangerSound.rule: SoundRule
  get() =
    when (this) {
      DangerSound.FIRE_ALARM -> FIRE_ALARM_RULE
      DangerSound.SIREN -> SIREN_RULE
      DangerSound.GLASS_BREAKING -> GLASS_BREAKING_RULE
      DangerSound.DOORBELL -> DOORBELL_RULE
      DangerSound.DOOR_KNOCK -> DOOR_KNOCK_RULE
      DangerSound.CAR_HORN -> CAR_HORN_RULE
    }

/** The danger sounds whose rule is met by YAMNet [scores], most dangerous first. */
internal fun detectedSounds(scores: FloatArray): List<DangerSound> {
  require(scores.size == YAMNET_CLASS_COUNT) { "Expected $YAMNET_CLASS_COUNT scores but got ${scores.size}" }
  return DangerSound.entries.filter { sound ->
    val rule = sound.rule
    rule.classIndices.any { scores[it] >= rule.threshold }
  }
}
