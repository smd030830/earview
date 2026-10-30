package com.example.earview.detection

/** A vibration must not run longer than this, so a pattern never outlasts the alert it announces. */
internal const val MAX_VIBRATION_MILLIS = 3_000L

/**
 * Vibration pattern per sound (FR-02), as Android waveform timings: off, on, off, on, ...
 * in milliseconds. Every pattern starts with 0 so it begins immediately, and none repeats.
 *
 * The patterns are proposals pending a check on real devices. More dangerous sounds feel longer.
 */
internal val DangerSound.vibrationTimings: LongArray
  get() =
    when (this) {
      // Three long pulses.
      DangerSound.FIRE_ALARM -> longArrayOf(0, 700, 150, 700, 150, 700)
      // Long, short, long.
      DangerSound.SIREN -> longArrayOf(0, 600, 150, 200, 150, 600)
      // Three rapid pulses followed by one long pulse.
      DangerSound.GLASS_BREAKING -> longArrayOf(0, 120, 100, 120, 100, 120, 150, 500)
      // Two short pulses ("ding-dong").
      DangerSound.DOORBELL -> longArrayOf(0, 150, 150, 150)
      // Three evenly spaced knocks.
      DangerSound.DOOR_KNOCK -> longArrayOf(0, 200, 250, 200, 250, 200)
      // One long pulse.
      DangerSound.CAR_HORN -> longArrayOf(0, 600)
    }
