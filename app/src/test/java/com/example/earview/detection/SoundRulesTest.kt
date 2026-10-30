package com.example.earview.detection

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SoundRulesTest {
  private val labels: List<String> =
    checkNotNull(javaClass.getResourceAsStream("/yamnet_label_list.txt")) { "label list missing" }
      .bufferedReader()
      .readLines()

  private fun scores(vararg pairs: Pair<Int, Float>) = FloatArray(YAMNET_CLASS_COUNT).also { s -> pairs.forEach { (i, v) -> s[i] = v } }

  private fun indexOf(label: String) = labels.indexOf(label).also { assertTrue("no class named $label", it >= 0) }

  @Test
  fun labelListMatchesTheModelOutput() {
    assertEquals(YAMNET_CLASS_COUNT, labels.size)
  }

  @Test
  fun everyRuleUsesTheIntendedYamnetClasses() {
    val expected =
      mapOf(
        DangerSound.FIRE_ALARM to listOf("Smoke detector, smoke alarm", "Fire alarm"),
        DangerSound.SIREN to
          listOf(
            "Siren",
            "Civil defense siren",
            "Emergency vehicle",
            "Police car (siren)",
            "Ambulance (siren)",
            "Fire engine, fire truck (siren)",
          ),
        DangerSound.GLASS_BREAKING to listOf("Shatter"),
        DangerSound.DOORBELL to listOf("Doorbell", "Ding-dong"),
        DangerSound.DOOR_KNOCK to listOf("Knock"),
        DangerSound.CAR_HORN to listOf("Vehicle horn, car horn, honking"),
      )

    DangerSound.entries.forEach { sound ->
      assertEquals("$sound", expected.getValue(sound).map(::indexOf).sorted(), sound.rule.classIndices.sorted())
    }
  }

  @Test
  fun thresholdsAreProbabilities() {
    DangerSound.entries.forEach { assertTrue("$it", it.rule.threshold > 0f && it.rule.threshold <= 1f) }
  }

  @Test
  fun silence_detectsNothing() {
    assertTrue(detectedSounds(FloatArray(YAMNET_CLASS_COUNT)).isEmpty())
  }

  @Test
  fun eachSound_isDetectedAtItsThreshold_andNotJustBelow() {
    DangerSound.entries.forEach { sound ->
      val index = sound.rule.classIndices.first()
      val at = sound.rule.threshold

      assertEquals("$sound at threshold", listOf(sound), detectedSounds(scores(index to at)))
      assertTrue("$sound below threshold", detectedSounds(scores(index to at - 0.01f)).isEmpty())
    }
  }

  @Test
  fun anyClassOfASound_triggersIt() {
    DangerSound.SIREN.rule.classIndices.forEach { index ->
      assertEquals(listOf(DangerSound.SIREN), detectedSounds(scores(index to 0.9f)))
    }
  }

  @Test
  fun unrelatedClass_detectsNothing() {
    val speech = indexOf("Speech")

    assertTrue(detectedSounds(scores(speech to 1f)).isEmpty())
  }

  @Test
  fun genericAlarmAndGooseHonk_doNotCountAsDangerSounds() {
    assertTrue(detectedSounds(scores(indexOf("Alarm") to 1f, indexOf("Honk") to 1f, indexOf("Alarm clock") to 1f)).isEmpty())
  }

  @Test
  fun babyCryingGenericGlassAndEngineKnocking_doNotTriggerNewSounds() {
    assertTrue(
      detectedSounds(
        scores(
          indexOf("Baby cry, infant cry") to 1f,
          indexOf("Glass") to 1f,
          indexOf("Chink, clink") to 1f,
          indexOf("Engine knocking") to 1f,
        ),
      ).isEmpty(),
    )
  }

  @Test
  fun targetSetContainsTheSixApprovedSoundsInPriorityOrder() {
    assertEquals(
      listOf(
        DangerSound.FIRE_ALARM,
        DangerSound.SIREN,
        DangerSound.GLASS_BREAKING,
        DangerSound.DOORBELL,
        DangerSound.DOOR_KNOCK,
        DangerSound.CAR_HORN,
      ),
      DangerSound.entries,
    )
  }

  @Test
  fun severalSounds_comeMostDangerousFirst() {
    val result =
      detectedSounds(
        scores(
          indexOf("Doorbell") to 0.9f,
          indexOf("Fire alarm") to 0.9f,
          indexOf("Siren") to 0.9f,
        ),
      )

    assertEquals(listOf(DangerSound.FIRE_ALARM, DangerSound.SIREN, DangerSound.DOORBELL), result)
  }

  @Test(expected = IllegalArgumentException::class)
  fun wrongNumberOfScores_isRejected() {
    detectedSounds(FloatArray(10))
  }
}
