package com.example.earview.detection

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test
import java.io.File
import java.security.MessageDigest

/** Guards the bundled model and the "audio never leaves the device" rule (BL-005, BL-007). */
class ModelAssetTest {
  // Unit tests run with the app module as the working directory.
  private val moduleDir = File(".")

  @Test
  fun bundledModelIsTheKnownYamnetFile() {
    val model = File(moduleDir, "src/main/assets/yamnet.tflite")

    val sha256 = MessageDigest.getInstance("SHA-256").digest(model.readBytes()).joinToString("") { "%02x".format(it) }

    assertEquals(YAMNET_SHA256, sha256)
  }

  @Test
  fun appCannotUseTheNetwork() {
    val manifest = File(moduleDir, "src/main/AndroidManifest.xml").readText()

    assertFalse(manifest.contains("android.permission.INTERNET"))
  }

  @Test
  fun ensembleAssetsMatchTheValidatedExports() {
    val expected = mapOf(
      "efficientat_mn04_as" to "06eae4a7c9c00652d292be324d0557b36c15e0fce08391d4b02d5b19cc62d31a",
      "panns_mobilenetv2" to "407c06bda7f61975a917a29a004fb679bbb304191edc72279c63c967c53348d6",
    )
    expected.forEach { (name, hash) ->
      val bytes = File(moduleDir, "src/main/assets/ensemble/$name.tflite").readBytes()
      assertEquals("TFL3", String(bytes, 4, 4, Charsets.US_ASCII))
      val actual = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
      assertEquals(name, hash, actual)
    }
  }

  private companion object {
    // lite-model_yamnet_classification_tflite_1.tflite, from
    // https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/audio_classification/android/
    const val YAMNET_SHA256 = "10c95ea3eb9a7bb4cb8bddf6feb023250381008177ac162ce169694d05c317de"
  }
}
