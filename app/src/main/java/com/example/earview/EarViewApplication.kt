package com.example.earview

import android.app.Application
import android.content.Context
import com.example.earview.detection.AndroidAlertVibrator
import com.example.earview.detection.AudioRecordSource
import com.example.earview.detection.DetectionController
import com.example.earview.detection.YamnetSoundDetector
import com.example.earview.detection.loadEnsemble
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob

class EarViewApplication : Application() {
  /** The one detection state holder shared by the UI and the detection service. */
  val detectionController: DetectionController by lazy {
    DetectionController(
      YamnetSoundDetector(AudioRecordSource(), { loadEnsemble(this) }),
      AndroidAlertVibrator(this),
      CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate),
    )
  }
}

val Context.detectionController: DetectionController
  get() = (applicationContext as EarViewApplication).detectionController
