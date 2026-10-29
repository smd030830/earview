package com.example.earview.detection

import android.content.Context
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Exported models include their own preprocessing and five-label mapping. */
class TfliteDangerClassifier private constructor(private val interpreter: Interpreter) : DangerScoreClassifier {
  override val windowSize = interpreter.getInputTensor(0).numElements()
  private val input = buffer(windowSize)
  private val output = buffer(DangerSound.entries.size)

  init {
    require(interpreter.inputTensorCount == 1 && interpreter.outputTensorCount == 1)
    require(interpreter.getInputTensor(0).dataType() == DataType.FLOAT32)
    require(interpreter.getOutputTensor(0).dataType() == DataType.FLOAT32)
    require(interpreter.getOutputTensor(0).numElements() == DangerSound.entries.size)
    require(windowSize in 1..(SAMPLE_RATE_HZ * 2)) { "Model window exceeds the latency budget" }
  }

  override fun dangerScores(window: FloatArray): FloatArray {
    require(window.size == windowSize)
    input.clear()
    input.asFloatBuffer().put(window)
    output.clear()
    interpreter.run(input, output)
    output.rewind()
    return FloatArray(DangerSound.entries.size).also { output.asFloatBuffer().get(it) }
  }

  override fun close() = interpreter.close()

  companion object {
    fun load(context: Context, asset: String): TfliteDangerClassifier {
      val bytes = context.assets.open(asset).use { it.readBytes() }
      val model = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
      model.put(bytes).rewind()
      val interpreter = Interpreter(model, Interpreter.Options().setNumThreads(2))
      return try {
        TfliteDangerClassifier(interpreter)
      } catch (e: Exception) {
        interpreter.close()
        throw e
      }
    }

    private fun buffer(size: Int): ByteBuffer = ByteBuffer.allocateDirect(size * Float.SIZE_BYTES).order(ByteOrder.nativeOrder())
  }
}
