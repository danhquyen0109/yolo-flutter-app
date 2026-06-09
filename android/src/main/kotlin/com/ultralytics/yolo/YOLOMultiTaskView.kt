// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.util.Log
import android.view.Gravity
import android.view.View
import android.widget.FrameLayout
import android.widget.ProgressBar
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

// MARK: - Detection box overlay

class DetectionOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    data class BoxInfo(val rect: RectF, val label: String, val color: Int)

    private var boxes: List<BoxInfo> = emptyList()

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
    }
    private val fillPaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 28f
        isAntiAlias = true
        isFakeBoldText = true
    }

    init {
        setWillNotDraw(false)
        setBackgroundColor(Color.TRANSPARENT)
    }

    fun setDetections(newBoxes: List<BoxInfo>) {
        boxes = newBoxes
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val labelPad = 6f
        for (info in boxes) {
            boxPaint.color = info.color
            canvas.drawRect(info.rect, boxPaint)

            val textWidth = textPaint.measureText(info.label)
            val textHeight = textPaint.descent() - textPaint.ascent()
            val labelTop = maxOf(0f, info.rect.top - textHeight - labelPad * 2)
            val labelRect = RectF(
                info.rect.left,
                labelTop,
                info.rect.left + textWidth + labelPad * 2,
                labelTop + textHeight + labelPad
            )
            fillPaint.color = Color.argb(192, Color.red(info.color), Color.green(info.color), Color.blue(info.color))
            canvas.drawRect(labelRect, fillPaint)
            canvas.drawText(
                info.label,
                labelRect.left + labelPad,
                labelRect.bottom - labelPad / 2 - textPaint.descent(),
                textPaint
            )
        }
    }
}

// MARK: - YOLOMultiTaskView

/**
 * Android equivalent of the iOS YOLOMultiTaskView. Hosts a single CameraX session and
 * dispatches each camera frame to three independent YOLO predictors (detect, segment,
 * classify) concurrently on their own background threads. Results are delivered to
 * [onMultiTaskStream] on the main thread.
 */
class YOLOMultiTaskView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : FrameLayout(context, attrs) {

    companion object {
        private const val TAG = "YOLOMultiTaskView"
    }

    // Camera
    private val previewView = PreviewView(context)
    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraExecutor: ExecutorService? = null

    // Predictors
    @Volatile private var detectPredictor: Predictor? = null
    @Volatile private var segmentPredictor: Predictor? = null
    @Volatile private var classifyPredictor: Predictor? = null

    // Per-predictor executors — each runs predict() concurrently
    private val detectExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val segmentExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val classifyExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    // One-frame-deep back-pressure flags (atomic so they're safe to read on cameraExecutor)
    private val detectBusy = AtomicBoolean(false)
    private val segmentBusy = AtomicBoolean(false)
    private val classifyBusy = AtomicBoolean(false)

    // Per-task FPS tracking (only accessed on their respective executor threads)
    private var detectLastResultTime: Long = 0
    private var segmentLastResultTime: Long = 0
    private var classifyLastResultTime: Long = 0

    // Camera FPS (accessed on cameraExecutor)
    private var camFrameCount = 0
    private var camFpsWindowStart: Long = System.currentTimeMillis()
    @Volatile private var camFps: Double = 0.0

    // Views
    private val overlayView = DetectionOverlayView(context)
    private val progressBar = ProgressBar(context)

    private var lifecycleOwner: LifecycleOwner? = null

    /** Called on the main thread for each inference result. */
    var onMultiTaskStream: ((Map<String, Any>) -> Unit)? = null

    init {
        setBackgroundColor(Color.BLACK)
        addView(previewView, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT))
        addView(overlayView, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT))
        val pbParams = LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT)
        pbParams.gravity = Gravity.CENTER
        addView(progressBar, pbParams)
    }

    fun onLifecycleOwnerAvailable(owner: LifecycleOwner) {
        lifecycleOwner = owner
    }

    // MARK: - Model loading

    fun loadModels(
        detectPath: String,
        segmentPath: String,
        classifyPath: String,
        useGpu: Boolean,
        lensFacing: Int = CameraSelector.LENS_FACING_BACK,
        onLoaded: (() -> Unit)? = null
    ) {
        var loadedCount = 0

        fun onOneLoaded() {
            val allDone = synchronized(this) {
                loadedCount++
                loadedCount == 3
            }
            if (allDone) {
                post {
                    progressBar.visibility = View.GONE
                    startCamera(lensFacing)
                    onLoaded?.invoke()
                }
            }
        }

        Executors.newSingleThreadExecutor().execute {
            try {
                detectPredictor = ObjectDetector(context = context, modelPath = detectPath, labels = emptyList(), useGpu = useGpu)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load detect model: $detectPath", e)
            }
            onOneLoaded()
        }

        Executors.newSingleThreadExecutor().execute {
            try {
                segmentPredictor = Segmenter(context, segmentPath, labels = emptyList(), useGpu = useGpu)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load segment model: $segmentPath", e)
            }
            onOneLoaded()
        }

        Executors.newSingleThreadExecutor().execute {
            try {
                classifyPredictor = Classifier(context, classifyPath, labels = emptyList(), useGpu = useGpu)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load classify model: $classifyPath", e)
            }
            onOneLoaded()
        }
    }

    // MARK: - Camera

    private fun startCamera(lensFacing: Int) {
        val owner = lifecycleOwner ?: return
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            cameraProvider = future.get()
            bindCamera(lensFacing, owner)
        }, ContextCompat.getMainExecutor(context))
    }

    private fun bindCamera(lensFacing: Int, owner: LifecycleOwner) {
        val provider = cameraProvider ?: return

        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(lensFacing)
            .build()

        val preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .build()
        preview.setSurfaceProvider(previewView.surfaceProvider)

        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        val exec = Executors.newSingleThreadExecutor()
        cameraExecutor = exec
        analysis.setAnalyzer(exec) { imageProxy -> onFrame(imageProxy) }

        try {
            provider.unbindAll()
            provider.bindToLifecycle(owner, cameraSelector, preview, analysis)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to bind camera", e)
        }
    }

    // MARK: - Frame processing

    private fun onFrame(imageProxy: ImageProxy) {
        // Track camera FPS
        camFrameCount++
        val now = System.currentTimeMillis()
        val elapsed = now - camFpsWindowStart
        if (elapsed >= 500) {
            camFps = camFrameCount * 1000.0 / elapsed
            camFrameCount = 0
            camFpsWindowStart = now
        }

        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        val isRotated = rotationDegrees % 180 != 0
        val w = imageProxy.width
        val h = imageProxy.height
        val orientedWidth = if (isRotated) h else w
        val orientedHeight = if (isRotated) w else h

        val bitmap = ImageUtils.toBitmap(imageProxy)
        imageProxy.close()
        if (bitmap == null) return

        val camFpsNow = camFps

        // Detect
        if (!detectBusy.getAndSet(true)) {
            val p = detectPredictor
            if (p != null) {
                detectExecutor.execute {
                    try {
                        val t0 = System.nanoTime()
                        val result = p.predict(bitmap, orientedWidth, orientedHeight, rotateForCamera = true)
                        val ms = (System.nanoTime() - t0) / 1_000_000.0
                        val nowMs = System.currentTimeMillis()
                        val fps = if (detectLastResultTime > 0L) 1000.0 / (nowMs - detectLastResultTime) else 0.0
                        detectLastResultTime = nowMs

                        val data = buildStreamData("detect", result, ms, fps, camFpsNow)
                        post {
                            val imgW = result.origShape.width.toFloat()
                            val imgH = result.origShape.height.toFloat()
                            val vW = width.toFloat()
                            val vH = height.toFloat()
                            overlayView.setDetections(result.boxes.take(30).map { box ->
                                DetectionOverlayView.BoxInfo(
                                    rect = aspectFillRect(box.xywhn, imgW, imgH, vW, vH),
                                    label = "${box.cls} ${(box.conf * 100).toInt()}%",
                                    color = Color.rgb(59, 130, 246)
                                )
                            })
                            onMultiTaskStream?.invoke(data)
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Detect inference error", e)
                    } finally {
                        detectBusy.set(false)
                    }
                }
            } else {
                detectBusy.set(false)
            }
        }

        // Segment
        if (!segmentBusy.getAndSet(true)) {
            val p = segmentPredictor
            if (p != null) {
                segmentExecutor.execute {
                    try {
                        val t0 = System.nanoTime()
                        val result = p.predict(bitmap, orientedWidth, orientedHeight, rotateForCamera = true)
                        val ms = (System.nanoTime() - t0) / 1_000_000.0
                        val nowMs = System.currentTimeMillis()
                        val fps = if (segmentLastResultTime > 0L) 1000.0 / (nowMs - segmentLastResultTime) else 0.0
                        segmentLastResultTime = nowMs

                        val data = buildStreamData("segment", result, ms, fps, camFpsNow)
                        post { onMultiTaskStream?.invoke(data) }
                    } catch (e: Exception) {
                        Log.e(TAG, "Segment inference error", e)
                    } finally {
                        segmentBusy.set(false)
                    }
                }
            } else {
                segmentBusy.set(false)
            }
        }

        // Classify
        if (!classifyBusy.getAndSet(true)) {
            val p = classifyPredictor
            if (p != null) {
                classifyExecutor.execute {
                    try {
                        val t0 = System.nanoTime()
                        val result = p.predict(bitmap, orientedWidth, orientedHeight, rotateForCamera = true)
                        val ms = (System.nanoTime() - t0) / 1_000_000.0
                        val nowMs = System.currentTimeMillis()
                        val fps = if (classifyLastResultTime > 0L) 1000.0 / (nowMs - classifyLastResultTime) else 0.0
                        classifyLastResultTime = nowMs

                        val data = buildStreamData("classify", result, ms, fps, camFpsNow)
                        post { onMultiTaskStream?.invoke(data) }
                    } catch (e: Exception) {
                        Log.e(TAG, "Classify inference error", e)
                    } finally {
                        classifyBusy.set(false)
                    }
                }
            } else {
                classifyBusy.set(false)
            }
        }
    }

    // MARK: - Coordinate mapping

    /** Maps a normalized box (0-1) to screen pixels under resizeAspectFill. */
    private fun aspectFillRect(normalized: RectF, imgW: Float, imgH: Float, viewW: Float, viewH: Float): RectF {
        if (imgW <= 0 || imgH <= 0 || viewW <= 0 || viewH <= 0) return RectF()
        val scale = maxOf(viewW / imgW, viewH / imgH)
        val offsetX = (imgW * scale - viewW) / 2f
        val offsetY = (imgH * scale - viewH) / 2f
        return RectF(
            normalized.left * imgW * scale - offsetX,
            normalized.top * imgH * scale - offsetY,
            normalized.right * imgW * scale - offsetX,
            normalized.bottom * imgH * scale - offsetY
        )
    }

    // MARK: - Stream data builder

    private fun buildStreamData(
        task: String,
        result: YOLOResult,
        processingMs: Double,
        fps: Double,
        cameraFps: Double
    ): Map<String, Any> {
        val map = HashMap<String, Any>()
        map["type"] = task
        map["fps"] = fps
        map["cameraFps"] = cameraFps
        map["processingTimeMs"] = processingMs

        when (task) {
            "classify" -> {
                result.probs?.let { probs ->
                    val top5 = ArrayList<Map<String, Any>>()
                    for (i in 0 until minOf(probs.top5Labels.size, probs.top5Confs.size)) {
                        top5.add(mapOf(
                            "name" to probs.top5Labels[i],
                            "confidence" to probs.top5Confs[i].toDouble()
                        ))
                    }
                    map["classification"] = mapOf(
                        "top1" to probs.top1Label,
                        "top1Confidence" to probs.top1Conf.toDouble(),
                        "top5" to top5
                    )
                }
            }
            else -> {
                val detections = ArrayList<Map<String, Any>>()
                for (box in result.boxes.take(50)) {
                    detections.add(mapOf(
                        "className" to box.cls,
                        "confidence" to box.conf.toDouble(),
                        "normalizedBox" to mapOf(
                            "left" to box.xywhn.left.toDouble(),
                            "top" to box.xywhn.top.toDouble(),
                            "right" to box.xywhn.right.toDouble(),
                            "bottom" to box.xywhn.bottom.toDouble()
                        )
                    ))
                }
                map["detections"] = detections
            }
        }
        return map
    }

    // MARK: - Lifecycle

    fun stopCamera() {
        cameraProvider?.unbindAll()
        cameraExecutor?.shutdown()
        cameraExecutor = null
    }

    fun release() {
        stopCamera()
        detectExecutor.shutdown()
        segmentExecutor.shutdown()
        classifyExecutor.shutdown()
        (detectPredictor as? BasePredictor)?.close()
        (segmentPredictor as? BasePredictor)?.close()
        (classifyPredictor as? BasePredictor)?.close()
        detectPredictor = null
        segmentPredictor = null
        classifyPredictor = null
    }
}
