// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.app.Activity
import android.content.Context
import android.util.Log
import android.view.View
import androidx.camera.core.CameraSelector
import androidx.lifecycle.LifecycleOwner
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.StandardMessageCodec
import io.flutter.plugin.platform.PlatformView
import io.flutter.plugin.platform.PlatformViewFactory

// MARK: - Platform view

class YOLOMultiTaskPlatformView(
    private val context: Context,
    private val viewId: Int,
    creationParams: Map<String?, Any?>?,
    messenger: BinaryMessenger
) : PlatformView, MethodChannel.MethodCallHandler {

    private val TAG = "YOLOMultiTaskPlatformView"
    private val multiTaskView = YOLOMultiTaskView(context)

    private val viewUniqueId: String = (creationParams?.get("viewId") as? String) ?: viewId.toString()

    private val eventChannel = EventChannel(messenger, "com.ultralytics.yolo/multiTaskResults_$viewUniqueId")
    private val methodChannel = MethodChannel(messenger, "com.ultralytics.yolo/multiTaskControl_$viewUniqueId")

    private var eventSink: EventChannel.EventSink? = null

    init {
        eventChannel.setStreamHandler(object : EventChannel.StreamHandler {
            override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
                eventSink = events
            }
            override fun onCancel(arguments: Any?) {
                eventSink = null
            }
        })

        methodChannel.setMethodCallHandler(this)

        multiTaskView.onMultiTaskStream = { data ->
            val sink = eventSink ?: return@onMultiTaskStream
            if (android.os.Looper.myLooper() == android.os.Looper.getMainLooper()) {
                sink.success(data)
            } else {
                android.os.Handler(android.os.Looper.getMainLooper()).post { sink.success(data) }
            }
        }

        // Attach lifecycle
        if (context is LifecycleOwner) {
            multiTaskView.onLifecycleOwnerAvailable(context)
        } else if (context is Activity) {
            // Wrap the Activity in a simple LifecycleOwner if needed; in practice Flutter Activity is a LifecycleOwner
            Log.w(TAG, "Context is Activity but not LifecycleOwner — camera may not start")
        }

        // Load models and start camera
        val detectPath = creationParams?.get("detectModel") as? String ?: return
        val segmentPath = creationParams?.get("segmentModel") as? String ?: return
        val classifyPath = creationParams?.get("classifyModel") as? String ?: return
        val useGpu = creationParams?.get("useGpu") as? Boolean ?: true
        val lensFacingStr = creationParams?.get("lensFacing") as? String ?: "back"
        val lensFacing = if (lensFacingStr == "front") CameraSelector.LENS_FACING_FRONT else CameraSelector.LENS_FACING_BACK

        multiTaskView.loadModels(
            detectPath = detectPath,
            segmentPath = segmentPath,
            classifyPath = classifyPath,
            useGpu = useGpu,
            lensFacing = lensFacing
        )
    }

    override fun getView(): View = multiTaskView

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "stop" -> {
                multiTaskView.stopCamera()
                result.success(null)
            }
            else -> result.notImplemented()
        }
    }

    override fun dispose() {
        eventSink = null
        eventChannel.setStreamHandler(null)
        methodChannel.setMethodCallHandler(null)
        multiTaskView.release()
    }
}

// MARK: - Factory

class YOLOMultiTaskPlatformViewFactory(
    private val messenger: BinaryMessenger
) : PlatformViewFactory(StandardMessageCodec.INSTANCE) {

    private var activity: Activity? = null

    fun setActivity(activity: Activity?) {
        this.activity = activity
    }

    override fun create(context: Context, viewId: Int, args: Any?): PlatformView {
        val creationParams = args as? Map<String?, Any?>
        val effectiveContext = activity ?: context
        return YOLOMultiTaskPlatformView(effectiveContext, viewId, creationParams, messenger)
    }
}
