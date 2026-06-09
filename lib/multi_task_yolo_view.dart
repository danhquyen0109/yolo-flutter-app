// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/gestures.dart';
import 'package:flutter/services.dart';
import 'package:flutter/widgets.dart';

import 'core/yolo_model_resolver.dart';

/// Callback fired on the main thread for each inference result from one of the
/// three simultaneous YOLO tasks.
///
/// [data] contains:
/// - `"type"`: `"detect"` | `"segment"` | `"classify"`
/// - `"fps"`: per-task FPS
/// - `"cameraFps"`: raw camera feed FPS
/// - `"processingTimeMs"`: CoreML inference time in ms
/// - `"detections"`: list of detection maps (detect / segment tasks)
/// - `"classification"`: map with `top1`, `top1Confidence`, `top5` (classify task)
typedef MultiTaskStreamCallback = void Function(Map<String, dynamic> data);

/// A Flutter widget that hosts a native iOS view running detection, segmentation,
/// and classification simultaneously on a single camera stream — no JPEG round-trip,
/// no MethodChannel serialisation per frame.
///
/// Android is not yet implemented; on Android the widget renders an empty black box.
class MultiTaskYOLOView extends StatefulWidget {
  const MultiTaskYOLOView({
    super.key,
    required this.detectModelPath,
    required this.segmentModelPath,
    required this.classifyModelPath,
    this.onStreamingData,
    this.lensFacing = 'back',
    this.useGpu = true,
  });

  final String detectModelPath;
  final String segmentModelPath;
  final String classifyModelPath;
  final MultiTaskStreamCallback? onStreamingData;
  final String lensFacing;
  final bool useGpu;

  @override
  State<MultiTaskYOLOView> createState() => _MultiTaskYOLOViewState();
}

class _MultiTaskYOLOViewState extends State<MultiTaskYOLOView> {
  static int _nextId = 0;
  late final String _viewId;
  EventChannel? _eventChannel;
  MethodChannel? _methodChannel;

  // Resolved absolute paths (null = still resolving)
  String? _detectResolved;
  String? _segmentResolved;
  String? _classifyResolved;
  String? _resolutionError;

  @override
  void initState() {
    super.initState();
    _viewId = 'multi_task_${_nextId++}';
    _resolveModels();
  }

  Future<void> _resolveModels() async {
    try {
      final results = await Future.wait([
        YOLOModelResolver.preparePath(widget.detectModelPath),
        YOLOModelResolver.preparePath(widget.segmentModelPath),
        YOLOModelResolver.preparePath(widget.classifyModelPath),
      ]);
      if (!mounted) return;
      setState(() {
        _detectResolved = results[0];
        _segmentResolved = results[1];
        _classifyResolved = results[2];
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _resolutionError = e.toString());
    }
  }

  void _onPlatformViewCreated(int id) {
    _eventChannel = EventChannel(
      'com.ultralytics.yolo/multiTaskResults_$_viewId',
    );
    _methodChannel = MethodChannel(
      'com.ultralytics.yolo/multiTaskControl_$_viewId',
    );

    _eventChannel!.receiveBroadcastStream().listen((event) {
      if (event is Map && widget.onStreamingData != null) {
        widget.onStreamingData!(Map<String, dynamic>.from(event));
      }
    });
  }

  Map<String, dynamic> get _creationParams => {
    'viewId': _viewId,
    'detectModel': _detectResolved!,
    'segmentModel': _segmentResolved!,
    'classifyModel': _classifyResolved!,
    'lensFacing': widget.lensFacing,
    'useGpu': widget.useGpu,
  };

  @override
  Widget build(BuildContext context) {
    if (!Platform.isIOS && !Platform.isAndroid) {
      return const ColoredBox(color: Color(0xFF000000));
    }

    if (_resolutionError != null) {
      return ColoredBox(
        color: const Color(0xFF000000),
        child: Center(
          child: Text(
            'Model error: $_resolutionError',
            style: const TextStyle(color: Color(0xFFFF4444), fontSize: 12),
            textAlign: TextAlign.center,
          ),
        ),
      );
    }

    if (_detectResolved == null ||
        _segmentResolved == null ||
        _classifyResolved == null) {
      // Still resolving (extracting zip from assets → Documents/files dir)
      return const ColoredBox(color: Color(0xFF000000));
    }

    if (Platform.isIOS) {
      return UiKitView(
        viewType: 'com.ultralytics.yolo/YOLOMultiTaskPlatformView',
        onPlatformViewCreated: _onPlatformViewCreated,
        creationParams: _creationParams,
        creationParamsCodec: const StandardMessageCodec(),
        gestureRecognizers: const <Factory<OneSequenceGestureRecognizer>>{},
      );
    }

    // Android
    return AndroidView(
      viewType: 'com.ultralytics.yolo/YOLOMultiTaskPlatformView',
      onPlatformViewCreated: _onPlatformViewCreated,
      creationParams: _creationParams,
      creationParamsCodec: const StandardMessageCodec(),
      gestureRecognizers: const <Factory<OneSequenceGestureRecognizer>>{},
    );
  }

  @override
  void dispose() {
    _methodChannel?.invokeMethod('stop');
    super.dispose();
  }
}
