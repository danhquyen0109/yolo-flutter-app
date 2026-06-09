// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:ultralytics_yolo/models/yolo_result.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';
import 'package:ultralytics_yolo/utils/map_converter.dart';
import 'package:ultralytics_yolo/yolo_performance_metrics.dart';
import 'package:ultralytics_yolo/yolo_streaming_config.dart';

/// Declarative model+task entry used to run multiple YOLO tasks in one [YOLOView].
class YOLOMultiTaskConfig {
  const YOLOMultiTaskConfig({
    required this.modelPath,
    this.task,
    this.useGpu = true,
    this.confidenceThreshold,
    this.iouThreshold,
    this.numItemsThreshold,
    this.streamingConfig,
  });

  final String modelPath;
  final YOLOTask? task;
  final bool useGpu;
  final double? confidenceThreshold;
  final double? iouThreshold;
  final int? numItemsThreshold;
  final YOLOStreamingConfig? streamingConfig;

  Map<String, dynamic> toMap() {
    final map = <String, dynamic>{
      'modelPath': modelPath,
      'useGpu': useGpu,
    };
    if (task != null) {
      map['task'] = task!.name;
    }
    if (confidenceThreshold != null) {
      map['confidenceThreshold'] = confidenceThreshold;
    }
    if (iouThreshold != null) {
      map['iouThreshold'] = iouThreshold;
    }
    if (numItemsThreshold != null) {
      map['numItemsThreshold'] = numItemsThreshold;
    }
    if (streamingConfig != null) {
      map['streamingConfig'] = {
        'includeDetections': streamingConfig!.includeDetections,
        'includeClassifications': streamingConfig!.includeClassifications,
        'includeProcessingTimeMs': streamingConfig!.includeProcessingTimeMs,
        'includeFps': streamingConfig!.includeFps,
        'includeMasks': streamingConfig!.includeMasks,
        'includePoses': streamingConfig!.includePoses,
        'includeOBB': streamingConfig!.includeOBB,
        'includeOriginalImage': streamingConfig!.includeOriginalImage,
        'maxFPS': streamingConfig!.maxFPS,
        'throttleIntervalMs': streamingConfig!.throttleInterval?.inMilliseconds,
        'inferenceFrequency': streamingConfig!.inferenceFrequency,
        'skipFrames': streamingConfig!.skipFrames,
      };
    }
    return map;
  }
}

/// Parsed per-task result entry emitted by multi-task camera inference streams.
class YOLOMultiTaskResult {
  const YOLOMultiTaskResult({
    required this.taskKey,
    required this.raw,
    this.task,
    this.modelPath,
    this.detections = const <YOLOResult>[],
    this.performance,
  });

  final String taskKey;
  final YOLOTask? task;
  final String? modelPath;
  final List<YOLOResult> detections;
  final YOLOPerformanceMetrics? performance;
  final Map<String, dynamic> raw;

  factory YOLOMultiTaskResult.fromMap(
    String taskKey,
    Map<dynamic, dynamic> raw,
  ) {
    final typed = MapConverter.convertToTypedMap(raw);

    final task = YOLOTaskParsing.tryParse(typed['task'] as String?);
    final modelPath = typed['modelPath'] as String?;

    final detectionsRaw = typed['detections'] as List<dynamic>? ?? const [];
    final detections = <YOLOResult>[];
    for (final detection in detectionsRaw) {
      if (detection is! Map) continue;
      if (detection['classIndex'] == null ||
          detection['className'] == null ||
          detection['confidence'] == null ||
          detection['boundingBox'] == null ||
          detection['normalizedBox'] == null) {
        continue;
      }
      detections.add(YOLOResult.fromMap(detection));
    }

    YOLOPerformanceMetrics? metrics;
    final perfRaw = typed['performance'];
    if (perfRaw is Map) {
      metrics = YOLOPerformanceMetrics.fromMap(
        MapConverter.convertToTypedMap(perfRaw),
      );
    } else if (typed.containsKey('fps') ||
        typed.containsKey('processingTimeMs') ||
        typed.containsKey('frameNumber')) {
      metrics = YOLOPerformanceMetrics.fromMap(typed);
    }

    return YOLOMultiTaskResult(
      taskKey: taskKey,
      task: task,
      modelPath: modelPath,
      detections: detections,
      performance: metrics,
      raw: typed,
    );
  }
}
