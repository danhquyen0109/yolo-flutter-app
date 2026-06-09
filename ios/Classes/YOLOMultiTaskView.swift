// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  YOLOMultiTaskView — runs detect, segment, and classify on a single camera stream.
//  Three BasePredictor instances each receive raw CVPixelBuffers directly; no JPEG
//  round-trip or MethodChannel serialisation overhead. Each predictor uses the iOS
//  Neural Engine / GPU concurrently via Apple's CoreML async scheduling.

import AVFoundation
import CoreVideo
import UIKit
import UltralyticsYOLO

// MARK: - Per-predictor result adapter

/// Bridges ResultsListener / InferenceTimeListener callbacks back to a labelled slot
/// in YOLOMultiTaskView. All callbacks are re-dispatched onto `cameraQueue` so the
/// busy-flag bookkeeping stays on one serial queue (no locks needed).
final class MultiTaskPredictorAdapter: ResultsListener, InferenceTimeListener,
  @unchecked Sendable
{
  let taskName: String
  private let cameraQueue: DispatchQueue

  // Set by owner before use
  var onResult: ((YOLOResult) -> Void)?
  var onTime: ((Double, Double) -> Void)?

  init(taskName: String, cameraQueue: DispatchQueue) {
    self.taskName = taskName
    self.cameraQueue = cameraQueue
  }

  func on(result: YOLOResult) {
    let r = result
    cameraQueue.async { [weak self] in self?.onResult?(r) }
  }

  func on(inferenceTime: Double, fpsRate: Double) {
    let ms = inferenceTime
    let fps = fpsRate
    cameraQueue.async { [weak self] in self?.onTime?(ms, fps) }
  }
}

// MARK: - Detection box overlay

/// Lightweight UIView that draws detection bounding boxes directly over the camera preview.
/// Updated on the main thread after each detection result; redraws via setNeedsDisplay.
final class DetectionOverlayView: UIView {

  struct BoxInfo {
    let rect: CGRect
    let label: String
    let color: UIColor
  }

  var boxInfos: [BoxInfo] = [] { didSet { setNeedsDisplay() } }

  override init(frame: CGRect) {
    super.init(frame: frame)
    backgroundColor = .clear
    isUserInteractionEnabled = false
    isOpaque = false
  }

  required init?(coder: NSCoder) { fatalError() }

  override func draw(_ rect: CGRect) {
    guard let ctx = UIGraphicsGetCurrentContext() else { return }
    let fontSize: CGFloat = 11
    let font = UIFont.systemFont(ofSize: fontSize, weight: .semibold)
    let attrs: [NSAttributedString.Key: Any] = [
      .font: font,
      .foregroundColor: UIColor.white,
    ]

    for info in boxInfos {
      // Box stroke
      ctx.setStrokeColor(info.color.cgColor)
      ctx.setLineWidth(2)
      ctx.stroke(info.rect)

      // Label background + text
      let textSize = (info.label as NSString).size(withAttributes: attrs)
      let labelPad: CGFloat = 4
      let labelRect = CGRect(
        x: info.rect.minX,
        y: max(0, info.rect.minY - textSize.height - labelPad * 2),
        width: textSize.width + labelPad * 2,
        height: textSize.height + labelPad
      )
      ctx.setFillColor(info.color.withAlphaComponent(0.75).cgColor)
      ctx.fill(labelRect)
      (info.label as NSString).draw(
        at: CGPoint(x: labelRect.minX + labelPad, y: labelRect.minY + labelPad / 2),
        withAttributes: attrs
      )
    }
  }
}

// MARK: - YOLOMultiTaskView

/// A UIView that hosts a single AVCaptureSession and dispatches each incoming
/// camera frame to three independent YOLO predictors simultaneously.
@MainActor
public class YOLOMultiTaskView: UIView {

  // MARK: Camera

  private let captureSession = AVCaptureSession()
  private var previewLayer: AVCaptureVideoPreviewLayer?

  /// Serial queue for camera delegate callbacks and busy-flag mutations only.
  let cameraQueue = DispatchQueue(label: "yolo.multi-task.camera", qos: .userInteractive)

  /// Per-predictor inference queues. predict() is synchronous (blocks the caller), so each
  /// predictor runs on its own queue so all three run concurrently instead of serially.
  private let detectQueue = DispatchQueue(label: "yolo.infer.detect", qos: .userInteractive)
  private let segmentQueue = DispatchQueue(label: "yolo.infer.segment", qos: .userInteractive)
  private let classifyQueue = DispatchQueue(label: "yolo.infer.classify", qos: .userInteractive)

  // MARK: Predictors

  var detectPredictor: BasePredictor?
  var segmentPredictor: BasePredictor?
  var classifyPredictor: BasePredictor?

  /// One-frame-deep back-pressure per predictor: if the previous frame is still
  /// being processed we skip rather than queue up. Accessed only on cameraQueue.
  var detectBusy = false
  var segmentBusy = false
  var classifyBusy = false

  private lazy var detectAdapter = MultiTaskPredictorAdapter(taskName: "detect", cameraQueue: cameraQueue)
  private lazy var segmentAdapter = MultiTaskPredictorAdapter(taskName: "segment", cameraQueue: cameraQueue)
  private lazy var classifyAdapter = MultiTaskPredictorAdapter(taskName: "classify", cameraQueue: cameraQueue)

  // MARK: Per-task FPS tracking (cameraQueue)

  private var detectLastResultTime: Double = 0
  private var segmentLastResultTime: Double = 0
  private var classifyLastResultTime: Double = 0

  private var detectFps: Double = 0
  private var segmentFps: Double = 0
  private var classifyFps: Double = 0

  // Camera FPS (cameraQueue)
  private var camFrameCount = 0
  private var camFpsWindowStart: Double = 0
  private var camFps: Double = 0

  // MARK: Callback

  /// Called on the main thread with a stream-data dict. Keys: "type" (detect/segment/classify),
  /// "detections", "processingTimeMs", "fps", "cameraFps".
  var onMultiTaskStream: (([String: Any]) -> Void)?

  // MARK: Detection overlay

  private let detectOverlay = DetectionOverlayView()

  // MARK: Loading indicator

  public let activityIndicator = UIActivityIndicatorView(style: .large)
  private var loadedCount = 0
  private var expectedCount = 0

  // MARK: Init

  public override init(frame: CGRect) {
    super.init(frame: frame)
    setupUI()
    wireAdapters()
  }

  required init?(coder: NSCoder) {
    super.init(coder: coder)
    setupUI()
    wireAdapters()
  }

  private func setupUI() {
    backgroundColor = .black
    addSubview(detectOverlay)
    activityIndicator.color = .white
    activityIndicator.startAnimating()
    addSubview(activityIndicator)
  }

  public override func layoutSubviews() {
    super.layoutSubviews()
    previewLayer?.frame = bounds
    detectOverlay.frame = bounds
    activityIndicator.center = CGPoint(x: bounds.midX, y: bounds.midY)
  }

  // MARK: - Adapter wiring

  private func wireAdapters() {
    detectAdapter.onResult = { [weak self] result in self?.handleResult(result, task: "detect") }
    segmentAdapter.onResult = { [weak self] result in self?.handleResult(result, task: "segment") }
    classifyAdapter.onResult = { [weak self] result in self?.handleResult(result, task: "classify") }

    detectAdapter.onTime = { [weak self] _, fps in self?.detectFps = fps }
    segmentAdapter.onTime = { [weak self] _, fps in self?.segmentFps = fps }
    classifyAdapter.onTime = { [weak self] _, fps in self?.classifyFps = fps }
  }

  // MARK: - Result handling (cameraQueue)

  private func handleResult(_ result: YOLOResult, task: String) {
    // Clear busy flag so the next frame can be dispatched.
    switch task {
    case "detect":
      detectBusy = false
      detectPredictor?.isUpdating = false
    case "segment":
      segmentBusy = false
      segmentPredictor?.isUpdating = false
    default:
      classifyBusy = false
      classifyPredictor?.isUpdating = false
    }

    // FPS from result interval
    let now = CACurrentMediaTime()
    var taskFps: Double = 0
    switch task {
    case "detect":
      if detectLastResultTime > 0 {
        let dt = now - detectLastResultTime
        if dt > 0 { detectFps = 1.0 / dt }
      }
      detectLastResultTime = now
      taskFps = detectFps
    case "segment":
      if segmentLastResultTime > 0 {
        let dt = now - segmentLastResultTime
        if dt > 0 { segmentFps = 1.0 / dt }
      }
      segmentLastResultTime = now
      taskFps = segmentFps
    default:
      if classifyLastResultTime > 0 {
        let dt = now - classifyLastResultTime
        if dt > 0 { classifyFps = 1.0 / dt }
      }
      classifyLastResultTime = now
      taskFps = classifyFps
    }

    let camFpsSnapshot = camFps
    let streamData = buildStreamData(result: result, task: task, fps: taskFps, cameraFps: camFpsSnapshot)

    // Snapshot detection data for the native overlay; rect computation deferred to main thread
    // so we have the correct view bounds.
    let rawBoxes: [(xywhn: CGRect, label: String)] = task == "detect"
      ? result.boxes.prefix(30).map { ($0.xywhn, "\($0.cls) \(Int($0.conf * 100))%") }
      : []
    let origShape = result.orig_shape

    DispatchQueue.main.async { [weak self] in
      guard let self else { return }
      if task == "detect" {
        let viewSize = self.bounds.size
        self.detectOverlay.boxInfos = rawBoxes.map { b in
          DetectionOverlayView.BoxInfo(
            rect: self.aspectFillRect(normalized: b.xywhn, imageSize: origShape, viewSize: viewSize),
            label: b.label,
            color: .systemBlue
          )
        }
      }
      self.onMultiTaskStream?(streamData)
    }
  }

  // MARK: - Coordinate mapping

  /// Maps a normalized bounding box (0–1 in image space) to screen points, accounting for
  /// the AVCaptureVideoPreviewLayer's resizeAspectFill scaling — identical to YOLOView's
  /// `aspectFillDisplayRect` helper in the UltralyticsYOLO framework.
  private func aspectFillRect(normalized: CGRect, imageSize: CGSize, viewSize: CGSize) -> CGRect {
    guard imageSize.width > 0, imageSize.height > 0,
      viewSize.width > 0, viewSize.height > 0
    else { return .zero }
    let scale = max(viewSize.width / imageSize.width, viewSize.height / imageSize.height)
    let offsetX = (imageSize.width * scale - viewSize.width) / 2
    let offsetY = (imageSize.height * scale - viewSize.height) / 2
    return CGRect(
      x: normalized.minX * imageSize.width * scale - offsetX,
      y: normalized.minY * imageSize.height * scale - offsetY,
      width: normalized.width * imageSize.width * scale,
      height: normalized.height * imageSize.height * scale
    )
  }

  // MARK: - Stream data builder

  private func buildStreamData(
    result: YOLOResult, task: String, fps: Double, cameraFps: Double
  ) -> [String: Any] {
    var map: [String: Any] = [
      "type": task,
      "fps": fps,
      "cameraFps": cameraFps,
      "processingTimeMs": result.speed * 1000,
    ]

    switch task {
    case "classify":
      if let probs = result.probs {
        var top5: [[String: Any]] = []
        for i in 0..<min(probs.top5Labels.count, probs.top5Confs.count) {
          top5.append(["name": probs.top5Labels[i], "confidence": Double(probs.top5Confs[i])])
        }
        map["classification"] = [
          "top1": probs.top1Label,
          "top1Confidence": Double(probs.top1Conf),
          "top5": top5,
        ]
      }
    default:
      var detections: [[String: Any]] = []
      for box in result.boxes.prefix(50) {
        detections.append([
          "className": box.cls,
          "confidence": Double(box.conf),
          "normalizedBox": [
            "left": Double(box.xywhn.minX),
            "top": Double(box.xywhn.minY),
            "right": Double(box.xywhn.maxX),
            "bottom": Double(box.xywhn.maxY),
          ],
        ])
      }
      map["detections"] = detections
    }

    return map
  }

  // MARK: - Model loading

  /// Load all three models. `completion` fires on the main thread once all three finish
  /// (or fail silently with a log). Camera starts automatically after all models are ready.
  func loadModels(
    detectPath: String,
    segmentPath: String,
    classifyPath: String,
    useGpu: Bool = true,
    cameraPosition: AVCaptureDevice.Position = .back,
    completion: @escaping () -> Void
  ) {
    expectedCount = 3
    loadedCount = 0

    func tryDone() {
      loadedCount += 1
      if loadedCount == expectedCount {
        DispatchQueue.main.async { [weak self] in
          self?.activityIndicator.stopAnimating()
          self?.startCamera(position: cameraPosition)
          completion()
        }
      }
    }

    load(path: detectPath, task: .detect, useGpu: useGpu) { [weak self] p in
      self?.detectPredictor = p
      tryDone()
    }
    load(path: segmentPath, task: .segment, useGpu: useGpu) { [weak self] p in
      self?.segmentPredictor = p
      tryDone()
    }
    load(path: classifyPath, task: .classify, useGpu: useGpu) { [weak self] p in
      self?.classifyPredictor = p
      tryDone()
    }
  }

  private func load(
    path: String, task: YOLOTask, useGpu: Bool,
    completion: @escaping (BasePredictor?) -> Void
  ) {
    guard let url = resolveModelURL(path) else {
      NSLog("YOLOMultiTaskView: model not found: %@", path)
      completion(nil)
      return
    }
    BasePredictor.create(for: task, modelURL: url, isRealTime: true, useGpu: useGpu) { result in
      switch result {
      case .success(let p):
        let bp = p as? BasePredictor
        bp?.capturesOriginalImage = false
        completion(bp)
      case .failure(let err):
        NSLog("YOLOMultiTaskView: load failed for %@: %@", path, err.localizedDescription)
        completion(nil)
      }
    }
  }

  private func resolveModelURL(_ nameOrPath: String) -> URL? {
    let lc = nameOrPath.lowercased()
    if lc.hasSuffix(".mlmodelc") || lc.hasSuffix(".mlpackage") || lc.hasSuffix(".mlmodel") {
      let u = URL(fileURLWithPath: nameOrPath)
      if FileManager.default.fileExists(atPath: u.path) { return u }
    }
    if let u = Bundle.main.url(forResource: nameOrPath, withExtension: "mlmodelc") { return u }
    if let u = Bundle.main.url(forResource: nameOrPath, withExtension: "mlpackage") { return u }
    return nil
  }

  // MARK: - Camera

  private func startCamera(position: AVCaptureDevice.Position) {
    let pos = position
    cameraQueue.async { [weak self] in self?.setupCamera(position: pos) }
  }

  private func setupCamera(position: AVCaptureDevice.Position) {
    captureSession.beginConfiguration()
    captureSession.sessionPreset = .hd1280x720

    guard let device = bestCaptureDevice(position: position),
      let input = try? AVCaptureDeviceInput(device: device),
      captureSession.canAddInput(input)
    else {
      captureSession.commitConfiguration()
      return
    }
    captureSession.addInput(input)

    let output = AVCaptureVideoDataOutput()
    output.videoSettings = [
      kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA)
    ]
    output.alwaysDiscardsLateVideoFrames = true
    output.setSampleBufferDelegate(self, queue: cameraQueue)
    if captureSession.canAddOutput(output) { captureSession.addOutput(output) }

    if let conn = output.connection(with: .video) {
      conn.videoOrientation = .portrait
    }

    captureSession.commitConfiguration()

    DispatchQueue.main.async { [weak self] in
      guard let self else { return }
      let preview = AVCaptureVideoPreviewLayer(session: self.captureSession)
      preview.videoGravity = .resizeAspectFill
      preview.frame = self.bounds
      self.layer.insertSublayer(preview, at: 0)
      self.previewLayer = preview
    }

    captureSession.startRunning()
    camFpsWindowStart = CACurrentMediaTime()
  }

  public func stopCamera() {
    cameraQueue.async { [weak self] in
      self?.captureSession.stopRunning()
    }
  }

  deinit {
    if captureSession.isRunning {
      captureSession.stopRunning()
    }
  }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension YOLOMultiTaskView: AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {

  public func captureOutput(
    _ output: AVCaptureOutput,
    didOutput sampleBuffer: CMSampleBuffer,
    from connection: AVCaptureConnection
  ) {
    // All mutable state accesses in this extension run on cameraQueue.

    // Track camera FPS
    let now = CACurrentMediaTime()
    camFrameCount += 1
    let elapsed = now - camFpsWindowStart
    if elapsed >= 0.5 {
      camFps = Double(camFrameCount) / elapsed
      camFrameCount = 0
      camFpsWindowStart = now
    }

    // Dispatch each predictor to its own queue so all three run concurrently.
    // predict() is synchronous — calling it on cameraQueue would serialise all three
    // and block new frames from arriving. Capture the adapters here (on cameraQueue)
    // so there is no cross-queue access to lazy vars.
    if let p = detectPredictor, !detectBusy, !p.isUpdating {
      detectBusy = true
      p.isUpdating = true
      let buf = sampleBuffer
      let adapter = detectAdapter
      detectQueue.async { p.predict(sampleBuffer: buf, onResultsListener: adapter, onInferenceTime: adapter) }
    }
    if let p = segmentPredictor, !segmentBusy, !p.isUpdating {
      segmentBusy = true
      p.isUpdating = true
      let buf = sampleBuffer
      let adapter = segmentAdapter
      segmentQueue.async { p.predict(sampleBuffer: buf, onResultsListener: adapter, onInferenceTime: adapter) }
    }
    if let p = classifyPredictor, !classifyBusy, !p.isUpdating {
      classifyBusy = true
      p.isUpdating = true
      let buf = sampleBuffer
      let adapter = classifyAdapter
      classifyQueue.async { p.predict(sampleBuffer: buf, onResultsListener: adapter, onInferenceTime: adapter) }
    }
  }
}
