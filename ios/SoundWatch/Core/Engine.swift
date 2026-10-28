import Foundation
import QuartzCore

struct StatusUpdate {
    let tAudio: Double
    let rmsDbfs: Double
    let probs: [Float]
    let gated: Bool
    let featureMs: Double
    let inferMs: Double
}

/// Ring buffer -> energy gate -> log-mel -> int8 model -> PostProcessor, one hop every 0.5 s
/// (port of src/soundwatch/engine/pipeline.py). `tAudio` = end of the 1 s window in stream seconds
/// (1.0, 1.5, 2.0, ...). All work runs on `queue`; callbacks are called on that queue.
final class Engine {
    let config: SoundWatchConfig
    let queue = DispatchQueue(label: "soundwatch.engine", qos: .userInitiated)
    var onStatus: ((StatusUpdate) -> Void)?
    var onAlert: ((AlertEvent) -> Void)?

    private let features: FeatureExtractor
    private let classifier: TFLiteClassifier
    private let post: PostProcessor
    private var rb: RingBuffer
    private var window: [Float]
    private var logmel: [Float] = []
    private var nextHopAt: Int
    private let win: Int
    private let hop: Int
    private let sr: Double

    init(config: SoundWatchConfig, classifier: TFLiteClassifier) {
        precondition(classifier.numClasses == config.numClasses, "model outputs != config classes")
        self.config = config
        self.classifier = classifier
        features = FeatureExtractor(config: config)
        post = PostProcessor(classes: config.classes, cfg: config.postprocess)
        win = config.audio.windowSamples
        hop = config.audio.hopSamples
        sr = Double(config.audio.sampleRate)
        rb = RingBuffer(capacity: win)
        window = [Float](repeating: 0, count: win)
        nextHopAt = win
    }

    // MARK: thread-safe entry points (async on the engine queue)

    func feed(_ samples: [Float]) { queue.async { self.process(samples) } }
    func ack(_ classId: Int) { queue.async { self.post.ack(classId) } }
    func setEnabled(_ classId: Int, _ on: Bool) { queue.async { self.post.setEnabled(classId, on) } }
    /// New stream: t_audio restarts at 0 (e.g. switching between microphone and demo file).
    func reset() { queue.async { self.resetNow() } }

    // MARK: synchronous (call on `queue`, or from tests)

    func resetNow() {
        rb.clear()
        post.reset()
        nextHopAt = win
    }

    func setEnabledNow(_ classId: Int, _ on: Bool) { post.setEnabled(classId, on) }

    /// Feed 16 kHz mono samples; runs every hop whose window is complete.
    func process(_ x: [Float]) {
        var off = 0
        while off < x.count {
            let take = min(x.count - off, nextHopAt - rb.totalWritten)
            x.withUnsafeBufferPointer { rb.write(UnsafeBufferPointer(rebasing: $0[off..<off + take])) }
            off += take
            if rb.totalWritten == nextHopAt {
                processHop(tAudio: Double(nextHopAt) / sr)
                nextHopAt += hop
            }
        }
    }

    /// Whole file as fast as possible (blocks of 1600 samples, like FileSource); returns all events.
    func runOffline(_ samples: [Float], block: Int = 1600) -> (events: [AlertEvent], status: [StatusUpdate]) {
        var events: [AlertEvent] = []
        var status: [StatusUpdate] = []
        let (oa, os) = (onAlert, onStatus)
        onAlert = { events.append($0) }
        onStatus = { status.append($0) }
        defer { onAlert = oa; onStatus = os }
        resetNow()
        var i = 0
        while i < samples.count {
            process(Array(samples[i..<min(i + block, samples.count)]))
            i += block
        }
        return (events, status)
    }

    private func processHop(tAudio: Double) {
        rb.latest(into: &window)
        let level = FeatureExtractor.rmsDbfs(window)
        var probs: [Float]
        var gated = false
        var featureMs = 0.0, inferMs = 0.0
        if level < config.audio.gateDbfs {
            gated = true
            probs = [Float](repeating: 0, count: config.numClasses)
            probs[config.backgroundId] = 1
        } else {
            let t0 = CACurrentMediaTime()
            features.logmel(window, into: &logmel)
            features.normalize(&logmel)
            let t1 = CACurrentMediaTime()
            do {
                probs = try classifier.predict(logmel)
            } catch {
                NSLog("SoundWatch inference failed: \(error)")
                probs = [Float](repeating: 0, count: config.numClasses)
                probs[config.backgroundId] = 1
            }
            let t2 = CACurrentMediaTime()
            featureMs = (t1 - t0) * 1000
            inferMs = (t2 - t1) * 1000
        }
        let events = post.update(probs, tAudio: tAudio)
        onStatus?(StatusUpdate(tAudio: tAudio, rmsDbfs: level, probs: probs, gated: gated,
                               featureMs: featureMs, inferMs: inferMs))
        for e in events { onAlert?(e) }
    }
}
