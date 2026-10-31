import Foundation
import QuartzCore

/// One model: its own feature stats, classifier and PostProcessor (config, thresholds, modes).
final class ModelPipeline {
    let name: String
    let config: SoundWatchConfig
    let classifier: TFLiteClassifier
    let post: PostProcessor
    private let mean: Float
    private let invStd: Float
    private var feats: [Float] = []

    init(name: String, config: SoundWatchConfig, classifier: TFLiteClassifier) {
        precondition(classifier.numClasses == config.numClasses, "model outputs != config classes")
        self.name = name
        self.config = config
        self.classifier = classifier
        post = PostProcessor(classes: config.classes, cfg: config.postprocess, modes: config.modes)
        mean = Float(config.featureStats.mean)
        invStd = Float(1.0 / max(config.featureStats.std, 1e-12))
    }

    /// Normalize the shared log-mel with this model's stats and run the classifier.
    func classify(_ logmel: [Float]) -> [Float] {
        feats = logmel
        FeatureExtractor.normalize(&feats, mean: mean, invStd: invStd)
        do {
            return try classifier.predict(feats)
        } catch {
            NSLog("SoundWatch inference failed (\(name)): \(error)")
            return backgroundOneHot
        }
    }

    var backgroundOneHot: [Float] {
        var p = [Float](repeating: 0, count: config.numClasses)
        p[config.backgroundId] = 1
        return p
    }
}

struct ModelStatus {
    let probs: [Float]
    let inferMs: Double
    let mode: String?  // active context mode (nil = no modes)
}

struct StatusUpdate {
    let tAudio: Double
    let rmsDbfs: Double
    let gated: Bool
    let featureMs: Double
    let models: [ModelStatus]
    /// First model (A), as before the dual-model engine.
    var probs: [Float] { models[0].probs }
    var inferMs: Double { models[0].inferMs }
}

/// Ring buffer -> energy gate -> log-mel (once per hop) -> per model: normalize, int8 model,
/// PostProcessor (port of src/soundwatch/engine/pipeline.py). One hop every 0.5 s; `tAudio` = end of
/// the 1 s window in stream seconds (1.0, 1.5, 2.0, ...). All work runs on `queue`; callbacks are
/// called on that queue. The models must share the front-end (audio params, window, filterbank).
final class Engine {
    let config: SoundWatchConfig  // front-end + gate (first model)
    let pipelines: [ModelPipeline]
    let queue = DispatchQueue(label: "soundwatch.engine", qos: .userInitiated)
    var onStatus: ((StatusUpdate) -> Void)?
    /// (model index, event)
    var onAlert: ((Int, AlertEvent) -> Void)?

    private let features: FeatureExtractor
    private var rb: RingBuffer
    private var window: [Float]
    private var logmel: [Float] = []
    private var nextHopAt: Int
    private let win: Int
    private let hop: Int
    private let sr: Double

    convenience init(config: SoundWatchConfig, classifier: TFLiteClassifier) {
        self.init(pipelines: [ModelPipeline(name: "A", config: config, classifier: classifier)])
    }

    init(pipelines: [ModelPipeline]) {
        precondition(!pipelines.isEmpty)
        let c = pipelines[0].config
        for p in pipelines.dropFirst() {
            let a = p.config.audio
            precondition(a.sampleRate == c.audio.sampleRate && a.windowSamples == c.audio.windowSamples
                         && a.hopSamples == c.audio.hopSamples && a.nFft == c.audio.nFft
                         && a.frameStep == c.audio.frameStep && a.nMels == c.audio.nMels
                         && a.gateDbfs == c.audio.gateDbfs && p.config.hann == c.hann
                         && p.config.melBands.map(\.weights) == c.melBands.map(\.weights),
                         "models must share the front-end")
        }
        self.pipelines = pipelines
        config = c
        features = FeatureExtractor(config: c)
        win = c.audio.windowSamples
        hop = c.audio.hopSamples
        sr = Double(c.audio.sampleRate)
        rb = RingBuffer(capacity: win)
        window = [Float](repeating: 0, count: win)
        nextHopAt = win
    }

    // MARK: thread-safe entry points (async on the engine queue)

    func feed(_ samples: [Float]) { queue.async { self.process(samples) } }
    func ack(_ classId: Int, model: Int? = nil) {
        queue.async { for (i, p) in self.pipelines.enumerated() where model == nil || model == i { p.post.ack(classId) } }
    }
    func setEnabled(_ classId: Int, _ on: Bool) {
        queue.async { self.pipelines.forEach { $0.post.setEnabled(classId, on) } }
    }
    func setMode(_ name: String, model: Int) {
        queue.async {
            guard model < self.pipelines.count else { return }
            self.pipelines[model].post.setMode(name, tAudio: Double(self.rb.totalWritten) / self.sr, reason: "command")
        }
    }
    /// New stream: t_audio restarts at 0 (e.g. switching between microphone and demo file).
    func reset() { queue.async { self.resetNow() } }

    // MARK: synchronous (call on `queue`, or from tests)

    func resetNow() {
        rb.clear()
        pipelines.forEach { $0.post.reset() }
        nextHopAt = win
    }

    func setEnabledNow(_ classId: Int, _ on: Bool) { pipelines.forEach { $0.post.setEnabled(classId, on) } }

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

    /// Whole file as fast as possible (blocks of 1600 samples, like FileSource); events of model A.
    func runOffline(_ samples: [Float], block: Int = 1600) -> (events: [AlertEvent], status: [StatusUpdate]) {
        let r = runOfflineAll(samples, block: block)
        return (r.events[0], r.status)
    }

    /// Same, with the events of every model.
    func runOfflineAll(_ samples: [Float], block: Int = 1600) -> (events: [[AlertEvent]], status: [StatusUpdate]) {
        var events = Array(repeating: [AlertEvent](), count: pipelines.count)
        var status: [StatusUpdate] = []
        let (oa, os) = (onAlert, onStatus)
        onAlert = { events[$0].append($1) }
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
        let gated = level < config.audio.gateDbfs
        var featureMs = 0.0
        if !gated {
            let t0 = CACurrentMediaTime()
            features.logmel(window, into: &logmel)
            featureMs = (CACurrentMediaTime() - t0) * 1000
        }
        var models: [ModelStatus] = []
        var allEvents: [(Int, AlertEvent)] = []
        for (i, p) in pipelines.enumerated() {
            var probs = p.backgroundOneHot
            var inferMs = 0.0
            if !gated {
                let t1 = CACurrentMediaTime()
                probs = p.classify(logmel)
                inferMs = (CACurrentMediaTime() - t1) * 1000
            }
            let events = p.post.update(probs, tAudio: tAudio, rmsDbfs: level)
            models.append(ModelStatus(probs: probs, inferMs: inferMs, mode: p.post.mode))
            allEvents += events.map { (i, $0) }
        }
        onStatus?(StatusUpdate(tAudio: tAudio, rmsDbfs: level, gated: gated, featureMs: featureMs, models: models))
        for (i, e) in allEvents { onAlert?(i, e) }
    }
}
