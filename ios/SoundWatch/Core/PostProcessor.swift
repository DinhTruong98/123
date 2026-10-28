import Foundation

struct AlertEvent: Equatable {
    let classId: Int
    let confidence: Float
    let tAudio: Double
    let repeatAlert: Bool  // DANGER alert re-emitted while waiting for ack
}

/// Port of src/soundwatch/engine/postprocess.py (see its docstring for the rules):
/// moving average over `smooth_n` hops, k-of-n trigger, per-class cooldown, DANGER repeat until ack,
/// acked occurrence silent until below threshold for `danger_clear_s`, disabled/background silent.
/// Rule 4b (optional `danger_fast_threshold`): a DANGER class also counts as triggered at a single hop
/// whose smoothed prob reaches the threshold (and `bg_margin`, if set), without waiting for k-of-n;
/// every other condition of a fresh alert still applies. Compared in Float, like numpy float32.
final class PostProcessor {
    let classes: [SoundClass]
    let cfg: SoundWatchConfig.Postprocess
    let numClasses: Int
    let backgroundId: Int
    private let smoothN: Int
    private let k: Int
    private let n: Int
    private let thr: [Float]
    private let isDanger: [Bool]
    private let cooldown: [Double]
    private var enabled: [Bool]
    private var hist: [[Float]]
    private var hits: [[Bool]]
    private(set) var smoothed: [Float]
    private var nSeen = 0
    private var histPos = 0
    private var hitsPos = 0
    private var lastEmit: [Double] = []
    private var lastAbove: [Double] = []
    private var active: [Bool] = []
    private var pending: [Bool] = []
    private var peak: [Float] = []

    init(classes: [SoundClass], cfg: SoundWatchConfig.Postprocess) {
        precondition(!classes.isEmpty && cfg.kOfN.count == 2)
        self.classes = classes
        self.cfg = cfg
        numClasses = classes.count
        backgroundId = numClasses - 1
        smoothN = max(1, cfg.smoothN)
        k = cfg.kOfN[0]
        n = cfg.kOfN[1]
        thr = classes.map(\.threshold)
        isDanger = classes.map { $0.level == .danger }
        cooldown = classes.map(\.cooldownS)
        enabled = classes.map(\.enabled)
        hist = Array(repeating: Array(repeating: 0, count: numClasses), count: smoothN)
        hits = Array(repeating: Array(repeating: false, count: numClasses), count: n)
        smoothed = Array(repeating: 0, count: numClasses)
        reset()
    }

    func reset() {
        hist = Array(repeating: Array(repeating: 0, count: numClasses), count: smoothN)
        hits = Array(repeating: Array(repeating: false, count: numClasses), count: n)
        smoothed = Array(repeating: 0, count: numClasses)
        nSeen = 0
        histPos = 0
        hitsPos = 0
        lastEmit = Array(repeating: -.infinity, count: numClasses)
        lastAbove = Array(repeating: -.infinity, count: numClasses)
        active = Array(repeating: false, count: numClasses)
        pending = Array(repeating: false, count: numClasses)
        peak = Array(repeating: 0, count: numClasses)
    }

    func ack(_ c: Int) {
        if c >= 0 && c < numClasses { pending[c] = false }
    }

    func setEnabled(_ c: Int, _ on: Bool) {
        guard c >= 0 && c < numClasses else { return }
        enabled[c] = on
        if !on {
            active[c] = false
            pending[c] = false
        }
    }

    func isEnabled(_ c: Int) -> Bool { enabled[c] }
    func isPending(_ c: Int) -> Bool { pending[c] }

    func update(_ probs: [Float], tAudio t: Double) -> [AlertEvent] {
        precondition(probs.count == numClasses)
        hist[histPos] = probs
        histPos = (histPos + 1) % smoothN
        nSeen += 1
        let m = Float(min(nSeen, smoothN))
        for c in 0..<numClasses {
            var s: Float = 0
            for r in 0..<smoothN { s += hist[r][c] }  // unfilled rows are zero
            smoothed[c] = s / m
        }

        var above = (0..<numClasses).map { smoothed[$0] >= thr[$0] }
        if let margin = cfg.bgMargin {
            let bg = smoothed[backgroundId]
            for c in 0..<numClasses where smoothed[c] - bg < Float(margin) { above[c] = false }
        }
        hits[hitsPos] = above
        hitsPos = (hitsPos + 1) % n
        var fastHit = [Bool](repeating: false, count: numClasses)
        if let fast = cfg.dangerFastThreshold {
            let bg = smoothed[backgroundId]
            for c in 0..<numClasses where isDanger[c] && smoothed[c] >= Float(fast) {
                fastHit[c] = cfg.bgMargin.map { smoothed[c] - bg >= Float($0) } ?? true
            }
        }
        for c in 0..<numClasses where above[c] { lastAbove[c] = t }

        var events: [AlertEvent] = []
        for c in 0..<numClasses {
            if c == backgroundId || !enabled[c] { continue }
            let triggered = (0..<n).reduce(0) { $0 + (hits[$1][c] ? 1 : 0) } >= k || fastHit[c]
            let danger = isDanger[c]
            if danger && active[c] && t - lastAbove[c] >= cfg.dangerClearS {
                active[c] = false
                pending[c] = false
            }
            let conf = smoothed[c]
            if danger && active[c] { peak[c] = max(peak[c], conf) }
            let since = t - lastEmit[c]
            if since < cooldown[c] { continue }
            if danger && active[c] {
                if pending[c] && since >= cfg.dangerRepeatS {
                    events.append(AlertEvent(classId: c, confidence: peak[c], tAudio: t, repeatAlert: true))
                    lastEmit[c] = t
                }
                continue
            }
            if triggered {
                events.append(AlertEvent(classId: c, confidence: conf, tAudio: t, repeatAlert: false))
                lastEmit[c] = t
                if danger {
                    active[c] = true
                    pending[c] = true
                    peak[c] = conf
                }
            }
        }
        // level desc, confidence desc (stable, like Python's sort)
        return events.enumerated().sorted { a, b in
            let la = classes[a.element.classId].level, lb = classes[b.element.classId].level
            if la != lb { return la > lb }
            if a.element.confidence != b.element.confidence { return a.element.confidence > b.element.confidence }
            return a.offset < b.offset
        }.map(\.element)
    }
}
