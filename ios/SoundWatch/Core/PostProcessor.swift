import Foundation

struct AlertEvent: Equatable {
    let classId: Int
    let confidence: Float
    let tAudio: Double
    let repeatAlert: Bool  // DANGER alert re-emitted while waiting for ack
    /// Effective level when context modes are configured (e.g. car_horn -> INFO in `street`);
    /// nil = the class's configured level (always nil without modes).
    var level: Level? = nil
}

/// Port of src/soundwatch/engine/postprocess.py (see its docstring for the rules):
/// moving average over `smooth_n` hops, k-of-n trigger, per-class cooldown, DANGER repeat until ack,
/// acked occurrence silent until below threshold for `danger_clear_s`, disabled/background silent.
/// Rule 4b (optional `danger_fast_threshold`): a DANGER class also counts as triggered at a single hop
/// whose smoothed prob reaches the threshold (and `bg_margin`, if set), without waiting for k-of-n;
/// every other condition of a fresh alert still applies, and a fast-path hit also keeps the
/// occurrence alive (clear timer of rule 5). Compared in Float, like numpy float32.
/// Rule 9 (optional `modes`): the active context mode switches classes off and overrides levels;
/// 'auto' follows the running median of the per-hop RMS. Without modes nothing changes.
final class PostProcessor {
    static let autoMode = "auto", autoLowMode = "indoor", autoHighMode = "street"

    let classes: [SoundClass]
    let cfg: SoundWatchConfig.Postprocess
    let modes: SoundWatchConfig.Modes?
    let numClasses: Int
    let backgroundId: Int
    private let smoothN: Int
    private let k: Int
    private let n: Int
    private let thr: [Float]
    private let baseLevel: [Level]
    private(set) var effectiveLevel: [Level]
    private var isDanger: [Bool]
    private let cooldown: [Double]
    private var enabled: [Bool]       // user settings
    private var modeOff: [Bool]       // switched off by the active mode
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
    private let keys: [String]
    private var rms: [(t: Double, v: Double)] = []
    private var autoT0: Double?
    /// Active named mode (nil = no modes configured) and the requested one ('auto' or a named mode).
    private(set) var mode: String?
    private(set) var modeSelector: String?
    private(set) var modeLog: [(t: Double, old: String?, new: String, reason: String)] = []

    init(classes: [SoundClass], cfg: SoundWatchConfig.Postprocess, modes: SoundWatchConfig.Modes? = nil,
         mode: String? = nil) {
        precondition(!classes.isEmpty && cfg.kOfN.count == 2)
        self.classes = classes
        self.cfg = cfg
        self.modes = modes
        numClasses = classes.count
        backgroundId = numClasses - 1
        smoothN = max(1, cfg.smoothN)
        k = cfg.kOfN[0]
        n = cfg.kOfN[1]
        thr = classes.map(\.threshold)
        baseLevel = classes.map(\.level)
        effectiveLevel = baseLevel
        isDanger = baseLevel.map { $0 == .danger }
        cooldown = classes.map(\.cooldownS)
        enabled = classes.map(\.enabled)
        modeOff = Array(repeating: false, count: classes.count)
        keys = classes.map(\.key)
        hist = Array(repeating: Array(repeating: 0, count: numClasses), count: smoothN)
        hits = Array(repeating: Array(repeating: false, count: numClasses), count: n)
        smoothed = Array(repeating: 0, count: numClasses)
        reset()
        if let modes {
            let ok = setMode(mode ?? modes.default, tAudio: 0, reason: "start")
            precondition(ok, "unknown mode \(mode ?? modes.default)")
        } else {
            precondition(mode == nil, "mode requested but no modes are configured")
        }
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
        rms.removeAll()
        autoT0 = nil
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
    /// Enabled by the user and not switched off by the current mode.
    func isActiveClass(_ c: Int) -> Bool { enabled[c] && !modeOff[c] }
    func level(_ c: Int) -> Level { effectiveLevel[c] }
    func isPending(_ c: Int) -> Bool { pending[c] }

    // MARK: modes (rule 9)

    /// Switch to a named mode or to 'auto'. Returns false for an unknown mode (or without modes).
    @discardableResult
    func setMode(_ name: String, tAudio: Double? = nil, reason: String = "manual") -> Bool {
        guard let modes, modes.names.contains(name) else { return false }
        modeSelector = name
        if name == Self.autoMode {
            rms.removeAll()
            autoT0 = nil
            applyMode(mode ?? Self.autoLowMode, tAudio: tAudio, reason: "\(reason): auto")
        } else {
            applyMode(name, tAudio: tAudio, reason: reason)
        }
        return true
    }

    private func applyMode(_ name: String, tAudio: Double?, reason: String) {
        guard let modes else { return }
        let old = mode
        if old == name && !modeLog.isEmpty { return }
        let spec = modes.spec(name)
        var off = Array(repeating: false, count: numClasses)
        for key in spec.disable { if let i = keys.firstIndex(of: key) { off[i] = true } }
        var level = baseLevel
        for (key, v) in spec.level { if let i = keys.firstIndex(of: key), let l = Level(rawValue: v) { level[i] = l } }
        let danger = level.map { $0 == .danger }
        // occurrences of classes that are switched off or stop being DANGER end now
        for c in 0..<numClasses where (off[c] && !modeOff[c]) || (isDanger[c] && !danger[c]) {
            active[c] = false
            pending[c] = false
        }
        modeOff = off
        effectiveLevel = level
        isDanger = danger
        mode = name
        modeLog.append((tAudio ?? .nan, old, name, reason))
    }

    private func autoUpdate(t: Double, rmsDbfs: Double) {
        guard let a = modes?.auto else { return }
        rms.append((t, rmsDbfs))
        while let first = rms.first, first.t <= t - a.windowS { rms.removeFirst() }
        if autoT0 == nil { autoT0 = t }
        if t - autoT0! + 1e-9 < min(a.minS, a.windowS) { return }
        let med = Self.median(rms.map(\.v))
        let h = a.hysteresisDb / 2
        if mode != Self.autoHighMode && med > a.streetAboveDbfs + h {
            applyMode(Self.autoHighMode, tAudio: t, reason: String(format: "auto: median RMS %.1f dBFS", med))
        } else if mode != Self.autoLowMode && med < a.streetAboveDbfs - h {
            applyMode(Self.autoLowMode, tAudio: t, reason: String(format: "auto: median RMS %.1f dBFS", med))
        }
    }

    /// numpy.median: mean of the two middle values for an even count.
    static func median(_ v: [Double]) -> Double {
        let s = v.sorted()
        guard !s.isEmpty else { return .nan }
        return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
    }

    // MARK: update

    /// `rmsDbfs` (level of the analysed window) is only used by the 'auto' mode.
    func update(_ probs: [Float], tAudio t: Double, rmsDbfs: Double? = nil) -> [AlertEvent] {
        precondition(probs.count == numClasses)
        if modeSelector == Self.autoMode, let r = rmsDbfs { autoUpdate(t: t, rmsDbfs: r) }

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
            // a fast-path hit also keeps the occurrence alive (rule 4b)
            for c in 0..<numClasses where fastHit[c] { lastAbove[c] = t }
        }
        for c in 0..<numClasses where above[c] { lastAbove[c] = t }

        var events: [AlertEvent] = []
        for c in 0..<numClasses {
            if c == backgroundId || !enabled[c] || modeOff[c] { continue }
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
            let level: Level? = modes == nil ? nil : effectiveLevel[c]
            if danger && active[c] {
                if pending[c] && since >= cfg.dangerRepeatS {
                    events.append(AlertEvent(classId: c, confidence: peak[c], tAudio: t, repeatAlert: true, level: level))
                    lastEmit[c] = t
                }
                continue
            }
            if triggered {
                events.append(AlertEvent(classId: c, confidence: conf, tAudio: t, repeatAlert: false, level: level))
                lastEmit[c] = t
                if danger {
                    active[c] = true
                    pending[c] = true
                    peak[c] = conf
                }
            }
        }
        // effective level desc, confidence desc (stable, like Python's sort)
        return events.enumerated().sorted { a, b in
            let la = effectiveLevel[a.element.classId], lb = effectiveLevel[b.element.classId]
            if la != lb { return la > lb }
            if a.element.confidence != b.element.confidence { return a.element.confidence > b.element.confidence }
            return a.offset < b.offset
        }.map(\.element)
    }
}
