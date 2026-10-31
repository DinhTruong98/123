import XCTest
@testable import SoundWatch

/// Context modes (postprocess.py rule 9) and the rule 4b fix, ported from
/// tests/test_v2_frontend_modes.py (MODES and helpers as there).
final class PostProcessorModesTests: XCTestCase {
    static let modesJSON = """
    {"default": "indoor", "modes": {"street": {"disable": ["dog_bark", "baby_cry"], "level": {"car_horn": 0}},
     "indoor": {}}, "auto": {"street_above_dbfs": -35, "window_s": 10, "hysteresis_db": 0, "min_s": 2}}
    """

    var cfg: SoundWatchConfig!

    override func setUpWithError() throws {
        cfg = try TestSupport.config()  // deployed model A config (no modes)
    }

    func modes(default d: String = "indoor") throws -> SoundWatchConfig.Modes {
        let s = Self.modesJSON.replacingOccurrences(of: "\"default\": \"indoor\"", with: "\"default\": \"\(d)\"")
        return try JSONDecoder.snake.decode(SoundWatchConfig.Modes.self, from: Data(s.utf8))
    }

    func id(_ key: String) -> Int { cfg.classId(key)! }

    func p(_ key: String, _ v: Float = 1) -> [Float] {
        var x = [Float](repeating: 0, count: cfg.numClasses)
        x[id(key)] = v
        x[cfg.backgroundId] = 1 - v
        return x
    }

    func run(_ pp: PostProcessor, _ seq: [[Float]], t0: Double = 1, rms: [Double]? = nil) -> [AlertEvent] {
        var out: [AlertEvent] = []
        for (i, x) in seq.enumerated() { out += pp.update(x, tAudio: t0 + 0.5 * Double(i), rmsDbfs: rms?[i]) }
        return out
    }

    /// Peaky random probability vectors (like numpy dirichlet(0.1)).
    func randomSeq(_ n: Int, seed: UInt64) -> [[Float]] {
        var rng = SplitMix64(seed: seed)
        return (0..<n).map { _ in
            let v = (0..<cfg.numClasses).map { _ -> Double in
                let u = Double(rng.next() % 1_000_000 + 1) / 1_000_001
                return pow(-log(u), 10)
            }
            let s = v.reduce(0, +)
            return v.map { Float($0 / s) }
        }
    }

    func testNoModesIdenticalEvents() {
        let seq = randomSeq(400, seed: 0)
        let a = run(PostProcessor(classes: cfg.classes, cfg: cfg.postprocess), seq)
        let b = run(PostProcessor(classes: cfg.classes, cfg: cfg.postprocess, modes: nil), seq,
                    rms: Array(repeating: -20, count: 400))
        XCTAssertEqual(a, b)
        XCTAssertFalse(a.isEmpty)
        XCTAssertTrue(a.allSatisfy { $0.level == nil })
    }

    func testIndoorModeIdentical() throws {
        let seq = randomSeq(400, seed: 1)
        let a = run(PostProcessor(classes: cfg.classes, cfg: cfg.postprocess), seq)
        let b = run(PostProcessor(classes: cfg.classes, cfg: cfg.postprocess, modes: try modes()), seq)
        XCTAssertEqual(a, b.map { AlertEvent(classId: $0.classId, confidence: $0.confidence, tAudio: $0.tAudio, repeatAlert: $0.repeatAlert) })
        XCTAssertTrue(b.allSatisfy { $0.level == cfg.classes[$0.classId].level })
    }

    func testStreetModeDisablesAndDowngrades() throws {
        let m = try modes()
        var pp = PostProcessor(classes: cfg.classes, cfg: cfg.postprocess, modes: m, mode: "street")
        let dog = id("dog_bark"), horn = id("car_horn")
        XCTAssertEqual(pp.mode, "street")
        XCTAssertFalse(pp.isActiveClass(dog))
        XCTAssertTrue(pp.isEnabled(dog))
        XCTAssertEqual(pp.level(horn), .info)
        XCTAssertTrue(run(pp, Array(repeating: p("dog_bark"), count: 6)).isEmpty)
        // car_horn as INFO: no fast path on a single saturated hop, no repeats
        pp = PostProcessor(classes: cfg.classes, cfg: cfg.postprocess, modes: m, mode: "street")
        let pulses = Array(repeating: [p("car_horn"), p("background")], count: 3).flatMap { $0 }
        XCTAssertTrue(run(pp, pulses).isEmpty)
        let ev = run(pp, Array(repeating: p("car_horn"), count: 30), t0: 10)
        XCTAssertFalse(ev.isEmpty)
        XCTAssertFalse(ev.contains { $0.repeatAlert })
        // indoor: same input alerts at once (fast path) and repeats
        pp = PostProcessor(classes: cfg.classes, cfg: cfg.postprocess, modes: m, mode: "indoor")
        let ev2 = run(pp, [p("car_horn")] + Array(repeating: p("background"), count: 20))
        XCTAssertEqual(ev2.first?.classId, horn)
        XCTAssertTrue(ev2.contains { $0.repeatAlert })
    }

    func testModeSwitchClearsOccurrence() throws {
        let horn = id("car_horn")
        let pp = PostProcessor(classes: cfg.classes, cfg: cfg.postprocess, modes: try modes())
        _ = run(pp, Array(repeating: p("car_horn"), count: 2))
        XCTAssertTrue(pp.isPending(horn))
        XCTAssertTrue(pp.setMode("street", tAudio: 2))
        XCTAssertFalse(pp.isPending(horn))
        XCTAssertEqual(pp.modeLog.last?.old, "indoor")
        XCTAssertEqual(pp.modeLog.last?.new, "street")
        XCTAssertEqual(pp.modeLog.last?.reason, "manual")
        XCTAssertFalse(pp.setMode("beach"))
    }

    func testAutoModeFollowsRunningMedian() throws {
        let pp = PostProcessor(classes: cfg.classes, cfg: cfg.postprocess, modes: try modes(default: "auto"))
        XCTAssertEqual(pp.modeSelector, "auto")
        XCTAssertEqual(pp.mode, "indoor")
        let bg = p("background")
        let n = 60
        let rms = Array(repeating: -50.0, count: n) + Array(repeating: -25.0, count: n) + Array(repeating: -50.0, count: n)
        _ = run(pp, Array(repeating: bg, count: 3 * n), rms: rms)
        let switches = pp.modeLog.filter { $0.reason.hasPrefix("auto:") }
        XCTAssertEqual(switches.map(\.new), ["street", "indoor"])
        // median over 10 s: flips about 5 s after each level change
        XCTAssertTrue(switches[0].t > 34 && switches[0].t < 37, "\(switches[0].t)")
        XCTAssertTrue(switches[1].t > 64 && switches[1].t < 67, "\(switches[1].t)")
        // a short loud burst (< half the window) does not switch
        let pp2 = PostProcessor(classes: cfg.classes, cfg: cfg.postprocess, modes: try modes(default: "auto"))
        _ = run(pp2, Array(repeating: bg, count: 60),
                rms: Array(repeating: -50.0, count: 20) + Array(repeating: -20.0, count: 8) + Array(repeating: -50.0, count: 32))
        XCTAssertEqual(pp2.mode, "indoor")
    }

    func testFastPathBelowThresholdKeepsOccurrence() {
        // danger_fast_threshold < threshold: the fast-path alert starts an occurrence kept alive by
        // further fast hits (no second fresh alert after the cooldown, repeats instead)
        let siren = id("siren")
        var classes = cfg.classes
        classes[siren].threshold = 0.99609375
        var pcfg = cfg.postprocess
        pcfg.dangerFastThreshold = 0.95
        let pp = PostProcessor(classes: classes, cfg: pcfg)
        let seq = [p("siren", 0.96)] + Array(repeating: p("siren", 0.99609375), count: 6) + [p("siren", 0.6)]
        let fresh = run(pp, seq).filter { !$0.repeatAlert && $0.classId == siren }
        XCTAssertEqual(fresh.count, 1)
        XCTAssertTrue(pp.isPending(siren))
    }

    func testFastPathBelowThresholdSequenceOfFastHitsOnly() {
        // only fast hits (0.96 < threshold): one occurrence, repeats, no fresh re-alert after cooldown_s
        let siren = id("siren")
        var classes = cfg.classes
        classes[siren].threshold = 0.99609375
        var pcfg = cfg.postprocess
        pcfg.dangerFastThreshold = 0.95
        let ev = run(PostProcessor(classes: classes, cfg: pcfg), Array(repeating: p("siren", 0.96), count: 30))
        XCTAssertEqual(ev.filter { !$0.repeatAlert }.count, 1)
        XCTAssertFalse(ev.filter(\.repeatAlert).isEmpty)
    }

    func testAlertEventLevelWithModes() throws {
        let ev = run(PostProcessor(classes: cfg.classes, cfg: cfg.postprocess, modes: try modes(), mode: "street"),
                     Array(repeating: p("car_horn"), count: 3))
        XCTAssertFalse(ev.isEmpty)
        XCTAssertTrue(ev.allSatisfy { $0.level == .info })
        let ev0 = run(PostProcessor(classes: cfg.classes, cfg: cfg.postprocess), Array(repeating: p("car_horn"), count: 3))
        XCTAssertFalse(ev0.isEmpty)
        XCTAssertTrue(ev0.allSatisfy { $0.level == nil })
    }
}
