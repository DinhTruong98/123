import XCTest
@testable import SoundWatch

/// Port of tests/test_wp5_postprocess.py. Like the Python tests, these pin the WP5/WP7 settings
/// from tests/_legacy_cfg.py (smooth_n 3, k_of_n 2/3, WP7 thresholds) instead of the tuned config.
final class PostProcessorTests: XCTestCase {
    static let hop = 0.5
    static let wp7Thresholds: [String: Float] = [
        "smoke_alarm": 0.60, "car_horn": 0.60, "siren": 0.60, "baby_cry": 0.70,
        "door_knock": 0.70, "dog_bark": 0.75, "water_running": 0.75,
    ]

    var cfg: SoundWatchConfig!

    override func setUpWithError() throws {
        var c = try TestSupport.config()
        c.postprocess.smoothN = 3
        c.postprocess.kOfN = [2, 3]
        c.postprocess.bgMargin = nil
        c.postprocess.dangerFastThreshold = nil  // like tests/_legacy_cfg.py
        for i in c.classes.indices { if let t = Self.wp7Thresholds[c.classes[i].key] { c.classes[i].threshold = t } }
        cfg = c
    }

    func id(_ key: String) -> Int { cfg.classId(key)! }
    func make(_ classes: [SoundClass]? = nil) -> PostProcessor {
        PostProcessor(classes: classes ?? cfg.classes, cfg: cfg.postprocess)
    }

    func probs(_ cid: Int, _ p: Float) -> [Float] {
        let n = cfg.numClasses
        var v = [Float](repeating: (1 - p) / Float(n - 1), count: n)
        v[cid] = p
        return v
    }

    func run(_ pp: PostProcessor, _ seq: [[Float]], t0: Double = 1.0) -> [(Double, AlertEvent)] {
        var out: [(Double, AlertEvent)] = []
        for (i, p) in seq.enumerated() {
            let t = t0 + Double(i) * Self.hop
            out += pp.update(p, tAudio: t).map { (t, $0) }
        }
        return out
    }

    func rep(_ v: [Float], _ n: Int) -> [[Float]] { Array(repeating: v, count: n) }

    func testKOfNAndSmoothing() {
        let dog = id("dog_bark")
        let pp = make()
        let bg = probs(cfg.backgroundId, 1), hi = probs(dog, 1)
        XCTAssertTrue(run(pp, rep(bg, 3) + [hi] + rep(bg, 5)).isEmpty)  // isolated spike
        pp.reset()
        let ev = run(pp, rep(bg, 3) + rep(hi, 6))
        XCTAssertEqual(ev.count, 1)
        XCTAssertEqual(ev[0].0, 1.0 + 6 * Self.hop)
        XCTAssertEqual(ev[0].1.classId, dog)
        XCTAssertFalse(ev[0].1.repeatAlert)
        XCTAssertEqual(ev[0].1.confidence, 1, accuracy: 1e-6)
    }

    func testCooldown() {
        let dog = id("dog_bark")
        let ev = run(make(), rep(probs(dog, 1), 70))  // 35 s continuous
        XCTAssertEqual(ev.count, 3)
        for i in 1..<ev.count { XCTAssertEqual(ev[i].0 - ev[i - 1].0, 15, accuracy: 1e-9) }  // cooldown_s
        XCTAssertTrue(ev.allSatisfy { !$0.1.repeatAlert })
    }

    func testDangerRepeatAckAndClear() {
        let fire = id("smoke_alarm")
        let bg = probs(cfg.backgroundId, 1), hi = probs(fire, 1)
        let repeatS = cfg.postprocess.dangerRepeatS, clearS = cfg.postprocess.dangerClearS

        // continuous alarm, never acked: fresh then repeat every danger_repeat_s
        var pp = make()
        var ev = run(pp, rep(hi, 40))
        XCTAssertEqual(ev.map(\.1.repeatAlert), [false, true, true, true])
        for i in 1..<ev.count { XCTAssertEqual(ev[i].0 - ev[i - 1].0, repeatS, accuracy: 1e-9) }

        // ack after the first alert: no repeats, no fresh alert while the alarm continues
        pp = make()
        XCTAssertEqual(run(pp, rep(hi, 3)).count, 1)
        XCTAssertTrue(pp.isPending(fire))
        pp.ack(fire)
        XCTAssertFalse(pp.isPending(fire))
        XCTAssertTrue(run(pp, rep(hi, 30), t0: 2.5).isEmpty)

        // repeats continue after the sound stops, until below threshold for danger_clear_s
        pp = make()
        ev = run(pp, rep(hi, 3) + rep(bg, 40))
        let ts = ev.map(\.0)
        XCTAssertTrue(ev.dropFirst().allSatisfy { $0.1.repeatAlert })
        XCTAssertTrue(ts.allSatisfy { $0 < 2.5 + clearS })
        XCTAssertEqual(ts, [1.5, 6.5, 11.5])
        XCTAssertFalse(pp.isPending(fire))

        // after the occurrence cleared, a new occurrence alerts again (fresh)
        let ev2 = run(pp, rep(hi, 3), t0: 30)
        XCTAssertEqual(ev2.count, 1)
        XCTAssertFalse(ev2[0].1.repeatAlert)
    }

    func testAckThenNewOccurrenceAfterClear() {
        let fire = id("smoke_alarm")
        let pp = make()
        let bg = probs(cfg.backgroundId, 1), hi = probs(fire, 1)
        _ = run(pp, rep(hi, 3))
        pp.ack(fire)
        let quiet = Int(cfg.postprocess.dangerClearS / Self.hop) + 2
        XCTAssertTrue(run(pp, rep(bg, quiet), t0: 2.5).isEmpty)
        let ev = run(pp, rep(hi, 3), t0: 2.5 + Double(quiet) * Self.hop)
        XCTAssertEqual(ev.count, 1)
        XCTAssertFalse(ev[0].1.repeatAlert)
    }

    func testDisabledAndBackgroundSilent() {
        let fire = id("smoke_alarm")
        let pp = make()
        pp.setEnabled(fire, false)
        XCTAssertTrue(run(pp, rep(probs(fire, 1), 20)).isEmpty)
        XCTAssertTrue(run(pp, rep(probs(cfg.backgroundId, 1), 20), t0: 20).isEmpty)
        // disabled in config
        var classes = cfg.classes
        classes[fire].enabled = false
        XCTAssertTrue(run(make(classes), rep(probs(fire, 1), 20)).isEmpty)
        // disabling while pending stops repeats
        let pp3 = make()
        XCTAssertEqual(run(pp3, rep(probs(fire, 1), 3)).count, 1)
        pp3.setEnabled(fire, false)
        XCTAssertTrue(run(pp3, rep(probs(fire, 1), 20), t0: 2.5).isEmpty)
    }

    func testMultipleClassesSameHopOrdered() {
        let fire = id("smoke_alarm"), dog = id("dog_bark"), horn = id("car_horn")
        var p = [Float](repeating: 0, count: cfg.numClasses)
        p[dog] = 0.9; p[fire] = 0.62; p[horn] = 0.8
        XCTAssertEqual(run(make(), rep(p, 2)).map(\.1.classId), [horn, fire, dog])
    }

    func testRepeatConfidenceIsOccurrencePeak() {
        let fire = id("smoke_alarm")
        let bg = probs(cfg.backgroundId, 1)
        let ev = run(make(), rep(probs(fire, 0.7), 2) + rep(probs(fire, 0.95), 4) + rep(bg, 10))
        XCTAssertEqual(ev.map(\.1.repeatAlert), [false, true])
        XCTAssertEqual(ev[0].1.confidence, 0.7, accuracy: 1e-6)
        XCTAssertEqual(ev[1].1.confidence, 0.95, accuracy: 1e-6)
    }

    func testBgMarginOption() {
        // optional rule (off by default): above only if smoothed[c] - smoothed[bg] >= bg_margin
        var pcfg = cfg.postprocess
        pcfg.bgMargin = 0.2
        let dog = id("dog_bark")
        var p = [Float](repeating: 0, count: cfg.numClasses)
        p[dog] = 0.8; p[cfg.backgroundId] = 0.7
        let pp = PostProcessor(classes: cfg.classes, cfg: pcfg)
        XCTAssertTrue(run(pp, rep(p, 6)).isEmpty)
        p[cfg.backgroundId] = 0.5
        XCTAssertEqual(run(PostProcessor(classes: cfg.classes, cfg: pcfg), rep(p, 6)).count, 1)
    }

    // MARK: DANGER fast path (rule 4b), port of test_fast_path_* in tests/test_wp5_postprocess.py

    static let sat: Float = 255.0 / 256.0  // saturated int8 output (scale 1/256, zero point -128)

    /// Tuned config (smooth_n 1, k_of_n 2/2, thresholds from JSON) with the fast path set explicitly.
    func deployed(_ fast: Double?, bgMargin: Double? = nil) throws -> SoundWatchConfig {
        var c = try TestSupport.config()
        c.postprocess.smoothN = 1
        c.postprocess.kOfN = [2, 2]
        c.postprocess.bgMargin = bgMargin
        c.postprocess.dangerFastThreshold = fast
        return c
    }

    func sat(_ c: SoundWatchConfig, _ cid: Int, _ p: Float = sat) -> [Float] {
        var v = [Float](repeating: 0, count: c.numClasses)
        v[cid] = p
        v[c.backgroundId] = 1 - p
        return v
    }

    func pp(_ c: SoundWatchConfig) -> PostProcessor { PostProcessor(classes: c.classes, cfg: c.postprocess) }

    func testFastPathSingleSaturatedHopAlertsAtOnce() throws {
        let off = try deployed(nil), on = try deployed(Double(Self.sat))
        let siren = on.classId("siren")!
        let bg = sat(on, on.backgroundId, 1)
        let seq = rep(bg, 3) + [sat(on, siren)] + rep(bg, 6)  // ambulance passing: one saturated hop
        XCTAssertTrue(run(pp(off), seq).isEmpty)
        let ev = run(pp(on), seq)
        XCTAssertEqual(ev.map(\.0), [1.0 + 3 * Self.hop])
        XCTAssertEqual(ev.map(\.1.classId), [siren])
        XCTAssertEqual(ev.map(\.1.repeatAlert), [false])
        XCTAssertEqual(ev[0].1.confidence, Self.sat)
    }

    func testFastPathThresholdIsExactInFloat32() throws {
        let on = try deployed(Double(Self.sat))
        let fire = on.classId("smoke_alarm")!
        XCTAssertTrue(pp(on).update(sat(on, fire, 254.0 / 256), tAudio: 1).isEmpty)  // one step below
        XCTAssertEqual(pp(on).update(sat(on, fire), tAudio: 1).map(\.classId), [fire])
        // the classifier's dequantized saturated output is exactly the threshold
        XCTAssertEqual((Float(Int8(127)) - Float(-128)) * Float(1.0 / 256), Self.sat)
    }

    func testFastPathOffOrUnreachedIsIdentical() throws {
        let off = try deployed(nil), on = try deployed(Double(Self.sat))
        var rng = SplitMix64(seed: 0)
        let levels: [Float] = [0.2, 0.92, 0.98, 0.99, 254.0 / 256]
        for _ in 0..<20 {
            let seq = (0..<80).map { _ in
                sat(off, Int(rng.next() % UInt64(off.numClasses)), levels[Int(rng.next() % UInt64(levels.count))])
            }
            let a = run(pp(off), seq), b = run(pp(on), seq)
            XCTAssertEqual(a.map(\.0), b.map(\.0))
            XCTAssertEqual(a.map(\.1), b.map(\.1))
        }
        // a config without the field decodes as "off"
        XCTAssertNil(try JSONDecoder.snake.decode(SoundWatchConfig.Postprocess.self, from: Data(
            #"{"smooth_n": 1, "k_of_n": [2, 2], "danger_repeat_s": 5, "danger_clear_s": 10}"#.utf8)).dangerFastThreshold)
    }

    func testFastPathDoesNotAffectWarnAndInfo() throws {
        let on = try deployed(Double(Self.sat))
        let bg = sat(on, on.backgroundId, 1)
        for key in ["baby_cry", "door_knock", "dog_bark", "water_running"] {
            XCTAssertTrue(run(pp(on), [bg, sat(on, on.classId(key)!), bg, bg]).isEmpty, key)
        }
    }

    func testFastPathNoFreshAlertDuringOccurrenceEvenIfAcked() throws {
        let on = try deployed(Double(Self.sat))
        let siren = on.classId("siren")!
        let bg = sat(on, on.backgroundId, 1), hi = sat(on, siren)
        let p = pp(on)
        XCTAssertEqual(run(p, [hi]).count, 1)
        XCTAssertTrue(p.isPending(siren))
        p.ack(siren)
        // isolated saturated hops every 2 s (< danger_clear_s apart): same occurrence, no new alert
        let seq = Array(repeating: rep(bg, 3) + [hi], count: 10).flatMap { $0 }
        XCTAssertTrue(run(p, seq, t0: 1.5).isEmpty)
    }

    func testFastPathKeepsCooldownRepeatAndClear() throws {
        let on = try deployed(Double(Self.sat))
        let pcfg = on.postprocess
        let siren = on.classId("siren")!
        let bg = sat(on, on.backgroundId, 1), hi = sat(on, siren)
        var p = pp(on)
        // one saturated hop, never acked: fresh + repeats every max(danger_repeat_s, cooldown_s)
        // until the sound has been below threshold for danger_clear_s
        var ev = run(p, [hi] + rep(bg, 40))
        let ts = ev.map(\.0)
        XCTAssertFalse(ev[0].1.repeatAlert)
        XCTAssertTrue(ev.dropFirst().allSatisfy { $0.1.repeatAlert })
        let step = max(pcfg.dangerRepeatS, on.classes[siren].cooldownS)
        for i in 1..<ts.count { XCTAssertEqual(ts[i] - ts[i - 1], step, accuracy: 1e-9) }
        XCTAssertLessThan(ts.last! - ts[0], pcfg.dangerClearS + step)
        XCTAssertFalse(p.isPending(siren))
        // after the occurrence cleared, a new single saturated hop alerts fresh again
        let ev2 = run(p, [hi], t0: 1.0 + 41 * Self.hop)
        XCTAssertEqual(ev2.map(\.1.classId), [siren])
        XCTAssertEqual(ev2.map(\.1.repeatAlert), [false])
        // cooldown: a fresh fast alert never comes < cooldown_s after the previous emission
        var classes = on.classes
        classes[siren] = SoundClass(key: classes[siren].key, labelVi: classes[siren].labelVi, level: classes[siren].level,
                                    color: classes[siren].color, vibration: classes[siren].vibration,
                                    threshold: classes[siren].threshold, cooldownS: 30, enabled: true,
                                    sfSymbol: classes[siren].sfSymbol)
        p = PostProcessor(classes: classes, cfg: pcfg)
        ev = run(p, [hi] + rep(bg, 25) + [hi] + rep(bg, 40) + [hi])
        let fresh = ev.filter { !$0.1.repeatAlert }.map(\.0)
        XCTAssertEqual(fresh.first, 1.0)
        for i in 1..<max(1, fresh.count) { XCTAssertGreaterThanOrEqual(fresh[i] - fresh[i - 1], 30) }
    }

    func testFastPathRespectsBgMargin() throws {
        let on = try deployed(Double(Self.sat), bgMargin: 0.2)
        let siren = on.classId("siren")!
        var v = [Float](repeating: 0, count: on.numClasses)
        v[siren] = Self.sat
        v[on.backgroundId] = 0.9  // (not normalized) margin 0.096 < 0.2
        XCTAssertTrue(pp(on).update(v, tAudio: 1).isEmpty)
        v[on.backgroundId] = 0.5  // margin 0.496 >= 0.2
        XCTAssertEqual(pp(on).update(v, tAudio: 1).map(\.classId), [siren])
    }
}

/// Small deterministic PRNG for the randomized test.
struct SplitMix64 {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

extension JSONDecoder {
    static var snake: JSONDecoder {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }
}
