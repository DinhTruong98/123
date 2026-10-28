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
}
