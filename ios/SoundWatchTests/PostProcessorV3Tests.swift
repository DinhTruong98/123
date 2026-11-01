import XCTest
@testable import SoundWatch

/// vn-v3 (docs/vn_v3_results.md section 11): mode display overrides (rule 9b) and 'auto' with sound
/// evidence; ported from tests/test_v3.py (MODES and helpers as there).
final class PostProcessorV3Tests: XCTestCase {
    var cfg: SoundWatchConfig!

    override func setUpWithError() throws {
        cfg = try TestSupport.config()  // app model A = v3
    }

    func modes(_ json: String) throws -> SoundWatchConfig.Modes {
        try JSONDecoder.snake.decode(SoundWatchConfig.Modes.self, from: Data(json.utf8))
    }

    static let displayModes = """
    {"default": "indoor", "modes": {
      "indoor": {"label": {"siren": "Còi hú báo cháy"}, "icon": {"siren": "fire.png"}, "sf_symbol": {"siren": "flame.fill"}},
      "street": {"disable": ["dog_bark"], "level": {"car_horn": 0}, "label": {"siren": "Xe ưu tiên"}}},
     "auto": {"street_above_dbfs": -35, "window_s": 30, "hysteresis_db": 3}}
    """

    func autoModes(_ auto: String) -> String {
        """
        {"default": "auto", "modes": {
          "indoor": {"label": {"siren": "Còi hú báo cháy"}, "sf_symbol": {"siren": "flame.fill"}},
          "street": {"disable": ["dog_bark"], "level": {"car_horn": 0}, "label": {"siren": "Xe ưu tiên"}}},
         "auto": \(auto)}
        """
    }

    func id(_ key: String) -> Int { cfg.classId(key)! }

    /// test_v3._p: background gets the rest (also for p("background")).
    func p(_ key: String, _ v: Float = 1) -> [Float] {
        var x = [Float](repeating: 0, count: cfg.numClasses)
        x[id(key)] = v
        x[cfg.backgroundId] += 1 - v
        return x
    }

    func run(_ pp: PostProcessor, _ seq: [[Float]], t0: Double = 1, rms: Double? = nil) -> [AlertEvent] {
        var out: [AlertEvent] = []
        for (i, x) in seq.enumerated() { out += pp.update(x, tAudio: t0 + 0.5 * Double(i), rmsDbfs: rms) }
        return out
    }

    func rep(_ v: [Float], _ n: Int) -> [[Float]] { Array(repeating: v, count: n) }

    func testV3ConfigDecodes() throws {
        let m = try XCTUnwrap(cfg.modes)
        XCTAssertEqual(cfg.classes[id("siren")].labelVi, "Còi hú")
        XCTAssertEqual(cfg.classes[id("smoke_alarm")].labelVi, "Chuông / bíp báo động")
        XCTAssertEqual(m.spec("street").label["siren"], "Xe ưu tiên (cấp cứu / cảnh sát / cứu hoả)")
        XCTAssertEqual(m.spec("indoor").sfSymbol["siren"], "flame.fill")
        let a = try XCTUnwrap(m.auto)
        XCTAssertTrue(a.usesSound && a.usesRms)
        XCTAssertEqual(a.streetEvidence["car_horn"], 1)
        XCTAssertEqual(a.minDwellS, 60)
        // a vn2-style auto block keeps the Python defaults for the new keys
        let old = try modes(#"{"default": "indoor", "modes": {"indoor": {}}, "auto": {"street_above_dbfs": -35, "window_s": 10, "hysteresis_db": 0, "min_s": 2}}"#)
        XCTAssertFalse(old.auto!.usesSound)
        XCTAssertEqual(old.auto!.evidenceWindowS, 60)
        XCTAssertEqual(old.auto!.minDwellS, 0)
        XCTAssertNil(try modes(#"{"default": "indoor", "modes": {}, "auto": {"street_above_dbfs": null}}"#).auto!.streetAboveDbfs)
    }

    func testEventsCarryModeDisplay() throws {
        let si = id("siren"), horn = id("car_horn")
        let pp = PostProcessor(classes: cfg.classes, cfg: cfg.postprocess, modes: try modes(Self.displayModes))
        let ev = run(pp, rep(p("siren"), 3))
        XCTAssertFalse(ev.isEmpty)
        XCTAssertTrue(ev.filter { $0.classId == si }.allSatisfy { $0.label == "Còi hú báo cháy" && $0.sfSymbol == "flame.fill" })
        XCTAssertEqual(pp.display(si).label, "Còi hú báo cháy")
        XCTAssertEqual(pp.display(si).sfSymbol, "flame.fill")
        XCTAssertEqual(pp.display(horn).label, cfg.classes[horn].labelVi)
        XCTAssertTrue(pp.setMode("street", tAudio: 3))
        XCTAssertEqual(pp.display(si).label, "Xe ưu tiên")
        XCTAssertEqual(pp.display(si).sfSymbol, cfg.classes[si].sfSymbol)
        let ev2 = run(pp, rep(p("background"), 30) + rep(p("siren"), 3), t0: 4)
        let fresh = ev2.filter { $0.classId == si && !$0.repeatAlert }
        XCTAssertEqual(fresh.last?.label, "Xe ưu tiên")
        XCTAssertNil(fresh.last?.sfSymbol)
        // the earlier events keep the label of their own time
        XCTAssertEqual(ev.first?.label, "Còi hú báo cháy")
    }

    func testNoDisplayOverridesIdenticalEvents() throws {
        let base = try modes(#"{"default": "indoor", "modes": {"indoor": {}, "street": {"disable": ["dog_bark"]}}}"#)
        let disp = try modes(#"{"default": "indoor", "modes": {"indoor": {"label": {"siren": "X"}}, "street": {"disable": ["dog_bark"]}}}"#)
        var rng = SplitMix64(seed: 3)
        let seq: [[Float]] = (0..<400).map { _ in
            let v = (0..<cfg.numClasses).map { _ in pow(-log(Double(rng.next() % 1_000_000 + 1) / 1_000_001), 10) }
            let s = v.reduce(0, +)
            return v.map { Float($0 / s) }
        }
        let a = run(PostProcessor(classes: cfg.classes, cfg: cfg.postprocess, modes: base), seq)
        let b = run(PostProcessor(classes: cfg.classes, cfg: cfg.postprocess, modes: disp), seq)
        XCTAssertFalse(a.isEmpty)
        XCTAssertEqual(a, b.map { var e = $0; e.label = nil; return e })
        XCTAssertTrue((a + b).allSatisfy { $0.sfSymbol == nil })
        let c = run(PostProcessor(classes: cfg.classes, cfg: cfg.postprocess), seq)  // no modes at all
        XCTAssertEqual(a.map { var e = $0; e.level = nil; return e }, c)
    }

    func testAutoSoundEvidenceSwitchesWithHysteresis() throws {
        let m = try modes(autoModes(#"{"street_above_dbfs": null, "street_evidence": {"car_horn": 1.0, "baby_cry": -1.0}, "street_score_above": 0.1, "score_hysteresis": 0.05, "evidence_window_s": 20, "min_s": 5}"#))
        let pp = PostProcessor(classes: cfg.classes, cfg: cfg.postprocess, modes: m)
        XCTAssertEqual(pp.mode, "indoor")
        XCTAssertEqual(pp.modeSelector, "auto")
        let horn = p("car_horn", 0.6), bg = p("background"), baby = p("baby_cry", 0.3)
        _ = run(pp, rep(bg, 40))  // 20 s of quiet indoor sound
        XCTAssertEqual(pp.mode, "indoor")
        XCTAssertEqual(pp.streetScore, 0)
        _ = run(pp, Array(repeating: [horn] + rep(bg, 3), count: 20).flatMap { $0 }, t0: 21)  // horn every 2 s
        XCTAssertEqual(pp.mode, "street")
        XCTAssertGreaterThan(pp.streetScore, 0.1)
        let last = try XCTUnwrap(pp.modeLog.last)
        XCTAssertTrue(last.t > 21 && last.t <= 61, "\(last.t)")
        XCTAssertTrue(last.reason.contains("street score"))
        _ = run(pp, rep(bg, 10), t0: 61)  // decays but stays above 0.05 for a while (hysteresis)
        XCTAssertEqual(pp.mode, "street")
        _ = run(pp, rep(baby, 60), t0: 66)  // household sound pulls the score below 0.05
        XCTAssertEqual(pp.mode, "indoor")
        XCTAssertLessThan(pp.streetScore, 0.05)
    }

    func testAutoMinDwellAndRmsVeto() throws {
        let m = try modes(autoModes(#"{"street_above_dbfs": -35, "window_s": 10, "hysteresis_db": 4, "street_evidence": {"car_horn": 1.0}, "street_score_above": 0.2, "score_hysteresis": 0.1, "evidence_window_s": 5, "min_s": 2, "min_dwell_s": 30}"#))
        let horn = p("car_horn", 0.9), bg = p("background")
        // loud street with horns -> street; then quiet without horns: stays street for >= 30 s (dwell)
        var pp = PostProcessor(classes: cfg.classes, cfg: cfg.postprocess, modes: m)
        _ = run(pp, Array(repeating: [horn, bg], count: 10).flatMap { $0 }, rms: -25)
        XCTAssertEqual(pp.mode, "street")
        let tSw = pp.modeLog.last!.t
        _ = run(pp, rep(bg, 40), t0: 11, rms: -50)
        XCTAssertTrue(pp.mode == "street" || pp.modeLog.last!.t - tSw >= 30)
        _ = run(pp, rep(bg, 80), t0: 31, rms: -50)
        XCTAssertEqual(pp.mode, "indoor")
        XCTAssertGreaterThanOrEqual(pp.modeLog.last!.t - tSw, 30)
        // quiet room hearing horns (e.g. through a window): the level vetoes street
        pp = PostProcessor(classes: cfg.classes, cfg: cfg.postprocess, modes: m)
        _ = run(pp, Array(repeating: [horn, bg], count: 40).flatMap { $0 }, rms: -55)
        XCTAssertEqual(pp.mode, "indoor")
        // loud room without horns (TV, karaoke): no switch to street
        pp = PostProcessor(classes: cfg.classes, cfg: cfg.postprocess, modes: m)
        _ = run(pp, rep(bg, 80), rms: -20)
        XCTAssertEqual(pp.mode, "indoor")
    }
}
