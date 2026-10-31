import XCTest
@testable import SoundWatch

/// Model B (VN v2) next to A: golden classifier, end-to-end demo vs Python
/// (demo_expected_events_vn2.json), and the dual-model engine giving each model the same events as
/// running it alone.
final class DualModelTests: XCTestCase {
    func testConfigBHasModesAndOwnPostprocess() throws {
        let a = try TestSupport.config(), b = try TestSupport.configB()
        XCTAssertNil(a.modes)
        XCTAssertEqual(b.modes?.default, "indoor")
        XCTAssertEqual(Set(b.modes!.names), ["indoor", "street", "auto"])
        XCTAssertEqual(b.modes!.spec("street").disable, ["dog_bark", "baby_cry"])
        XCTAssertEqual(b.modes!.spec("street").level, ["car_horn": 0])
        XCTAssertEqual(b.postprocess.kOfN, [2, 3])
        XCTAssertEqual(b.postprocess.dangerFastThreshold, 0.95)
        XCTAssertEqual(b.featureStats.mean, -2.148079443279617, accuracy: 1e-12)
        // same front-end: the log-mel can be shared
        XCTAssertEqual(a.hann, b.hann)
        XCTAssertEqual(a.melBands.map(\.weights), b.melBands.map(\.weights))
        XCTAssertEqual(a.classes.map(\.key), b.classes.map(\.key))
    }

    func testClassifierBMatchesPython() throws {
        let ref = try TestSupport.json("golden_features_vn2")["classifier"] as! [String: Any]
        let probs = try TestSupport.classifierB().predict(TestSupport.floats(ref["features"]))
        let expected = TestSupport.floats(ref["probs"])
        print("CLASSIFIER_B swift \(probs) python \(expected)")
        for (a, b) in zip(probs, expected) { XCTAssertEqual(a, b, accuracy: 1.0 / 256 + 1e-6) }
    }

    func testDemoScenarioMatchesPythonB() throws {
        let cfg = try TestSupport.configB()
        let engine = Engine(pipelines: [ModelPipeline(name: "B", config: cfg, classifier: try TestSupport.classifierB())])
        let (events, status) = engine.runOffline(try TestSupport.demoAudio())
        let exp = try TestSupport.json("demo_expected_events_vn2")
        let expected = (exp["events"] as! [[String: Any]])
        let hops = exp["hops"] as! [[String: Any]]
        XCTAssertEqual(status.count, hops.count)
        var maxProbDiff: Float = 0
        for (s, h) in zip(status, hops) {
            maxProbDiff = max(maxProbDiff, zip(s.probs, TestSupport.floats(h["probs"])).map { abs($0 - $1) }.max()!)
        }
        let keys = cfg.classes.map(\.key)
        print(String(format: "E2E_B max |dprob| per hop %.4f", maxProbDiff))
        for e in events { print(String(format: "E2E_B swift %6.1f s  %@%@", e.tAudio, keys[e.classId], e.repeatAlert ? "  repeat" : "")) }
        XCTAssertEqual(events.count, expected.count)
        for (s, p) in zip(events, expected) {
            XCTAssertEqual(keys[s.classId], p["class_key"] as! String)
            XCTAssertEqual(s.tAudio, (p["t_audio"] as! NSNumber).doubleValue, accuracy: 0.25)
            XCTAssertEqual(s.repeatAlert, p["repeat"] as! Bool)
            XCTAssertEqual(s.level, cfg.classes[s.classId].level)  // indoor = configured levels
        }
    }

    func testDualEngineMatchesSingleRuns() throws {
        let a = try TestSupport.config(), b = try TestSupport.configB()
        let x = try TestSupport.demoAudio()
        let single = [
            Engine(pipelines: [ModelPipeline(name: "A", config: a, classifier: try TestSupport.classifier())]).runOffline(x).events,
            Engine(pipelines: [ModelPipeline(name: "B", config: b, classifier: try TestSupport.classifierB())]).runOffline(x).events,
        ]
        let dual = Engine(pipelines: [
            ModelPipeline(name: "A", config: a, classifier: try TestSupport.classifier()),
            ModelPipeline(name: "B", config: b, classifier: try TestSupport.classifierB()),
        ])
        let t0 = CACurrentMediaTime()
        let r = dual.runOfflineAll(x)
        let wall = CACurrentMediaTime() - t0
        XCTAssertEqual(r.events[0], single[0])
        XCTAssertEqual(r.events[1], single[1])
        XCTAssertFalse(r.events[0].isEmpty)
        XCTAssertFalse(r.events[1].isEmpty)
        let inferred = r.status.filter { !$0.gated }
        func mean(_ v: [Double]) -> Double { v.reduce(0, +) / Double(max(1, v.count)) }
        print(String(format: "DUAL feature %.3f ms, infer A %.3f ms, infer B %.3f ms (mean of %d hops), wall %.2f s",
                     mean(inferred.map(\.featureMs)), mean(inferred.map { $0.models[0].inferMs }),
                     mean(inferred.map { $0.models[1].inferMs }), inferred.count, wall))
        XCTAssertTrue(r.status.allSatisfy { $0.models[1].mode == "indoor" && $0.models[0].mode == nil })
    }

    func testDualEngineModeCommand() throws {
        let a = try TestSupport.config(), b = try TestSupport.configB()
        let dual = Engine(pipelines: [
            ModelPipeline(name: "A", config: a, classifier: try TestSupport.classifier()),
            ModelPipeline(name: "B", config: b, classifier: try TestSupport.classifierB()),
        ])
        dual.setMode("street", model: 1)
        dual.queue.sync {}
        XCTAssertEqual(dual.pipelines[1].post.mode, "street")
        let r = dual.runOfflineAll(try TestSupport.demoAudio())
        let keys = b.classes.map(\.key)
        // street: no dog_bark / baby_cry, car_horn as INFO without repeats; A unaffected
        XCTAssertFalse(r.events[1].contains { ["dog_bark", "baby_cry"].contains(keys[$0.classId]) })
        XCTAssertTrue(r.events[1].filter { keys[$0.classId] == "car_horn" }.allSatisfy { $0.level == .info && !$0.repeatAlert })
        XCTAssertTrue(r.events[0].contains { keys[$0.classId] == "baby_cry" })
    }
}
