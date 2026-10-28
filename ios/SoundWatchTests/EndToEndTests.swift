import XCTest
@testable import SoundWatch

/// Engine offline on demo_scenario.wav vs the Python pipeline (demo_expected_events.json):
/// same non-repeat events (class, |dt| <= 0.5 s).
final class EndToEndTests: XCTestCase {
    func testDemoScenarioMatchesPython() throws {
        let cfg = try TestSupport.config()
        let engine = Engine(config: cfg, classifier: try TestSupport.classifier())
        let (x, sr) = try WavReader.read(Resources.require("demo_scenario", "wav", bundles: [TestSupport.bundle]))
        XCTAssertEqual(sr, 16000)
        XCTAssertEqual(x.count, 60 * 16000)

        let t0 = CACurrentMediaTime()
        let (events, status) = engine.runOffline(x)
        let wall = CACurrentMediaTime() - t0

        let expectedJSON = try TestSupport.json("demo_expected_events")
        let expected = (expectedJSON["events"] as! [[String: Any]])
        let hops = expectedJSON["hops"] as! [[String: Any]]
        XCTAssertEqual(status.count, hops.count)

        // per-hop agreement (diagnostic)
        var maxProbDiff: Float = 0
        var gatedMismatch = 0
        for (s, h) in zip(status, hops) {
            XCTAssertEqual(s.tAudio, (h["t_audio"] as! NSNumber).doubleValue, accuracy: 1e-9)
            if s.gated != (h["gated"] as! Bool) { gatedMismatch += 1 }
            maxProbDiff = max(maxProbDiff, zip(s.probs, TestSupport.floats(h["probs"])).map { abs($0 - $1) }.max()!)
        }
        let inferred = status.filter { !$0.gated }
        let featMs = inferred.map(\.featureMs).reduce(0, +) / Double(max(1, inferred.count))
        let inferMs = inferred.map(\.inferMs).reduce(0, +) / Double(max(1, inferred.count))
        print(String(format: "E2E hops %d, gated mismatch %d, max |dprob| per hop %.4f, wall %.2f s for 60 s audio",
                     status.count, gatedMismatch, maxProbDiff, wall))
        print(String(format: "E2E mean feature %.3f ms, inference %.3f ms over %d hops", featMs, inferMs, inferred.count))

        let keys = cfg.classes.map(\.key)
        for e in events {
            print(String(format: "E2E swift %6.1f s  %@%@", e.tAudio, keys[e.classId], e.repeatAlert ? "  repeat" : ""))
        }
        for e in expected {
            print(String(format: "E2E python %6.1f s  %@%@", (e["t_audio"] as! NSNumber).doubleValue,
                         e["class_key"] as! String, (e["repeat"] as! Bool) ? "  repeat" : ""))
        }

        let fresh = events.filter { !$0.repeatAlert }
        let expFresh = expected.filter { !($0["repeat"] as! Bool) }
        XCTAssertEqual(fresh.count, expFresh.count)
        for (s, p) in zip(fresh, expFresh) {
            XCTAssertEqual(keys[s.classId], p["class_key"] as! String)
            XCTAssertEqual(s.tAudio, (p["t_audio"] as! NSNumber).doubleValue, accuracy: 0.5)
        }
    }
}
