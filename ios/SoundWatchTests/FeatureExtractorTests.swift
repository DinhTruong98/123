import XCTest
@testable import SoundWatch

/// Swift log-mel vs Python golden vectors (`export-golden`): max |diff| < 1e-3 on every case.
final class FeatureExtractorTests: XCTestCase {
    func testGoldenLogmel() throws {
        let cfg = try TestSupport.config()
        let fx = FeatureExtractor(config: cfg)
        let golden = try TestSupport.json("golden_features")
        let cases = golden["cases"] as! [[String: Any]]
        XCTAssertEqual(cases.count, 5)
        var out: [Float] = []
        for c in cases {
            let pcm = (c["input_int16"] as! [NSNumber]).map { Float($0.int16Value) / 32768 }
            let ref = TestSupport.floats(c["logmel"])
            fx.logmel(pcm, into: &out)
            XCTAssertEqual(out.count, ref.count)
            let maxDiff = zip(out, ref).map { abs($0 - $1) }.max()!
            print(String(format: "GOLDEN %@ max|diff| = %.3g", c["name"] as! String, maxDiff))
            XCTAssertLessThan(maxDiff, 1e-3, "case \(c["name"]!)")
        }
    }

    func testSilenceIsLogEps() throws {
        let fx = FeatureExtractor(config: try TestSupport.config())
        var out: [Float] = []
        fx.logmel([Float](repeating: 0, count: 16000), into: &out)
        for v in out { XCTAssertEqual(v, logf(1e-6), accuracy: 1e-5) }
    }

    func testRmsDbfs() {
        XCTAssertEqual(FeatureExtractor.rmsDbfs([Float](repeating: 0, count: 100)), -120)
        XCTAssertEqual(FeatureExtractor.rmsDbfs([Float](repeating: 0.5, count: 100)), 20 * log10(0.5), accuracy: 1e-9)
    }

    func testRingBufferOrder() {
        var rb = RingBuffer(capacity: 5)
        var out: [Float] = []
        rb.write([1, 2, 3])
        rb.latest(into: &out)
        XCTAssertEqual(out, [0, 0, 1, 2, 3])
        rb.write([4, 5, 6, 7])
        rb.latest(into: &out)
        XCTAssertEqual(out, [3, 4, 5, 6, 7])
        rb.write([8, 9, 10, 11, 12, 13])
        rb.latest(into: &out)
        XCTAssertEqual(out, [9, 10, 11, 12, 13])
        XCTAssertEqual(rb.totalWritten, 13)
    }

    func testFeaturePlusInferenceTime() throws {
        let cfg = try TestSupport.config()
        let fx = FeatureExtractor(config: cfg)
        let clf = try TestSupport.classifier()
        let golden = try TestSupport.json("golden_features")
        let c = (golden["cases"] as! [[String: Any]])[2]
        let pcm = (c["input_int16"] as! [NSNumber]).map { Float($0.int16Value) / 32768 }
        var lm: [Float] = []
        var tf = 0.0, ti = 0.0
        let n = 50
        for _ in 0..<n {
            let t0 = CACurrentMediaTime()
            fx.logmel(pcm, into: &lm)
            fx.normalize(&lm)
            let t1 = CACurrentMediaTime()
            _ = try clf.predict(lm)
            ti += CACurrentMediaTime() - t1
            tf += t1 - t0
        }
        print(String(format: "TIMING feature %.3f ms, inference %.3f ms (mean of %d)", tf / Double(n) * 1000, ti / Double(n) * 1000, n))
        XCTAssertLessThan((tf + ti) / Double(n) * 1000, 20)
    }
}
