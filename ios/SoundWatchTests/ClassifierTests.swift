import XCTest
@testable import SoundWatch

/// Same normalized input as Python TFLiteClassifier (golden case 4) -> probs within one
/// quantization step (1/256).
final class ClassifierTests: XCTestCase {
    func testMatchesPython() throws {
        let clf = try TestSupport.classifier()
        XCTAssertEqual(clf.numClasses, 8)
        XCTAssertEqual(clf.outputScale, 1.0 / 256, accuracy: 1e-9)
        XCTAssertEqual(clf.outputZeroPoint, -128)
        print("MODEL input scale \(clf.inputScale) zp \(clf.inputZeroPoint), output scale \(clf.outputScale) zp \(clf.outputZeroPoint)")
        let ref = try TestSupport.json("golden_features")["classifier"] as! [String: Any]
        let probs = try clf.predict(TestSupport.floats(ref["features"]))
        let expected = TestSupport.floats(ref["probs"])
        print("CLASSIFIER swift \(probs) python \(expected)")
        for (a, b) in zip(probs, expected) { XCTAssertEqual(a, b, accuracy: 1.0 / 256 + 1e-6) }
    }

    func testSwiftFeaturesGiveSameProbs() throws {
        // full Swift path (features + normalize + model) on golden case 4
        let cfg = try TestSupport.config()
        let fx = FeatureExtractor(config: cfg)
        let golden = try TestSupport.json("golden_features")
        let c = (golden["cases"] as! [[String: Any]])[4]
        var lm: [Float] = []
        fx.logmel((c["input_int16"] as! [NSNumber]).map { Float($0.int16Value) / 32768 }, into: &lm)
        fx.normalize(&lm)
        let ref = golden["classifier"] as! [String: Any]
        let refFeatures = TestSupport.floats(ref["features"])
        XCTAssertLessThan(zip(lm, refFeatures).map { abs($0 - $1) }.max()!, 1e-3)
        let probs = try TestSupport.classifier().predict(lm)
        for (a, b) in zip(probs, TestSupport.floats(ref["probs"])) { XCTAssertEqual(a, b, accuracy: 1.0 / 256 + 1e-6) }
    }

    func testSfSymbolsExist() throws {
        for c in try TestSupport.config().classes {
            XCTAssertNotNil(UIImage(systemName: c.sfSymbol), c.sfSymbol)
        }
    }
}
