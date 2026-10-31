import XCTest
@testable import SoundWatch

final class TestBundleToken {}

enum TestSupport {
    static let bundle = Bundle(for: TestBundleToken.self)

    static func config() throws -> SoundWatchConfig {
        try SoundWatchConfig.load(from: Resources.require("soundwatch_config", "json", bundles: [bundle]))
    }

    static func json(_ name: String) throws -> [String: Any] {
        let url = try Resources.require(name, "json", bundles: [bundle])
        return try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
    }

    static func floats(_ v: Any?) -> [Float] { (v as! [NSNumber]).map(\.floatValue) }

    static func classifier() throws -> TFLiteClassifier {
        try TFLiteClassifier(modelURL: Resources.require("student_int8", "tflite", bundles: [bundle]))
    }

    /// Model B (VN v2): config with modes, int8 model.
    static func configB() throws -> SoundWatchConfig {
        try SoundWatchConfig.load(from: Resources.require("soundwatch_config_vn2", "json", bundles: [bundle]))
    }

    static func classifierB() throws -> TFLiteClassifier {
        try TFLiteClassifier(modelURL: Resources.require("student_int8_vn2", "tflite", bundles: [bundle]))
    }

    static func demoAudio() throws -> [Float] {
        try WavReader.read(Resources.require("demo_scenario", "wav", bundles: [bundle])).samples
    }
}
