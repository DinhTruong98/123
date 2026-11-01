import Foundation

/// Mirror of `soundwatch_config.json` (written by ios/scripts/export_ios_assets.py from config/soundwatch.yaml).
struct SoundWatchConfig: Decodable {
    struct Audio: Decodable {
        let sampleRate: Int
        let windowSamples: Int
        let hopSamples: Int
        let nFft: Int
        let frameStep: Int
        let nFrames: Int
        let nMels: Int
        let fmin: Double
        let fmax: Double
        let logEps: Double
        let gateDbfs: Double
    }

    struct Postprocess: Decodable {
        var smoothN: Int
        var kOfN: [Int]
        var dangerRepeatS: Double
        var dangerClearS: Double
        var bgMargin: Double?
        var dangerFastThreshold: Double?  // rule 4b, nil = off
    }

    /// Context modes (vn-v2, postprocess.py rule 9); absent = no modes.
    struct Modes: Decodable {
        struct Spec: Decodable {
            var disable: [String] = []
            var level: [String: Int] = [:]
            // vn-v3 display overrides (rule 9b): class key -> label / PNG icon / SF Symbol
            var label: [String: String] = [:]
            var icon: [String: String] = [:]
            var sfSymbol: [String: String] = [:]

            init() {}

            init(from decoder: Decoder) throws {  // every key optional, like the YAML
                let c = try decoder.container(keyedBy: CodingKeys.self)
                disable = try c.decodeIfPresent([String].self, forKey: .disable) ?? []
                level = try c.decodeIfPresent([String: Int].self, forKey: .level) ?? [:]
                label = try c.decodeIfPresent([String: String].self, forKey: .label) ?? [:]
                icon = try c.decodeIfPresent([String: String].self, forKey: .icon) ?? [:]
                sfSymbol = try c.decodeIfPresent([String: String].self, forKey: .sfSymbol) ?? [:]
            }

            private enum CodingKeys: String, CodingKey { case disable, level, label, icon, sfSymbol }
        }

        /// config.AutoModeConfig: level evidence (median RMS, vn-v2; off when `streetAboveDbfs` is nil)
        /// and/or sound evidence (vn-v3: EMA of a weighted sum of the class probabilities).
        struct Auto: Decodable {
            var streetAboveDbfs: Double? = -35
            var windowS: Double = 30
            var hysteresisDb: Double = 0
            var minS: Double = 5
            var streetEvidence: [String: Double] = [:]
            var streetScoreAbove: Double = 0
            var scoreHysteresis: Double = 0
            var evidenceWindowS: Double = 60
            var minDwellS: Double = 0

            var usesRms: Bool { streetAboveDbfs != nil }
            var usesSound: Bool { !streetEvidence.isEmpty }

            init() {}

            init(from decoder: Decoder) throws {  // defaults as in Python
                let c = try decoder.container(keyedBy: CodingKeys.self)
                if c.contains(.streetAboveDbfs) { streetAboveDbfs = try c.decodeIfPresent(Double.self, forKey: .streetAboveDbfs) }
                windowS = try c.decodeIfPresent(Double.self, forKey: .windowS) ?? windowS
                hysteresisDb = try c.decodeIfPresent(Double.self, forKey: .hysteresisDb) ?? hysteresisDb
                minS = try c.decodeIfPresent(Double.self, forKey: .minS) ?? minS
                streetEvidence = try c.decodeIfPresent([String: Double].self, forKey: .streetEvidence) ?? [:]
                streetScoreAbove = try c.decodeIfPresent(Double.self, forKey: .streetScoreAbove) ?? streetScoreAbove
                scoreHysteresis = try c.decodeIfPresent(Double.self, forKey: .scoreHysteresis) ?? scoreHysteresis
                evidenceWindowS = try c.decodeIfPresent(Double.self, forKey: .evidenceWindowS) ?? evidenceWindowS
                minDwellS = try c.decodeIfPresent(Double.self, forKey: .minDwellS) ?? minDwellS
            }

            private enum CodingKeys: String, CodingKey {
                case streetAboveDbfs, windowS, hysteresisDb, minS, streetEvidence, streetScoreAbove, scoreHysteresis,
                     evidenceWindowS, minDwellS
            }
        }

        let `default`: String
        let modes: [String: Spec]
        let auto: Auto?

        var names: [String] { modes.keys.sorted() + (auto == nil ? [] : [PostProcessor.autoMode]) }
        func spec(_ name: String) -> Spec { modes[name] ?? Spec() }
    }

    struct UI: Decodable {
        let alertTimeoutS: Double
        let historyLen: Int
    }

    struct FeatureStats: Decodable {
        let mean: Double
        let std: Double
    }

    struct MelBand: Decodable {
        let start: Int
        let len: Int
        let weights: [Float]
    }

    let audio: Audio
    var postprocess: Postprocess
    let ui: UI
    var classes: [SoundClass]
    let featureStats: FeatureStats
    let hann: [Float]
    let melBands: [MelBand]
    var modes: Modes?

    var numClasses: Int { classes.count }
    var backgroundId: Int { classes.count - 1 }

    func classId(_ key: String) -> Int? { classes.firstIndex { $0.key == key } }

    static func load(from url: URL) throws -> SoundWatchConfig {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let cfg = try decoder.decode(SoundWatchConfig.self, from: Data(contentsOf: url))
        guard cfg.classes.last?.key == "background" else {
            throw NSError(domain: "SoundWatch", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "last class must be 'background'"])
        }
        return cfg
    }
}

enum Level: Int, Decodable, Comparable {
    case info = 0, warn = 1, danger = 2
    static func < (a: Level, b: Level) -> Bool { a.rawValue < b.rawValue }
}

struct SoundClass: Decodable {
    let key: String
    let labelVi: String
    let level: Level
    let color: [Int]
    let vibration: [Int]
    var threshold: Float
    let cooldownS: Double
    var enabled: Bool
    let sfSymbol: String
}

/// Locates bundled resources (app bundle, or the test bundle for test-only files).
enum Resources {
    static func url(_ name: String, _ ext: String, bundles: [Bundle] = []) -> URL? {
        for b in bundles + [Bundle.main] + Bundle.allBundles {
            if let u = b.url(forResource: name, withExtension: ext) { return u }
        }
        return nil
    }

    static func require(_ name: String, _ ext: String, bundles: [Bundle] = []) throws -> URL {
        guard let u = url(name, ext, bundles: bundles) else {
            throw NSError(domain: "SoundWatch", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "missing resource \(name).\(ext)"])
        }
        return u
    }
}
