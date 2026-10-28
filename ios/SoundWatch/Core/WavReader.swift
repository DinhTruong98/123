import Foundation

/// Minimal reader for 16-bit PCM WAV (the demo file): returns mono float32 (int16 / 32768,
/// channel mean) and the sample rate, like soundfile.read(dtype="float32") + to_mono_float32.
enum WavReader {
    static func read(_ url: URL) throws -> (samples: [Float], sampleRate: Int) {
        let d = try Data(contentsOf: url)
        func u32(_ o: Int) -> Int { Int(d[o]) | Int(d[o + 1]) << 8 | Int(d[o + 2]) << 16 | Int(d[o + 3]) << 24 }
        func u16(_ o: Int) -> Int { Int(d[o]) | Int(d[o + 1]) << 8 }
        guard d.count > 12, String(data: d[0..<4], encoding: .ascii) == "RIFF",
              String(data: d[8..<12], encoding: .ascii) == "WAVE" else { throw err("not a WAV file") }
        var o = 12
        var channels = 0, rate = 0, bits = 0
        while o + 8 <= d.count {
            let id = String(data: d[o..<o + 4], encoding: .ascii)
            let size = u32(o + 4)
            let body = o + 8
            if id == "fmt " {
                guard u16(body) == 1 else { throw err("only PCM WAV is supported") }
                channels = u16(body + 2)
                rate = u32(body + 4)
                bits = u16(body + 14)
            } else if id == "data" {
                guard bits == 16, channels > 0 else { throw err("only 16-bit PCM WAV is supported") }
                let frames = min(size, d.count - body) / (2 * channels)
                var out = [Float](repeating: 0, count: frames)
                d.withUnsafeBytes { raw in
                    for f in 0..<frames {
                        var s: Float = 0
                        for c in 0..<channels {
                            let p = body + (f * channels + c) * 2
                            s += Float(Int16(bitPattern: UInt16(raw[p]) | UInt16(raw[p + 1]) << 8)) / 32768
                        }
                        out[f] = s / Float(channels)
                    }
                }
                return (out, rate)
            }
            o = body + size + (size & 1)
        }
        throw err("no data chunk")
    }

    private static func err(_ m: String) -> NSError {
        NSError(domain: "SoundWatch", code: 3, userInfo: [NSLocalizedDescriptionKey: m])
    }
}
