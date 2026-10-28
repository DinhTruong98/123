import Accelerate
import Foundation

/// Log-mel features, port of src/soundwatch/audio/features.py (plan 4.2):
/// 49 frames x 512 samples (step 320, no padding), periodic Hann, |rfft|^2 (unscaled, 257 bins),
/// sparse HTK mel filterbank (40 bands, norm=None), natural log(mel + 1e-6).
/// The window and filterbank come from soundwatch_config.json so they match Python bit for bit.
/// FFT, power and mel are computed in double precision: near-silent bands (mel ~ eps) are very
/// sensitive to FFT rounding, and float32 vDSP differs from numpy's float32 FFT by up to ~1.2e-3
/// there, while double stays within ~3e-4 of numpy (numpy's own float32 error).
final class FeatureExtractor {
    let nFft: Int
    let step: Int
    let nFrames: Int
    let nMels: Int
    let windowSamples: Int
    private let logEps: Double
    private let mean: Float
    private let invStd: Float
    private let hann: [Float]
    private let bands: [(start: Int, weights: [Double])]
    private let log2n: vDSP_Length
    private let fftSetup: FFTSetupD
    private var frame: [Float]
    private var frameD: [Double]
    private var re: [Double]
    private var im: [Double]
    private var power: [Double]

    init(config: SoundWatchConfig) {
        let a = config.audio
        nFft = a.nFft
        step = a.frameStep
        nFrames = a.nFrames
        nMels = a.nMels
        windowSamples = a.windowSamples
        logEps = a.logEps
        mean = Float(config.featureStats.mean)
        invStd = Float(1.0 / max(config.featureStats.std, 1e-12))
        hann = config.hann
        bands = config.melBands.map { (start: $0.start, weights: $0.weights.map(Double.init)) }
        precondition(hann.count == nFft && bands.count == nMels && nFft.nonzeroBitCount == 1)
        log2n = vDSP_Length(nFft.trailingZeroBitCount)
        fftSetup = vDSP_create_fftsetupD(log2n, FFTRadix(kFFTRadix2))!
        frame = [Float](repeating: 0, count: nFft)
        frameD = [Double](repeating: 0, count: nFft)
        re = [Double](repeating: 0, count: nFft / 2)
        im = [Double](repeating: 0, count: nFft / 2)
        power = [Double](repeating: 0, count: nFft / 2 + 1)
    }

    deinit { vDSP_destroy_fftsetupD(fftSetup) }

    /// window (windowSamples floats in [-1, 1]) -> logmel, row-major (nFrames, nMels), NOT normalized.
    func logmel(_ window: [Float], into out: inout [Float]) {
        precondition(window.count == windowSamples)
        if out.count != nFrames * nMels { out = [Float](repeating: 0, count: nFrames * nMels) }
        let half = nFft / 2
        window.withUnsafeBufferPointer { x in
            for t in 0..<nFrames {
                // frame * w in float32 (as numpy), then double
                vDSP_vmul(x.baseAddress! + t * step, 1, hann, 1, &frame, 1, vDSP_Length(nFft))
                vDSP_vspdp(frame, 1, &frameD, 1, vDSP_Length(nFft))
                re.withUnsafeMutableBufferPointer { rp in
                    im.withUnsafeMutableBufferPointer { ip in
                        var split = DSPDoubleSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                        frameD.withUnsafeBufferPointer { f in
                            f.baseAddress!.withMemoryRebound(to: DSPDoubleComplex.self, capacity: half) {
                                vDSP_ctozD($0, 2, &split, 1, vDSP_Length(half))
                            }
                        }
                        vDSP_fft_zripD(fftSetup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))
                    }
                }
                // zrip returns 2 * X[k]; bin 0 holds DC in realp[0] and Nyquist in imagp[0].
                let dc = re[0] * 0.5, nyq = im[0] * 0.5
                power[0] = dc * dc
                power[half] = nyq * nyq
                for k in 1..<half {
                    let r = re[k] * 0.5, i = im[k] * 0.5
                    power[k] = r * r + i * i
                }
                power.withUnsafeBufferPointer { p in
                    for m in 0..<nMels {
                        let b = bands[m]
                        var acc = 0.0
                        if !b.weights.isEmpty {
                            vDSP_dotprD(p.baseAddress! + b.start, 1, b.weights, 1, &acc, vDSP_Length(b.weights.count))
                        }
                        out[t * nMels + m] = Float(log(acc + logEps))
                    }
                }
            }
        }
    }

    /// (logmel - mean) / std, in place.
    func normalize(_ x: inout [Float]) {
        var negMean = -mean
        var s = invStd
        x.withUnsafeMutableBufferPointer { p in
            let n = vDSP_Length(p.count)
            vDSP_vsadd(p.baseAddress!, 1, &negMean, p.baseAddress!, 1, n)
            vDSP_vsmul(p.baseAddress!, 1, &s, p.baseAddress!, 1, n)
        }
    }

    /// RMS level in dBFS (full scale 1.0), -120 for silence (features.rms_dbfs).
    static func rmsDbfs(_ x: [Float]) -> Double {
        guard !x.isEmpty else { return -120 }
        var ss = 0.0
        for v in x { let d = Double(v); ss += d * d }
        let ms = ss / Double(x.count)
        if ms <= 1e-12 { return -120 }
        return max(-120, 10 * log10(ms))
    }
}
