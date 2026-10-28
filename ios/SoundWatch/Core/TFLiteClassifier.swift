import Foundation
import TensorFlowLiteC

/// Runs student_int8.tflite (port of TFLiteClassifier in src/soundwatch/engine/classifier.py):
/// quantize the normalized features with the model's input scale/zero-point, invoke, dequantize.
/// Uses the LiteRT C API directly (the official Swift wrapper has no int8 tensor type).
final class TFLiteClassifier {
    let numClasses: Int
    let inputScale: Float
    let inputZeroPoint: Int
    let outputScale: Float
    let outputZeroPoint: Int
    private let model: OpaquePointer
    private let options: OpaquePointer
    private let delegate: UnsafeMutablePointer<TfLiteDelegate>?
    private let interpreter: OpaquePointer
    private let input: UnsafeMutablePointer<TfLiteTensor>
    private let inputCount: Int
    private let inputIsInt8: Bool
    private let outputIsInt8: Bool
    private var qbuf: [Int8]
    private var outRaw: [UInt8]

    init(modelURL: URL, threads: Int = 1) throws {
        guard let m = TfLiteModelCreateFromFile(modelURL.path) else { throw Self.err("cannot load \(modelURL.lastPathComponent)") }
        model = m
        options = TfLiteInterpreterOptionsCreate()
        TfLiteInterpreterOptionsSetNumThreads(options, Int32(threads))
        // XNNPACK, as the Python LiteRT interpreter uses by default
        var xo = TfLiteXNNPackDelegateOptionsDefault()
        xo.num_threads = Int32(threads)
        delegate = TfLiteXNNPackDelegateCreate(&xo)
        if let delegate { TfLiteInterpreterOptionsAddDelegate(options, delegate) }
        guard let interp = TfLiteInterpreterCreate(model, options) else { throw Self.err("cannot create interpreter") }
        interpreter = interp
        guard TfLiteInterpreterAllocateTensors(interpreter) == kTfLiteOk,
              let inp = TfLiteInterpreterGetInputTensor(interpreter, 0),
              let out = TfLiteInterpreterGetOutputTensor(interpreter, 0) else { throw Self.err("cannot allocate tensors") }
        input = inp
        inputCount = (0..<TfLiteTensorNumDims(inp)).reduce(1) { $0 * Int(TfLiteTensorDim(inp, $1)) }
        numClasses = Int(TfLiteTensorDim(out, TfLiteTensorNumDims(out) - 1))
        inputIsInt8 = TfLiteTensorType(inp) == kTfLiteInt8
        outputIsInt8 = TfLiteTensorType(out) == kTfLiteInt8
        let qi = TfLiteTensorQuantizationParams(inp), qo = TfLiteTensorQuantizationParams(out)
        inputScale = qi.scale
        inputZeroPoint = Int(qi.zero_point)
        outputScale = qo.scale
        outputZeroPoint = Int(qo.zero_point)
        qbuf = [Int8](repeating: 0, count: inputCount)
        outRaw = [UInt8](repeating: 0, count: TfLiteTensorByteSize(out))
    }

    deinit {
        TfLiteInterpreterDelete(interpreter)
        if let delegate { TfLiteXNNPackDelegateDelete(delegate) }
        TfLiteInterpreterOptionsDelete(options)
        TfLiteModelDelete(model)
    }

    /// features: normalized log-mel, row-major (49, 40). Returns class probabilities.
    func predict(_ features: [Float]) throws -> [Float] {
        precondition(features.count == inputCount, "features do not fit the model input")
        let status: TfLiteStatus
        if inputIsInt8 {
            let scale = inputScale == 0 ? 1 : inputScale
            let zp = Float(inputZeroPoint)
            for i in 0..<inputCount {
                // np.round(x / scale + zp): float32 math, round half to even, clip to int8
                let q = (features[i] / scale + zp).rounded(.toNearestOrEven)
                qbuf[i] = Int8(min(127, max(-128, q)))
            }
            status = qbuf.withUnsafeBytes { TfLiteTensorCopyFromBuffer(input, $0.baseAddress, $0.count) }
        } else {
            status = features.withUnsafeBytes { TfLiteTensorCopyFromBuffer(input, $0.baseAddress, $0.count) }
        }
        guard status == kTfLiteOk else { throw Self.err("copy input failed") }
        guard TfLiteInterpreterInvoke(interpreter) == kTfLiteOk,
              let out = TfLiteInterpreterGetOutputTensor(interpreter, 0) else { throw Self.err("invoke failed") }
        _ = outRaw.withUnsafeMutableBytes { TfLiteTensorCopyToBuffer(out, $0.baseAddress, $0.count) }
        let y: [Float]
        if outputIsInt8 {
            let scale = outputScale == 0 ? 1 : outputScale
            let zp = Float(outputZeroPoint)
            y = outRaw.map { (Float(Int8(bitPattern: $0)) - zp) * scale }
        } else {
            y = outRaw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        return Self.asProbs(y)
    }

    /// Unchanged (clipped to [0, 1]) if it already looks like probabilities, else softmax.
    static func asProbs(_ y: [Float], tol: Float = 0.05) -> [Float] {
        let sum = y.reduce(0, +)
        if y.allSatisfy({ $0 >= -1e-6 }) && abs(sum - 1) <= tol {
            return y.map { min(1, max(0, $0)) }
        }
        let m = Double(y.max() ?? 0)
        let e = y.map { exp(Double($0) - m) }
        let s = e.reduce(0, +)
        return e.map { Float($0 / s) }
    }

    private static func err(_ m: String) -> NSError {
        NSError(domain: "SoundWatch", code: 7, userInfo: [NSLocalizedDescriptionKey: "TFLite: \(m)"])
    }
}
