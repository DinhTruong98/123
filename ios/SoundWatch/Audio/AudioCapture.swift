import AVFoundation

/// Microphone -> mono float32 16 kHz blocks. AVAudioSession mode .measurement turns off AGC,
/// noise suppression and voice processing so the signal is close to the training data.
final class AudioCapture {
    var onSamples: (([Float]) -> Void)?
    private let engine = AVAudioEngine()
    private var converter: AVAudioConverter?
    private let outFormat: AVAudioFormat
    private(set) var isRunning = false

    init(sampleRate: Int) {
        outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate),
                                  channels: 1, interleaved: false)!
    }

    static func configureSession() throws {
        let s = AVAudioSession.sharedInstance()
        try s.setCategory(.playAndRecord, mode: .measurement,
                          options: [.defaultToSpeaker, .mixWithOthers, .allowBluetoothA2DP])
        // iOS mutes Core Haptics and system vibration while recording unless this is set
        try s.setAllowHapticsAndSystemSoundsDuringRecording(true)
        try s.setActive(true)
    }

    static func requestPermission() async -> Bool {
        await AVAudioApplication.requestRecordPermission()
    }

    func start() throws {
        guard !isRunning else { return }
        try Self.configureSession()
        let input = engine.inputNode
        let inFormat = input.outputFormat(forBus: 0)
        guard inFormat.sampleRate > 0, inFormat.channelCount > 0 else {
            throw NSError(domain: "SoundWatch", code: 4, userInfo: [NSLocalizedDescriptionKey: "Không có micro"])
        }
        guard let conv = AVAudioConverter(from: inFormat, to: outFormat) else {
            throw NSError(domain: "SoundWatch", code: 5,
                          userInfo: [NSLocalizedDescriptionKey: "Không tạo được bộ chuyển 16 kHz"])
        }
        conv.downmix = true
        converter = conv
        let ratio = outFormat.sampleRate / inFormat.sampleRate
        input.installTap(onBus: 0, bufferSize: 4096, format: inFormat) { [weak self] buf, _ in
            self?.convert(buf, ratio: ratio)
        }
        engine.prepare()
        try engine.start()
        isRunning = true
    }

    func stop() {
        guard isRunning else { return }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        converter = nil
        isRunning = false
    }

    private func convert(_ buf: AVAudioPCMBuffer, ratio: Double) {
        guard let conv = converter else { return }
        let cap = AVAudioFrameCount(Double(buf.frameLength) * ratio + 32)
        guard let out = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: cap) else { return }
        var fed = false
        var error: NSError?
        conv.convert(to: out, error: &error) { _, status in
            if fed {
                status.pointee = .noDataNow
                return nil
            }
            fed = true
            status.pointee = .haveData
            return buf
        }
        guard error == nil, out.frameLength > 0, let ch = out.floatChannelData else { return }
        onSamples?(Array(UnsafeBufferPointer(start: ch[0], count: Int(out.frameLength))))
    }
}

/// Feeds the bundled demo WAV into the engine at real-time speed (blocks of 0.1 s on an absolute
/// schedule, like FileSource(realtime=True)), optionally playing it on the speaker at the same time.
final class DemoFilePlayer {
    var onSamples: (([Float]) -> Void)?
    var onFinished: (() -> Void)?
    private var timer: DispatchSourceTimer?
    private var player: AVAudioPlayer?
    private(set) var isPlaying = false

    func start(url: URL, sampleRate: Int, playAloud: Bool) throws {
        stop()
        let (x, sr) = try WavReader.read(url)
        guard sr == sampleRate else {
            throw NSError(domain: "SoundWatch", code: 6, userInfo: [NSLocalizedDescriptionKey: "demo WAV phải là \(sampleRate) Hz"])
        }
        if playAloud {
            if AVAudioSession.sharedInstance().category != .playAndRecord { try AudioCapture.configureSession() }
            player = try AVAudioPlayer(contentsOf: url)
            player?.play()
        }
        let block = sampleRate / 10
        var pos = 0
        let t0 = DispatchTime.now()
        let q = DispatchQueue(label: "soundwatch.demo")
        let tm = DispatchSource.makeTimerSource(queue: q)
        tm.schedule(deadline: t0 + .milliseconds(100), repeating: .milliseconds(100), leeway: .milliseconds(5))
        tm.setEventHandler { [weak self] in
            guard let self else { return }
            // every block whose last sample is "recorded" by now
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
            let due = min(x.count, Int(elapsed * Double(sampleRate)))
            while pos + block <= due || (due == x.count && pos < x.count) {
                let end = min(pos + block, x.count)
                self.onSamples?(Array(x[pos..<end]))
                pos = end
            }
            if pos >= x.count {
                self.stop()
                DispatchQueue.main.async { self.onFinished?() }
            }
        }
        timer = tm
        isPlaying = true
        tm.resume()
    }

    func stop() {
        timer?.cancel()
        timer = nil
        player?.stop()
        player = nil
        isPlaying = false
    }
}
