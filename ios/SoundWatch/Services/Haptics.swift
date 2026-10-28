import AudioToolbox
import CoreHaptics
import Foundation

/// Vibration for alerts. A class's `vibration` pattern is ms on/off alternating, starting with on.
///
/// - WARN/INFO, and DANGER repeats while nothing is looping: the pattern plays once.
/// - DANGER: the pattern loops until `stopContinuous` (ack), or until the optional time limit.
///   In the foreground it loops with Core Haptics (full intensity). In the background Core Haptics
///   cannot play, so the system vibration (`kSystemSoundID_Vibrate`) is repeated on a timer instead;
///   the app stays alive there because it keeps recording (UIBackgroundModes audio).
///
/// Both only work while the microphone records if the audio session allows haptics during
/// recording (`AudioCapture.configureSession`); iOS mutes them otherwise.
final class Haptics {
    var log: ((String) -> Void)?
    /// DANGER vibration stops by itself after this many seconds (nil = until ack).
    var continuousLimit: TimeInterval?

    private var engine: CHHapticEngine?
    private var loopPlayer: CHHapticAdvancedPatternPlayer?
    private var bgTimer: DispatchSourceTimer?
    private var limitTimer: DispatchSourceTimer?
    private var active: [(classId: Int, pattern: [Int])] = []  // looping DANGER classes, newest last
    private var inBackground = false
    private let q = DispatchQueue(label: "soundwatch.haptics")

    init() {
        guard CHHapticEngine.capabilitiesForHardware().supportsHaptics else { return }
        do {
            let e = try CHHapticEngine()
            e.playsHapticsOnly = true
            e.isAutoShutdownEnabled = false
            e.stoppedHandler = { [weak self] reason in
                self?.log?("engine stopped, reason \(reason.rawValue)")
                // e.g. audio session interruption: restart a running DANGER loop in the foreground
                self?.q.async { if let self, !self.active.isEmpty, !self.inBackground { self.apply() } }
            }
            e.resetHandler = { [weak self] in
                self?.log?("engine reset")
                self?.q.async { self?.loopPlayer = nil; self?.apply() }
            }
            try e.start()
            engine = e
        } catch {
            log?("engine unavailable: \(error)")
        }
    }

    var loopingClassIds: [Int] { q.sync { active.map(\.classId) } }
    func isLooping(_ classId: Int) -> Bool { q.sync { active.contains { $0.classId == classId } } }

    func playOnce(pattern ms: [Int], level: Level) {
        guard !ms.isEmpty else { return }
        q.async { [self] in
            if inBackground || engine == nil {
                AudioServicesPlaySystemSound(kSystemSoundID_Vibrate)
                return
            }
            do {
                try engine!.start()
                try engine!.makePlayer(with: Self.pattern(ms, level: level, gap: 0)).start(atTime: CHHapticTimeImmediate)
            } catch {
                log?("play failed: \(error)")
                AudioServicesPlaySystemSound(kSystemSoundID_Vibrate)
            }
        }
    }

    func startContinuous(classId: Int, pattern ms: [Int]) {
        guard !ms.isEmpty else { return }
        q.async { [self] in
            active.removeAll { $0.classId == classId }
            active.append((classId, ms))
            startLimitTimer()
            apply()
        }
    }

    func stopContinuous(classId: Int) {
        q.async { [self] in
            guard active.contains(where: { $0.classId == classId }) else { return }
            active.removeAll { $0.classId == classId }
            if active.isEmpty { limitTimer?.cancel(); limitTimer = nil }
            apply()
        }
    }

    func stopAll() {
        q.async { [self] in
            active.removeAll()
            limitTimer?.cancel()
            limitTimer = nil
            apply()
        }
    }

    func setBackground(_ bg: Bool) {
        q.async { [self] in
            guard inBackground != bg else { return }
            inBackground = bg
            apply()
        }
    }

    // MARK: on q

    /// Bring the loop in line with `active` / `inBackground`.
    private func apply() {
        try? loopPlayer?.stop(atTime: CHHapticTimeImmediate)
        loopPlayer = nil
        bgTimer?.cancel()
        bgTimer = nil
        guard let cur = active.last else { return }
        let total = Double(cur.pattern.reduce(0, +)) / 1000
        let gap = 0.6
        if inBackground || engine == nil {
            let t = DispatchSource.makeTimerSource(queue: q)
            t.schedule(deadline: .now(), repeating: max(1.0, total + gap))
            t.setEventHandler { AudioServicesPlaySystemSound(kSystemSoundID_Vibrate) }
            t.resume()
            bgTimer = t
            log?("DANGER vibration loop: system vibrate every \(max(1.0, total + gap)) s (background)")
            return
        }
        do {
            try engine!.start()
            let p = try engine!.makeAdvancedPlayer(with: Self.pattern(cur.pattern, level: .danger, gap: gap))
            p.loopEnabled = true
            p.loopEnd = total + gap
            p.completionHandler = { [weak self] err in
                if let err { self?.log?("loop ended: \(err)") }
            }
            try p.start(atTime: CHHapticTimeImmediate)
            loopPlayer = p
            log?("DANGER vibration loop: Core Haptics, period \(total + gap) s (foreground)")
        } catch {
            log?("loop failed: \(error); falling back to system vibrate")
            let t = DispatchSource.makeTimerSource(queue: q)
            t.schedule(deadline: .now(), repeating: max(1.0, total + gap))
            t.setEventHandler { AudioServicesPlaySystemSound(kSystemSoundID_Vibrate) }
            t.resume()
            bgTimer = t
        }
    }

    private func startLimitTimer() {
        limitTimer?.cancel()
        limitTimer = nil
        guard let limit = continuousLimit, limit > 0 else { return }
        let t = DispatchSource.makeTimerSource(queue: q)
        t.schedule(deadline: .now() + limit)
        t.setEventHandler { [weak self] in
            self?.log?("DANGER vibration stopped after the \(Int(limit)) s limit")
            self?.active.removeAll()
            self?.apply()
        }
        t.resume()
        limitTimer = t
    }

    private static func pattern(_ ms: [Int], level: Level, gap: Double) throws -> CHHapticPattern {
        let intensity: Float = level == .danger ? 1.0 : (level == .warn ? 0.7 : 0.5)
        var events: [CHHapticEvent] = []
        var t = 0.0
        for (i, d) in ms.enumerated() {
            let dur = Double(d) / 1000
            if i % 2 == 0 {
                events.append(CHHapticEvent(
                    eventType: .hapticContinuous,
                    parameters: [CHHapticEventParameter(parameterID: .hapticIntensity, value: intensity),
                                 CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.6)],
                    relativeTime: t, duration: dur))
            }
            t += dur
        }
        return try CHHapticPattern(events: events, parameters: [])
    }
}
