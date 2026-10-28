import AudioToolbox
import CoreHaptics

/// Plays a class's `vibration` pattern (ms on/off alternating, starting with on) with Core Haptics.
/// DANGER at full intensity, WARN/INFO softer. Falls back to the system vibration without haptics
/// hardware or while the app is in the background (Core Haptics does not play there).
final class Haptics {
    private var engine: CHHapticEngine?

    init() {
        guard CHHapticEngine.capabilitiesForHardware().supportsHaptics else { return }
        engine = try? CHHapticEngine()
        engine?.isAutoShutdownEnabled = true
        engine?.resetHandler = { [weak self] in try? self?.engine?.start() }
        try? engine?.start()
    }

    func play(pattern ms: [Int], level: Level, background: Bool) {
        guard !ms.isEmpty else { return }
        guard let engine, !background else {
            AudioServicesPlayAlertSound(SystemSoundID(kSystemSoundID_Vibrate))
            return
        }
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
        do {
            try engine.start()
            let player = try engine.makePlayer(with: CHHapticPattern(events: events, parameters: []))
            try player.start(atTime: CHHapticTimeImmediate)
        } catch {
            AudioServicesPlayAlertSound(SystemSoundID(kSystemSoundID_Vibrate))
        }
    }
}
