import AVFoundation
import os
import SwiftUI
import UserNotifications

struct ShownAlert: Identifiable, Equatable {
    let id = UUID()
    let classId: Int
    let confidence: Float
    let date: Date
    let tAudio: Double
}

@MainActor
final class AppState: ObservableObject {
    enum Source: Equatable { case idle, mic, demo }

    @Published private(set) var status: StatusUpdate?
    @Published private(set) var currentAlert: ShownAlert?
    @Published private(set) var history: [ShownAlert] = []
    @Published private(set) var source: Source = .idle
    @Published private(set) var errorMessage: String?
    @Published private(set) var micDenied = false
    @Published private(set) var timing = (featureMs: 0.0, inferMs: 0.0, maxTotalMs: 0.0)
    @Published var enabled: [Bool] = []
    @Published private(set) var notificationsAllowed: Bool?
    @Published var showDebug = UserDefaults.standard.bool(forKey: "showDebug") {
        didSet { UserDefaults.standard.set(showDebug, forKey: "showDebug") }
    }
    @Published var playDemoAloud = true
    /// DANGER vibration of a class stops by itself after this many seconds (default 30); 0 = until ack.
    @Published var dangerVibrationLimitS = UserDefaults.standard.object(forKey: "dangerVibrationLimitS") as? Int ?? 30 {
        didSet {
            UserDefaults.standard.set(dangerVibrationLimitS, forKey: "dangerVibrationLimitS")
            haptics.continuousLimit = dangerVibrationLimitS > 0 ? Double(dangerVibrationLimitS) : nil
        }
    }

    let config: SoundWatchConfig
    private let engine: Engine?
    private let capture: AudioCapture
    private let demo = DemoFilePlayer()
    private let haptics = Haptics()
    private let notifier = Notifier()
    private var dismissTask: Task<Void, Never>?
    private var isActive = true
    private static let enabledKey = "enabledClasses"
    private let micFeeds = OSAllocatedUnfairLock(initialState: true)

    static var isRunningTests: Bool { ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil }

    init() {
        // A missing/corrupt bundled resource is a build error, so crash loudly.
        let cfg = try! SoundWatchConfig.load(from: Resources.require("soundwatch_config", "json"))
        config = cfg
        capture = AudioCapture(sampleRate: cfg.audio.sampleRate)
        do {
            let clf = try TFLiteClassifier(modelURL: Resources.require("student_int8", "tflite"))
            engine = Engine(config: cfg, classifier: clf)
        } catch {
            engine = nil
            errorMessage = "Không nạp được model: \(error.localizedDescription)"
        }
        let saved = UserDefaults.standard.array(forKey: Self.enabledKey) as? [Bool]
        enabled = saved?.count == cfg.numClasses ? saved! : cfg.classes.map(\.enabled)
        for (c, on) in enabled.enumerated() { engine?.setEnabled(c, on) }

        engine?.onStatus = { [weak self] s in Task { @MainActor in self?.handle(status: s) } }
        engine?.onAlert = { [weak self] e in
            let date = Date()
            Task { @MainActor in self?.handle(alert: e, date: date) }
        }
        // The microphone keeps running while the demo plays (its samples are dropped), so switching
        // sources never restarts recording, which iOS refuses while the app is in the background.
        let micFeeds = micFeeds
        capture.onSamples = { [weak self] x in
            if micFeeds.withLock({ $0 }) { self?.engine?.feed(x) }
        }
        demo.onSamples = { [weak self] x in self?.engine?.feed(x) }
        demo.onFinished = { [weak self] in Task { @MainActor in self?.demoFinished() } }
        haptics.log = { print("SW_HAPTIC \($0)") }
        haptics.continuousLimit = dangerVibrationLimitS > 0 ? Double(dangerVibrationLimitS) : nil
        notifier.onAck = { [weak self] c in Task { @MainActor in self?.acknowledge(classId: c) } }
        notifier.onOpen = { [weak self] c in Task { @MainActor in self?.reopen(classId: c) } }
    }

    func class_(_ id: Int) -> SoundClass { config.classes[id] }

    // MARK: lifecycle

    func start() async {
        guard !Self.isRunningTests, source == .idle else { return }
        notifier.setUp()
        await refreshNotificationStatus()
        guard await AudioCapture.requestPermission() else {
            micDenied = true
            return
        }
        micDenied = false
        startMic()
        // launch arguments -demo [-demoDelay N]: start with the demo file, optionally after N s
        // (presentations, screenshots, lock-screen tests)
        let args = ProcessInfo.processInfo.arguments
        if args.contains("-demo") {
            if let i = args.firstIndex(of: "-demoDelay"), i + 1 < args.count, let s = Double(args[i + 1]) {
                try? await Task.sleep(for: .seconds(s))
            }
            toggleDemo()
        }
    }

    func setActive(_ active: Bool) {
        isActive = active
        haptics.setBackground(!active)
        if active { Task { await refreshNotificationStatus() } }
    }

    func refreshNotificationStatus() async {
        let s = await UNUserNotificationCenter.current().notificationSettings()
        switch s.authorizationStatus {
        case .authorized, .provisional, .ephemeral: notificationsAllowed = true
        case .denied: notificationsAllowed = false
        default: notificationsAllowed = nil
        }
    }

    private func startMic() {
        engine?.reset()
        micFeeds.withLock { $0 = true }
        do {
            try capture.start()  // no-op if already running
            source = .mic
            errorMessage = nil
        } catch {
            source = .idle
            errorMessage = "Không mở được micro: \(error.localizedDescription)"
        }
    }

    // MARK: demo file

    func toggleDemo() {
        if source == .demo {
            demo.stop()
            demoFinished()
            return
        }
        micFeeds.withLock { $0 = false }
        engine?.reset()
        timing = (0, 0, 0)
        do {
            try demo.start(url: Resources.require("demo_scenario", "wav"), sampleRate: config.audio.sampleRate,
                           playAloud: playDemoAloud)
            source = .demo
        } catch {
            errorMessage = "Không phát được file demo: \(error.localizedDescription)"
            startMic()
        }
    }

    private func demoFinished() {
        source = .idle
        startMic()
    }

    // MARK: engine events

    private func handle(status s: StatusUpdate) {
        status = s
        if !s.gated {
            let total = s.featureMs + s.inferMs
            timing = (s.featureMs, s.inferMs, max(timing.maxTotalMs, total))
        }
    }

    private func handle(alert e: AlertEvent, date: Date) {
        let cls = class_(e.classId)
        print(String(format: "SW_ALERT t_audio=%.1f %@ conf=%.3f%@", e.tAudio, cls.key, e.confidence,
                     e.repeatAlert ? " repeat" : ""))
        let shown = ShownAlert(classId: e.classId, confidence: e.confidence, date: date, tAudio: e.tAudio)
        let background = !isActive
        if cls.level == .danger && !e.repeatAlert {
            haptics.startContinuous(classId: e.classId, pattern: cls.vibration)  // until ack
        } else if !(cls.level == .danger && haptics.isVibrationHandled(e.classId)) {
            haptics.playOnce(pattern: cls.vibration, level: cls.level)
        }
        if background {
            // after the vibration limit, DANGER repeats update the notification silently
            let silent = cls.level == .danger && e.repeatAlert && !haptics.isLooping(e.classId)
            notifier.post(classId: e.classId, cls: cls, confidence: e.confidence, at: date,
                          repeatAlert: e.repeatAlert, silent: silent)
        }
        if !e.repeatAlert {
            history.insert(shown, at: 0)
            if history.count > config.ui.historyLen { history.removeLast(history.count - config.ui.historyLen) }
        }
        // A pending DANGER alert is not covered by a lower-level one.
        if let cur = currentAlert, class_(cur.classId).level == .danger, cls.level < .danger { return }
        if e.repeatAlert, currentAlert?.classId == e.classId { return }
        show(shown)
    }

    private func show(_ a: ShownAlert) {
        currentAlert = a
        dismissTask?.cancel()
        guard class_(a.classId).level != .danger else { return }
        let timeout = config.ui.alertTimeoutS
        dismissTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(timeout))
            guard !Task.isCancelled else { return }
            if self?.currentAlert?.id == a.id { self?.currentAlert = nil }
        }
    }

    /// Tap on the alert: DANGER -> ack (stops repeats and vibration), others -> dismiss.
    func acknowledgeCurrent() {
        guard let a = currentAlert else { return }
        acknowledge(classId: a.classId)
    }

    /// A DANGER ack (in the app or "Xác nhận" on a notification) also acks every other DANGER class
    /// that is still vibrating: the face shows only the newest alert, and one "I know" should stop
    /// the phone vibrating instead of leaving an older, hidden alert running.
    func acknowledge(classId: Int) {
        if class_(classId).level == .danger {
            for c in haptics.loopingClassIds where c != classId { ack(classId: c) }
        }
        ack(classId: classId)
    }

    func ack(classId: Int) {
        if class_(classId).level == .danger { engine?.ack(classId) }
        haptics.stopContinuous(classId: classId)
        notifier.clear(classKey: class_(classId).key)
        if currentAlert?.classId == classId {
            dismissTask?.cancel()
            currentAlert = nil
        }
    }

    private func reopen(classId: Int) {
        if let a = history.first(where: { $0.classId == classId }), currentAlert == nil { show(a) }
    }

    func clearHistory() { history.removeAll() }

    // MARK: settings

    func setEnabled(_ classId: Int, _ on: Bool) {
        enabled[classId] = on
        engine?.setEnabled(classId, on)
        UserDefaults.standard.set(enabled, forKey: Self.enabledKey)
        if !on {
            haptics.stopContinuous(classId: classId)
            if currentAlert?.classId == classId { currentAlert = nil }
        }
    }
}
