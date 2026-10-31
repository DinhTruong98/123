import AVFoundation
import os
import SwiftUI
import UserNotifications

struct ShownAlert: Identifiable, Equatable {
    enum Verdict: String { case correct, incorrect }

    let id = UUID()
    let classId: Int
    let confidence: Float
    let date: Date
    let tAudio: Double
    var model = 0                 // 0 = A (Hiện tại), 1 = B (VN v2)
    var level: Level? = nil       // effective level (context mode); nil = configured level
    var mode: String? = nil       // context mode of the model when it alerted
    var verdict: Verdict? = nil   // user feedback (Đúng / Sai)
    var disagree = false          // the other model did not report this class within ±2 s
}

/// One model shown in the app.
struct ModelInfo {
    let short: String  // "A" / "B"
    let name: String   // "Hiện tại" / "VN v2"
    let config: SoundWatchConfig
}

@MainActor
final class AppState: ObservableObject {
    enum Source: Equatable { case idle, mic, demo }

    static let agreeWindowS = 2.0

    @Published private(set) var status: StatusUpdate?
    @Published private(set) var currentAlerts: [ShownAlert?] = []
    @Published private(set) var history: [ShownAlert] = []
    @Published private(set) var source: Source = .idle
    @Published private(set) var errorMessage: String?
    @Published private(set) var micDenied = false
    @Published private(set) var timing = (featureMs: 0.0, inferMs: [Double](), maxTotalMs: 0.0)
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
    /// The one model whose alerts vibrate and post notifications (default B, so no double vibration).
    @Published var alertModel: Int {
        didSet {
            UserDefaults.standard.set(alertModel, forKey: "alertModel")
            haptics.stopAll()
        }
    }
    /// Context mode requested for B: indoor / street / auto.
    @Published var modeB = UserDefaults.standard.string(forKey: "modeB") ?? "indoor" {
        didSet {
            UserDefaults.standard.set(modeB, forKey: "modeB")
            if models.count > 1 { engine?.setMode(modeB, model: 1) }
            session.log(["type": "mode_request", "model": "B", "mode": modeB])
        }
    }
    @Published var saveDisagreementAudio = UserDefaults.standard.bool(forKey: "saveDisagreementAudio") {
        didSet { UserDefaults.standard.set(saveDisagreementAudio, forKey: "saveDisagreementAudio") }
    }
    @Published private(set) var sessionActive = false
    @Published private(set) var sessionStart: Date?
    @Published private(set) var truths: [(classId: Int, date: Date)] = []

    let models: [ModelInfo]
    var config: SoundWatchConfig { models[0].config }
    private let engine: Engine?
    private let capture: AudioCapture
    private let demo = DemoFilePlayer()
    private let haptics = Haptics()
    private let notifier = Notifier()
    let session = SessionRecorder()
    private let audioHistory: AudioHistory
    private var sessionAlerts: [ShownAlert] = []  // every fresh alert of the running session
    private var dismissTasks: [Task<Void, Never>?] = []
    private var isActive = true
    private var rmsMinute: (minute: Int, sum: Double, n: Int)?
    private var timingAcc: (n: Int, feature: Double, infer: [Double], maxTotal: Double) = (0, 0, [], 0)
    private static let enabledKey = "enabledClasses"
    private let micFeeds = OSAllocatedUnfairLock(initialState: true)

    static var isRunningTests: Bool { ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil }

    init() {
        // A missing/corrupt bundled resource is a build error, so crash loudly.
        let cfgA = try! SoundWatchConfig.load(from: Resources.require("soundwatch_config", "json"))
        var infos = [ModelInfo(short: "A", name: "Hiện tại", config: cfgA)]
        var pipelines: [ModelPipeline] = []
        var err: String?
        do {
            let clf = try TFLiteClassifier(modelURL: Resources.require("student_int8", "tflite"))
            pipelines.append(ModelPipeline(name: "A", config: cfgA, classifier: clf))
            // model B (VN v2) runs next to A on the same audio, if bundled
            if let u = Resources.url("soundwatch_config_vn2", "json"), let m = Resources.url("student_int8_vn2", "tflite") {
                let cfgB = try SoundWatchConfig.load(from: u)
                precondition(cfgB.classes.map(\.key) == cfgA.classes.map(\.key), "models must share the classes")
                pipelines.append(ModelPipeline(name: "B", config: cfgB, classifier: try TFLiteClassifier(modelURL: m)))
                infos.append(ModelInfo(short: "B", name: "VN v2", config: cfgB))
            }
        } catch {
            err = "Không nạp được model: \(error.localizedDescription)"
        }
        models = infos
        engine = pipelines.isEmpty ? nil : Engine(pipelines: pipelines)
        errorMessage = err
        alertModel = min(UserDefaults.standard.object(forKey: "alertModel") as? Int ?? 1, infos.count - 1)
        currentAlerts = Array(repeating: nil, count: infos.count)
        dismissTasks = Array(repeating: nil, count: infos.count)
        capture = AudioCapture(sampleRate: cfgA.audio.sampleRate)
        audioHistory = AudioHistory(capacity: 15 * cfgA.audio.sampleRate)
        let saved = UserDefaults.standard.array(forKey: Self.enabledKey) as? [Bool]
        enabled = saved?.count == cfgA.numClasses ? saved! : cfgA.classes.map(\.enabled)
        for (c, on) in enabled.enumerated() { engine?.setEnabled(c, on) }
        if infos.count > 1 { engine?.setMode(modeB, model: 1) }

        engine?.onStatus = { [weak self] s in Task { @MainActor in self?.handle(status: s) } }
        engine?.onAlert = { [weak self] m, e in
            let date = Date()
            Task { @MainActor in self?.handle(alert: e, model: m, date: date) }
        }
        // The microphone keeps running while the demo plays (its samples are dropped), so switching
        // sources never restarts recording, which iOS refuses while the app is in the background.
        let micFeeds = micFeeds
        let audioHistory = audioHistory
        capture.onSamples = { [weak self] x in
            if micFeeds.withLock({ $0 }) {
                audioHistory.append(x)
                self?.engine?.feed(x)
            }
        }
        demo.onSamples = { [weak self] x in
            audioHistory.append(x)
            self?.engine?.feed(x)
        }
        demo.onFinished = { [weak self] in Task { @MainActor in self?.demoFinished() } }
        haptics.log = { print("SW_HAPTIC \($0)") }
        haptics.continuousLimit = dangerVibrationLimitS > 0 ? Double(dangerVibrationLimitS) : nil
        notifier.onAck = { [weak self] c in Task { @MainActor in self?.acknowledge(classId: c) } }
        notifier.onOpen = { [weak self] c in Task { @MainActor in self?.reopen(classId: c) } }
    }

    func class_(_ id: Int) -> SoundClass { config.classes[id] }

    /// The class as shown for an alert: its effective level (context mode) decides colour and wording.
    func displayClass(_ a: ShownAlert) -> SoundClass {
        let c = class_(a.classId)
        guard let l = a.level, l != c.level else { return c }
        return c.with(level: l)
    }

    /// A's current alert (main "Nghe" face, unchanged single-model view).
    var currentAlert: ShownAlert? { currentAlerts.first ?? nil }

    /// Active context mode of a model (nil = no modes).
    func activeMode(_ model: Int) -> String? {
        guard let s = status, model < s.models.count else { return nil }
        return s.models[model].mode
    }

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
        if args.contains("-session") { startSession() }
        if args.contains("-debug") { showDebug = true }
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
        timing = (0, [], 0)
        timingAcc = (0, 0, Array(repeating: 0, count: models.count), 0)
        do {
            try demo.start(url: Resources.require("demo_scenario", "wav"), sampleRate: config.audio.sampleRate,
                           playAloud: playDemoAloud)
            source = .demo
            session.log(["type": "source", "source": "demo"])
        } catch {
            errorMessage = "Không phát được file demo: \(error.localizedDescription)"
            startMic()
        }
    }

    private func demoFinished() {
        let a = timingAcc
        if a.n > 0 {
            let infer = a.infer.enumerated().map { String(format: "%@ %.2f", models[$0.offset].short, $0.element / Double(a.n)) }
            print(String(format: "SW_TIMING hops=%d feature %.2f ms, infer %@ ms, total mean %.2f ms, max %.2f ms", a.n,
                         a.feature / Double(a.n), infer.joined(separator: " / "),
                         (a.feature + a.infer.reduce(0, +)) / Double(a.n), a.maxTotal))
        }
        source = .idle
        session.log(["type": "source", "source": "mic"])
        startMic()
    }

    // MARK: engine events

    private func handle(status s: StatusUpdate) {
        status = s
        if !s.gated {
            let infer = s.models.map(\.inferMs)
            let total = s.featureMs + infer.reduce(0, +)
            timing = (s.featureMs, infer, max(timing.maxTotalMs, total))
            if timingAcc.infer.count == infer.count {
                timingAcc = (timingAcc.n + 1, timingAcc.feature + s.featureMs, zip(timingAcc.infer, infer).map(+),
                             max(timingAcc.maxTotal, total))
            }
        }
        guard sessionActive else { return }
        let minute = Int(Date().timeIntervalSince1970 / 60)
        if let m = rmsMinute, m.minute != minute {
            session.log(["type": "rms_minute", "minute": Date(timeIntervalSince1970: Double(m.minute) * 60).iso,
                         "mean_dbfs": m.sum / Double(max(1, m.n)), "hops": m.n])
            rmsMinute = nil
        }
        let m = rmsMinute ?? (minute, 0, 0)
        rmsMinute = (minute, m.sum + s.rmsDbfs, m.n + 1)
    }

    private func handle(alert e: AlertEvent, model m: Int, date: Date) {
        let cls = class_(e.classId)
        let level = e.level ?? models[m].config.classes[e.classId].level
        let mode = activeMode(m)
        print(String(format: "SW_ALERT model=%@ t_audio=%.1f %@ level=%d conf=%.3f%@", models[m].short, e.tAudio,
                     cls.key, level.rawValue, e.confidence, e.repeatAlert ? " repeat" : ""))
        var shown = ShownAlert(classId: e.classId, confidence: e.confidence, date: date, tAudio: e.tAudio,
                               model: m, level: e.level, mode: mode)
        if !e.repeatAlert {
            history.insert(shown, at: 0)
            if history.count > 2 * config.ui.historyLen { history.removeLast(history.count - 2 * config.ui.historyLen) }
            if sessionActive {
                sessionAlerts.append(shown)
                session.log(["type": "alert", "id": shown.id.uuidString, "model": models[m].short,
                             "class": cls.key, "level": level.rawValue, "confidence": Double(e.confidence),
                             "t_audio": e.tAudio, "mode": mode ?? "", "source": source == .demo ? "demo" : "mic"])
            }
            if models.count > 1 { checkAgreement(shown) }
        }
        if m == alertModel { vibrateAndNotify(e, level: level, date: date) }
        // A pending DANGER alert is not covered by a lower-level one.
        if let cur = currentAlerts[m], effectiveLevel(cur) == .danger, level < .danger { return }
        // a repeat only brings an alert back when the face is idle (the newest alert stays in front)
        if e.repeatAlert, currentAlerts[m] != nil { return }
        shown.verdict = nil
        show(shown, model: m)
    }

    func effectiveLevel(_ a: ShownAlert) -> Level { a.level ?? models[a.model].config.classes[a.classId].level }

    private func vibrateAndNotify(_ e: AlertEvent, level: Level, date: Date) {
        let cls = class_(e.classId)
        if level == .danger && !e.repeatAlert {
            haptics.startContinuous(classId: e.classId, pattern: cls.vibration)  // until ack / limit
        } else if !(level == .danger && haptics.isVibrationHandled(e.classId)) {
            haptics.playOnce(pattern: cls.vibration, level: level)
        }
        if !isActive {
            // after the vibration limit, DANGER repeats update the notification silently
            let silent = level == .danger && e.repeatAlert && !haptics.isLooping(e.classId)
            notifier.post(classId: e.classId, cls: level == cls.level ? cls : cls.with(level: level),
                          confidence: e.confidence, at: date, repeatAlert: e.repeatAlert, silent: silent)
        }
    }

    private func show(_ a: ShownAlert, model m: Int) {
        currentAlerts[m] = a
        dismissTasks[m]?.cancel()
        guard effectiveLevel(a) != .danger else { return }
        let timeout = config.ui.alertTimeoutS
        dismissTasks[m] = Task { [weak self] in
            try? await Task.sleep(for: .seconds(timeout))
            guard !Task.isCancelled else { return }
            if self?.currentAlerts[m]?.id == a.id { self?.currentAlerts[m] = nil }
        }
    }

    /// Two models agree on an alert if the other one reported the same class within ±2 s. Checked
    /// 2 s after the alert; a disagreement is marked on the alert, logged, and (if enabled) the last
    /// 10 s of audio are saved with the session.
    private func checkAgreement(_ a: ShownAlert) {
        Task { [weak self] in
            try? await Task.sleep(for: .seconds(Self.agreeWindowS))
            guard let self else { return }
            let other = self.history.contains {
                $0.model != a.model && $0.classId == a.classId && abs($0.date.timeIntervalSince(a.date)) <= Self.agreeWindowS
            }
            guard !other else { return }
            self.mark(a.id) { $0.disagree = true }
            guard self.sessionActive else { return }
            var rec: [String: Any] = ["type": "disagreement", "alert_id": a.id.uuidString,
                                      "model": self.models[a.model].short, "class": self.class_(a.classId).key]
            if self.saveDisagreementAudio, let name = self.session.saveWav(
                self.audioHistory.last(10 * self.config.audio.sampleRate), sampleRate: self.config.audio.sampleRate,
                name: "disagree-\(self.models[a.model].short)-\(self.class_(a.classId).key)") {
                rec["audio"] = name
            }
            self.session.log(rec)
        }
    }

    private func mark(_ id: UUID, _ f: (inout ShownAlert) -> Void) {
        if let i = history.firstIndex(where: { $0.id == id }) { f(&history[i]) }
        if let i = sessionAlerts.firstIndex(where: { $0.id == id }) { f(&sessionAlerts[i]) }
        for m in currentAlerts.indices where currentAlerts[m]?.id == id { f(&currentAlerts[m]!) }
    }

    /// Tap on A's alert ("Nghe" face): DANGER -> ack (stops repeats and vibration), others -> dismiss.
    func acknowledgeCurrent() {
        guard let a = currentAlert else { return }
        acknowledge(classId: a.classId)
    }

    func acknowledge(model m: Int) {
        guard let a = currentAlerts[m] else { return }
        acknowledge(classId: a.classId)
    }

    /// An ack is about the sound, so it acks the class in both models. A DANGER ack (in the app or
    /// "Xác nhận" on a notification) also acks every other DANGER class that is still vibrating: the
    /// face shows only the newest alert, and one "I know" should stop the phone vibrating.
    func acknowledge(classId: Int) {
        for c in haptics.loopingClassIds where c != classId { ack(classId: c) }
        ack(classId: classId)
    }

    func ack(classId: Int) {
        engine?.ack(classId)
        haptics.stopContinuous(classId: classId)
        notifier.clear(classKey: class_(classId).key)
        for m in currentAlerts.indices where currentAlerts[m]?.classId == classId {
            dismissTasks[m]?.cancel()
            currentAlerts[m] = nil
        }
    }

    private func reopen(classId: Int) {
        if let a = history.first(where: { $0.classId == classId && $0.model == alertModel }),
           currentAlerts[alertModel] == nil { show(a, model: alertModel) }
    }

    func clearHistory() { history.removeAll() }

    // MARK: comparison session

    func startSession() {
        guard !sessionActive else { return }
        session.start(models: models.map { "\($0.short) \($0.name)" })
        sessionAlerts = []
        truths = []
        rmsMinute = nil
        sessionStart = Date()
        sessionActive = true
        session.log(["type": "settings", "alert_model": models[alertModel].short, "mode_b": modeB,
                     "save_audio": saveDisagreementAudio])
    }

    func stopSession() {
        guard sessionActive else { return }
        session.stop()
        sessionActive = false
    }

    /// "Có âm thật": the user heard / saw this sound now (ground truth).
    func markTruth(classId: Int) {
        truths.append((classId, Date()))
        session.log(["type": "truth", "class": class_(classId).key])
    }

    func setVerdict(_ v: ShownAlert.Verdict, for id: UUID) {
        mark(id) { $0.verdict = v }
        session.log(["type": "feedback", "alert_id": id.uuidString, "verdict": v.rawValue])
    }

    struct ModelStats {
        var alerts = 0, correct = 0, incorrect = 0, disagree = 0, truthsCaught = 0
        var incorrectPerHour = 0.0
    }

    /// Per model: fresh alerts, Đúng / Sai, disagreements, "Có âm thật" events caught (an alert of the
    /// same class within [t - 2 s, t + 5 s]), and false alarms (Sai) per hour of session.
    func sessionStats(now: Date = Date()) -> [ModelStats] {
        let hours = max(1e-9, now.timeIntervalSince(sessionStart ?? now) / 3600)
        return models.indices.map { m in
            var s = ModelStats()
            let mine = sessionAlerts.filter { $0.model == m }
            s.alerts = mine.count
            s.correct = mine.filter { $0.verdict == .correct }.count
            s.incorrect = mine.filter { $0.verdict == .incorrect }.count
            s.disagree = mine.filter(\.disagree).count
            s.truthsCaught = truths.filter { t in
                mine.contains { $0.classId == t.classId && $0.date.timeIntervalSince(t.date) >= -2
                    && $0.date.timeIntervalSince(t.date) <= 5 }
            }.count
            s.incorrectPerHour = Double(s.incorrect) / hours
            return s
        }
    }

    /// Files to share: the JSONL log, a CSV summary and saved audio clips.
    func exportFiles() -> [URL] {
        let stats = sessionStats()
        var csv = "model,alerts,correct,incorrect,disagree,truths,truths_caught,incorrect_per_hour\n"
        for (m, s) in stats.enumerated() {
            csv += "\(models[m].short) \(models[m].name),\(s.alerts),\(s.correct),\(s.incorrect),\(s.disagree),"
            csv += "\(truths.count),\(s.truthsCaught),\(String(format: "%.2f", s.incorrectPerHour))\n"
        }
        return session.files(summaryCSV: csv)
    }

    // MARK: settings

    func setEnabled(_ classId: Int, _ on: Bool) {
        enabled[classId] = on
        engine?.setEnabled(classId, on)
        UserDefaults.standard.set(enabled, forKey: Self.enabledKey)
        if !on {
            haptics.stopContinuous(classId: classId)
            for m in currentAlerts.indices where currentAlerts[m]?.classId == classId { currentAlerts[m] = nil }
        }
    }
}

extension SoundClass {
    /// The same class shown at another level (context mode), with that level's standard colour.
    func with(level l: Level) -> SoundClass {
        let color: [Int] = switch l {
        case .danger: [230, 40, 40]
        case .warn: [255, 140, 0]
        case .info: [40, 140, 255]
        }
        return SoundClass(key: key, labelVi: labelVi, level: l, color: color, vibration: vibration,
                          threshold: threshold, cooldownS: cooldownS, enabled: enabled, sfSymbol: sfSymbol)
    }
}

extension Date {
    var iso: String { ISO8601DateFormatter.withFraction.string(from: self) }
}

extension ISO8601DateFormatter {
    static let withFraction: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()
}

/// Last N samples fed to the engine (thread-safe), for saving audio around a disagreement.
final class AudioHistory: @unchecked Sendable {
    private let lock = OSAllocatedUnfairLock(initialState: ())
    private var rb: RingBuffer

    init(capacity: Int) { rb = RingBuffer(capacity: capacity) }

    func append(_ x: [Float]) { lock.withLock { _ in rb.write(x) } }

    func last(_ n: Int) -> [Float] {
        lock.withLock { _ in
            var out: [Float] = []
            rb.latest(into: &out)
            return Array(out.suffix(min(n, rb.totalWritten)))
        }
    }
}
