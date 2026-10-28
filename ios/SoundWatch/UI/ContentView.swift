import SwiftUI

struct ContentView: View {
    @EnvironmentObject var state: AppState
    @State private var tab = Self.launchTab ?? .listen

    enum Tab: String, Hashable { case listen, history, settings }

    /// Launch argument `-tab history|settings` (screenshots): open that tab and stay there.
    private static let launchTab: Tab? = {
        let args = ProcessInfo.processInfo.arguments
        guard let i = args.firstIndex(of: "-tab"), i + 1 < args.count else { return nil }
        return Tab(rawValue: args[i + 1])
    }()

    var body: some View {
        TabView(selection: $tab) {
            ListenView()
                .tabItem { Label("Nghe", systemImage: "waveform.circle.fill") }
                .tag(Tab.listen)
            HistoryView()
                .tabItem { Label("Lịch sử", systemImage: "clock.arrow.circlepath") }
                .tag(Tab.history)
            SettingsView(onPlayDemo: { tab = .listen })
                .tabItem { Label("Cài đặt", systemImage: "gearshape.fill") }
                .tag(Tab.settings)
        }
        // a new alert always brings the watch face to the front
        .onChange(of: state.currentAlert) { _, a in
            if a != nil, Self.launchTab == nil { withAnimation { tab = .listen } }
        }
    }
}

// MARK: listen tab

struct ListenView: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 22) {
                    StatusChip()
                    WatchFace()
                        .containerRelativeFrame(.horizontal) { w, _ in min(w - 32, 380) }
                        .aspectRatio(1, contentMode: .fit)
                    if state.micDenied {
                        Notice(icon: "mic.slash.fill",
                               text: "Chưa có quyền dùng micro. Vào Cài đặt > SoundWatch > Micro để bật.")
                    }
                    if let err = state.errorMessage {
                        Notice(icon: "exclamationmark.octagon.fill", text: err)
                    }
                    RecentAlerts()
                    if state.showDebug { DebugPanel() }
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 24)
            }
            .background(PageBackground())
            .navigationTitle("SoundWatch")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

/// Grouped background; flashes in the alert colour while a DANGER alert waits for ack, so the alert
/// is visible from across the room.
private struct PageBackground: View {
    @EnvironmentObject var state: AppState
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        if let a = state.currentAlert, state.class_(a.classId).level == .danger {
            let c = state.class_(a.classId).alertColor
            TimelineView(.periodic(from: .now, by: 0.25)) { ctx in
                let on = reduceMotion || Int(ctx.date.timeIntervalSinceReferenceDate * 4) % 2 == 0
                c.opacity(on ? 0.35 : 0.12).ignoresSafeArea()
            }
        } else {
            Color(.systemGroupedBackground).ignoresSafeArea()
        }
    }
}

private struct StatusChip: View {
    @EnvironmentObject var state: AppState

    private var info: (String, String, Color) {
        switch state.source {
        case .mic: ("Đang nghe", "mic.fill", .green)
        case .demo: ("Đang phát file demo", "play.circle.fill", .blue)
        case .idle: state.micDenied ? ("Micro đang tắt", "mic.slash.fill", .red) : ("Đang khởi động…", "hourglass", .secondary)
        }
    }

    var body: some View {
        let (text, icon, color) = info
        HStack(spacing: 8) {
            Image(systemName: icon)
                .foregroundStyle(color)
                .symbolEffect(.pulse, options: .repeating, isActive: state.source != .idle)
            Text(text).font(.subheadline.weight(.semibold))
            if let s = state.status, state.source != .idle {
                Text(String(format: "%.0f dB", s.rmsDbfs))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 8)
        .background(.thinMaterial, in: Capsule())
        .accessibilityElement(children: .combine)
    }
}

private struct Notice: View {
    let icon: String
    let text: String

    var body: some View {
        Label(text, systemImage: icon)
            .font(.callout)
            .foregroundStyle(.red)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(14)
            .background(.red.opacity(0.12), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
}

// MARK: watch face

struct WatchFace: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        ZStack {
            if let a = state.currentAlert {
                AlertFace(alert: a, cls: state.class_(a.classId)) { state.acknowledgeCurrent() }
                    .id(a.id)
                    .transition(.asymmetric(insertion: .scale(scale: 0.6).combined(with: .opacity),
                                            removal: .opacity))
            } else {
                IdleFace()
                    .transition(.opacity)
            }
        }
        .animation(.spring(response: 0.45, dampingFraction: 0.7), value: state.currentAlert)
    }
}

private struct IdleFace: View {
    @EnvironmentObject var state: AppState
    @ScaledMetric(relativeTo: .largeTitle) private var clockSize: CGFloat = 68

    private var level: Double {
        guard let s = state.status, state.source != .idle else { return 0 }
        return min(1, max(0.02, (s.rmsDbfs + 80) / 80))  // -80..0 dBFS
    }

    private var caption: String {
        switch state.source {
        case .mic: "Đang nghe…"
        case .demo: "Đang phát demo…"
        case .idle: state.micDenied ? "Micro đang tắt" : "Đang khởi động…"
        }
    }

    var body: some View {
        ZStack {
            Circle().fill(.regularMaterial)
            Circle().stroke(Color.primary.opacity(0.08), lineWidth: 16).padding(16)
            Circle()
                .trim(from: 0, to: level)
                .stroke(AngularGradient(colors: [.green, .mint, .yellow, .orange, .red],
                                        center: .center, startAngle: .degrees(0), endAngle: .degrees(360)),
                        style: StrokeStyle(lineWidth: 16, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .padding(16)
                .animation(.smooth(duration: 0.45), value: level)
            VStack(spacing: 6) {
                Image(systemName: "waveform")
                    .font(.title.weight(.semibold))
                    .foregroundStyle(.tint)
                    .symbolEffect(.variableColor.iterative.dimInactiveLayers, options: .repeating,
                                  isActive: state.source != .idle)
                TimelineView(.everyMinute) { ctx in
                    Text(ctx.date, format: .dateTime.hour().minute())
                        .font(.system(size: clockSize, weight: .bold, design: .rounded))
                        .monospacedDigit()
                        .minimumScaleFactor(0.5)
                        .lineLimit(1)
                }
                Text(caption)
                    .font(.headline)
                    .foregroundStyle(.secondary)
            }
            .padding(40)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("SoundWatch \(caption)")
    }
}

private struct AlertFace: View {
    let alert: ShownAlert
    let cls: SoundClass
    let onTap: () -> Void
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @ScaledMetric(relativeTo: .largeTitle) private var iconSize: CGFloat = 80

    private var danger: Bool { cls.level == .danger }

    var body: some View {
        TimelineView(.periodic(from: .now, by: 0.25)) { ctx in
            // DANGER blinks at 2 Hz (0.25 s bright, 0.25 s dim); with Reduce Motion it stays bright
            let dim = danger && !reduceMotion && Int(ctx.date.timeIntervalSinceReferenceDate * 4) % 2 == 1
            ZStack {
                Circle()
                    .fill(cls.alertColor.gradient)
                    .opacity(dim ? 0.7 : 1)
                    .shadow(color: cls.alertColor.opacity(0.5), radius: dim ? 8 : 24)
                if danger {
                    Circle().strokeBorder(cls.alertForeground.opacity(dim ? 0.25 : 0.9), lineWidth: 8).padding(8)
                }
                VStack(spacing: 8) {
                    Image(systemName: cls.sfSymbol)
                        .font(.system(size: iconSize, weight: .bold))
                        .symbolEffect(.bounce, options: .repeat(2), value: alert.id)
                    Text(cls.labelVi)
                        .font(.system(.largeTitle, design: .rounded, weight: .heavy))
                        .multilineTextAlignment(.center)
                        .minimumScaleFactor(0.5)
                        .lineLimit(2)
                    Label(cls.level.name, systemImage: cls.level.symbol)
                        .font(.headline)
                    Text("\(Int((alert.confidence * 100).rounded()))% · \(alert.date.formatted(date: .omitted, time: .standard))")
                        .font(.subheadline.weight(.medium).monospacedDigit())
                        .opacity(0.9)
                    if danger {
                        Text("Nhấn để xác nhận")
                            .font(.headline)
                            .foregroundStyle(cls.alertColor)
                            .padding(.horizontal, 18).padding(.vertical, 9)
                            .background(cls.alertForeground, in: Capsule())
                            .padding(.top, 4)
                    } else {
                        Text("Nhấn để đóng").font(.footnote).opacity(0.8)
                    }
                }
                .foregroundStyle(cls.alertForeground)
                .padding(36)
            }
        }
        .contentShape(Circle())
        .onTapGesture(perform: onTap)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(cls.level.name): \(cls.labelVi), độ tin cậy \(Int((alert.confidence * 100).rounded())) phần trăm")
        .accessibilityHint(danger ? "Nhấn đúp để xác nhận" : "Nhấn đúp để đóng")
        .accessibilityAddTraits(.isButton)
    }
}

// MARK: recent alerts / rows

private struct RecentAlerts: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Gần đây").font(.title3.weight(.bold))
            if state.history.isEmpty {
                Label("Chưa có cảnh báo nào", systemImage: "bell.slash")
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(16)
                    .background(.background, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            } else {
                VStack(spacing: 0) {
                    ForEach(Array(state.history.prefix(3).enumerated()), id: \.element.id) { i, a in
                        if i > 0 { Divider().padding(.leading, 70) }
                        AlertRow(alert: a, cls: state.class_(a.classId)).padding(.horizontal, 14).padding(.vertical, 10)
                    }
                }
                .background(.background, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            }
        }
    }
}

struct AlertRow: View {
    let alert: ShownAlert
    let cls: SoundClass

    var body: some View {
        HStack(spacing: 14) {
            Image(systemName: cls.sfSymbol)
                .font(.title3.weight(.semibold))
                .foregroundStyle(cls.alertForeground)
                .frame(width: 44, height: 44)
                .background(cls.alertColor.gradient, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            VStack(alignment: .leading, spacing: 2) {
                Text(cls.labelVi).font(.headline)
                Text("\(cls.level.name) · \(Int((alert.confidence * 100).rounded()))%")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 8)
            VStack(alignment: .trailing, spacing: 2) {
                Text(alert.date, format: .dateTime.hour().minute().second())
                    .font(.subheadline.monospacedDigit())
                Text(alert.date, style: .relative)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .accessibilityElement(children: .combine)
    }
}

// MARK: history

struct HistoryView: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        NavigationStack {
            Group {
                if state.history.isEmpty {
                    ContentUnavailableView("Chưa có cảnh báo", systemImage: "bell.slash",
                                           description: Text("Các âm thanh được phát hiện sẽ hiện ở đây (\(state.config.ui.historyLen) lần gần nhất)."))
                } else {
                    List {
                        Section {
                            ForEach(state.history) { a in AlertRow(alert: a, cls: state.class_(a.classId)) }
                        } footer: {
                            Text("Giữ \(state.config.ui.historyLen) cảnh báo gần nhất. Lần nhắc lại của cảnh báo nguy hiểm không được tính.")
                        }
                    }
                    .listStyle(.insetGrouped)
                }
            }
            .navigationTitle("Lịch sử")
            .toolbar {
                if !state.history.isEmpty {
                    Button("Xoá", role: .destructive) { withAnimation { state.clearHistory() } }
                }
            }
        }
    }
}

// MARK: settings

struct SettingsView: View {
    @EnvironmentObject var state: AppState
    let onPlayDemo: () -> Void

    var body: some View {
        NavigationStack {
            Form {
                ForEach([Level.danger, .warn, .info], id: \.self) { level in
                    Section {
                        ForEach(classes(of: level), id: \.self) { c in
                            let cls = state.class_(c)
                            Toggle(isOn: Binding(get: { state.enabled[c] }, set: { state.setEnabled(c, $0) })) {
                                Label {
                                    Text(cls.labelVi)
                                } icon: {
                                    Image(systemName: cls.sfSymbol)
                                        .foregroundStyle(cls.alertForeground)
                                        .frame(width: 30, height: 30)
                                        .background(cls.alertColor, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                                }
                            }
                        }
                    } header: {
                        Label(level.name, systemImage: level.symbol)
                    } footer: {
                        if level == .info { Text("Âm bị tắt sẽ không bao giờ cảnh báo.") }
                    }
                }

                Section {
                    LabeledContent("Thông báo") {
                        switch state.notificationsAllowed {
                        case true?: Label("Đã bật", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
                        case false?: Label("Đã tắt", systemImage: "xmark.circle.fill").foregroundStyle(.red)
                        case nil: Text("Chưa hỏi").foregroundStyle(.secondary)
                        }
                    }
                    Button("Mở cài đặt thông báo") {
                        if let url = URL(string: UIApplication.openNotificationSettingsURLString) {
                            UIApplication.shared.open(url)
                        }
                    }
                } header: {
                    Text("Khi khoá màn hình")
                } footer: {
                    Text("Khi app chạy nền hoặc khoá màn hình, mỗi cảnh báo được gửi thành thông báo kèm rung. Cảnh báo nguy hiểm có nút Xác nhận để dừng nhắc lại.")
                }

                Section("Demo & debug") {
                    Toggle("Hiện bảng debug", isOn: $state.showDebug)
                    Toggle("Phát demo ra loa", isOn: $state.playDemoAloud)
                        .disabled(state.source == .demo)
                    Button {
                        state.toggleDemo()
                        onPlayDemo()
                    } label: {
                        Label(state.source == .demo ? "Dừng file demo" : "Phát file demo",
                              systemImage: state.source == .demo ? "stop.fill" : "play.fill")
                    }
                }

                Section("Thông tin") {
                    LabeledContent("Model", value: "DS-CNN int8 · 208 KB")
                    LabeledContent("Nhận diện", value: "\(state.config.numClasses - 1) âm + nền")
                    LabeledContent("Cửa sổ", value: "1 s, cập nhật mỗi 0.5 s")
                    LabeledContent("Phiên bản", value: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "–")
                }
            }
            .navigationTitle("Cài đặt")
        }
    }

    private func classes(of level: Level) -> [Int] {
        (0..<(state.config.numClasses - 1)).filter { state.class_($0).level == level }
    }
}

// MARK: debug

struct DebugPanel: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Debug", systemImage: "waveform.path.ecg").font(.headline)
            let s = state.status
            ForEach(0..<state.config.numClasses, id: \.self) { c in
                let cls = state.class_(c)
                ProbBar(label: cls.key, value: Double(s?.probs[c] ?? 0),
                        threshold: Double(cls.threshold), color: cls.uiColor)
            }
            Divider()
            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 4) {
                row("RMS", String(format: "%.1f dBFS", s?.rmsDbfs ?? -120))
                row("Gated", (s?.gated ?? true) ? "có (dưới \(Int(state.config.audio.gateDbfs)) dBFS)" : "không")
                row("Feature", String(format: "%.2f ms", state.timing.featureMs))
                row("Suy luận", String(format: "%.2f ms", state.timing.inferMs))
                row("Tổng max", String(format: "%.2f ms / hop", state.timing.maxTotalMs))
                row("t_audio", String(format: "%.1f s", s?.tAudio ?? 0))
            }
            .font(.footnote.monospacedDigit())
            Button {
                state.toggleDemo()
            } label: {
                Label(state.source == .demo ? "Dừng file demo" : "Phát file demo",
                      systemImage: state.source == .demo ? "stop.fill" : "play.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
        }
        .padding(16)
        .background(.background, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    private func row(_ k: String, _ v: String) -> some View {
        GridRow {
            Text(k).foregroundStyle(.secondary)
            Text(v)
        }
    }
}

private struct ProbBar: View {
    let label: String
    let value: Double
    let threshold: Double
    let color: Color

    var body: some View {
        HStack(spacing: 8) {
            Text(label).font(.caption.monospaced()).frame(width: 104, alignment: .leading).lineLimit(1)
            GeometryReader { g in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.primary.opacity(0.1))
                    Capsule().fill(value >= threshold ? color : color.opacity(0.45))
                        .frame(width: max(0, g.size.width * min(1, value)))
                        .animation(.smooth(duration: 0.3), value: value)
                    if threshold <= 1 {
                        Rectangle().fill(Color.primary).frame(width: 2, height: 14)
                            .offset(x: g.size.width * threshold - 1)
                    }
                }
            }
            .frame(height: 10)
            Text(String(format: "%.2f", value)).font(.caption.monospacedDigit()).frame(width: 36, alignment: .trailing)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(label) \(Int(value * 100)) phần trăm, ngưỡng \(Int(threshold * 100))")
    }
}
