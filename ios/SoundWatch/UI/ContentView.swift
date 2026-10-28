import SwiftUI

extension SoundClass {
    var uiColor: Color {
        Color(red: Double(color[0]) / 255, green: Double(color[1]) / 255, blue: Double(color[2]) / 255)
    }
}

struct ContentView: View {
    @EnvironmentObject var state: AppState
    @State private var showHistory = false
    @State private var showSettings = false

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                WatchFace()
                    .frame(width: 320, height: 320)
                    .padding(.top, 24)
                if state.micDenied {
                    Label("Chưa có quyền dùng micro. Vào Cài đặt > SoundWatch để bật.", systemImage: "mic.slash")
                        .font(.footnote).foregroundStyle(.red).multilineTextAlignment(.center)
                }
                if let err = state.errorMessage {
                    Text(err).font(.footnote).foregroundStyle(.red).multilineTextAlignment(.center)
                }
                HStack(spacing: 12) {
                    BarButton(title: "Lịch sử", icon: "clock.arrow.circlepath") { showHistory = true }
                    BarButton(title: "Cài đặt", icon: "slider.horizontal.3") { showSettings = true }
                    BarButton(title: state.showDebug ? "Ẩn debug" : "Debug", icon: "waveform.path.ecg") {
                        withAnimation { state.showDebug.toggle() }
                    }
                }
                if state.showDebug { DebugPanel() }
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 24)
        }
        .background(Color.black.ignoresSafeArea())
        .preferredColorScheme(.dark)
        .sheet(isPresented: $showHistory) { HistoryView() }
        .sheet(isPresented: $showSettings) { SettingsView() }
    }
}

private struct BarButton: View {
    let title: String
    let icon: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: icon).font(.title3)
                Text(title).font(.caption)
            }
            .frame(maxWidth: .infinity, minHeight: 56)
        }
        .buttonStyle(.bordered)
    }
}

// MARK: watch face

struct WatchFace: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        ZStack {
            if let a = state.currentAlert {
                AlertFace(alert: a, cls: state.class_(a.classId)) { state.acknowledgeCurrent() }
                    .transition(.scale.combined(with: .opacity))
            } else {
                IdleFace()
                    .transition(.opacity)
            }
        }
        .animation(.easeOut(duration: 0.2), value: state.currentAlert)
    }
}

private struct IdleFace: View {
    @EnvironmentObject var state: AppState

    private var level: Double {
        guard let s = state.status else { return 0 }
        return min(1, max(0, (s.rmsDbfs + 80) / 80))  // -80..0 dBFS
    }

    private var caption: String {
        switch state.source {
        case .mic: "Đang nghe…"
        case .demo: "Đang phát file demo…"
        case .idle: state.micDenied ? "Micro đang tắt" : "Đang khởi động…"
        }
    }

    var body: some View {
        ZStack {
            Circle().fill(Color(white: 0.08))
            Circle().stroke(Color(white: 0.2), lineWidth: 10).padding(10)
            Circle()
                .trim(from: 0, to: level)
                .stroke(level > 0.7 ? Color.orange : Color.green, style: StrokeStyle(lineWidth: 10, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .padding(10)
                .animation(.linear(duration: 0.3), value: level)
            VStack(spacing: 8) {
                Image(systemName: "ear").font(.system(size: 34)).foregroundStyle(.secondary)
                TimelineView(.periodic(from: .now, by: 1)) { ctx in
                    Text(ctx.date, format: .dateTime.hour().minute())
                        .font(.system(size: 64, weight: .semibold, design: .rounded))
                        .monospacedDigit()
                }
                Text(caption).font(.title3).foregroundStyle(.secondary)
            }
        }
    }
}

private struct AlertFace: View {
    let alert: ShownAlert
    let cls: SoundClass
    let onTap: () -> Void

    var body: some View {
        TimelineView(.periodic(from: .now, by: 0.25)) { ctx in
            // DANGER blinks at 2 Hz: 0.25 s bright, 0.25 s dim
            let dim = cls.level == .danger && Int(ctx.date.timeIntervalSinceReferenceDate * 4) % 2 == 1
            ZStack {
                Circle().fill(cls.uiColor.opacity(dim ? 0.45 : 1))
                VStack(spacing: 6) {
                    Image(systemName: cls.sfSymbol)
                        .font(.system(size: 84, weight: .bold))
                        .frame(height: 100)
                    Text(cls.labelVi)
                        .font(.system(size: 30, weight: .bold, design: .rounded))
                        .multilineTextAlignment(.center)
                        .minimumScaleFactor(0.6)
                        .lineLimit(2)
                    Text("\(Int((alert.confidence * 100).rounded()))% · \(alert.date.formatted(date: .omitted, time: .standard))")
                        .font(.headline).monospacedDigit()
                    if cls.level == .danger {
                        Text("Nhấn để xác nhận")
                            .font(.subheadline.weight(.semibold))
                            .padding(.horizontal, 14).padding(.vertical, 6)
                            .background(.white.opacity(0.25), in: Capsule())
                            .padding(.top, 4)
                    }
                }
                .foregroundStyle(.white)
                .padding(36)
            }
        }
        .contentShape(Circle())
        .onTapGesture(perform: onTap)
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isButton)
    }
}

// MARK: history / settings

struct HistoryView: View {
    @EnvironmentObject var state: AppState
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                if state.history.isEmpty {
                    Text("Chưa có cảnh báo nào").foregroundStyle(.secondary)
                }
                ForEach(state.history) { a in
                    let c = state.class_(a.classId)
                    HStack(spacing: 14) {
                        Image(systemName: c.sfSymbol)
                            .font(.title2).foregroundStyle(.white)
                            .frame(width: 44, height: 44)
                            .background(c.uiColor, in: Circle())
                        VStack(alignment: .leading) {
                            Text(c.labelVi).font(.headline)
                            Text("\(Int((a.confidence * 100).rounded()))%").font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Text(a.date, format: .dateTime.hour().minute().second()).monospacedDigit()
                    }
                }
            }
            .navigationTitle("Lịch sử")
            .toolbar { Button("Xong") { dismiss() } }
        }
    }
}

struct SettingsView: View {
    @EnvironmentObject var state: AppState
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section("Âm thanh cần cảnh báo") {
                    ForEach(0..<(state.config.numClasses - 1), id: \.self) { c in
                        let cls = state.class_(c)
                        Toggle(isOn: Binding(get: { state.enabled[c] }, set: { state.setEnabled(c, $0) })) {
                            Label {
                                Text(cls.labelVi)
                            } icon: {
                                Image(systemName: cls.sfSymbol).foregroundStyle(cls.uiColor)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Cài đặt")
            .toolbar { Button("Xong") { dismiss() } }
        }
    }
}

// MARK: debug

struct DebugPanel: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            let s = state.status
            ForEach(0..<state.config.numClasses, id: \.self) { c in
                let cls = state.class_(c)
                ProbBar(label: cls.key, value: Double(s?.probs[c] ?? 0),
                        threshold: Double(cls.threshold), color: cls.uiColor)
            }
            Divider()
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
                GridRow {
                    Text("RMS"); Text(String(format: "%.1f dBFS", s?.rmsDbfs ?? -120))
                }
                GridRow {
                    Text("Gated"); Text((s?.gated ?? true) ? "có (dưới \(Int(state.config.audio.gateDbfs)) dBFS)" : "không")
                }
                GridRow {
                    Text("Feature"); Text(String(format: "%.2f ms", state.timing.featureMs))
                }
                GridRow {
                    Text("Suy luận"); Text(String(format: "%.2f ms", state.timing.inferMs))
                }
                GridRow {
                    Text("Tổng max"); Text(String(format: "%.2f ms / hop", state.timing.maxTotalMs))
                }
                GridRow {
                    Text("t_audio"); Text(String(format: "%.1f s", s?.tAudio ?? 0))
                }
            }
            .font(.caption.monospacedDigit())
            Divider()
            Toggle("Phát ra loa khi chạy demo", isOn: $state.playDemoAloud)
                .font(.footnote)
                .disabled(state.source == .demo)
            Button {
                state.toggleDemo()
            } label: {
                Label(state.source == .demo ? "Dừng file demo" : "Phát file demo",
                      systemImage: state.source == .demo ? "stop.fill" : "play.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
        }
        .padding(14)
        .background(Color(white: 0.1), in: RoundedRectangle(cornerRadius: 14))
    }
}

private struct ProbBar: View {
    let label: String
    let value: Double
    let threshold: Double
    let color: Color

    var body: some View {
        HStack(spacing: 8) {
            Text(label).font(.caption2.monospaced()).frame(width: 96, alignment: .leading)
            GeometryReader { g in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color(white: 0.2))
                    Capsule().fill(value >= threshold ? color : color.opacity(0.5))
                        .frame(width: g.size.width * min(1, value))
                    if threshold <= 1 {
                        Rectangle().fill(.white).frame(width: 2)
                            .offset(x: g.size.width * threshold - 1)
                    }
                }
            }
            .frame(height: 10)
            Text(String(format: "%.2f", value)).font(.caption2.monospacedDigit()).frame(width: 34, alignment: .trailing)
        }
    }
}
