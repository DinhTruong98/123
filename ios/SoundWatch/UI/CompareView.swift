import SwiftUI

/// Side-by-side comparison of model A (Hiện tại) and B (VN v2) on the same microphone: two small
/// faces (yellow border when the other model did not report the same sound within ±2 s), B's
/// context mode, "Có âm thật" ground truth, Đúng/Sai feedback and the session controls.
struct CompareView: View {
    @EnvironmentObject var state: AppState
    @State private var showStats = false
    @State private var truthFlash: String?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 18) {
                    SessionBar(showStats: $showStats)
                    HStack(alignment: .top, spacing: 12) {
                        ForEach(state.models.indices, id: \.self) { m in MiniFace(model: m) }
                    }
                    if state.models.count > 1 { ModePicker() }
                    TruthButton(flash: $truthFlash)
                    if let f = truthFlash {
                        Label("Đã ghi: \(f)", systemImage: "checkmark.circle.fill")
                            .font(.subheadline).foregroundStyle(.green)
                    }
                    CompareRecent()
                    if state.showDebug { DebugPanel() }
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 24)
            }
            .background(Color(.systemGroupedBackground).ignoresSafeArea())
            .navigationTitle("So sánh 2 model")
            .navigationBarTitleDisplayMode(.inline)
            .sheet(isPresented: $showStats) { SessionStatsView() }
        }
    }
}

private struct SessionBar: View {
    @EnvironmentObject var state: AppState
    @Binding var showStats: Bool

    var body: some View {
        HStack(spacing: 10) {
            if state.sessionActive, let t0 = state.sessionStart {
                Image(systemName: "record.circle").foregroundStyle(.red)
                    .symbolEffect(.pulse, options: .repeating)
                Text(t0, style: .timer).font(.headline.monospacedDigit())
                Spacer()
                Button("Thống kê") { showStats = true }.buttonStyle(.bordered)
                Button("Dừng", role: .destructive) { state.stopSession() }.buttonStyle(.borderedProminent)
            } else {
                Image(systemName: "chart.bar.doc.horizontal").foregroundStyle(.secondary)
                Text("Chưa ghi phiên đo").font(.subheadline).foregroundStyle(.secondary)
                Spacer()
                if state.session.dir != nil {
                    Button("Thống kê") { showStats = true }.buttonStyle(.bordered)
                }
                Button("Bắt đầu phiên") { state.startSession() }.buttonStyle(.borderedProminent)
            }
        }
        .padding(12)
        .background(.background, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
    }
}

/// Small watch face of one model.
private struct MiniFace: View {
    @EnvironmentObject var state: AppState
    let model: Int

    var body: some View {
        let info = state.models[model]
        let alert = state.currentAlerts[model]
        VStack(spacing: 8) {
            HStack(spacing: 4) {
                Text(info.short).font(.caption.weight(.heavy))
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background(Color.primary.opacity(0.12), in: Capsule())
                Text(info.name).font(.subheadline.weight(.semibold))
                if model == state.alertModel {
                    Image(systemName: "iphone.radiowaves.left.and.right").font(.caption).foregroundStyle(.tint)
                        .accessibilityLabel("Model này rung và gửi thông báo")
                }
            }
            ZStack {
                if let a = alert {
                    let cls = state.displayClass(a)
                    Circle().fill(cls.alertColor.gradient)
                    VStack(spacing: 4) {
                        Image(systemName: cls.sfSymbol).font(.system(size: 34, weight: .bold))
                        Text(cls.labelVi).font(.headline).multilineTextAlignment(.center)
                            .minimumScaleFactor(0.6).lineLimit(2)
                        Text("\(cls.level.name) · \(Int((a.confidence * 100).rounded()))%")
                            .font(.caption.weight(.medium))
                        Text(a.date, format: .dateTime.hour().minute().second()).font(.caption2.monospacedDigit())
                    }
                    .foregroundStyle(cls.alertForeground)
                    .padding(14)
                } else {
                    Circle().fill(.regularMaterial)
                    VStack(spacing: 4) {
                        Image(systemName: "waveform").font(.title2).foregroundStyle(.tint)
                            .symbolEffect(.variableColor.iterative, options: .repeating, isActive: state.source != .idle)
                        Text("Đang nghe").font(.subheadline).foregroundStyle(.secondary)
                        if let mode = state.activeMode(model) {
                            Text(modeName(mode)).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .aspectRatio(1, contentMode: .fit)
            .overlay(Circle().strokeBorder(.yellow, lineWidth: alert?.disagree == true ? 6 : 0))
            .contentShape(Circle())
            .onTapGesture { state.acknowledge(model: model) }
            .animation(.spring(response: 0.4, dampingFraction: 0.75), value: alert)
            if let s = state.status, model < s.models.count, !s.gated {
                Text(String(format: "%.1f ms", s.models[model].inferMs)).font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity)
        .accessibilityElement(children: .combine)
    }
}

func modeName(_ m: String) -> String {
    switch m {
    case "indoor": "Trong nhà"
    case "street": "Ngoài đường"
    case "auto": "Tự động"
    default: m
    }
}

private struct ModePicker: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Chế độ của B").font(.subheadline.weight(.semibold))
            Picker("Chế độ của B", selection: $state.modeB) {
                Text("Trong nhà").tag("indoor")
                Text("Ngoài đường").tag("street")
                Text("Tự động").tag("auto")
            }
            .pickerStyle(.segmented)
            if state.modeB == "auto", let m = state.activeMode(1) {
                Text("Đang chạy: \(modeName(m))").font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}

/// "Có âm thật": the user marks a real event now (ground truth for the session statistics).
private struct TruthButton: View {
    @EnvironmentObject var state: AppState
    @Binding var flash: String?

    var body: some View {
        Menu {
            ForEach(classOrder, id: \.self) { c in
                let cls = state.class_(c)
                Button {
                    state.markTruth(classId: c)
                    flash = cls.labelVi
                    Task { try? await Task.sleep(for: .seconds(3)); flash = nil }
                } label: {
                    Label(cls.labelVi, systemImage: cls.sfSymbol)
                }
            }
        } label: {
            Label("Có âm thật", systemImage: "hand.raised.fill")
                .font(.headline)
                .frame(maxWidth: .infinity, minHeight: 50)
        }
        .buttonStyle(.borderedProminent)
        .tint(.orange)
    }

    private var classOrder: [Int] {
        (0..<(state.config.numClasses - 1)).sorted { state.class_($0).level > state.class_($1).level }
    }
}

private struct CompareRecent: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Cảnh báo gần đây").font(.title3.weight(.bold))
            if state.history.isEmpty {
                Label("Chưa có cảnh báo nào", systemImage: "bell.slash")
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(16)
                    .background(.background, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            } else {
                VStack(spacing: 0) {
                    ForEach(Array(state.history.prefix(6).enumerated()), id: \.element.id) { i, a in
                        if i > 0 { Divider().padding(.leading, 70) }
                        AlertRow(alert: a, cls: state.displayClass(a), showFeedback: true)
                            .padding(.horizontal, 14).padding(.vertical, 10)
                    }
                }
                .background(.background, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            }
        }
    }
}

/// Đúng / Sai buttons for one alert.
struct VerdictButtons: View {
    @EnvironmentObject var state: AppState
    let alert: ShownAlert

    var body: some View {
        HStack(spacing: 6) {
            button(.correct, "checkmark", .green, "Đúng")
            button(.incorrect, "xmark", .red, "Sai")
        }
    }

    private func button(_ v: ShownAlert.Verdict, _ icon: String, _ color: Color, _ label: String) -> some View {
        Button {
            state.setVerdict(v, for: alert.id)
        } label: {
            Image(systemName: icon).font(.caption.weight(.bold))
                .frame(width: 30, height: 26)
                .foregroundStyle(alert.verdict == v ? .white : color)
                .background(alert.verdict == v ? color : color.opacity(0.12), in: Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
    }
}

struct SessionStatsView: View {
    @EnvironmentObject var state: AppState
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            TimelineView(.periodic(from: .now, by: 5)) { ctx in
                let stats = state.sessionStats(now: ctx.date)
                List {
                    Section {
                        if let t0 = state.sessionStart {
                            LabeledContent("Bắt đầu", value: t0.formatted(date: .omitted, time: .shortened))
                            LabeledContent("Thời lượng", value: Duration.seconds(ctx.date.timeIntervalSince(t0))
                                .formatted(.time(pattern: .hourMinuteSecond)))
                        }
                        LabeledContent("Âm thật đã đánh dấu", value: "\(state.truths.count)")
                    }
                    ForEach(stats.indices, id: \.self) { m in
                        let s = stats[m]
                        Section("\(state.models[m].short) · \(state.models[m].name)") {
                            LabeledContent("Cảnh báo", value: "\(s.alerts)")
                            LabeledContent("Đúng / Sai", value: "\(s.correct) / \(s.incorrect)")
                            LabeledContent("Bắt được âm thật", value: "\(s.truthsCaught) / \(state.truths.count)")
                            LabeledContent("Báo nhầm (Sai) / giờ", value: String(format: "%.1f", s.incorrectPerHour))
                            LabeledContent("Không khớp model kia", value: "\(s.disagree)")
                        }
                    }
                    Section {
                        ShareLink(items: state.exportFiles()) {
                            Label("Xuất phiên đo (JSONL + CSV)", systemImage: "square.and.arrow.up")
                        }
                    } footer: {
                        Text("Âm thật được tính là bắt được nếu model báo cùng lớp trong khoảng 2 s trước đến 5 s sau lúc bấm.")
                    }
                }
            }
            .navigationTitle("Thống kê phiên")
            .toolbar { Button("Xong") { dismiss() } }
        }
    }
}
