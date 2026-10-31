import Foundation

/// A comparison session on disk: Documents/sessions/<yyyyMMdd-HHmmss>/session.jsonl (one JSON object
/// per line: alerts of both models, "Có âm thật", Đúng/Sai, RMS per minute, mode changes, ...),
/// optional WAV clips and, on export, summary.csv. Only kept inside the app; shared by the user.
final class SessionRecorder {
    private(set) var dir: URL?
    private var handle: FileHandle?
    private var t0 = Date()
    private let q = DispatchQueue(label: "soundwatch.session")

    var isRunning: Bool { handle != nil }

    func start(models: [String]) {
        stop()
        let f = DateFormatter()
        f.dateFormat = "yyyyMMdd-HHmmss"
        f.locale = Locale(identifier: "en_US_POSIX")
        t0 = Date()
        let d = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("sessions/\(f.string(from: t0))", isDirectory: true)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        let file = d.appendingPathComponent("session.jsonl")
        FileManager.default.createFile(atPath: file.path, contents: nil)
        dir = d
        handle = try? FileHandle(forWritingTo: file)
        log(["type": "start", "models": models, "app": Bundle.main.infoDictionary?["CFBundleShortVersionString"] ?? ""])
    }

    func stop() {
        guard handle != nil else { return }
        log(["type": "stop"])
        q.sync {
            try? handle?.close()
            handle = nil
        }
    }

    /// Appends one record with wall time ("t", ISO 8601) and seconds since the start ("t_s").
    func log(_ rec: [String: Any]) {
        guard let h = handle else { return }
        var r = rec
        let now = Date()
        r["t"] = now.iso
        r["t_s"] = (now.timeIntervalSince(t0) * 1000).rounded() / 1000
        guard JSONSerialization.isValidJSONObject(r),
              let data = try? JSONSerialization.data(withJSONObject: r, options: [.sortedKeys]) else { return }
        q.async {
            h.write(data)
            h.write(Data("\n".utf8))
        }
    }

    /// 16-bit PCM mono WAV in the session folder; returns its file name.
    func saveWav(_ x: [Float], sampleRate: Int, name: String) -> String? {
        guard let d = dir, !x.isEmpty else { return nil }
        let f = DateFormatter()
        f.dateFormat = "HHmmss"
        let file = "\(name)-\(f.string(from: Date())).wav"
        var data = Data()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        let bytes = UInt32(x.count * 2)
        data.append(contentsOf: Array("RIFF".utf8)); u32(36 + bytes)
        data.append(contentsOf: Array("WAVEfmt ".utf8)); u32(16); u16(1); u16(1)
        u32(UInt32(sampleRate)); u32(UInt32(sampleRate * 2)); u16(2); u16(16)
        data.append(contentsOf: Array("data".utf8)); u32(bytes)
        for v in x {
            let s = Int16(max(-32768, min(32767, (v * 32768).rounded())))
            withUnsafeBytes(of: s.littleEndian) { data.append(contentsOf: $0) }
        }
        do {
            try data.write(to: d.appendingPathComponent(file))
            return file
        } catch {
            return nil
        }
    }

    /// Files of the current (or last) session, after writing summary.csv.
    func files(summaryCSV: String) -> [URL] {
        guard let d = dir else { return [] }
        q.sync { try? handle?.synchronize() }
        try? summaryCSV.write(to: d.appendingPathComponent("summary.csv"), atomically: true, encoding: .utf8)
        let all = (try? FileManager.default.contentsOfDirectory(at: d, includingPropertiesForKeys: nil)) ?? []
        return all.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }
}
