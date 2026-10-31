import UIKit
import UserNotifications

/// Local notifications for alerts while the app is in the background or the screen is locked.
/// Title = emoji + Vietnamese label, subtitle = level · confidence · time, a rendered class icon as
/// attachment, grouped per class (threadIdentifier). DANGER notifications carry "Xác nhận" (acks the
/// PostProcessor, stops the repeats), WARN/INFO carry "Đã thấy". DANGER uses the time-sensitive
/// interruption level (needs the entitlement, see ios/README.md).
final class Notifier: NSObject, UNUserNotificationCenterDelegate {
    static let ackAction = "ACK"
    static let seenAction = "SEEN"
    static let dangerCategory = "SW_DANGER"
    static let alertCategory = "SW_ALERT"
    var onAck: ((Int) -> Void)?
    var onOpen: ((Int) -> Void)?

    func setUp() {
        let c = UNUserNotificationCenter.current()
        c.delegate = self
        let ack = UNNotificationAction(identifier: Self.ackAction, title: "Xác nhận",
                                       options: [], icon: UNNotificationActionIcon(systemImageName: "checkmark.circle.fill"))
        let seen = UNNotificationAction(identifier: Self.seenAction, title: "Đã thấy",
                                        options: [], icon: UNNotificationActionIcon(systemImageName: "eye"))
        c.setNotificationCategories([
            UNNotificationCategory(identifier: Self.dangerCategory, actions: [ack], intentIdentifiers: [],
                                   hiddenPreviewsBodyPlaceholder: "Cảnh báo nguy hiểm",
                                   categorySummaryFormat: "%u cảnh báo %@", options: []),
            UNNotificationCategory(identifier: Self.alertCategory, actions: [seen], intentIdentifiers: [],
                                   hiddenPreviewsBodyPlaceholder: "Cảnh báo âm thanh",
                                   categorySummaryFormat: "%u lần %@", options: []),
        ])
        c.requestAuthorization(options: [.alert, .sound, .badge]) { _, _ in }
    }

    func post(classId: Int, cls: SoundClass, confidence: Float, at date: Date, repeatAlert: Bool, silent: Bool = false) {
        let content = UNMutableNotificationContent()
        content.title = "\(cls.emoji) \(cls.labelVi)"
        let time = date.formatted(date: .omitted, time: .standard)
        content.subtitle = "\(cls.level.name) · \(Int((confidence * 100).rounded()))% · \(time)"
        switch (cls.level, repeatAlert) {
        case (.danger, true): content.body = "Vẫn chưa xác nhận. Nhấn giữ thông báo, chọn Xác nhận để dừng nhắc lại."
        case (.danger, false): content.body = "Kiểm tra xung quanh ngay. Nhấn giữ thông báo để xác nhận."
        case (.warn, _): content.body = "Có thể cần bạn chú ý."
        case (.info, _): content.body = "Âm thanh xung quanh, không khẩn cấp."
        }
        content.sound = silent ? nil : .default
        content.userInfo = ["classId": classId]
        content.threadIdentifier = cls.key
        content.summaryArgument = cls.labelVi
        content.categoryIdentifier = cls.level == .danger ? Self.dangerCategory : Self.alertCategory
        content.interruptionLevel = cls.level == .danger ? .timeSensitive : .active
        content.relevanceScore = cls.level == .danger ? 1.0 : (cls.level == .warn ? 0.6 : 0.3)
        if let a = Self.attachment(for: cls) { content.attachments = [a] }
        // DANGER: one notification per class (a repeat replaces it); others keep one per occurrence
        let id = cls.level == .danger ? "alert.\(cls.key)" : "alert.\(cls.key).\(Int(date.timeIntervalSince1970))"
        UNUserNotificationCenter.current().add(UNNotificationRequest(identifier: id, content: content, trigger: nil))
    }

    func clear(classKey: String) {
        let c = UNUserNotificationCenter.current()
        c.getDeliveredNotifications { list in
            let ids = list.map(\.request.identifier).filter { $0 == "alert.\(classKey)" || $0.hasPrefix("alert.\(classKey).") }
            c.removeDeliveredNotifications(withIdentifiers: ids)
        }
    }

    /// Class icon (white SF Symbol on the level colour, rounded square) as a PNG attachment.
    private static func attachment(for cls: SoundClass) -> UNNotificationAttachment? {
        let size = CGSize(width: 180, height: 180)
        let img = UIGraphicsImageRenderer(size: size).image { ctx in
            let rect = CGRect(origin: .zero, size: size)
            UIBezierPath(roundedRect: rect, cornerRadius: 40).addClip()
            let base = cls.uiColorUIKit
            let colors = [base.withAlphaComponent(1).cgColor, base.darker(0.25).cgColor] as CFArray
            if let g = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: colors, locations: [0, 1]) {
                ctx.cgContext.drawLinearGradient(g, start: .zero, end: CGPoint(x: size.width, y: size.height), options: [])
            }
            let cfg = UIImage.SymbolConfiguration(pointSize: 88, weight: .bold)
            if let sym = UIImage(systemName: cls.sfSymbol, withConfiguration: cfg)?
                .withTintColor(cls.foregroundUIKit, renderingMode: .alwaysOriginal) {
                let s = sym.size
                sym.draw(in: CGRect(x: (size.width - s.width) / 2, y: (size.height - s.height) / 2, width: s.width, height: s.height))
            }
        }
        guard let data = img.pngData() else { return nil }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("sw-\(cls.key)-\(UUID().uuidString).png")
        do {
            try data.write(to: url)
            return try UNNotificationAttachment(identifier: cls.key, url: url,
                                                options: [UNNotificationAttachmentOptionsThumbnailHiddenKey: false])
        } catch {
            return nil
        }
    }

    func userNotificationCenter(_ center: UNUserNotificationCenter, didReceive response: UNNotificationResponse,
                                withCompletionHandler done: @escaping () -> Void) {
        if let id = response.notification.request.content.userInfo["classId"] as? Int {
            switch response.actionIdentifier {
            case Self.ackAction, Self.seenAction: onAck?(id)
            default: onOpen?(id)
            }
        }
        done()
    }

    func userNotificationCenter(_ center: UNUserNotificationCenter, willPresent notification: UNNotification,
                                withCompletionHandler done: @escaping (UNNotificationPresentationOptions) -> Void) {
        done([])  // in the foreground the watch face shows the alert
    }
}
