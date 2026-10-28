import UserNotifications

/// Local notifications for DANGER/WARN alerts while the app is in the background or the screen is
/// locked. DANGER notifications carry a "Xác nhận" action that acks the alert.
final class Notifier: NSObject, UNUserNotificationCenterDelegate {
    static let ackAction = "ACK"
    static let category = "SOUNDWATCH_ALERT"
    var onAck: ((Int) -> Void)?
    var onOpen: ((Int) -> Void)?

    func setUp() {
        let c = UNUserNotificationCenter.current()
        c.delegate = self
        let ack = UNNotificationAction(identifier: Self.ackAction, title: "Xác nhận", options: [])
        c.setNotificationCategories([UNNotificationCategory(identifier: Self.category, actions: [ack],
                                                            intentIdentifiers: [], options: [])])
        c.requestAuthorization(options: [.alert, .sound, .badge]) { _, _ in }
    }

    func post(classId: Int, cls: SoundClass, confidence: Float, at date: Date, repeatAlert: Bool) {
        let content = UNMutableNotificationContent()
        content.title = (cls.level == .danger ? "⚠️ " : "") + cls.labelVi
        let time = date.formatted(date: .omitted, time: .standard)
        content.body = "\(repeatAlert ? "Vẫn chưa xác nhận · " : "")Độ tin cậy \(Int((confidence * 100).rounded()))% · \(time)"
        content.sound = .default
        content.userInfo = ["classId": classId]
        content.interruptionLevel = cls.level == .danger ? .timeSensitive : .active
        if cls.level == .danger { content.categoryIdentifier = Self.category }
        // one notification per class: a repeat replaces the previous one
        let req = UNNotificationRequest(identifier: "alert.\(cls.key)", content: content, trigger: nil)
        UNUserNotificationCenter.current().add(req)
    }

    func clear(classKey: String) {
        UNUserNotificationCenter.current().removeDeliveredNotifications(withIdentifiers: ["alert.\(classKey)"])
    }

    func userNotificationCenter(_ center: UNUserNotificationCenter, didReceive response: UNNotificationResponse,
                                withCompletionHandler done: @escaping () -> Void) {
        if let id = response.notification.request.content.userInfo["classId"] as? Int {
            if response.actionIdentifier == Self.ackAction { onAck?(id) } else { onOpen?(id) }
        }
        done()
    }

    func userNotificationCenter(_ center: UNUserNotificationCenter, willPresent notification: UNNotification,
                                withCompletionHandler done: @escaping (UNNotificationPresentationOptions) -> Void) {
        done([])  // in the foreground the watch face shows the alert
    }
}
