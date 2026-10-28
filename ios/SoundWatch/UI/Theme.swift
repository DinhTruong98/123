import SwiftUI
import UIKit

extension Level {
    var name: String {
        switch self {
        case .danger: "Nguy hiểm"
        case .warn: "Cảnh báo"
        case .info: "Thông tin"
        }
    }

    var symbol: String {
        switch self {
        case .danger: "exclamationmark.triangle.fill"
        case .warn: "exclamationmark.circle.fill"
        case .info: "info.circle.fill"
        }
    }
}

extension SoundClass {
    var emoji: String {
        switch key {
        case "smoke_alarm": "🔥"
        case "car_horn": "🚗"
        case "siren": "🚨"
        case "baby_cry": "👶"
        case "door_knock": "🚪"
        case "dog_bark": "🐕"
        case "water_running": "💧"
        default: "👂"
        }
    }

    private var rgb: (Double, Double, Double) {
        (Double(color[0]) / 255, Double(color[1]) / 255, Double(color[2]) / 255)
    }

    /// Alert background with readable text: the config colour, darkened for DANGER until white text
    /// reaches WCAG AA (4.5:1); WARN/INFO keep their colour and use black text if that reads better.
    private var display: (bg: (Double, Double, Double), white: Bool) {
        var c = rgb
        if Theme.contrast(c, (1, 1, 1)) >= 4.5 { return (c, true) }
        if level == .danger {
            while Theme.contrast(c, (1, 1, 1)) < 4.5 { c = (c.0 * 0.95, c.1 * 0.95, c.2 * 0.95) }
            return (c, true)
        }
        return (c, Theme.contrast(c, (1, 1, 1)) >= Theme.contrast(c, (0, 0, 0)))
    }

    /// Accent (icons, bars) in the original config colour.
    var uiColor: Color { Color(red: rgb.0, green: rgb.1, blue: rgb.2) }
    var alertColor: Color { let c = display.bg; return Color(red: c.0, green: c.1, blue: c.2) }
    var alertForeground: Color { display.white ? .white : .black }
    var uiColorUIKit: UIColor { let c = display.bg; return UIColor(red: c.0, green: c.1, blue: c.2, alpha: 1) }
    var foregroundUIKit: UIColor { display.white ? .white : .black }
}

enum Theme {
    static func luminance(_ c: (Double, Double, Double)) -> Double {
        func lin(_ v: Double) -> Double { v <= 0.03928 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4) }
        return 0.2126 * lin(c.0) + 0.7152 * lin(c.1) + 0.0722 * lin(c.2)
    }

    static func contrast(_ a: (Double, Double, Double), _ b: (Double, Double, Double)) -> Double {
        let la = luminance(a), lb = luminance(b)
        return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)
    }
}

extension UIColor {
    func darker(_ amount: CGFloat) -> UIColor {
        var h: CGFloat = 0, s: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        getHue(&h, saturation: &s, brightness: &b, alpha: &a)
        return UIColor(hue: h, saturation: s, brightness: b * (1 - amount), alpha: a)
    }
}
