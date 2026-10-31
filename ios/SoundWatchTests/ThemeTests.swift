import XCTest
@testable import SoundWatch

/// UI checks: every SF Symbol used exists, alert text is readable (WCAG AA 4.5:1).
final class ThemeTests: XCTestCase {
    func testSymbolsExist() throws {
        let ui = ["waveform.circle.fill", "clock.arrow.circlepath", "gearshape.fill", "mic.fill", "mic.slash.fill",
                  "play.circle.fill", "hourglass", "exclamationmark.octagon.fill", "waveform", "bell.slash",
                  "waveform.path.ecg", "play.fill", "stop.fill", "checkmark.circle.fill", "xmark.circle.fill", "eye"]
        let levels = [Level.danger, .warn, .info].map(\.symbol)
        for s in ui + levels { XCTAssertNotNil(UIImage(systemName: s), s) }
    }

    func testAlertContrast() throws {
        for c in try TestSupport.config().classes {
            var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
            c.uiColorUIKit.getRed(&r, green: &g, blue: &b, alpha: &a)
            let fg: (Double, Double, Double) = c.foregroundUIKit == .white ? (1, 1, 1) : (0, 0, 0)
            let ratio = Theme.contrast((Double(r), Double(g), Double(b)), fg)
            XCTAssertGreaterThanOrEqual(ratio, 4.5, c.key)
            if c.level == .danger { XCTAssertEqual(c.foregroundUIKit, .white, c.key) }
        }
    }
}
