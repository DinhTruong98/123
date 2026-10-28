import SwiftUI

@main
struct SoundWatchApp: App {
    @StateObject private var state = AppState()
    @Environment(\.scenePhase) private var phase

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(state)
                .task { await state.start() }
                .onChange(of: phase) { _, p in state.setActive(p == .active) }
        }
    }
}
