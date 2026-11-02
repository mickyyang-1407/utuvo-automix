// The native window around the Automix engine.
//
// The engine (the PyInstaller build of app.py) is a local web server.  This
// shell starts it, waits for the address it announces, and shows the page in
// a WKWebView -- so the App is one window of its own instead of a browser tab.
//
//   Contents/MacOS/Automix          this shell
//   Contents/MacOS/automix-engine   the engine
//
// Things that are easy to get wrong:
//  * Any argument means the command line (--cli, --validate-plan): the shell
//    execs the engine with the same arguments and gets out of the way, so the
//    App's command line and its exit codes stay exactly what they were.
//  * The engine runs in its own process group, and the render children it
//    spawns join that group.  Quitting asks the engine to finish a running
//    publication and end its group; the engine also watches this process, so
//    even a Force Quit leaves nothing behind.
//  * A web view does nothing on its own for <input type=file>, downloads,
//    target=_blank or alert/confirm: each is wired up below.

import AppKit
import WebKit

let engineName = "automix-engine"
let readyPrefix = "automix UI  ->  "
let popupTitles = ["plan": "mixplan.json", "report": "report.md"]

// MARK: - command line passthrough

// a process launched without stdin/stdout/stderr would hand those numbers to
// the first files it opens, and the engine's stdout pipe would land on them
for fd: Int32 in 0...2 where fcntl(fd, F_GETFD) == -1 {
    _ = open("/dev/null", O_RDWR)
}

// Every argument goes to the engine untouched.  (macOS 10.9 and later never
// add the old "-psn_" process serial number to a launched App.)
let shellArgs = Array(CommandLine.arguments.dropFirst())
let macOSDir = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath().deletingLastPathComponent()
let enginePath = macOSDir.appendingPathComponent(engineName).path

if !shellArgs.isEmpty {
    var argv: [UnsafeMutablePointer<CChar>?] = [strdup(enginePath)] + shellArgs.map { strdup($0) } + [nil]
    execv(enginePath, &argv)
    perror("Automix: cannot start \(engineName)")
    exit(127)
}

// MARK: - the engine process

final class Engine {
    private(set) var pid: pid_t = 0
    let logURL: URL
    var onReady: ((URL) -> Void)?
    var onExit: ((Int32) -> Void)?
    private var ready = false

    init() {
        let logs = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/UTUVO Automix", isDirectory: true)
        try? FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
        logURL = logs.appendingPathComponent("engine.log")
    }

    func start() throws {
        // a fresh log for every launch; the engine's stderr goes straight in
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        let logFd = open(logURL.path, O_WRONLY | O_APPEND)
        guard logFd >= 0 else { throw err("cannot open \(logURL.path)") }
        defer { close(logFd) }

        var fds: [Int32] = [0, 0]
        guard pipe(&fds) == 0 else { throw err("cannot create a pipe") }
        let (readFd, writeFd) = (fds[0], fds[1])

        var actions: posix_spawn_file_actions_t?
        posix_spawn_file_actions_init(&actions)
        defer { posix_spawn_file_actions_destroy(&actions) }
        posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", O_RDONLY, 0)
        posix_spawn_file_actions_adddup2(&actions, writeFd, 1)
        posix_spawn_file_actions_adddup2(&actions, logFd, 2)
        posix_spawn_file_actions_addclose(&actions, readFd)
        posix_spawn_file_actions_addclose(&actions, writeFd)

        var attr: posix_spawnattr_t?
        posix_spawnattr_init(&attr)
        defer { posix_spawnattr_destroy(&attr) }
        posix_spawnattr_setflags(&attr, Int16(POSIX_SPAWN_SETPGROUP))
        posix_spawnattr_setpgroup(&attr, 0)

        var env = ProcessInfo.processInfo.environment
        env["AUTOMIX_NO_BROWSER"] = "1"
        env["AUTOMIX_SHELL_PID"] = String(getpid())
        var envp: [UnsafeMutablePointer<CChar>?] = env.map { strdup("\($0.key)=\($0.value)") } + [nil]
        var argv: [UnsafeMutablePointer<CChar>?] = [strdup(enginePath), nil]
        defer { (envp + argv).forEach { free($0) } }

        let rc = posix_spawn(&pid, enginePath, &actions, &attr, &argv, &envp)
        close(writeFd)
        guard rc == 0 else {
            close(readFd)
            throw err("cannot start \(engineName): \(String(cString: strerror(rc)))")
        }
        readAnnouncements(from: readFd)
        watchExit()
    }

    /// The engine prints its address once the socket is bound; everything it
    /// prints also goes to the log.
    private func readAnnouncements(from fd: Int32) {
        let log = FileHandle(forWritingAtPath: logURL.path)
        Thread.detachNewThread { [weak self] in
            let input = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
            var pending = Data()
            while true {
                let chunk = input.availableData
                if chunk.isEmpty { break }
                log?.seekToEndOfFile()
                log?.write(chunk)
                pending.append(chunk)
                while let nl = pending.firstIndex(of: 0x0A) {
                    let line = String(decoding: pending[pending.startIndex..<nl], as: UTF8.self)
                    pending.removeSubrange(pending.startIndex...nl)
                    if line.hasPrefix(readyPrefix),
                       let url = URL(string: line.dropFirst(readyPrefix.count).trimmingCharacters(in: .whitespaces)),
                       url.scheme == "http", url.host == "127.0.0.1", url.port != nil {
                        DispatchQueue.main.async {
                            guard let self, !self.ready else { return }
                            self.ready = true
                            self.onReady?(url)
                        }
                    }
                }
            }
        }
    }

    // The engine's pid is also its process group id.  It stays ours until it
    // is reaped: the watcher first waits WITHOUT reaping, and only under
    // `lock` marks the engine gone, sweeps what is left of its group and
    // reaps it.  stop() signals under the same lock, so neither number can
    // belong to some other process by the time it is used.
    private let lock = NSLock()
    private var exited = false
    private let exitSignal = DispatchSemaphore(value: 0)

    private func watchExit() {
        let pid = self.pid
        Thread.detachNewThread { [weak self] in
            var info = siginfo_t()
            while waitid(P_PID, id_t(pid), &info, WEXITED | WNOWAIT) < 0 && errno == EINTR {}
            var status: Int32 = 0
            self?.lock.lock()
            self?.exited = true
            // render children of an engine that crashed must not outlive it
            killpg(pid, SIGKILL)
            while waitpid(pid, &status, 0) < 0 && errno == EINTR {}
            self?.lock.unlock()
            self?.exitSignal.signal()
            DispatchQueue.main.async { self?.onExit?(status) }
        }
    }

    /// Stop the engine and every render child it started.  The engine is
    /// asked first: it lets a publication that is running finish (app.py),
    /// then ends its own group.  Only if it does not, the group is killed.
    func stop() {
        guard pid > 0 else { return }
        lock.lock()
        let running = !exited
        if running { kill(pid, SIGTERM) }
        lock.unlock()
        guard running else { return }
        if exitSignal.wait(timeout: .now() + 65) == .timedOut {
            lock.lock()
            if !exited { killpg(pid, SIGKILL) }
            lock.unlock()
        }
    }

    private func err(_ text: String) -> NSError {
        NSError(domain: "Automix", code: 1, userInfo: [NSLocalizedDescriptionKey: text])
    }
}

// MARK: - the App

final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate,
                         WKNavigationDelegate, WKUIDelegate, WKDownloadDelegate {
    let engine = Engine()
    var window: NSWindow!
    var webView: WKWebView!
    var status: NSTextField!
    var origin: URL?
    var popups: [NSWindow] = []
    var pendingDownloads: [ObjectIdentifier: (partial: URL, final: URL)] = [:]
    var quitting = false

    func applicationDidFinishLaunching(_ note: Notification) {
        buildMenu()
        buildWindow()
        engine.onReady = { [weak self] url in self?.load(url) }
        engine.onExit = { [weak self] status in self?.engineExited(status) }
        do {
            try engine.start()
        } catch {
            fail("Automix could not start.", detail: error.localizedDescription)
            return
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 60) { [weak self] in
            guard let self, self.origin == nil, !self.quitting else { return }
            self.fail("Automix did not start in time.",
                      detail: "The details are in \(self.engine.logURL.path).")
        }
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ app: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ note: Notification) {
        quitting = true
        // a download that has not finished is abandoned with the App
        for (partial, _) in pendingDownloads.values {
            try? FileManager.default.removeItem(at: partial)
        }
        pendingDownloads.removeAll()
        engine.stop()
    }

    func windowWillClose(_ note: Notification) {
        guard let closing = note.object as? NSWindow else { return }
        if closing === window {
            NSApp.terminate(nil)
        } else {
            popups.removeAll { $0 === closing }
        }
    }

    // MARK: window

    func makeConfiguration() -> WKWebViewConfiguration {
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .default()
        config.mediaTypesRequiringUserActionForPlayback = []
        return config
    }

    func buildWindow() {
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1320, height: 880),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered, defer: false)
        window.title = "Automix"
        window.minSize = NSSize(width: 900, height: 600)
        window.isReleasedWhenClosed = false
        window.delegate = self
        window.center()
        window.setFrameAutosaveName("AutomixMain")
        // the page is dark-only (--bg in web/index.html); no white flash while it starts
        window.appearance = NSAppearance(named: .darkAqua)
        window.backgroundColor = NSColor(srgbRed: 0x0e / 255.0, green: 0x10 / 255.0, blue: 0x13 / 255.0, alpha: 1)

        webView = WKWebView(frame: window.contentView!.bounds, configuration: makeConfiguration())
        webView.autoresizingMask = [.width, .height]
        webView.navigationDelegate = self
        webView.uiDelegate = self
        webView.allowsMagnification = true
        webView.isHidden = true
        webView.underPageBackgroundColor = window.backgroundColor
        window.contentView!.addSubview(webView)

        status = NSTextField(labelWithString: "Starting Automix…")
        status.font = .systemFont(ofSize: 15)
        status.textColor = .secondaryLabelColor
        status.alignment = .center
        status.frame = NSRect(x: 0, y: window.contentView!.bounds.midY - 12,
                              width: window.contentView!.bounds.width, height: 24)
        status.autoresizingMask = [.width, .minYMargin, .maxYMargin]
        window.contentView!.addSubview(status)

        window.makeKeyAndOrderFront(nil)
    }

    func load(_ url: URL) {
        origin = url
        webView.load(URLRequest(url: url))
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        if webView === self.webView, webView.isHidden {
            webView.isHidden = false
            status.isHidden = true
        }
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        loadFailed(webView, error)
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        loadFailed(webView, error)
    }

    func loadFailed(_ webView: WKWebView, _ error: Error) {
        // a navigation turned into a download ends with this error; it is not a failure
        let ns = error as NSError
        if ns.domain == "WebKitErrorDomain" && ns.code == 102 { return }
        if ns.domain == NSURLErrorDomain && ns.code == NSURLErrorCancelled { return }
        if webView === self.webView && webView.isHidden {
            fail("Automix could not open its window.", detail: error.localizedDescription)
        }
    }

    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        webView.reload()
    }

    // MARK: engine

    func engineExited(_ status: Int32) {
        guard !quitting else { return }
        fail("Automix stopped unexpectedly.",
             detail: "The details are in \(engine.logURL.path).")
    }

    func fail(_ message: String, detail: String) {
        guard !quitting else { return }
        quitting = true
        let alert = NSAlert()
        alert.alertStyle = .critical
        alert.messageText = message
        alert.informativeText = detail
        alert.addButton(withTitle: "Quit")
        alert.addButton(withTitle: "Show Log")
        if alert.runModal() == .alertSecondButtonReturn {
            NSWorkspace.shared.activateFileViewerSelecting([engine.logURL])
        }
        NSApp.terminate(nil)
    }

    // MARK: navigation

    func isOwn(_ url: URL?) -> Bool {
        guard let url, let origin else { return false }
        return url.scheme == origin.scheme && url.host == origin.host && url.port == origin.port
    }

    func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        let url = action.request.url
        if action.shouldPerformDownload {
            decisionHandler(.download)
        } else if isOwn(url) || url?.absoluteString == "about:blank" {
            decisionHandler(.allow)
        } else {
            // anything outside the engine belongs in the user's browser
            if let url, ["http", "https", "mailto"].contains(url.scheme ?? "") {
                NSWorkspace.shared.open(url)
            }
            decisionHandler(.cancel)
        }
    }

    func webView(_ webView: WKWebView, decidePolicyFor response: WKNavigationResponse,
                 decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void) {
        let http = response.response as? HTTPURLResponse
        let disposition = http?.value(forHTTPHeaderField: "Content-Disposition") ?? ""
        if disposition.lowercased().hasPrefix("attachment") || !response.canShowMIMEType {
            decisionHandler(.download)
        } else {
            decisionHandler(.allow)
        }
    }

    func webView(_ webView: WKWebView, navigationAction: WKNavigationAction, didBecome download: WKDownload) {
        download.delegate = self
    }

    func webView(_ webView: WKWebView, navigationResponse: WKNavigationResponse, didBecome download: WKDownload) {
        download.delegate = self
    }

    // MARK: downloads

    func download(_ download: WKDownload, decideDestinationUsing response: URLResponse,
                  suggestedFilename: String, completionHandler: @escaping (URL?) -> Void) {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = suggestedFilename
        panel.directoryURL = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first
        panel.canCreateDirectories = true
        panel.beginSheetModal(for: window) { [weak self] answer in
            guard answer == .OK, let url = panel.url, let self else { completionHandler(nil); return }
            // The file arrives in the App's own temporary folder and only
            // replaces the chosen one once it is complete: a failed or
            // interrupted download leaves a file the user already had
            // untouched.  (A scratch name next to the chosen file would not
            // do: in Downloads or Documents macOS asks for access to the whole
            // folder, because only the chosen name was granted.)
            let partial = FileManager.default.temporaryDirectory
                .appendingPathComponent("automix-download-\(UUID().uuidString)")
            self.pendingDownloads[ObjectIdentifier(download)] = (partial, url)
            completionHandler(partial)
        }
    }

    func downloadDidFinish(_ download: WKDownload) {
        guard let (partial, final) = pendingDownloads.removeValue(forKey: ObjectIdentifier(download)) else { return }
        do {
            // an existing file is swapped for the new one in a single step,
            // so if anything fails the user still has the old file
            if FileManager.default.fileExists(atPath: final.path) {
                _ = try FileManager.default.replaceItemAt(final, withItemAt: partial)
            } else {
                try FileManager.default.moveItem(at: partial, to: final)
            }
        } catch {
            try? FileManager.default.removeItem(at: partial)
            let alert = NSAlert(error: error)
            alert.messageText = "The file could not be saved."
            alert.beginSheetModal(for: window)
        }
    }

    func download(_ download: WKDownload, didFailWithError error: Error, resumeData: Data?) {
        if let (partial, _) = pendingDownloads.removeValue(forKey: ObjectIdentifier(download)) {
            try? FileManager.default.removeItem(at: partial)
        }
        let ns = error as NSError
        if ns.domain == NSURLErrorDomain && ns.code == NSURLErrorCancelled { return }
        let alert = NSAlert(error: error)
        alert.messageText = "The file could not be saved."
        alert.beginSheetModal(for: window)
    }

    // MARK: panels

    func webView(_ webView: WKWebView, runOpenPanelWith parameters: WKOpenPanelParameters,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping ([URL]?) -> Void) {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = parameters.allowsMultipleSelection
        panel.canChooseDirectories = parameters.allowsDirectories
        panel.canChooseFiles = true
        panel.beginSheetModal(for: webView.window ?? window) { answer in
            completionHandler(answer == .OK ? panel.urls : nil)
        }
    }

    func webView(_ webView: WKWebView, runJavaScriptAlertPanelWithMessage message: String,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping () -> Void) {
        let alert = NSAlert()
        alert.messageText = message
        alert.addButton(withTitle: "OK")
        alert.beginSheetModal(for: webView.window ?? window) { _ in completionHandler() }
    }

    func webView(_ webView: WKWebView, runJavaScriptConfirmPanelWithMessage message: String,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping (Bool) -> Void) {
        let alert = NSAlert()
        alert.messageText = message
        alert.addButton(withTitle: "OK")
        alert.addButton(withTitle: "Cancel")
        alert.beginSheetModal(for: webView.window ?? window) { completionHandler($0 == .alertFirstButtonReturn) }
    }

    func webView(_ webView: WKWebView, runJavaScriptTextInputPanelWithPrompt prompt: String, defaultText: String?,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping (String?) -> Void) {
        let alert = NSAlert()
        alert.messageText = prompt
        let field = NSTextField(frame: NSRect(x: 0, y: 0, width: 280, height: 24))
        field.stringValue = defaultText ?? ""
        alert.accessoryView = field
        alert.addButton(withTitle: "OK")
        alert.addButton(withTitle: "Cancel")
        alert.window.initialFirstResponder = field
        alert.beginSheetModal(for: webView.window ?? window) {
            completionHandler($0 == .alertFirstButtonReturn ? field.stringValue : nil)
        }
    }

    /// target=_blank (the plan and the report) opens in a window of its own.
    func webView(_ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration,
                 for action: WKNavigationAction, windowFeatures: WKWindowFeatures) -> WKWebView? {
        guard isOwn(action.request.url) else {
            if let url = action.request.url, ["http", "https"].contains(url.scheme ?? "") {
                NSWorkspace.shared.open(url)
            }
            return nil
        }
        let popup = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 820, height: 720),
                             styleMask: [.titled, .closable, .miniaturizable, .resizable],
                             backing: .buffered, defer: false)
        popup.isReleasedWhenClosed = false
        popup.delegate = self
        let view = WKWebView(frame: popup.contentView!.bounds, configuration: configuration)
        view.autoresizingMask = [.width, .height]
        view.navigationDelegate = self
        view.uiDelegate = self
        popup.contentView!.addSubview(view)
        popup.title = popupTitles[action.request.url?.lastPathComponent ?? ""] ?? "Automix"
        popup.cascadeTopLeft(from: NSPoint(x: window.frame.minX + 40, y: window.frame.maxY - 40))
        popup.makeKeyAndOrderFront(nil)
        popups.append(popup)
        return view
    }

    func webViewDidClose(_ webView: WKWebView) {
        if let w = webView.window, w !== window { w.close() }
    }

    // MARK: menu

    func buildMenu() {
        let main = NSMenu()

        let app = NSMenu()
        app.addItem(withTitle: "About Automix", action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
        app.addItem(.separator())
        app.addItem(withTitle: "Hide Automix", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        let others = app.addItem(withTitle: "Hide Others", action: #selector(NSApplication.hideOtherApplications(_:)), keyEquivalent: "h")
        others.keyEquivalentModifierMask = [.command, .option]
        app.addItem(withTitle: "Show All", action: #selector(NSApplication.unhideAllApplications(_:)), keyEquivalent: "")
        app.addItem(.separator())
        app.addItem(withTitle: "Quit Automix", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        main.addItem(submenu(app, "Automix"))

        let edit = NSMenu(title: "Edit")
        edit.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
        let redo = edit.addItem(withTitle: "Redo", action: Selector(("redo:")), keyEquivalent: "z")
        redo.keyEquivalentModifierMask = [.command, .shift]
        edit.addItem(.separator())
        edit.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        edit.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        edit.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        edit.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        main.addItem(submenu(edit, "Edit"))

        let view = NSMenu(title: "View")
        let full = view.addItem(withTitle: "Enter Full Screen", action: #selector(NSWindow.toggleFullScreen(_:)), keyEquivalent: "f")
        full.keyEquivalentModifierMask = [.command, .control]
        main.addItem(submenu(view, "View"))

        let win = NSMenu(title: "Window")
        win.addItem(withTitle: "Minimize", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
        win.addItem(withTitle: "Zoom", action: #selector(NSWindow.performZoom(_:)), keyEquivalent: "")
        win.addItem(withTitle: "Close", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
        win.addItem(.separator())
        win.addItem(withTitle: "Bring All to Front", action: #selector(NSApplication.arrangeInFront(_:)), keyEquivalent: "")
        main.addItem(submenu(win, "Window"))
        NSApp.windowsMenu = win

        NSApp.mainMenu = main
    }

    func submenu(_ menu: NSMenu, _ title: String) -> NSMenuItem {
        let item = NSMenuItem(title: title, action: nil, keyEquivalent: "")
        item.submenu = menu
        return item
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
