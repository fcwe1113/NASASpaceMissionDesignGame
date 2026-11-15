"""Keeps Tk's window messages away from Panda3D on Windows.

Why: on Windows every window made by one thread shares one message queue.
Panda3D empties that whole queue each frame (it doesn't only take its own
window's messages), and it does that with Python's lock released. If Panda
picks up a message meant for a Tk window, Tk handles it right there. Most of
the time that's harmless, but some messages (dragging or resizing a window by
its title bar or edge, the window menu) start a loop inside Windows that makes
Tk run Python code. Python isn't ready for that at that point, so the program
dies with "Fatal Python error: PyEval_RestoreThread ... GIL is released".

The fix: a message hook. Whenever Tk isn't the one reading the queue, the
messages that would start one of those loops (clicks on a Tk window's title
bar or edges, Alt/F10 and the window menu, clicks on a native scrollbar) are
set aside instead of being handled, then put back on the queue just before
Tk's next turn (root.update()). So those loops only ever run in Tk's own turn,
when Python is ready for them. Ordinary clicks and typing are left alone: Tk
needs them delivered normally to know which mouse buttons are down, and they
are safe.

On other systems this does nothing.
"""
import sys
from contextlib import contextmanager

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    WH_GETMESSAGE = 3
    HC_ACTION = 0
    PM_REMOVE = 1
    WM_NULL = 0x0000
    # Messages that make Windows run a loop of its own (moving, sizing, menus)
    HOLD = {
        0x00A1, 0x00A3,          # WM_NCLBUTTONDOWN / DBLCLK: title bar, edges, menu bar, buttons
        0x00A4, 0x00A5, 0x00A6,  # WM_NCRBUTTON*: right-click the title bar (window menu)
        0x00A7, 0x00A9,          # WM_NCMBUTTONDOWN / DBLCLK
        0x0104, 0x0105, 0x0106,  # WM_SYSKEYDOWN / UP / SYSCHAR: Alt, F10, Alt+Space
        0x0112,                  # WM_SYSCOMMAND (posted)
    }
    # Clicks on a native scrollbar start its own drag loop too
    SCROLLBAR_CLICKS = {0x0201, 0x0203}      # WM_LBUTTONDOWN / DBLCLK

    LRESULT = ctypes.c_ssize_t
    HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
    user32.SetWindowsHookExW.restype = wintypes.HHOOK
    user32.SetWindowsHookExW.argtypes = [ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD]
    user32.CallNextHookEx.restype = LRESULT
    user32.CallNextHookEx.argtypes = [wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
    user32.UnhookWindowsHookEx.argtypes = [wintypes.HHOOK]
    user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.GetAncestor.restype = wintypes.HWND
    user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    user32.IsChild.argtypes = [wintypes.HWND, wintypes.HWND]
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]


class MessageGuard:
    """Install once Panda's window exists; wrap every root.update() in `with guard.tk_turn():`."""

    def __init__(self, panda_hwnd):
        self.panda_hwnd = int(panda_hwnd or 0)
        self.held = []                  # (hwnd, message, wparam, lparam) set aside for Tk
        self.in_tk = False
        self.hook = None
        if sys.platform != "win32" or not self.panda_hwnd:
            return
        self._thread = kernel32.GetCurrentThreadId()
        self._proc = HOOKPROC(self._on_message)          # keep a reference, or it gets freed
        self.hook = user32.SetWindowsHookExW(WH_GETMESSAGE, self._proc, None, self._thread)

    def _is_panda(self, hwnd):
        return hwnd == self.panda_hwnd or bool(user32.IsChild(self.panda_hwnd, hwnd))

    def _must_hold(self, hwnd, message):
        if message in HOLD:
            return not self._is_panda(hwnd)
        if message in SCROLLBAR_CLICKS and hwnd:
            name = ctypes.create_unicode_buffer(32)
            user32.GetClassNameW(hwnd, name, 32)
            return name.value.lower() == "scrollbar"
        return False

    def _on_message(self, code, wparam, lparam):
        try:
            if code == HC_ACTION and wparam == PM_REMOVE and not self.in_tk:
                msg = wintypes.MSG.from_address(lparam)
                hwnd = msg.hWnd or 0
                if self._must_hold(hwnd, msg.message):
                    self.held.append((hwnd, msg.message, msg.wParam, msg.lParam))
                    msg.message = WM_NULL    # Panda sees nothing; Tk gets it on its turn
        except Exception:
            pass                             # never let an error escape into Windows
        return user32.CallNextHookEx(None, code, wparam, lparam)

    @contextmanager
    def tk_turn(self):
        """Tk's turn: put back what was set aside, then let Tk read the queue itself."""
        held, self.held = self.held, []
        for hwnd, message, wparam, lparam in held:
            if hwnd:
                user32.PostMessageW(hwnd, message, wparam, lparam)
            else:
                user32.PostThreadMessageW(self._thread, message, wparam, lparam)
        self.in_tk = True
        try:
            yield
        finally:
            self.in_tk = False

    def remove(self):
        if self.hook:
            user32.UnhookWindowsHookEx(self.hook)
            self.hook = None
