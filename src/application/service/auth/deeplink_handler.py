"""Deep-link + single-instance plumbing for ``unitport://`` URLs.

When the user clicks a ``unitport://auth-callback?code=...`` link:
1. The OS launches Python with the URL as argv[1].
2. If a primary UnitPort is already running, the secondary forwards the URL
   via QLocalServer/QLocalSocket and exits.
3. The primary receives the URL via Qt signal -> AuthManager.handle_oauth_callback.

Only one function is meant to be called from main.py:
:func:`install_single_instance_guard`.
"""

from __future__ import annotations

import sys
from typing import List, Optional

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtNetwork import QLocalServer, QLocalSocket

from unitport_sdk import log_info, log_warning

_SINGLE_INSTANCE_NAME = "unitport.singleton"
_DEEPLINK_SCHEME = "unitport://"


def _find_deeplink_url(argv: List[str]) -> Optional[str]:
    """Scan argv for a unitport://... argument. Returns the first match."""
    for arg in argv[1:]:
        if isinstance(arg, str) and arg.lower().startswith(_DEEPLINK_SCHEME):
            return arg
    return None


class DeeplinkHandler(QObject):
    """Runs in the primary process; receives deep-link URLs over QLocalServer."""

    url_received = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._server = QLocalServer(self)
        self._server.newConnection.connect(self._on_new_connection)

    def listen(self) -> bool:
        QLocalServer.removeServer(_SINGLE_INSTANCE_NAME)
        if not self._server.listen(_SINGLE_INSTANCE_NAME):
            log_warning(
                f"[deeplink] QLocalServer.listen failed: {self._server.errorString()}"
            )
            return False
        return True

    def dispatch(self, url: str) -> None:
        """Emit a URL as if just arrived — for the case where the primary
        process itself was launched with unitport://... in argv."""
        if url:
            self.url_received.emit(url)

    def _on_new_connection(self) -> None:
        socket: QLocalSocket = self._server.nextPendingConnection()
        if socket is None:
            return
        socket.readyRead.connect(lambda s=socket: self._consume(s))
        socket.disconnected.connect(socket.deleteLater)
        if socket.bytesAvailable() > 0:
            self._consume(socket)

    def _consume(self, socket: QLocalSocket) -> None:
        data = bytes(socket.readAll()).decode("utf-8", errors="replace").strip()
        if data:
            log_info(f"[deeplink] received URL from secondary instance: {data[:80]}")
            self.url_received.emit(data)


def install_single_instance_guard(argv: List[str]) -> Optional[DeeplinkHandler]:
    """Return a live DeeplinkHandler if primary; None if a primary is already running.

    When None is returned, caller should sys.exit(0) — pending URL was forwarded.
    """
    pending_url = _find_deeplink_url(argv)

    probe = QLocalSocket()
    probe.connectToServer(_SINGLE_INSTANCE_NAME)
    if probe.waitForConnected(200):
        if pending_url:
            probe.write(pending_url.encode("utf-8"))
            probe.flush()
            probe.waitForBytesWritten(200)
            log_info(
                f"[deeplink] forwarded deep-link to running instance: {pending_url[:80]}"
            )
        probe.disconnectFromServer()
        return None

    handler = DeeplinkHandler()
    if not handler.listen():
        log_warning(
            "[deeplink] running without single-instance guard — "
            "subsequent launches may not forward OAuth callbacks"
        )
    return handler


def find_deeplink_url(argv: Optional[List[str]] = None) -> Optional[str]:
    """Public helper — used by main.py to dispatch the URL that started the primary."""
    return _find_deeplink_url(argv if argv is not None else sys.argv)


__all__ = [
    "DeeplinkHandler",
    "install_single_instance_guard",
    "find_deeplink_url",
]
