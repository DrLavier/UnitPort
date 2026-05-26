# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

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
from typing import List, Optional, Tuple

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


def install_single_instance_guard(
    argv: List[str],
) -> Tuple[Optional[DeeplinkHandler], bool]:
    """Set up deep-link routing and decide whether to keep running.

    Returns ``(handler, should_exit)``:

    * ``(DeeplinkHandler, False)`` — we are the primary. Use the handler
      to receive deep-link URLs from future secondaries.
    * ``(None, True)`` — we are a secondary launched WITH a
      ``unitport://`` URL in argv. The URL has been forwarded to the
      primary; the caller should ``sys.exit(0)`` before constructing
      any windows.
    * ``(None, False)`` — we are a secondary launched WITHOUT a
      deep-link URL (a parallel install / dev test / second `start.bat`).
      The caller should keep running as a standalone window. Note that
      OS-level ``unitport://`` URL scheme registration is a single
      global — OAuth callbacks will land in whichever primary is alive
      at the moment the user clicks the link, not in this parallel
      instance.

    Compatibility note: prior versions returned ``Optional[DeeplinkHandler]``
    and callers treated ``None`` as "exit". The split was needed because
    the old shape conflated "we forwarded a URL" with "we are a secondary",
    causing parallel ``start.bat`` launches to silently exit.
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
                f"[deeplink] forwarded deep-link to running primary: "
                f"{pending_url[:80]}"
            )
            probe.disconnectFromServer()
            return None, True
        probe.disconnectFromServer()
        log_info(
            "[deeplink] another UnitPort primary is already running; "
            "this process will continue as a parallel instance. "
            "OAuth callbacks will route to the primary."
        )
        return None, False

    handler = DeeplinkHandler()
    if not handler.listen():
        log_warning(
            "[deeplink] running without single-instance guard — "
            "subsequent launches may not forward OAuth callbacks"
        )
    return handler, False


def find_deeplink_url(argv: Optional[List[str]] = None) -> Optional[str]:
    """Public helper — used by main.py to dispatch the URL that started the primary."""
    return _find_deeplink_url(argv if argv is not None else sys.argv)


__all__ = [
    "DeeplinkHandler",
    "install_single_instance_guard",
    "find_deeplink_url",
]
