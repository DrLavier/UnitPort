"""Short-lived localhost HTTP server that catches the OAuth redirect.

Loopback server (RFC 8252) is preferred over the custom URL scheme because it
can serve the success page back into the browser. AuthManager falls back to
the unitport:// scheme if the loopback port is unavailable.

Flow:
    AuthManager.sign_in_oauth()
        -> OAuthCallbackServer.start()                   (binds 127.0.0.1:54321)
        -> build oauth URL with redirect_to = http://127.0.0.1:54321/auth-callback
        -> open browser
        -> user approves
        -> browser hits /auth-callback?code=<code>
        -> server emits callback_received(query) on the Qt thread
        -> server responds with a self-contained HTML page
        -> AuthManager runs PKCE exchange, then server.stop()
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import parse_qs, urlparse

from PyQt6.QtCore import QByteArray, QObject, QTimer, pyqtSignal
from PyQt6.QtNetwork import QHostAddress, QTcpServer, QTcpSocket

from unitport_sdk import log_info


# Fixed port — Supabase rejects any redirect URL not pre-registered under
# "Additional Redirect URLs". 54321 is above the privileged range, unlikely
# to clash, memorable.
DEFAULT_CALLBACK_PORT = 54321
CALLBACK_PATH = "/auth-callback"


_SUCCESS_HTML = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>UnitPort — Signed In</title>
<style>
 body {
   margin: 0; min-height: 100vh;
   display: flex; align-items: center; justify-content: center;
   font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
   background: #0f172a; color: #e2e8f0;
 }
 .card {
   text-align: center; padding: 48px 64px;
   background: #1e293b; border-radius: 12px;
   box-shadow: 0 10px 30px rgba(0,0,0,0.35);
   border: 1px solid #334155;
 }
 .check {
   width: 56px; height: 56px; margin: 0 auto 20px;
   border-radius: 50%; background: #22C55E;
   display: flex; align-items: center; justify-content: center;
 }
 .check svg { width: 32px; height: 32px; stroke: #0a0a0a; stroke-width: 3; fill: none; }
 h1 { margin: 0 0 8px; font-size: 22px; font-weight: 600; }
 p  { margin: 4px 0; color: #94a3b8; font-size: 14px; }
</style>
</head>
<body>
 <div class="card">
   <div class="check">
     <svg viewBox="0 0 24 24" aria-hidden="true">
       <polyline points="4,12 10,18 20,6" stroke-linecap="round" stroke-linejoin="round"></polyline>
     </svg>
   </div>
   <h1>Signed in to UnitPort</h1>
   <p>Authentication complete. You can close this tab.</p>
   <p>Return to the UnitPort app to continue.</p>
 </div>
 <script>setTimeout(function(){ try { window.close(); } catch(e){} }, 1500);</script>
</body>
</html>
"""

_ERROR_HTML = """\
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>UnitPort — Sign-in failed</title>
<style>
 body {{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
   font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
   background:#0f172a;color:#e2e8f0;}}
 .card {{text-align:center;padding:48px 64px;background:#1e293b;border-radius:12px;
   box-shadow:0 10px 30px rgba(0,0,0,.35);border:1px solid #334155;}}
 h1 {{margin:0 0 8px;font-size:22px;font-weight:600;color:#ef4444;}}
 p {{margin:4px 0;color:#94a3b8;font-size:14px;}}
 code {{background:#0f172a;padding:2px 6px;border-radius:4px;font-size:12px;}}
</style></head>
<body><div class="card">
<h1>Sign-in failed</h1>
<p>{message}</p>
<p>Close this tab and try again from UnitPort.</p>
</div></body></html>
"""


class OAuthCallbackServer(QObject):
    """One-shot localhost HTTP server. Emits once, then idles until stopped."""

    callback_received = pyqtSignal(dict)
    timed_out = pyqtSignal()

    _TIMEOUT_MS = 5 * 60 * 1000  # 5 min

    def __init__(self, port: int = DEFAULT_CALLBACK_PORT, parent=None) -> None:
        super().__init__(parent)
        self._port = port
        self._server: Optional[QTcpServer] = None
        self._timer: Optional[QTimer] = None
        self._consumed = False

    def start(self) -> str:
        """Bind the listen socket. Returns the redirect URL. Raises OSError on bind fail."""
        server = QTcpServer(self)
        # Loopback only — never 0.0.0.0 (would expose OAuth code to LAN).
        if not server.listen(QHostAddress(QHostAddress.SpecialAddress.LocalHost), self._port):
            err = server.errorString()
            server.deleteLater()
            raise OSError(f"Cannot bind 127.0.0.1:{self._port} — {err}")
        server.newConnection.connect(self._on_connection)
        self._server = server

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)
        self._timer.start(self._TIMEOUT_MS)

        url = f"http://127.0.0.1:{self._port}{CALLBACK_PATH}"
        log_info(f"[oauth:server] listening on {url}")
        return url

    def stop(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer.deleteLater()
            self._timer = None
        if self._server is not None:
            self._server.close()
            self._server.deleteLater()
            self._server = None

    @property
    def redirect_url(self) -> str:
        return f"http://127.0.0.1:{self._port}{CALLBACK_PATH}"

    def _on_connection(self) -> None:
        if self._server is None:
            return
        socket = self._server.nextPendingConnection()
        if socket is None:
            return
        socket.readyRead.connect(lambda s=socket: self._read_request(s))
        socket.disconnected.connect(socket.deleteLater)

    def _read_request(self, socket: QTcpSocket) -> None:
        raw = bytes(socket.readAll())
        if not raw:
            return
        try:
            header_end = raw.index(b"\r\n\r\n")
            request_block = raw[:header_end].decode("latin-1", errors="replace")
        except ValueError:
            return

        request_line = request_block.split("\r\n", 1)[0]
        parts = request_line.split(" ")
        path = parts[1] if len(parts) >= 2 else "/"

        parsed = urlparse(path)
        if parsed.path.rstrip("/") != CALLBACK_PATH.rstrip("/"):
            self._respond(socket, 404, "text/plain", b"Not found")
            return

        if self._consumed:
            self._respond(socket, 200, "text/html; charset=utf-8", _SUCCESS_HTML.encode("utf-8"))
            return

        flat = {k: v[0] for k, v in parse_qs(parsed.query or "").items() if v}
        err = flat.get("error") or flat.get("error_code") or ""
        if err:
            msg = flat.get("error_description") or err
            html = _ERROR_HTML.format(message=_escape(msg))
            self._respond(socket, 400, "text/html; charset=utf-8", html.encode("utf-8"))
        else:
            self._respond(socket, 200, "text/html; charset=utf-8", _SUCCESS_HTML.encode("utf-8"))

        self._consumed = True
        if self._timer is not None:
            self._timer.stop()
        self.callback_received.emit(flat)

    def _respond(self, socket: QTcpSocket, status: int, content_type: str, body: bytes) -> None:
        reason = {200: "OK", 400: "Bad Request", 404: "Not Found"}.get(status, "OK")
        head = (
            f"HTTP/1.1 {status} {reason}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: close\r\n"
            f"Cache-Control: no-store\r\n"
            f"\r\n"
        ).encode("latin-1")
        socket.write(QByteArray(head + body))
        socket.flush()
        socket.disconnectFromHost()

    def _on_timeout(self) -> None:
        if self._consumed:
            return
        log_info("[oauth:server] timed out waiting for callback")
        self.timed_out.emit()
        self.stop()


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


__all__ = [
    "OAuthCallbackServer",
    "DEFAULT_CALLBACK_PORT",
    "CALLBACK_PATH",
]
