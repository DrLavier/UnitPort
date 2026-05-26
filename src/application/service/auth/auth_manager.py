# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""AuthManager — Qt-facing facade over SupabaseClient + secure store + profile cache.

Holds the currently signed-in user in memory, emits signals whenever the auth
state changes, exposes worker-backed methods for every UI-side operation
(sign-in, sign-up, OAuth, sign-out, password reset).

Persistence:
    - refresh token  -> OS keyring via secure.SecureCredentialStore.auth_refresh_token()
    - cached profile -> ``Paths.USER_CONFIG_DIR / "session.json"`` via profile_cache
    - access token   -> memory only

Usage:
    from application.service.auth import get_auth_manager
    mgr = get_auth_manager()
    mgr.authenticated.connect(lambda u: ...)
    mgr.try_restore_session()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import parse_qs, urlparse

from PyQt6.QtCore import QObject, QThread, QTimer, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QDesktopServices, QPixmap

from unitport_sdk import Config, log_error, log_info, log_warning

from .avatar_cache import AvatarCache, make_initials_pixmap
from .oauth_callback_server import OAuthCallbackServer
from .profile_cache import (
    CachedProfile,
    clear_cached_profile,
    load_cached_profile,
    save_cached_profile,
)
from .secure import get_secure_store
from .supabase_client import (
    AuthError,
    AuthNetworkError,
    AuthSession,
    AuthUser,
    PkceChallenge,
    SupabaseClient,
)


# ---------------------------------------------------------------------------
# Public user profile (UI consumes this)
# ---------------------------------------------------------------------------


@dataclass
class UserProfile:
    """What the sidebar card renders. Subset of the Supabase user dict."""

    user_id: str
    email: str
    display_name: str = ""
    avatar_url: str = ""
    provider: str = ""

    @property
    def label(self) -> str:
        if self.display_name:
            return self.display_name
        if self.email:
            return self.email.split("@", 1)[0]
        return "User"

    @classmethod
    def from_auth_user(cls, u: AuthUser) -> "UserProfile":
        return cls(
            user_id=u.id,
            email=u.email,
            display_name=u.display_name,
            avatar_url=u.avatar_url,
            provider=u.provider,
        )

    @classmethod
    def from_cached(cls, c: CachedProfile) -> "UserProfile":
        return cls(
            user_id=c.user_id,
            email=c.email,
            display_name=c.display_name,
            avatar_url=c.avatar_url,
            provider=c.provider,
        )


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------


class _AuthWorker(QThread):
    """Generic worker: runs a zero-arg callable producing AuthSession (or None)."""

    done = pyqtSignal(object)          # AuthSession | None
    failed = pyqtSignal(str, str)      # code, message

    def __init__(self, fn, parent=None) -> None:
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:  # type: ignore[override]
        try:
            result = self._fn()
        except AuthError as exc:
            self.failed.emit(exc.code, exc.message)
            return
        except AuthNetworkError as exc:
            self.failed.emit("network", str(exc))
            return
        except Exception as exc:
            log_error(f"[auth:worker] Unexpected failure: {exc}")
            self.failed.emit("internal", str(exc))
            return
        self.done.emit(result)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class AuthManager(QObject):
    """Singleton. Owns the live session + cached profile + Supabase client."""

    authenticated = pyqtSignal(object)                  # UserProfile
    signed_out = pyqtSignal()
    auth_error = pyqtSignal(str, str)                   # code, message
    email_confirmation_pending = pyqtSignal(str)        # email
    info_message = pyqtSignal(str)
    avatar_updated = pyqtSignal(str, QPixmap)           # user_id, pixmap
    # Per-user workspace was just hot-switched (no restart). Emitted with
    # the new active user_id (or "" for guest) after the shim has been
    # written, ``Paths.USER_CONFIG_DIR`` has been re-resolved, and data
    # caches re-hydrated. UI components (UserPanel, Sidebar) listen and
    # refresh their view of the workspace.
    workspace_changed = pyqtSignal(str)
    # Linked-identity signals (Phase B of identity-linking feature).
    # ``identities_changed`` carries the freshly fetched list[dict] (each
    # dict is a raw Supabase identity row with provider/identity_id/
    # identity_data fields). ``identity_linked`` / ``identity_unlinked``
    # are fired with the provider name BEFORE refresh_identities is
    # dispatched, so UI can show a flash of confirmation immediately.
    identities_changed = pyqtSignal(object)             # list[dict]
    identity_linked = pyqtSignal(str)                   # provider
    identity_unlinked = pyqtSignal(str)                 # provider

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        url = Config.get_value("auth", "supabase_url", fallback="") or ""
        anon = Config.get_value("auth", "supabase_anon_key", fallback="") or ""
        redirect = (
            Config.get_value("auth", "redirect_url", fallback="unitport://auth-callback")
            or "unitport://auth-callback"
        )
        try:
            self._client: Optional[SupabaseClient] = SupabaseClient(
                url=url, anon_key=anon, redirect_url=redirect,
            )
        except ValueError as exc:
            log_error(f"[auth:init] {exc}")
            self._client = None

        self._secure = get_secure_store()

        # live session
        self._access_token: str = ""
        self._refresh_token: str = ""
        self._user: Optional[UserProfile] = None

        # OAuth in-flight. ``_pending_oauth_mode`` is "sign_in" while a
        # login-OAuth dance is open and "link" while an
        # identity-link-OAuth dance is open. Used by _consume_oauth_query
        # to route the returning code, and as a mutex so one mode can't
        # interrupt the other.
        self._pending_pkce: Optional[PkceChallenge] = None
        self._oauth_server: Optional[OAuthCallbackServer] = None
        self._pending_oauth_mode: Optional[str] = None    # "sign_in" | "link"
        self._pending_link_provider: str = ""
        # Identity cache populated after sign-in (and after every
        # successful link/unlink). UserPanel reads this to render its
        # three status buttons without doing its own REST calls.
        self._identities: list = []
        # "Has email/password sign-in" capability bit. Supabase doesn't
        # always add a provider="email" entry to user.identities[] when
        # a password is set on an OAuth-only account — it does set
        # "email" in app_metadata.providers, though. We reconcile both
        # sources (server-truth + optimistic-on-set_password) so the
        # [Email] chip in UserPanel reflects what the user can actually
        # do, not just what Supabase chose to materialise as an
        # identity row.
        self._email_password_set: bool = False

        # keep live workers alive while they run (GC would kill QThread)
        self._workers: list[QThread] = []
        self._shutting_down: bool = False

        # Network-failure retry for try_restore_session. We *only* re-arm on
        # transient (code == "network") failures — invalid_grant et al. mean
        # the refresh token is dead and retrying is pointless. Cap at 5 min
        # because beyond that the user has almost certainly noticed the
        # signed-out state and will sign in manually.
        self._restore_retry_timer = QTimer(self)
        self._restore_retry_timer.setSingleShot(True)
        self._restore_retry_timer.timeout.connect(self._on_restore_retry)
        self._restore_retry_attempt: int = 0
        # Backoff schedule (seconds): 5s, 15s, 45s, 2m, 5m, then steady 5m.
        self._restore_retry_schedule_s: tuple[int, ...] = (5, 15, 45, 120, 300)

        # Avatar pipeline — one cache instance shared across the app.
        self._avatar_cache = AvatarCache(parent=self)
        self._avatar_cache.avatar_ready.connect(self._on_avatar_ready)

    # -- public state ------------------------------------------------------

    def is_signed_in(self) -> bool:
        return self._user is not None and bool(self._access_token)

    def current_user(self) -> Optional[UserProfile]:
        return self._user

    def access_token(self) -> str:
        return self._access_token

    def cached_user(self) -> Optional[UserProfile]:
        cached = load_cached_profile()
        if cached.is_empty:
            return None
        return UserProfile.from_cached(cached)

    def cached_avatar(self) -> Optional[QPixmap]:
        cached = load_cached_profile()
        if cached.is_empty:
            return None
        if cached.avatar_url:
            pm = self._avatar_cache.load_cached(cached.avatar_url)
            if pm is not None:
                return pm
        label = cached.display_name or cached.email
        if not label:
            return None
        return make_initials_pixmap(cached.user_id, label, 128)

    # -- lifecycle ---------------------------------------------------------

    def try_restore_session(self) -> None:
        """Silent session restore on app start. Emits authenticated or signed_out.

        The keyring slot is per-user (account = Supabase user_id); we read
        ``system.ini[Session].active_user`` to know which slot to look at.
        No active_user → guest workspace → no auto-restore.

        Logs every decision point so when "auto sign-in didn't work" gets
        reported we can see exactly which branch returned.
        """
        if self._client is None:
            log_warning(
                "[auth:restore] Supabase client not configured — "
                "auto sign-in disabled. Check system.ini[auth] supabase_url / "
                "supabase_anon_key."
            )
            self.signed_out.emit()
            return
        from application.service.user_workspace import (
            read_active_user,
            read_active_username,
        )
        uid = read_active_user()
        uname = read_active_username() or ""
        if not uid:
            log_info(
                f"[auth:restore] no active_user in system.ini[Session] "
                f"(active_username={uname!r}) — staying as guest"
            )
            self.signed_out.emit()
            return
        token = self._secure.auth_refresh_token(uid).get()
        if not token:
            log_warning(
                f"[auth:restore] no refresh_token in keyring for user_id="
                f"{uid!r} (username={uname!r}) — cannot auto sign-in. "
                f"User must sign in manually to repopulate the keyring."
            )
            self.signed_out.emit()
            return
        log_info(
            f"[auth:restore] attempting silent refresh for user "
            f"{uname!r} (uid={uid})"
        )
        self._run(
            lambda: self._client.refresh_session(token),
            on_ok=self._accept_session,
            on_fail=self._reject_restore,
        )

    def sign_in_password(self, email: str, password: str) -> None:
        if self._client is None:
            self.auth_error.emit("config", "Supabase client is not configured.")
            return
        self._run(
            lambda: self._client.sign_in_password(email, password),
            on_ok=self._accept_session,
            on_fail=self._emit_auth_error,
        )

    def sign_up(self, email: str, password: str) -> None:
        if self._client is None:
            self.auth_error.emit("config", "Supabase client is not configured.")
            return

        def _do() -> Optional[AuthSession]:
            try:
                return self._client.sign_up(email, password)
            except AuthError as exc:
                if exc.code == "email_confirmation_required":
                    return None
                raise

        def _handle(session: Optional[AuthSession]) -> None:
            if session is None:
                self.email_confirmation_pending.emit(email)
            else:
                self._accept_session(session)

        self._run(_do, on_ok=_handle, on_fail=self._emit_auth_error)

    def sign_in_oauth(self, provider: str) -> None:
        """Open the browser for OAuth.

        Prefers a short-lived loopback HTTP server so the browser can render a
        success page. Falls back to the unitport:// custom scheme if the
        loopback port is bound or the OS denies the socket.
        """
        if self._client is None:
            self.auth_error.emit("config", "Supabase client is not configured.")
            return
        if self._pending_oauth_mode == "link":
            self.auth_error.emit(
                "oauth_busy",
                "Another sign-in operation is still pending — please wait.",
            )
            return
        self._reset_pending_oauth()
        self._pending_oauth_mode = "sign_in"

        self._pending_pkce = SupabaseClient.generate_pkce()
        redirect_override = self._start_oauth_server_or_fallback()

        url = self._client.build_oauth_url(
            provider,
            pkce=self._pending_pkce,
            redirect_override=redirect_override,
        )
        QDesktopServices.openUrl(QUrl(url))

    def _start_oauth_server_or_fallback(self) -> Optional[str]:
        """Start the loopback callback server; on bind failure return None
        so the caller uses the unitport:// custom scheme. Shared by
        sign_in_oauth and link_identity_oauth."""
        redirect_override: Optional[str] = None
        server = OAuthCallbackServer(parent=self)
        try:
            redirect_override = server.start()
        except OSError as exc:
            log_warning(
                f"[auth:oauth] loopback unavailable ({exc}); falling back "
                f"to custom scheme {self._client.redirect_url}"
            )
            server.deleteLater()
            server = None

        if server is not None:
            server.callback_received.connect(self._on_oauth_server_callback)
            server.timed_out.connect(self._on_oauth_server_timeout)
            self._oauth_server = server
        return redirect_override

    def send_password_reset(self, email: str) -> None:
        if self._client is None:
            self.auth_error.emit("config", "Supabase client is not configured.")
            return

        def _do() -> None:
            self._client.send_password_reset(email)
            return None

        def _handle(_session) -> None:
            self.info_message.emit(
                f"Password reset email sent to {email}. Check your inbox."
            )

        self._run(_do, on_ok=_handle, on_fail=self._emit_auth_error)

    # -- identity linking --------------------------------------------------

    def current_identities(self) -> list:
        """Return a defensive copy of the cached identities list.

        Populated by ``refresh_identities`` (called automatically on
        sign-in and after every link/unlink). UI components should
        prefer this over hitting the REST API directly.
        """
        return list(self._identities)

    def has_email_password(self) -> bool:
        """Whether the current user can sign in with email + password.

        True when EITHER the cached identities contain a
        provider="email" row OR ``app_metadata.providers`` from the
        last refresh included "email" OR the user set a password
        through ``set_password`` this session. See the
        ``_email_password_set`` doc-comment in ``__init__`` for why
        this fallback exists.
        """
        if self._email_password_set:
            return True
        return any(
            isinstance(i, dict) and i.get("provider") == "email"
            for i in self._identities
        )

    def refresh_identities(self) -> None:
        """Re-fetch identities for the current user; emit identities_changed.

        Pulls the full ``/user`` payload (not just identities[]) so we
        can also pick up ``app_metadata.providers``, which is where
        Supabase actually records the "email" sign-in capability for
        users who set a password without ever signing UP via email.
        """
        if not self._access_token or self._client is None:
            return
        access = self._access_token

        def _do() -> dict:
            return self._client.get_user_raw(access)

        def _on_ok(raw: dict) -> None:
            ids = list(raw.get("identities") or [])
            self._identities = ids
            # Server-reported capability — promote to True only.
            # ``set_password`` may have flipped this on optimistically
            # before Supabase materialised the flag; don't clobber that.
            app_md = raw.get("app_metadata") or {}
            providers = app_md.get("providers")
            if isinstance(providers, list) and "email" in providers:
                self._email_password_set = True
            self.identities_changed.emit(ids)

        self._run(_do, on_ok=_on_ok, on_fail=self._emit_auth_error)

    def link_identity_oauth(self, provider: str) -> None:
        """Start an OAuth flow that adds ``provider`` to the current user.

        Identical callback plumbing as ``sign_in_oauth`` (loopback server
        preferred, ``unitport://`` fallback); routing happens in
        ``_consume_oauth_query`` based on ``_pending_oauth_mode``.
        """
        log_info(f"[auth:link] link_identity_oauth(provider={provider!r}) invoked")
        if self._client is None or not self._access_token:
            log_warning(
                "[auth:link] aborted: client missing or not signed in"
            )
            self.auth_error.emit("config", "Not signed in.")
            return
        if self._pending_oauth_mode is not None:
            log_warning(
                f"[auth:link] aborted: another oauth flow in-flight "
                f"(mode={self._pending_oauth_mode!r})"
            )
            self.auth_error.emit(
                "oauth_busy",
                "Another sign-in operation is still pending — please wait.",
            )
            return

        self._reset_pending_oauth()
        self._pending_oauth_mode = "link"
        self._pending_link_provider = provider
        self._pending_pkce = SupabaseClient.generate_pkce()
        redirect_override = self._start_oauth_server_or_fallback()
        log_info(
            f"[auth:link] pkce ready; redirect_override="
            f"{redirect_override or '(unitport://)'}; dispatching build_link_url"
        )

        access = self._access_token
        pkce = self._pending_pkce

        def _do_build_url() -> str:
            return self._client.build_link_url(
                provider,
                access,
                pkce=pkce,
                redirect_override=redirect_override,
            )

        def _on_url(url: str) -> None:
            if not url:
                log_warning(
                    "[auth:link] Supabase returned empty url from "
                    "/user/identities/authorize — aborting"
                )
                self._reset_pending_oauth()
                self.auth_error.emit(
                    "oauth_url", "Failed to build link URL.",
                )
                return
            log_info(
                f"[auth:link] opening browser for {provider!r} link: "
                f"{url[:80]}..."
            )
            QDesktopServices.openUrl(QUrl(url))

        def _on_url_fail(code: str, message: str) -> None:
            log_warning(
                f"[auth:link] build_link_url failed: code={code!r} "
                f"message={message!r}"
            )
            self._reset_pending_oauth()
            self.auth_error.emit(code, message)

        self._run(_do_build_url, on_ok=_on_url, on_fail=_on_url_fail)

    def unlink_identity_by_provider(self, provider: str) -> None:
        """DELETE the identity that matches ``provider`` for the current user.

        Caller is expected to have shown a confirm dialog. The Supabase
        server still enforces the "can't delete the only identity" rule
        as a final defence — that case bubbles up via auth_error.
        """
        if self._client is None or not self._access_token:
            self.auth_error.emit("config", "Not signed in.")
            return
        target = next(
            (i for i in self._identities if i.get("provider") == provider),
            None,
        )
        if target is None:
            self.auth_error.emit("not_linked", f"{provider} is not linked.")
            return
        identity_id = str(
            target.get("identity_id") or target.get("id") or ""
        )
        if not identity_id:
            self.auth_error.emit("internal", "Identity has no id.")
            return
        access = self._access_token
        current_provider = self._user.provider if self._user else ""

        def _do() -> None:
            self._client.unlink_identity(access, identity_id)
            return None

        def _on_ok(_r) -> None:
            log_info(f"[auth:unlink] {provider!r} identity removed")
            self.identity_unlinked.emit(provider)
            self.refresh_identities()
            if current_provider == provider:
                self.info_message.emit(
                    "You unlinked your current sign-in method. "
                    "You may be signed out shortly."
                )

        self._run(_do, on_ok=_on_ok, on_fail=self._emit_auth_error)

    def mark_email_password_pending(self) -> None:
        """Optimistically mark the email/password capability as available.

        Called by EmailIdentityDialog right after a successful
        ``send_password_reset`` dispatch: we can't observe whether the
        user actually clicks the verification link in their inbox, but
        the *intent* is unambiguous, so flip the local capability bit
        so the UI reflects what the user is about to set up.

        Session-scoped: ``_clear_session`` and ``_accept_session``
        reset it to False, and the next refresh from Supabase
        ``app_metadata.providers`` will reconcile (can only promote
        to True, see ``refresh_identities``).

        Note: this replaces the previous ``set_password`` / ``change_email``
        public methods. Direct in-app password mutation was removed —
        all password operations must now go through Supabase's
        email-verification flow (``send_password_reset``).
        """
        if self._email_password_set:
            return
        self._email_password_set = True
        # Re-emit identities_changed so UserPanel re-renders the
        # [Email] chip immediately based on the flipped capability.
        self.identities_changed.emit(self._identities)

    def sign_out(self) -> None:
        """Clear local session immediately, then best-effort call /logout in background."""
        self._reset_pending_oauth()
        # Explicit sign-out cancels any armed restore retry — the user
        # made an intentional choice and we should stop trying to log
        # them back in automatically.
        self._reset_restore_retry()
        token = self._access_token
        # Snapshot the leaving uid BEFORE clearing the in-memory session
        # so we delete the correct per-user keyring slot.
        leaving_uid = self._user.user_id if self._user is not None else ""
        if not leaving_uid:
            from application.service.user_workspace import read_active_user
            leaving_uid = read_active_user() or ""
        self._clear_session()
        self._secure.auth_refresh_token(leaving_uid).delete()
        clear_cached_profile()
        self.signed_out.emit()

        # Hot-switch back to the _guest workspace. The leaving user's data
        # stays in ``{slug_A}/`` on disk so signing back in restores it
        # instantly. Machine-level state (setup_state, install_state) is
        # at the WORKSPACE root and is unaffected.
        try:
            from application.service.user_workspace import (
                set_user_workspace,
                reload_workspace_data,
            )
            set_user_workspace(None)
            reload_workspace_data()
            self.workspace_changed.emit("")
            log_info("[auth:sign_out] hot-switched workspace back to _guest")
        except Exception as exc:                                  # noqa: BLE001
            log_warning(f"[auth:sign_out] workspace hot-switch failed: {exc}")

        if self._client is None or not token:
            return

        def _do() -> None:
            try:
                self._client.sign_out(token)
            except (AuthError, AuthNetworkError) as exc:
                log_info(f"[auth:sign_out] server call failed (ignored): {exc}")
            return None

        self._run(_do, on_ok=lambda _r: None, on_fail=lambda _c, _m: None)

    # -- oauth callbacks ---------------------------------------------------

    @pyqtSlot(dict)
    def _on_oauth_server_callback(self, query: dict) -> None:
        self._shutdown_oauth_server()
        self._consume_oauth_query(query, query_fragment={})

    @pyqtSlot()
    def _on_oauth_server_timeout(self) -> None:
        self._shutdown_oauth_server()
        self._pending_pkce = None
        self.auth_error.emit(
            "oauth_timeout",
            "OAuth sign-in timed out. Please try again.",
        )

    def _shutdown_oauth_server(self) -> None:
        if self._oauth_server is not None:
            self._oauth_server.stop()
            self._oauth_server.deleteLater()
            self._oauth_server = None

    def _reset_pending_oauth(self) -> None:
        self._shutdown_oauth_server()
        self._pending_pkce = None
        self._pending_oauth_mode = None
        self._pending_link_provider = ""

    def _consume_oauth_query(self, query: dict, query_fragment: dict) -> None:
        err = query.get("error") or query_fragment.get("error")
        if err:
            msg = (
                query.get("error_description")
                or query_fragment.get("error_description")
                or err
            )
            log_warning(
                f"[auth:callback] OAuth callback returned error: "
                f"code={err!r} message={msg!r} mode={self._pending_oauth_mode!r}"
            )
            # Clear ALL pending OAuth state, not just PKCE — otherwise a
            # failed link attempt leaves _pending_oauth_mode="link" set
            # and blocks future link / sign-in attempts with oauth_busy.
            self._pending_pkce = None
            self._pending_oauth_mode = None
            self._pending_link_provider = ""
            self.auth_error.emit(str(err), str(msg))
            return

        code = query.get("code")
        if code and self._pending_pkce is not None and self._pending_oauth_mode == "link":
            # Identity-linking exchange. Uses the currently-signed-in
            # access token as Bearer; Supabase associates the new
            # identity with the existing user and returns a refreshed
            # session for the SAME user_id. We discard the returned
            # tokens here and only refresh the identities list — the
            # live session remains valid and avoid resetting workspace
            # state mid-link.
            verifier = self._pending_pkce.verifier
            provider = self._pending_link_provider
            self._pending_pkce = None
            self._pending_oauth_mode = None
            self._pending_link_provider = ""
            access = self._access_token

            if not access:
                self.auth_error.emit(
                    "config",
                    "Sign-in expired during link. Please sign in again.",
                )
                return

            def _do_link_exchange() -> dict:
                return self._client.link_identity_pkce_exchange(
                    str(code), verifier, access,
                )

            def _on_linked(resp: dict) -> None:
                # Supabase rotates the session tokens when an identity is
                # added. Adopt the fresh tokens so subsequent calls (e.g.
                # refresh_identities) use the now-canonical pair.
                new_access = str(resp.get("access_token") or "")
                new_refresh = str(resp.get("refresh_token") or "")
                if new_access:
                    self._access_token = new_access
                if new_refresh:
                    self._refresh_token = new_refresh
                    if self._user is not None:
                        self._secure.auth_refresh_token(
                            self._user.user_id
                        ).set(new_refresh)
                log_info(
                    f"[auth:link] {provider!r} identity linked; "
                    f"tokens rotated: access={bool(new_access)} "
                    f"refresh={bool(new_refresh)}"
                )
                self.identity_linked.emit(provider)
                self.refresh_identities()

            self._run(
                _do_link_exchange, on_ok=_on_linked, on_fail=self._emit_auth_error,
            )
            return

        if code and self._pending_pkce is not None:
            verifier = self._pending_pkce.verifier
            self._pending_pkce = None
            self._pending_oauth_mode = None

            def _do_pkce_exchange() -> AuthSession:
                session = self._client.exchange_pkce_code(str(code), verifier)
                # Some Supabase token responses ship a sparse user object
                # (no user_metadata) — hit /user with the fresh access token
                # so avatar_url / display_name are populated before persist.
                if not session.user.avatar_url or not session.user.display_name:
                    try:
                        raw = self._client.get_user_raw(session.access_token)
                        full_user = AuthUser.from_supabase(raw)
                        session.user = full_user
                        log_info(
                            f"[auth:avatar] enriched user via /user "
                            f"— avatar_url set: {bool(full_user.avatar_url)}"
                        )
                        if not full_user.avatar_url:
                            md = raw.get("user_metadata") or {}
                            log_warning(
                                f"[auth:avatar] avatar still missing — "
                                f"full user_metadata={md!r}"
                            )
                    except (AuthError, AuthNetworkError) as exc:
                        log_info(
                            f"[auth:avatar] /user enrichment failed "
                            f"(non-fatal): {exc}"
                        )
                return session

            self._run(
                _do_pkce_exchange,
                on_ok=self._accept_session,
                on_fail=self._emit_auth_error,
            )
            return

        access = query_fragment.get("access_token")
        refresh = query_fragment.get("refresh_token")
        if access and refresh:
            self._pending_pkce = None
            self._pending_oauth_mode = None
            self._pending_link_provider = ""
            expires_in = int(query_fragment.get("expires_in") or 0)
            token_type = query_fragment.get("token_type") or "bearer"

            def _do() -> AuthSession:
                user = self._client.get_user(access)
                return AuthSession(
                    access_token=access,
                    refresh_token=refresh,
                    expires_in=expires_in,
                    token_type=token_type,
                    user=user,
                )
            self._run(_do, on_ok=self._accept_session, on_fail=self._emit_auth_error)
            return

        log_warning(
            f"[auth:callback] Unrecognised OAuth response — query={query!r} "
            f"fragment={query_fragment!r}"
        )

    # -- deeplink callback -------------------------------------------------

    def handle_oauth_callback(self, url: str) -> None:
        """Entry point for the unitport:// custom-scheme fallback."""
        if self._client is None:
            return
        parsed = urlparse(url)
        flat_q = {k: v[0] for k, v in parse_qs(parsed.query or "").items() if v}
        flat_f = {k: v[0] for k, v in parse_qs(parsed.fragment or "").items() if v}
        self._consume_oauth_query(flat_q, flat_f)

    # -- internals ---------------------------------------------------------

    def _run(self, fn, on_ok, on_fail) -> None:
        worker = _AuthWorker(fn, parent=self)
        worker.done.connect(on_ok)
        worker.failed.connect(on_fail)
        worker.finished.connect(lambda w=worker: self._drop_worker(w))
        self._workers.append(worker)
        worker.start()

    def _drop_worker(self, worker: QThread) -> None:
        try:
            self._workers.remove(worker)
        except ValueError:
            pass
        worker.deleteLater()

    @pyqtSlot(object)
    def _accept_session(self, session: AuthSession) -> None:
        # Any successful auth cancels a pending backoff retry, since we
        # are now signed in and the keyring entry has been refreshed.
        self._reset_restore_retry()
        self._access_token = session.access_token
        self._refresh_token = session.refresh_token
        self._user = UserProfile.from_auth_user(session.user)
        # Reset the email-password capability bit; refresh_identities
        # below will repopulate it from app_metadata.providers.
        self._email_password_set = False
        # Per-user keyring slot keyed by Supabase user_id — different
        # accounts no longer overwrite each other's refresh tokens.
        self._secure.auth_refresh_token(session.user.id).set(session.refresh_token)
        save_cached_profile(
            CachedProfile(
                user_id=session.user.id,
                email=session.user.email,
                display_name=session.user.display_name,
                avatar_url=session.user.avatar_url,
                provider=session.user.provider,
            )
        )
        self.authenticated.emit(self._user)

        # Workspace hot-switch. Three cases:
        #
        # 1. Silent restore (shim active_user already equals this uid):
        #    no-op; the live USER_CONFIG_DIR is already correct.
        #
        # 2. Guest → first-login: rename the live ``_guest/`` directory to
        #    the new ``{slug}/`` so the user "claims" the data they were
        #    just working on. reload_paths + reload_workspace_data so the
        #    in-process state follows.
        #
        # 3. Account switch (A → B): flip the shim to ``{slug_B}/``
        #    (created if missing). A's data stays in ``{slug_A}/`` on
        #    disk; B's working set is independent. reload_paths +
        #    reload_workspace_data so projects panel etc. show B's view.
        #
        # No restart is ever required — machine-level state (setup_state,
        # install_state) lives in the WORKSPACE root and is shared.
        try:
            self._handle_workspace_on_accept(session.user)
        except Exception as exc:                                  # noqa: BLE001
            log_warning(f"[auth:accept] workspace hot-switch failed: {exc}")

        if session.user.avatar_url:
            log_info(
                f"[auth:avatar] fetching {session.user.avatar_url[:80]} "
                f"for user {session.user.id[:8]}..."
            )
            self._avatar_cache.fetch(session.user.id, session.user.avatar_url)
        else:
            label = session.user.display_name or session.user.email
            log_info(
                f"[auth:avatar] no avatar_url — synthesizing initials "
                f"bubble for {label!r} (provider={session.user.provider!r})"
            )
            pm = make_initials_pixmap(session.user.id, label, 128)
            self._on_avatar_ready(session.user.id, pm)

        # Pull identities so UserPanel's three status buttons can render
        # the correct linked/unlinked state on first paint after sign-in.
        # Best-effort: any failure logs via auth_error but doesn't block
        # the sign-in itself.
        self.refresh_identities()

    def _handle_workspace_on_accept(self, user: "AuthUser") -> None:
        """Hot-switch USER_CONFIG_DIR to this user's slug directory.

        ``user`` carries id + email + display_name; we prefer email for
        the slug (stable, unique in Supabase), falling back to display_name
        or user_id when email is unavailable.
        """
        from pathlib import Path
        from application.service.user_workspace import (
            read_active_user,
            set_user_workspace,
            reload_workspace_data,
        )
        from unitport_sdk import Paths

        prev_uid = read_active_user()
        # Silent-restore fast path: same uid AND user_config_dir is already
        # anchored at the UUID slug (post-migration layout). When the
        # filesystem still carries the legacy email-derived slug we must
        # fall through to set_user_workspace so the rename triggers.
        if prev_uid == user.id and Path(Paths.USER_CONFIG_DIR).name == user.id:
            return

        username = user.email or user.display_name or user.id
        set_user_workspace(username, user_id=user.id)
        reload_workspace_data()
        log_info(
            f"[auth:accept] hot-switched workspace for {username!r} "
            f"(uid={user.id})"
        )
        self.workspace_changed.emit(user.id)

    @pyqtSlot(str, QPixmap)
    def _on_avatar_ready(self, user_id: str, pixmap: QPixmap) -> None:
        if self._user is not None and self._user.user_id != user_id:
            log_info(
                f"[auth:avatar] dropping late arrival for {user_id[:8]} "
                f"— active user is now {self._user.user_id[:8]}"
            )
            return
        log_info(
            f"[auth:avatar] ready for {user_id[:8]}: {pixmap.width()}x{pixmap.height()}"
        )
        self.avatar_updated.emit(user_id, pixmap)

    @pyqtSlot(str, str)
    def _emit_auth_error(self, code: str, message: str) -> None:
        self.auth_error.emit(code, message)

    @pyqtSlot(str, str)
    def _reject_restore(self, code: str, message: str) -> None:
        log_info(f"[auth:restore] refresh failed ({code}: {message})")
        if code != "network":
            # Token is permanently dead (revoked, expired beyond grace,
            # invalid_grant). Drop the keyring entry so we don't keep
            # retrying with a known-bad token. Use the active workspace
            # uid (the slot we read in try_restore_session).
            from application.service.user_workspace import read_active_user
            uid = read_active_user() or ""
            self._secure.auth_refresh_token(uid).delete()
            self._reset_restore_retry()
            self._clear_session()
            self.signed_out.emit()
            return

        # Transient network failure: keep the keyring entry, emit signed_out
        # so the UI reflects the disconnected state, and arm a backoff timer
        # to retry once the network is presumably back.
        self._clear_session()
        self.signed_out.emit()
        if self._shutting_down:
            return
        idx = min(
            self._restore_retry_attempt, len(self._restore_retry_schedule_s) - 1
        )
        delay_s = self._restore_retry_schedule_s[idx]
        self._restore_retry_attempt += 1
        log_info(
            f"[auth:restore] network failure — retry #{self._restore_retry_attempt} "
            f"in {delay_s}s"
        )
        self._restore_retry_timer.start(delay_s * 1000)

    @pyqtSlot()
    def _on_restore_retry(self) -> None:
        """Fired by the backoff timer; only attempts if we still have a refresh token."""
        if self._shutting_down or self._client is None:
            return
        if self._user is not None:
            # User signed in manually while we were waiting — drop the retry.
            self._reset_restore_retry()
            return
        from application.service.user_workspace import read_active_user
        uid = read_active_user()
        if not uid:
            # Workspace flipped to guest while we were waiting (user signed
            # out via UI). Nothing to retry against.
            self._reset_restore_retry()
            return
        token = self._secure.auth_refresh_token(uid).get()
        if not token:
            self._reset_restore_retry()
            return
        log_info(
            f"[auth:restore] retrying silent refresh (attempt "
            f"#{self._restore_retry_attempt})"
        )
        self._run(
            lambda: self._client.refresh_session(token),
            on_ok=self._accept_session,
            on_fail=self._reject_restore,
        )

    def _reset_restore_retry(self) -> None:
        self._restore_retry_timer.stop()
        self._restore_retry_attempt = 0

    def _clear_session(self) -> None:
        self._access_token = ""
        self._refresh_token = ""
        self._user = None
        # Identities are user-scoped — wipe so a subsequent sign-in
        # doesn't briefly render the previous user's chip state.
        if self._identities:
            self._identities = []
            self.identities_changed.emit([])
        self._email_password_set = False
        self._pending_oauth_mode = None
        self._pending_link_provider = ""

    # -- shutdown ---------------------------------------------------------

    def shutdown(self, timeout_ms: int = 1500) -> None:
        """Cancel pending retries and join all live worker QThreads.

        Called by ``UnitPortMain._shutdown_tasks`` so that an OAuth/sign-out
        call mid-flight at quit cannot trigger "QThread: Destroyed while
        thread is still running". Idempotent.
        """
        self._shutting_down = True
        self._reset_restore_retry()
        self._reset_pending_oauth()
        for w in list(self._workers):
            try:
                w.quit()
                w.wait(timeout_ms)
            except RuntimeError:
                # QThread already deleted by Qt — finished + deleteLater
                # already cleaned it up.
                pass
        self._workers.clear()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


_instance: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """Return the process-wide AuthManager (constructed on first call)."""
    global _instance
    if _instance is None:
        _instance = AuthManager()
    return _instance
