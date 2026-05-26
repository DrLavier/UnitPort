# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Thin REST wrapper around the Supabase Auth (gotrue) REST API.

No Qt, no global state. All methods are synchronous and meant to be
called from a worker thread (``AuthManager`` wraps them in ``QThread``
subclasses).

Endpoints exercised (all under ``<url>/auth/v1/``):
    POST  /signup                         sign_up
    POST  /token?grant_type=password      sign_in_password
    POST  /token?grant_type=refresh_token refresh_session
    POST  /token?grant_type=pkce          exchange_pkce_code
    POST  /logout                         sign_out
    GET   /user                           get_user / list_identities
    PUT   /user                           update_user (password / email)
    POST  /recover                        send_password_reset
    GET   /authorize                      build_oauth_url (browser only)
    GET   /user/identities/authorize      build_link_url (Bearer-authed)
    DEL   /user/identities/{id}           unlink_identity
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import httpx


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AuthError(Exception):
    """Raised when Supabase returns a 4xx/5xx with a structured error body."""

    def __init__(self, status: int, code: str = "", message: str = "") -> None:
        super().__init__(message or code or f"HTTP {status}")
        self.status = status
        self.code = code
        self.message = message or code or f"HTTP {status}"


class AuthNetworkError(Exception):
    """Raised for connection / timeout / DNS errors (no response received)."""


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AuthSession:
    """Tokens + the user profile returned by a successful auth call."""

    access_token: str
    refresh_token: str
    expires_in: int
    token_type: str
    user: "AuthUser"


@dataclass
class AuthUser:
    """Subset of the Supabase user object that the UI cares about."""

    id: str
    email: str
    display_name: str = ""
    avatar_url: str = ""
    provider: str = ""

    @classmethod
    def from_supabase(cls, d: dict) -> "AuthUser":
        md = d.get("user_metadata") or {}
        app_md = d.get("app_metadata") or {}
        provider = app_md.get("provider") or ""
        display_name = (
            md.get("full_name")
            or md.get("name")
            or md.get("preferred_username")
            or md.get("user_name")
            or ""
        )
        avatar_url = (
            md.get("avatar_url")
            or md.get("picture")
            or md.get("profile_image_url")
            or ""
        )
        return cls(
            id=str(d.get("id", "")),
            email=str(d.get("email", "")),
            display_name=str(display_name),
            avatar_url=str(avatar_url),
            provider=str(provider),
        )


@dataclass
class PkceChallenge:
    """Verifier + challenge pair for PKCE-protected OAuth flow."""

    verifier: str
    challenge: str


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class SupabaseClient:
    """Thin REST wrapper — constructed once with project URL + anon key."""

    _AUTH_PREFIX = "/auth/v1"
    _DEFAULT_TIMEOUT = 15.0

    def __init__(
        self,
        url: str,
        anon_key: str,
        *,
        redirect_url: str = "unitport://auth-callback",
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        if not url or not anon_key:
            raise ValueError(
                "SupabaseClient requires non-empty url and anon_key — "
                "check [auth] section in src/config/system.ini"
            )
        self._base_url = url.rstrip("/")
        self._anon_key = anon_key
        self._redirect_url = redirect_url
        self._timeout = timeout

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def redirect_url(self) -> str:
        return self._redirect_url

    # -- auth flows ---------------------------------------------------------

    def sign_up(self, email: str, password: str) -> AuthSession:
        body = {
            "email": email,
            "password": password,
            "options": {"email_redirect_to": self._redirect_url},
        }
        data = self._post("/signup", json=body)
        return self._parse_session(data)

    def sign_in_password(self, email: str, password: str) -> AuthSession:
        data = self._post(
            "/token",
            params={"grant_type": "password"},
            json={"email": email, "password": password},
        )
        return self._parse_session(data)

    def refresh_session(self, refresh_token: str) -> AuthSession:
        data = self._post(
            "/token",
            params={"grant_type": "refresh_token"},
            json={"refresh_token": refresh_token},
        )
        return self._parse_session(data)

    def exchange_pkce_code(self, code: str, verifier: str) -> AuthSession:
        data = self._post(
            "/token",
            params={"grant_type": "pkce"},
            json={"auth_code": code, "code_verifier": verifier},
        )
        return self._parse_session(data)

    def sign_out(self, access_token: str) -> None:
        self._post(
            "/logout",
            headers={"Authorization": f"Bearer {access_token}"},
            expect_empty=True,
        )

    def get_user(self, access_token: str) -> AuthUser:
        data = self._get(
            "/user",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        return AuthUser.from_supabase(data)

    def get_user_raw(self, access_token: str) -> dict:
        return self._get(
            "/user",
            headers={"Authorization": f"Bearer {access_token}"},
        )

    def send_password_reset(self, email: str) -> None:
        self._post(
            "/recover",
            json={
                "email": email,
                "options": {"email_redirect_to": self._redirect_url},
            },
            expect_empty=True,
        )

    # -- identity linking --------------------------------------------------

    def list_identities(self, access_token: str) -> list:
        """Return ``user.identities[]`` for the currently signed-in user.

        Each entry is a Supabase identity dict with at least
        ``provider`` / ``identity_id`` / ``identity_data`` (which carries
        ``email``, ``sub``, ...). Used by AuthManager to render the
        three "linked methods" status buttons in UserPanel.
        """
        raw = self._get(
            "/user",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        ids = raw.get("identities") or []
        return list(ids) if isinstance(ids, list) else []

    def build_link_url(
        self,
        provider: str,
        access_token: str,
        *,
        pkce: PkceChallenge,
        redirect_override: Optional[str] = None,
    ) -> str:
        """Authorize URL for linking a new identity to the *current* user.

        Hits ``GET /user/identities/authorize`` server-side (with Bearer
        in the Authorization header — the OS browser cannot set custom
        headers). By default Supabase responds with a 303 redirect to
        the provider's OAuth URL, which is useless for us — we want to
        open the URL ourselves via QDesktopServices. So we set BOTH
        ``skip_http_redirect=true`` (newer GoTrue) AND the
        ``X-Skip-Http-Redirect`` header (older GoTrue) — whichever the
        backend understands, the response becomes JSON ``{"url": "..."}``.
        """
        params: Dict[str, str] = {
            "provider": provider,
            "redirect_to": redirect_override or self._redirect_url,
            "code_challenge": pkce.challenge,
            "code_challenge_method": "s256",
            "flow_type": "pkce",
            "skip_http_redirect": "true",
        }
        data = self._get(
            "/user/identities/authorize",
            params=params,
            headers={
                "Authorization": f"Bearer {access_token}",
                "X-Skip-Http-Redirect": "true",
            },
        )
        return str(data.get("url") or "")

    def link_identity_pkce_exchange(
        self,
        code: str,
        verifier: str,
        access_token: str,
    ) -> dict:
        """Exchange the link-flow ``code`` for an updated session.

        Same endpoint as sign-in PKCE exchange — Supabase distinguishes
        sign-in vs link by the OAuth state it set when we hit
        /user/identities/authorize with Bearer. The response carries the
        same user_id (no new account is created); the new identity row
        is appended server-side.
        """
        return self._post(
            "/token",
            params={"grant_type": "pkce"},
            json={"auth_code": code, "code_verifier": verifier},
            headers={"Authorization": f"Bearer {access_token}"},
        )

    def unlink_identity(self, access_token: str, identity_id: str) -> None:
        """Remove one identity from the current user.

        Supabase rejects deletion of the only remaining identity with a
        422 carrying ``single_identity_not_deletable`` — that error
        bubbles up via AuthError and AuthManager translates it into a
        friendly status-bar message in UserPanel.
        """
        self._delete(
            f"/user/identities/{identity_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            expect_empty=True,
        )

    def update_user(
        self,
        access_token: str,
        *,
        password: Optional[str] = None,
        email: Optional[str] = None,
    ) -> dict:
        """Set/change password or change the primary email.

        Email changes send a confirmation mail and only take effect once
        the user clicks the link; the response here is the unmodified
        user object (the change is pending server-side).
        """
        body: Dict[str, Any] = {}
        if password is not None:
            body["password"] = password
        if email is not None:
            body["email"] = email
        return self._put(
            "/user",
            json=body,
            headers={"Authorization": f"Bearer {access_token}"},
        )

    # -- OAuth URL (browser only) ------------------------------------------

    @staticmethod
    def generate_pkce() -> PkceChallenge:
        verifier = _urlsafe_b64(secrets.token_bytes(32))
        digest = hashlib.sha256(verifier.encode("ascii")).digest()
        challenge = _urlsafe_b64(digest)
        return PkceChallenge(verifier=verifier, challenge=challenge)

    def build_oauth_url(
        self,
        provider: str,
        *,
        pkce: Optional[PkceChallenge] = None,
        redirect_override: Optional[str] = None,
    ) -> str:
        params: Dict[str, str] = {
            "provider": provider,
            "redirect_to": redirect_override or self._redirect_url,
        }
        if pkce is not None:
            params["code_challenge"] = pkce.challenge
            params["code_challenge_method"] = "s256"
            params["flow_type"] = "pkce"
        return f"{self._base_url}{self._AUTH_PREFIX}/authorize?{urlencode(params)}"

    # -- low-level HTTP ----------------------------------------------------

    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        h = {
            "apikey": self._anon_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if extra:
            h.update(extra)
        return h

    def _url(self, path: str) -> str:
        return f"{self._base_url}{self._AUTH_PREFIX}{path}"

    def _post(
        self,
        path: str,
        *,
        json: Optional[dict] = None,
        params: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        expect_empty: bool = False,
    ) -> dict:
        try:
            resp = httpx.post(
                self._url(path),
                json=json,
                params=params,
                headers=self._headers(headers),
                timeout=self._timeout,
            )
        except httpx.RequestError as exc:
            raise AuthNetworkError(str(exc)) from exc
        return self._parse_response(resp, expect_empty=expect_empty)

    def _get(
        self,
        path: str,
        *,
        params: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> dict:
        try:
            resp = httpx.get(
                self._url(path),
                params=params,
                headers=self._headers(headers),
                timeout=self._timeout,
            )
        except httpx.RequestError as exc:
            raise AuthNetworkError(str(exc)) from exc
        return self._parse_response(resp)

    def _put(
        self,
        path: str,
        *,
        json: Optional[dict] = None,
        params: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        expect_empty: bool = False,
    ) -> dict:
        try:
            resp = httpx.put(
                self._url(path),
                json=json,
                params=params,
                headers=self._headers(headers),
                timeout=self._timeout,
            )
        except httpx.RequestError as exc:
            raise AuthNetworkError(str(exc)) from exc
        return self._parse_response(resp, expect_empty=expect_empty)

    def _delete(
        self,
        path: str,
        *,
        params: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        expect_empty: bool = False,
    ) -> dict:
        try:
            resp = httpx.delete(
                self._url(path),
                params=params,
                headers=self._headers(headers),
                timeout=self._timeout,
            )
        except httpx.RequestError as exc:
            raise AuthNetworkError(str(exc)) from exc
        return self._parse_response(resp, expect_empty=expect_empty)

    @staticmethod
    def _parse_response(resp: httpx.Response, *, expect_empty: bool = False) -> dict:
        if resp.status_code == 204 or (expect_empty and resp.status_code < 300):
            return {}
        try:
            body: Any = resp.json()
        except ValueError:
            body = {"msg": resp.text[:500]}

        if resp.is_success:
            return body if isinstance(body, dict) else {"data": body}

        code = ""
        msg = ""
        if isinstance(body, dict):
            code = str(
                body.get("error_code")
                or body.get("error")
                or body.get("code")
                or ""
            )
            msg = str(
                body.get("error_description")
                or body.get("msg")
                or body.get("message")
                or ""
            )
        raise AuthError(status=resp.status_code, code=code, message=msg)

    @staticmethod
    def _parse_session(data: dict) -> AuthSession:
        access_token = str(data.get("access_token", ""))
        refresh_token = str(data.get("refresh_token", ""))
        expires_in = int(data.get("expires_in", 0) or 0)
        token_type = str(data.get("token_type", "bearer"))

        user_dict = data.get("user")
        if not isinstance(user_dict, dict):
            user_dict = data if "id" in data else {}

        if not access_token:
            raise AuthError(
                status=200,
                code="email_confirmation_required",
                message=(
                    "Account created. Please confirm your email to sign in."
                ),
            )

        return AuthSession(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=expires_in,
            token_type=token_type,
            user=AuthUser.from_supabase(user_dict),
        )


def _urlsafe_b64(data: bytes) -> str:
    """Base64url-encode without padding — matches RFC 7636 PKCE spec."""
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")
