# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Thin REST wrapper around the Supabase Storage REST API.

Mirrors ``supabase_client.SupabaseClient`` in shape (no Qt, no global
state, ``access_token`` passed as a Bearer per call) so we keep one
HTTP style across the auth subsystem. We deliberately do NOT pull in
``supabase-py`` — its postgrest / realtime / gotrue subdeps would
balloon the venv for two endpoints.

Endpoints exercised (all under ``<url>/storage/v1/``):
    POST    /object/{bucket}/{key}            upload (binary body)
    PUT     /object/{bucket}/{key}            upload (upsert via header)
    GET     /object/{bucket}/{key}            download (binary body)
    HEAD    /object/{bucket}/{key}            head (size + etag)
    POST    /object/list/{bucket}             list (json body: prefix, limit, ...)
    DELETE  /object/{bucket}                  remove (json body: prefixes[])

And one PostgREST helper for the quota table:
    GET   /rest/v1/user_storage_quota         quota_get
    PATCH /rest/v1/user_storage_quota         quota_update

Errors:
    - 4xx/5xx with a structured body  -> StorageError(status, code, message)
    - connection / timeout / DNS     -> StorageNetworkError

This wrapper does NOT enforce the 50 MB file-size limit or the
``{user_id}/`` prefix RLS rule — both are server-side. Clients are
expected to obey those policies before hitting the wire.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class StorageError(Exception):
    """Raised on a 4xx/5xx response with a structured body."""

    def __init__(self, status: int, code: str = "", message: str = "") -> None:
        super().__init__(message or code or f"HTTP {status}")
        self.status = status
        self.code = code
        self.message = message or code or f"HTTP {status}"


class StorageNetworkError(Exception):
    """Raised when no response was received (timeout, DNS, refused, ...)."""


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RemoteObject:
    """Trimmed projection of a Supabase Storage ``list`` row.

    Matches what we actually need for diff / progress UI; raw row carries
    more metadata that we ignore.
    """

    name: str                # path relative to bucket root (object key)
    size: int = 0
    etag: str = ""
    content_type: str = ""
    updated_at: str = ""

    @classmethod
    def from_supabase(cls, parent_prefix: str, d: dict) -> "RemoteObject":
        meta = d.get("metadata") or {}
        name = str(d.get("name") or "")
        full = f"{parent_prefix}/{name}" if parent_prefix else name
        # Storage list() returns directories with name only + null metadata.
        size = 0
        if isinstance(meta, dict):
            size = int(meta.get("size") or 0)
        return cls(
            name=full,
            size=size,
            etag=str((meta or {}).get("eTag") or d.get("id") or ""),
            content_type=str((meta or {}).get("mimetype") or ""),
            updated_at=str(d.get("updated_at") or ""),
        )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class SupabaseStorageClient:
    """Stateless Storage wrapper — constructed once with project URL + anon key.

    All operations take ``access_token`` (the user's JWT, plumbed by
    AuthManager.access_token()); the anon key is sent as ``apikey``
    alongside. This keeps the client safe to share across threads — the
    only state it owns is the URL + key + bucket name.
    """

    _STORAGE_PREFIX = "/storage/v1"
    _REST_PREFIX = "/rest/v1"
    _DEFAULT_TIMEOUT = 30.0          # uploads can be a few MB; 15s too tight

    def __init__(
        self,
        url: str,
        anon_key: str,
        *,
        bucket: str = "user-data",
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        if not url or not anon_key:
            raise ValueError(
                "SupabaseStorageClient requires non-empty url and anon_key — "
                "check [auth] section in src/config/system.ini"
            )
        self._base_url = url.rstrip("/")
        self._anon_key = anon_key
        self._bucket = bucket
        self._timeout = timeout

    @property
    def bucket(self) -> str:
        return self._bucket

    # -- Storage object operations -----------------------------------------

    def list(
        self,
        access_token: str,
        prefix: str,
        *,
        limit: int = 1000,
    ) -> List[RemoteObject]:
        """List objects under ``prefix`` recursively.

        Supabase's POST /object/list/{bucket} is shallow by default — it
        only returns the immediate children of ``prefix``. We walk
        breadth-first across "folder" entries (rows with no metadata) so
        the caller gets a flat list of every file under ``prefix``.
        """
        seen: List[RemoteObject] = []
        stack: List[str] = [prefix.rstrip("/")]
        while stack:
            current = stack.pop()
            body = {
                "prefix": current,
                "limit": limit,
                "offset": 0,
                "sortBy": {"column": "name", "order": "asc"},
            }
            data = self._post(
                f"{self._STORAGE_PREFIX}/object/list/{self._bucket}",
                json=body,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            # Supabase returns ``{"data": [...]}`` envelope in some
            # versions; in newer ones it's a top-level list. Be lenient.
            rows: list = data if isinstance(data, list) else list(
                data.get("data") or []
            )
            for row in rows:
                if not isinstance(row, dict):
                    continue
                meta = row.get("metadata")
                if not meta:
                    # Folder row. Recurse if its name is non-empty.
                    sub = str(row.get("name") or "").rstrip("/")
                    if sub:
                        nested = f"{current}/{sub}" if current else sub
                        stack.append(nested)
                    continue
                seen.append(RemoteObject.from_supabase(current, row))
        return seen

    def upload(
        self,
        access_token: str,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
        upsert: bool = True,
    ) -> bool:
        """Upload ``data`` to ``{bucket}/{key}``. Returns True on success."""
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": content_type,
            "x-upsert": "true" if upsert else "false",
        }
        try:
            resp = httpx.post(
                self._url(f"{self._STORAGE_PREFIX}/object/{self._bucket}/{key}"),
                content=data,
                headers=self._headers(headers, content_type=content_type),
                timeout=self._timeout,
            )
        except httpx.RequestError as exc:
            raise StorageNetworkError(str(exc)) from exc
        self._parse_response(resp, expect_empty=True)
        return True

    def download(self, access_token: str, key: str) -> Optional[bytes]:
        """Fetch ``{bucket}/{key}`` raw bytes; None if 404."""
        try:
            resp = httpx.get(
                self._url(f"{self._STORAGE_PREFIX}/object/{self._bucket}/{key}"),
                headers=self._headers(
                    {"Authorization": f"Bearer {access_token}"},
                ),
                timeout=self._timeout,
            )
        except httpx.RequestError as exc:
            raise StorageNetworkError(str(exc)) from exc
        if resp.status_code == 404:
            return None
        if not resp.is_success:
            self._parse_response(resp)         # raises StorageError
        return resp.content

    def head(self, access_token: str, key: str) -> Optional[dict]:
        """HEAD a single object; returns ``{size, etag, content_type}`` or None."""
        try:
            resp = httpx.head(
                self._url(f"{self._STORAGE_PREFIX}/object/{self._bucket}/{key}"),
                headers=self._headers(
                    {"Authorization": f"Bearer {access_token}"},
                ),
                timeout=self._timeout,
            )
        except httpx.RequestError as exc:
            raise StorageNetworkError(str(exc)) from exc
        if resp.status_code == 404:
            return None
        if not resp.is_success:
            self._parse_response(resp)         # raises StorageError
        return {
            "size": int(resp.headers.get("content-length", 0) or 0),
            "etag": str(resp.headers.get("etag", "")).strip('"'),
            "content_type": str(resp.headers.get("content-type", "")),
        }

    def remove(self, access_token: str, keys: List[str]) -> bool:
        """Bulk delete. Returns True if request returned 200/204."""
        if not keys:
            return True
        body = {"prefixes": list(keys)}
        try:
            resp = httpx.request(
                "DELETE",
                self._url(f"{self._STORAGE_PREFIX}/object/{self._bucket}"),
                json=body,
                headers=self._headers(
                    {"Authorization": f"Bearer {access_token}"},
                ),
                timeout=self._timeout,
            )
        except httpx.RequestError as exc:
            raise StorageNetworkError(str(exc)) from exc
        self._parse_response(resp, expect_empty=True)
        return True

    # -- PostgREST quota table ---------------------------------------------

    def quota_get(self, access_token: str, user_id: str) -> Optional[dict]:
        """SELECT row from ``public.user_storage_quota`` for the current user.

        Returns None if no row exists (the registration trigger should
        have created one — its absence usually means the user signed up
        before the trigger was deployed).
        """
        try:
            resp = httpx.get(
                self._url(f"{self._REST_PREFIX}/user_storage_quota"),
                params={"user_id": f"eq.{user_id}", "select": "*"},
                headers=self._rest_headers(access_token),
                timeout=self._timeout,
            )
        except httpx.RequestError as exc:
            raise StorageNetworkError(str(exc)) from exc
        if not resp.is_success:
            self._parse_response(resp)
        try:
            rows = resp.json()
        except ValueError:
            return None
        if isinstance(rows, list) and rows:
            return dict(rows[0])
        return None

    def quota_update(
        self,
        access_token: str,
        user_id: str,
        used_bytes: int,
    ) -> bool:
        """PATCH ``used_bytes`` on the quota row for ``user_id``."""
        try:
            resp = httpx.patch(
                self._url(f"{self._REST_PREFIX}/user_storage_quota"),
                params={"user_id": f"eq.{user_id}"},
                json={"used_bytes": int(used_bytes)},
                headers=self._rest_headers(access_token),
                timeout=self._timeout,
            )
        except httpx.RequestError as exc:
            raise StorageNetworkError(str(exc)) from exc
        if not resp.is_success:
            self._parse_response(resp)
        return True

    def storage_usage_rpc(
        self,
        access_token: str,
        user_id: str,
    ) -> Optional[dict]:
        """Call the server-side ``get_storage_usage`` Postgres function.

        Returns ``{"used_bytes": int, "max_bytes": int}`` or ``None`` if
        the row is missing / RPC errors out. The function is deployed on
        Supabase and computes the user's current usage server-side, which
        is authoritative — we never compute it from local sync state.
        """
        try:
            resp = httpx.post(
                self._url(f"{self._REST_PREFIX}/rpc/get_storage_usage"),
                json={"p_user_id": user_id},
                headers=self._rest_headers(access_token),
                timeout=self._timeout,
            )
        except httpx.RequestError as exc:
            raise StorageNetworkError(str(exc)) from exc
        if not resp.is_success:
            self._parse_response(resp)
        try:
            body = resp.json()
        except ValueError:
            return None
        if isinstance(body, dict):
            return body
        if isinstance(body, list) and body and isinstance(body[0], dict):
            return dict(body[0])
        return None

    # -- low-level HTTP ----------------------------------------------------

    def _headers(
        self,
        extra: Optional[Dict[str, str]] = None,
        *,
        content_type: Optional[str] = None,
    ) -> Dict[str, str]:
        h: Dict[str, str] = {
            "apikey": self._anon_key,
            "Accept": "application/json",
        }
        # Default Content-Type only for json POSTs. Binary uploads supply
        # their own (audio/png/octet-stream/...) and we must not clobber it.
        if content_type is None:
            h["Content-Type"] = "application/json"
        if extra:
            h.update(extra)
        return h

    def _rest_headers(self, access_token: str) -> Dict[str, str]:
        return {
            "apikey": self._anon_key,
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Prefer": "return=representation",
        }

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self._base_url}{path}"

    def _post(
        self,
        path: str,
        *,
        json: Optional[dict] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        try:
            resp = httpx.post(
                self._url(path),
                json=json,
                headers=self._headers(headers),
                timeout=self._timeout,
            )
        except httpx.RequestError as exc:
            raise StorageNetworkError(str(exc)) from exc
        if not resp.is_success:
            self._parse_response(resp)
        try:
            return resp.json()
        except ValueError:
            return {}

    @staticmethod
    def _parse_response(
        resp: httpx.Response,
        *,
        expect_empty: bool = False,
    ) -> dict:
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
                body.get("error")
                or body.get("code")
                or body.get("statusCode")
                or ""
            )
            msg = str(
                body.get("message")
                or body.get("msg")
                or body.get("error_description")
                or ""
            )
        raise StorageError(status=resp.status_code, code=code, message=msg)


__all__ = [
    "SupabaseStorageClient",
    "RemoteObject",
    "StorageError",
    "StorageNetworkError",
]
