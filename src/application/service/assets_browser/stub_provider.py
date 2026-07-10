# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""DefaultAssetBrowserProvider — frontend-round implementation of the seam.

This adapts what already exists into the :mod:`model` DTOs:

* **Motion / Policy** packages come from the download registry
  (``ResourceManager`` over ``custom_mods/.resources.json``) plus the
  scanned community packs (``library.list_motion_packs`` /
  ``scan_policy_packs``) — the same two sources the current Resources panel
  reads, now unioned behind one call.
* **Model** packages come from ``registers.robots``.
* **Canvas** packages are the shipped/downloaded template canvases under
  ``custom_mods/canvas/**/*.canvas.json``.
* **Clips** inside a Motion package are enumerated by walking the package's
  on-disk folder (LAFAN1-style downloads, whose ``<pkg>/<robot>/*.csv``
  layout the ``library`` pack-scanner cannot see) or its ``pack:`` file list
  (community AMP packs).
* **Segments** are real user state, persisted under ``USER_CONFIG_DIR`` via
  the SDK ``DataManager`` and keyed by a hash of the clip ref.

``probe_clip`` now loads the clip for real ``(n_frames, fps)`` (the Clip
Motion Editor's timeline + offscreen render need them). Deferred to the
backend round (see the plan): unifying the two motion-discovery paths into
one index, and acting on ``classify_package``. Those are isolated here so
the UI never has to change when they land.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from unitport_sdk import Paths, load_data, log_warning, save_data

from application.service.resources import ResourceKind, get_resource_manager

from .model import AssetCategory, ClipRow, PackageRow, SegmentRow

_SEGMENT_REL = ("scripts", "training_motion", "segments")
_CLASSIFY_REL = ("scripts", "training_motion", "classifications.json")
_CANVAS_SUFFIX = ".canvas.json"


class DefaultAssetBrowserProvider:
    """Concrete provider — implements the three seam protocols."""

    def __init__(self) -> None:
        self._clip_exts_cache: Optional[frozenset] = None
        self._probe_cache: Dict[str, Tuple[int, float]] = {}
        # IR-role caches for the render-robot compatibility check. Clip roles
        # are a property of the source file (robot-independent), so they cache
        # by clip_ref; robot roles cache by SKU. ``clip_role_overlap`` combines
        # them per (clip_ref, sku). ``False`` = "already looked up, no roles".
        self._clip_roles_cache: Dict[str, frozenset] = {}
        self._robot_roles_cache: Dict[str, frozenset] = {}

    # ------------------------------------------------------------------
    # Read side — categories / packages
    # ------------------------------------------------------------------

    def list_categories(self) -> List[AssetCategory]:
        return [
            AssetCategory.MOTION,
            AssetCategory.POLICY,
            AssetCategory.MODEL,
            AssetCategory.CANVAS,
        ]

    def list_packages(self, category: AssetCategory) -> List[PackageRow]:
        if category == AssetCategory.MOTION:
            return self._motion_packages()
        if category == AssetCategory.POLICY:
            return self._policy_packages()
        if category == AssetCategory.MODEL:
            return self._model_packages()
        if category == AssetCategory.CANVAS:
            return self._canvas_packages()
        raise ValueError(f"[assets_browser] unknown category: {category!r}")

    def _motion_packages(self) -> List[PackageRow]:
        mgr = get_resource_manager()
        rows: List[PackageRow] = []
        seen_dirs: List[Path] = []

        # 1. Download-registry motion entries — provenance + the real folder
        #    (this is where LAFAN1 lives; the pack-scanner below can't see it
        #    because its layout has no datasets/<subdir>).
        for e in mgr.list_motion_entries():
            rows.append(self._entry_row(e, AssetCategory.MOTION, clip_capable=True))
            try:
                seen_dirs.append(e.absolute_path().resolve())
            except OSError:
                pass

        # 2. Scanned community AMP packs — grouped by TOP-LEVEL package so
        #    each package (amp_go2, rl_amp, …) is ONE card; every subdir's
        #    clips surface together in that card's expanded clip table.
        by_pkg: Dict[str, list] = {}
        for pack in mgr.list_motion_packs():
            if pack.files and self._under_any(pack.files[0], seen_dirs):
                continue
            by_pkg.setdefault(pack.package, []).append(pack)
        for package in sorted(by_pkg):
            group = by_pkg[package]
            prov: Dict[str, str] = {"clips": str(sum(p.file_count for p in group))}
            if len(group) > 1:
                prov["subdirs"] = str(len(group))
            rows.append(PackageRow(
                package_id=f"motionpkg:{package}",
                category=AssetCategory.MOTION,
                name=package,
                state="local",
                provenance=prov,
                transport=None,
                clip_capable=True,
                raw_entry_id=None,
                # Install-bundled community packs are managed too (the whole
                # point of ResourceManager) — inspectable + removable.
                openable=True,
                removable=True,
            ))
        return rows

    def _policy_packages(self) -> List[PackageRow]:
        mgr = get_resource_manager()
        rows: List[PackageRow] = []
        indexed = mgr.list_policy_entries()
        indexed_paths = {Path(e.local_path).as_posix() for e in indexed}
        for e in indexed:
            rows.append(self._entry_row(e, AssetCategory.POLICY, clip_capable=False))
        for pack in mgr.scan_policy_packs():
            rel = f"policies/{pack.pack_id}"
            if rel in indexed_paths:
                continue
            rows.append(PackageRow(
                package_id=f"policy:{pack.pack_id}",
                category=AssetCategory.POLICY,
                name=pack.label(),
                state="local",
                provenance={"path": rel},
                transport=None,
                clip_capable=False,
                raw_entry_id=None,
                openable=True,
                removable=True,
            ))
        return rows

    def _model_packages(self) -> List[PackageRow]:
        from registers import robots

        rows: List[PackageRow] = []
        for sku in robots.list_skus():
            spec = robots.get_robot_spec(sku)
            name = spec.name if spec and spec.name else sku
            prov: Dict[str, str] = {"sku": sku}
            if spec is not None:
                if spec.brand:
                    prov["brand"] = spec.brand
                if spec.model:
                    prov["model"] = spec.model
                if spec.families:
                    prov["families"] = ", ".join(spec.families)
            rows.append(PackageRow(
                package_id=f"model:{sku}",
                category=AssetCategory.MODEL,
                name=name,
                state="local",
                provenance=prov,
                transport=None,
                clip_capable=False,
                raw_entry_id=None,
            ))
        return rows

    def _canvas_packages(self) -> List[PackageRow]:
        root = Paths.CUSTOM_MODS_DIR / "canvas"
        rows: List[PackageRow] = []
        if not root.is_dir():
            return rows
        for f in sorted(root.rglob(f"*{_CANVAS_SUFFIX}")):
            if not f.is_file():
                continue
            try:
                rel = f.relative_to(Paths.CUSTOM_MODS_DIR).as_posix()
            except ValueError:
                rel = f.name
            name = f.name[: -len(_CANVAS_SUFFIX)]
            rows.append(PackageRow(
                package_id=f"canvas:{rel}",
                category=AssetCategory.CANVAS,
                name=name or f.stem,
                state="local",
                provenance={"backend": f.parent.name, "path": rel},
                transport=None,
                clip_capable=False,
                raw_entry_id=None,
                # Inspectable; not removable (shipped templates).
                openable=True,
                removable=False,
            ))
        return rows

    def _entry_row(self, e, category: AssetCategory, *, clip_capable: bool) -> PackageRow:
        """Build a PackageRow from a download-registry ``ResourceEntry``."""
        return PackageRow(
            package_id=e.id,
            category=category,
            name=e.name or e.id[:8],
            state=e.state.value,
            provenance=self._entry_provenance(e),
            transport=e.transport.value,
            clip_capable=clip_capable,
            raw_entry_id=e.id,
            error=(e.error if e.state.value == "error" else ""),
            openable=True,
            removable=True,
        )

    @staticmethod
    def _entry_provenance(e) -> Dict[str, str]:
        prov: Dict[str, str] = {"source": e.source.url or "(local import)"}
        if e.source.revision:
            prov["revision"] = e.source.revision
        if e.source.subpath:
            prov["subpath"] = e.source.subpath
        if e.downloaded_at:
            prov["downloaded_at"] = e.downloaded_at
        if e.size_bytes:
            prov["size"] = f"{e.size_bytes // (1024 * 1024)} MiB"
        return prov

    @staticmethod
    def _under_any(path: Path, roots: List[Path]) -> bool:
        try:
            rp = path.resolve()
        except OSError:
            return False
        for root in roots:
            try:
                if rp.is_relative_to(root):
                    return True
            except (ValueError, OSError):
                continue
        return False

    # ------------------------------------------------------------------
    # Read side — clips
    # ------------------------------------------------------------------

    def list_clips(self, package_id: str) -> List[ClipRow]:
        if package_id.startswith("motionpkg:"):
            return self._clips_from_package(package_id[len("motionpkg:"):])
        entry = get_resource_manager().index.get(package_id)
        if entry is not None and entry.asset_kind == ResourceKind.MOTION:
            return self._clips_from_dir(entry.absolute_path())
        # model: / canvas: / policy: → not clip-capable.
        return []

    def _clips_from_package(self, package: str) -> List[ClipRow]:
        """All clips across every ``datasets/<subdir>`` of one AMP package."""
        from scripts.training_motion.labels import infer_task_tag
        from scripts.training_motion.library import build_pack_ref, list_motion_packs

        clips: List[ClipRow] = []
        for pack in list_motion_packs():
            if pack.package != package:
                continue
            for f in pack.files:
                ref = build_pack_ref(pack.package, pack.subdir, f.name)
                clips.append(ClipRow(
                    clip_ref=ref,
                    name=f.stem,
                    task_tag=infer_task_tag(f.name),
                    segment_count=len(self.list_segments(ref)),
                ))
        return clips

    def _clips_from_dir(self, dir_path: Path) -> List[ClipRow]:
        # Frontend-round listing is by extension only. A package may contain a
        # sidecar that shares a clip extension (e.g. LAFAN1's
        # ``dataset_infos.json`` vs the ``.json`` clips amp_legged_gym uses) —
        # it is listed here but fails loud when actually parsed (the backend
        # round's ``probe_clip`` is where content validation belongs, §8).
        from scripts.training_motion.labels import infer_task_tag

        exts = self._clip_exts()
        clips: List[ClipRow] = []
        if not dir_path.is_dir():
            return clips
        for f in sorted(dir_path.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in exts:
                continue
            ref = str(f.resolve())
            clips.append(ClipRow(
                clip_ref=ref,
                name=f.stem,
                task_tag=infer_task_tag(f.name),
                segment_count=len(self.list_segments(ref)),
            ))
        return clips

    def _clip_exts(self) -> frozenset:
        """Recognised motion-clip extensions — SSOT is the loader registry.

        Cached after first use so a panel expand pays the (numpy-pulling)
        loaders import at most once.
        """
        if self._clip_exts_cache is None:
            try:
                from application.training.motion.loaders import _EXT_TO_FORMAT
            except Exception as exc:  # fail loud (§8) — env without the stack
                raise RuntimeError(
                    "[assets_browser] cannot import motion loaders to resolve "
                    f"clip file extensions: {exc}"
                ) from exc
            self._clip_exts_cache = frozenset(_EXT_TO_FORMAT)
        return self._clip_exts_cache

    def probe_clip(self, clip_ref: str) -> Tuple[int, float]:
        """Return the clip's real ``(n_frames, fps)`` by loading it.

        ``n_frames`` / ``fps`` are properties of the *source file* (row
        count + the format's declared sampling rate); the target robot
        only affects joint *reordering*, which a probe ignores. So we load
        with the first robot SKU that the loader accepts — guessing one
        from the clip ref first (a ``…/g1/…`` path loads under the G1 SKU),
        then falling back to the remaining SKUs. Caching makes the
        (numpy-parsing) load cost land once per clip ref.

        Fails loud (CLAUDE.md §8): a missing file, an unknown extension, or
        a clip no registered robot can interpret raises rather than
        returning a sentinel. The Clip Motion Editor wraps this call and
        degrades to a fallback timeline range on failure, so the provider
        never has to lie about the count.
        """
        cached = self._probe_cache.get(clip_ref)
        if cached is not None:
            return cached
        from application.service.runtime.simulation.mujoco.clip_loading import (
            load_clip,
            resolve_clip_path,
            resolve_target_family,
        )

        path = resolve_clip_path(clip_ref)  # fail loud on empty / missing
        candidates = self._probe_candidate_skus(clip_ref)
        if not candidates:
            raise RuntimeError(
                "[assets_browser] probe_clip: no robots are registered to "
                "interpret the clip skeleton."
            )
        last_exc: Optional[Exception] = None
        for sku in candidates:
            try:
                family = resolve_target_family(sku)
                clip = load_clip(path, sku=sku, family=family)
            except Exception as exc:  # try the next robot
                last_exc = exc
                continue
            result = (int(clip.n_frames), float(clip.fps))
            self._probe_cache[clip_ref] = result
            return result
        raise RuntimeError(
            f"[assets_browser] probe_clip could not load {clip_ref!r} with "
            f"any of the {len(candidates)} registered robot(s); last error: "
            f"{last_exc}"
        )

    # ------------------------------------------------------------------
    # Manage — resolve a package's folder / delete its files
    # ------------------------------------------------------------------

    def package_folder(self, package_id: str) -> Optional[Path]:
        """On-disk folder for a package, or ``None`` (registry models have none)."""
        entry = get_resource_manager().index.get(package_id)
        if entry is not None:
            return entry.absolute_path()
        if package_id.startswith("motionpkg:"):
            return self._motion_pack_root(package_id[len("motionpkg:"):])
        if package_id.startswith("policy:"):
            folder = Paths.CUSTOM_MODS_DIR / "policies" / package_id[len("policy:"):]
            return folder if folder.is_dir() else None
        if package_id.startswith("canvas:"):
            f = Paths.CUSTOM_MODS_DIR / package_id[len("canvas:"):]
            return f.parent if f.is_file() else None
        return None  # model: — registry-only, no folder

    def remove_package(self, package_id: str) -> None:
        """Delete a package's files. Raises (§8) when it is not safely removable."""
        mgr = get_resource_manager()
        entry = mgr.index.get(package_id)
        if entry is not None:
            mgr.remove_entry(package_id, delete_files=True)
            return
        if package_id.startswith("motionpkg:"):
            self._rmtree_guarded(
                self._motion_pack_root(package_id[len("motionpkg:"):]),
                self._motion_archive_roots(), package_id,
            )
            return
        if package_id.startswith("policy:"):
            self._rmtree_guarded(
                Paths.CUSTOM_MODS_DIR / "policies" / package_id[len("policy:"):],
                [Paths.CUSTOM_MODS_DIR / "policies"], package_id,
            )
            return
        raise RuntimeError(
            f"[assets_browser] package {package_id!r} is not removable "
            "(registry models / shipped canvases have no deletable files)."
        )

    @staticmethod
    def _motion_archive_roots() -> List[Path]:
        from scripts.training_motion.library import _archive_roots
        return [r.resolve() for r in _archive_roots()]

    def _motion_pack_root(self, package: str) -> Optional[Path]:
        for root in self._motion_archive_roots():
            cand = root / package
            if cand.is_dir():
                return cand.resolve()
        return None

    @staticmethod
    def _rmtree_guarded(
        folder: Optional[Path], allowed_parents: List[Path], package_id: str,
    ) -> None:
        import shutil

        if folder is None or not folder.is_dir():
            raise RuntimeError(
                f"[assets_browser] no folder found to remove for {package_id!r}."
            )
        folder = folder.resolve()
        # Safety: only ever delete a single pack folder sitting DIRECTLY under a
        # known archive root — never a root itself nor an arbitrary path.
        if not any(folder.parent == p.resolve() for p in allowed_parents):
            raise RuntimeError(
                f"[assets_browser] refusing to remove {folder}: not a pack "
                f"folder directly under a managed archive root."
            )
        shutil.rmtree(folder)

    def suggest_sku(self, clip_ref: str) -> Optional[str]:
        """Registered SKU whose MJCF IR roles overlap the clip's the most.

        SKUs are opaque ids, so the editor can't guess the robot from the
        path; overlap is the robust signal (a LAFAN G1 clip's 29 roles match
        the G1 robot fully and quadrupeds not at all). Returns ``None`` when
        the clip can't load or nothing overlaps.
        """
        from registers import robots

        skus = list(robots.list_skus())
        if not skus:
            return None
        clip_roles = self._clip_roles(clip_ref)
        if not clip_roles:
            return None
        best_sku: Optional[str] = None
        best_overlap = 0
        for sku in skus:
            robot_roles = self._robot_roles(sku)
            if not robot_roles:
                continue  # robot has no MJCF joint table — can't render it
            overlap = len(robot_roles & clip_roles)
            if overlap > best_overlap:
                best_sku, best_overlap = sku, overlap
        return best_sku

    def clip_role_overlap(self, clip_ref: str, sku: str) -> Optional[Tuple[int, int]]:
        """``(overlap, clip_role_count)`` between the clip's IR roles and ``sku``'s.

        The clip's IR roles come from the source-format mapping (independent of
        the target robot); ``sku``'s roles come from its MJCF joint table. The
        Clip Motion Editor picker uses the ratio to flag a clip the bound robot
        mostly lacks joints for (``danger_zone`` — playback would be garbage).

        Returns ``None`` when the clip can't be loaded or the robot has no
        joint table. Cached per ``(clip_ref, sku)`` (both role sets memoised)
        so repeated picker rebuilds don't re-parse the clip.
        """
        if not clip_ref or not sku:
            return None
        clip_roles = self._clip_roles(clip_ref)
        if not clip_roles:
            return None
        robot_roles = self._robot_roles(sku)
        if not robot_roles:
            return None
        return len(robot_roles & clip_roles), len(clip_roles)

    def _clip_roles(self, clip_ref: str) -> frozenset:
        """IR roles present in ``clip_ref`` (cached; empty set on load failure)."""
        cached = self._clip_roles_cache.get(clip_ref)
        if cached is not None:
            return cached
        from application.service.runtime.simulation.mujoco.clip_loading import (
            load_clip,
            resolve_clip_path,
            resolve_target_family,
        )

        roles: frozenset = frozenset()
        try:
            path = resolve_clip_path(clip_ref)
        except Exception:
            self._clip_roles_cache[clip_ref] = roles
            return roles
        for sku in self._probe_candidate_skus(clip_ref):
            try:
                clip = load_clip(path, sku=sku, family=resolve_target_family(sku))
            except Exception:
                continue
            roles = frozenset(str(r) for r in (clip.ir_roles or []))
            break
        self._clip_roles_cache[clip_ref] = roles
        return roles

    def _robot_roles(self, sku: str) -> frozenset:
        """IR roles the robot ``sku`` actuates (cached; empty on missing table)."""
        cached = self._robot_roles_cache.get(sku)
        if cached is not None:
            return cached
        from application.service.runtime.simulation.mujoco.clip_frame_apply import (
            build_name_to_role,
        )

        roles: frozenset = frozenset()
        try:
            roles = frozenset(build_name_to_role(sku).values())
        except Exception:
            roles = frozenset()
        self._robot_roles_cache[sku] = roles
        return roles

    @staticmethod
    def _probe_candidate_skus(clip_ref: str) -> List[str]:
        """Registered SKUs ordered guess-first (token match on the ref)."""
        from registers import robots

        skus = list(robots.list_skus())
        low = clip_ref.lower().replace("\\", "/")
        guessed = [s for s in skus if s.split(":")[-1].lower() in low]
        rest = [s for s in skus if s not in guessed]
        return guessed + rest

    # ------------------------------------------------------------------
    # Segments — real user state under USER_CONFIG_DIR
    # ------------------------------------------------------------------

    def list_segments(self, clip_ref: str) -> List[SegmentRow]:
        out: List[SegmentRow] = []
        for s in self._read_raw(clip_ref):
            try:
                out.append(SegmentRow(
                    segment_id=str(s["segment_id"]),
                    clip_ref=clip_ref,
                    name=str(s.get("name", "")),
                    start_frame=int(s["start_frame"]),
                    end_frame=int(s["end_frame"]),
                    task_tag=str(s.get("task_tag", "")),
                ))
            except (KeyError, ValueError, TypeError) as exc:
                log_warning(
                    f"[assets_browser] dropping malformed segment in {clip_ref!r}: {exc}"
                )
        return out

    def add_segment(
        self,
        clip_ref: str,
        *,
        name: str,
        start_frame: int,
        end_frame: int,
        task_tag: str = "",
    ) -> SegmentRow:
        name = (name or "").strip()
        if not name:
            raise ValueError("[assets_browser] segment name is required")
        if int(end_frame) <= int(start_frame):
            raise ValueError(
                f"[assets_browser] segment end_frame ({end_frame}) must be > "
                f"start_frame ({start_frame})"
            )
        segs = self._read_raw(clip_ref)
        record = {
            "segment_id": uuid.uuid4().hex,
            "name": name,
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "task_tag": str(task_tag or ""),
        }
        segs.append(record)
        self._write_raw(clip_ref, segs)
        return self._row(clip_ref, record)

    def rename_segment(self, clip_ref: str, segment_id: str, name: str) -> SegmentRow:
        name = (name or "").strip()
        if not name:
            raise ValueError("[assets_browser] segment name is required")
        segs = self._read_raw(clip_ref)
        target = self._find(segs, segment_id, clip_ref)
        target["name"] = name
        self._write_raw(clip_ref, segs)
        return self._row(clip_ref, target)

    def set_segment_tag(self, clip_ref: str, segment_id: str, task_tag: str) -> SegmentRow:
        segs = self._read_raw(clip_ref)
        target = self._find(segs, segment_id, clip_ref)
        target["task_tag"] = str(task_tag or "")
        self._write_raw(clip_ref, segs)
        return self._row(clip_ref, target)

    def delete_segment(self, clip_ref: str, segment_id: str) -> None:
        segs = self._read_raw(clip_ref)
        kept = [s for s in segs if s.get("segment_id") != segment_id]
        if len(kept) == len(segs):
            raise KeyError(
                f"[assets_browser] segment {segment_id!r} not found for clip {clip_ref!r}"
            )
        self._write_raw(clip_ref, kept)

    @staticmethod
    def _find(segs: List[dict], segment_id: str, clip_ref: str) -> dict:
        for s in segs:
            if s.get("segment_id") == segment_id:
                return s
        raise KeyError(
            f"[assets_browser] segment {segment_id!r} not found for clip {clip_ref!r}"
        )

    @staticmethod
    def _row(clip_ref: str, record: dict) -> SegmentRow:
        return SegmentRow(
            segment_id=str(record["segment_id"]),
            clip_ref=clip_ref,
            name=str(record.get("name", "")),
            start_frame=int(record["start_frame"]),
            end_frame=int(record["end_frame"]),
            task_tag=str(record.get("task_tag", "")),
        )

    def _read_raw(self, clip_ref: str) -> List[dict]:
        if not Paths.is_user_config_dir_set():
            return []
        path = self._segment_file(Paths.USER_CONFIG_DIR, clip_ref)
        if not path.exists():
            return []
        try:
            data = load_data(path)
        except Exception as exc:
            log_warning(
                f"[assets_browser] failed to read segments for {clip_ref!r}: {exc}"
            )
            return []
        if not isinstance(data, dict):
            return []
        raw = data.get("segments") or []
        if not isinstance(raw, list):
            return []
        return [dict(s) for s in raw if isinstance(s, dict)]

    def _write_raw(self, clip_ref: str, segments: List[dict]) -> None:
        root = Paths.require_user_config_dir()  # raises if unset (§8/§1.4)
        path = self._segment_file(root, clip_ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_data(path, {"version": 1, "clip_ref": clip_ref, "segments": segments})

    @staticmethod
    def _segment_file(user_root: Path, clip_ref: str) -> Path:
        digest = hashlib.sha1(clip_ref.encode("utf-8")).hexdigest()
        return user_root.joinpath(*_SEGMENT_REL) / f"{digest}.json"

    # ------------------------------------------------------------------
    # Classify — stub (records the choice, does not yet route the asset)
    # ------------------------------------------------------------------

    def classify_package(
        self, package_id: str, *, asset_kind: str, robot_family: str
    ) -> None:
        asset_kind = (asset_kind or "").strip()
        if not asset_kind:
            raise ValueError("[assets_browser] asset_kind is required")
        path = Paths.require_user_config_dir().joinpath(*_CLASSIFY_REL)
        path.parent.mkdir(parents=True, exist_ok=True)
        data: Dict[str, dict] = {}
        if path.exists():
            loaded = load_data(path)
            if isinstance(loaded, dict) and isinstance(loaded.get("classifications"), dict):
                data = dict(loaded["classifications"])
        # NOTE (frontend round): recorded but NOT yet routed into the right
        # registry — the backend round consumes this overlay. The dialog
        # tells the user it is recorded but not active (no §8 illusion).
        data[package_id] = {
            "asset_kind": asset_kind,
            "robot_family": (robot_family or "").strip(),
        }
        save_data(path, {"version": 1, "classifications": data})


__all__ = ["DefaultAssetBrowserProvider"]
