"""application.ui.canvas.undo — QUndoStack 命令类 / Canvas undo commands.

每个用户可见的 mutating 操作对应一个 ``QUndoCommand`` 子类。命令在 ``CanvasPage``
的 ``_undo_stack`` 上 push；执行体调 page 的 raw helper 路径，不再走 user-facing
入口（避免命令在自身 redo 中递归 push）。

所有命令在 redo / undo 期间通过 ``page._in_undo = True`` 抑制 ParamRow.set_value
和 NodeItem.mouseRelease 等回调路径再次 push。

复合操作（粘贴 / 复制选中 / 节点删除带边 / 整节点断边）走 PasteCmd /
DeleteSelectionCmd / DisconnectAllEdgesCmd 单条命令——不要拆成多条原子命令再
``beginMacro``，否则 undo 一次只回滚一条；用户期望"一次操作 = 一次撤销"。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QUndoCommand


# =============================================================================
# 内部小工具
# =============================================================================

def _enter_undo(page) -> None:
    page._in_undo = True


def _leave_undo(page) -> None:
    page._in_undo = False


# =============================================================================
# SpawnNodeCmd —— 用户从 NodeLibrary / 右键菜单创建节点
# =============================================================================

class SpawnNodeCmd(QUndoCommand):
    """单节点创建。``instance_id`` 在首次 redo 时锁定，后续 redo / undo 复用同 id."""

    def __init__(
        self,
        page,
        schema_id: str,
        x: float,
        y: float,
        params: Optional[dict] = None,
        instance_id: Optional[str] = None,
        text: str = "",
    ) -> None:
        super().__init__(text or f"Spawn {schema_id}")
        self._page = page
        self._schema_id = schema_id
        self._x = float(x)
        self._y = float(y)
        self._params = dict(params or {})
        self._instance_id = instance_id

    def redo(self) -> None:
        _enter_undo(self._page)
        try:
            item = self._page._spawn_node_raw(
                self._schema_id,
                self._x,
                self._y,
                instance_id=self._instance_id,
                params=self._params,
            )
        finally:
            _leave_undo(self._page)
        if item is None:
            self.setObsolete(True)
            return
        self._instance_id = item.node_id

    def undo(self) -> None:
        if self._instance_id is None:
            return
        _enter_undo(self._page)
        try:
            self._page._delete_node_raw(self._instance_id)
        finally:
            _leave_undo(self._page)


# =============================================================================
# DeleteSelectionCmd —— 删除选中节点 + 选中边
# =============================================================================

class DeleteSelectionCmd(QUndoCommand):
    """删除一组节点 + 一组独立选中边。

    ``snapshot`` 是 ``_serialize_full_selection`` 的结果，含原 ``id`` / ``ui`` /
    ``params``、内部边、loose_edges（独立选中、两端不全在节点选中集中的边）。
    redo 按快照逐个删除；undo 通过 ``_apply_snapshot_raw`` 原地恢复（同 id）。
    """

    def __init__(self, page, snapshot: Dict[str, Any]) -> None:
        super().__init__("Delete selection")
        self._page = page
        self._snapshot = snapshot

    def redo(self) -> None:
        _enter_undo(self._page)
        try:
            for n in self._snapshot.get("nodes", []):
                self._page._delete_node_raw(n.get("id"))
            for e in self._snapshot.get("loose_edges", []):
                self._page._disconnect_edge_by_id_raw(e.get("id"))
            # 残留 group(成员快照与实际不一致等)显式清掉。_delete_node_raw 在
            # 成员清空时会自动销毁 group,这里兜底。
            for g in self._snapshot.get("groups", []):
                gid = g.get("id")
                if gid in getattr(self._page, "_groups", {}):
                    self._page._delete_group_raw(gid)
        finally:
            _leave_undo(self._page)

    def undo(self) -> None:
        _enter_undo(self._page)
        try:
            self._page._apply_snapshot_raw(
                self._snapshot,
                offset=QPointF(0.0, 0.0),
                fixed_node_ids=[n["id"] for n in self._snapshot.get("nodes", [])],
                fixed_edge_ids=[e.get("id") for e in self._snapshot.get("edges", [])],
            )
            for e in self._snapshot.get("loose_edges", []):
                self._page._reconnect_edge_raw(e)
            # 节点回来之后,恢复被删除的 group(直接选中的 + 因级联而消失的)
            for g in self._snapshot.get("groups", []):
                self._page._restore_group_raw(g)
        finally:
            _leave_undo(self._page)


# =============================================================================
# MoveNodeCmd —— 拖动结束去抖
# =============================================================================

class MoveNodeCmd(QUndoCommand):
    """一次鼠标按下→释放期间所有被拖动节点的位移。"""

    def __init__(
        self,
        page,
        moves: Sequence[Tuple[str, QPointF, QPointF]],
    ) -> None:
        super().__init__("Move node(s)")
        self._page = page
        self._moves: List[Tuple[str, QPointF, QPointF]] = [
            (str(inst), QPointF(old), QPointF(new)) for inst, old, new in moves
        ]
        if not self._moves:
            self.setObsolete(True)

    def redo(self) -> None:
        self._apply(use_old=False)

    def undo(self) -> None:
        self._apply(use_old=True)

    def _apply(self, use_old: bool) -> None:
        _enter_undo(self._page)
        try:
            for inst_id, old, new in self._moves:
                target = old if use_old else new
                # ``__group__:<id>`` 占位 key → 解析到 page._groups
                if isinstance(inst_id, str) and inst_id.startswith("__group__:"):
                    gid = inst_id.split(":", 1)[1]
                    item = getattr(self._page, "_groups", {}).get(gid)
                    if item is None:
                        continue
                    # group 在 undo 中移动自身时,临时挂起成员联动,避免
                    # 与同命令中成员的显式 setPos 互相叠加
                    item._suspend_member_sync = True
                    try:
                        item.setPos(target)
                        item._last_pos = QPointF(target)
                    finally:
                        item._suspend_member_sync = False
                    continue
                item = self._page._instances.get(inst_id)
                if item is None:
                    continue
                item.setPos(target)
        finally:
            _leave_undo(self._page)


# =============================================================================
# ConnectCmd —— 创建一条边
# =============================================================================

class ConnectCmd(QUndoCommand):
    """新建一条边。

    端口拖拽释放路径下，``ConnectionItem`` 已经被 ``PortDragController`` 加进 scene
    了——传 ``already_created=True``，首次 redo 跳过实际创建（避免 double-add）。
    后续 undo / redo 走标准创建/断开循环。
    """

    def __init__(
        self,
        page,
        edge_id: str,
        src_inst: str,
        src_port: str,
        dst_inst: str,
        dst_port: str,
        already_created: bool = False,
    ) -> None:
        super().__init__("Connect")
        self._page = page
        self._edge_id = edge_id
        self._src_inst = src_inst
        self._src_port = src_port
        self._dst_inst = dst_inst
        self._dst_port = dst_port
        self._first_redo_skip = bool(already_created)

    def redo(self) -> None:
        if self._first_redo_skip:
            self._first_redo_skip = False
            return
        _enter_undo(self._page)
        try:
            edge = self._page._connect_raw(
                self._src_inst, self._src_port,
                self._dst_inst, self._dst_port,
                edge_id=self._edge_id,
            )
            if edge is None:
                self.setObsolete(True)
        finally:
            _leave_undo(self._page)

    def undo(self) -> None:
        _enter_undo(self._page)
        try:
            self._page._disconnect_edge_by_id_raw(self._edge_id)
        finally:
            _leave_undo(self._page)


# =============================================================================
# DisconnectCmd —— 用户主动断一条边（边右键 / 节点右键 disconnect_all 单条）
# =============================================================================

class DisconnectCmd(QUndoCommand):
    """断一条边。redo 走 raw 断开；undo 通过 (src/dst) 数据重建并保留 edge_id."""

    def __init__(
        self,
        page,
        edge_id: str,
        src_inst: str,
        src_port: str,
        dst_inst: str,
        dst_port: str,
    ) -> None:
        super().__init__("Disconnect")
        self._page = page
        self._edge_id = edge_id
        self._src_inst = src_inst
        self._src_port = src_port
        self._dst_inst = dst_inst
        self._dst_port = dst_port

    def redo(self) -> None:
        _enter_undo(self._page)
        try:
            self._page._disconnect_edge_by_id_raw(self._edge_id)
        finally:
            _leave_undo(self._page)

    def undo(self) -> None:
        _enter_undo(self._page)
        try:
            self._page._connect_raw(
                self._src_inst, self._src_port,
                self._dst_inst, self._dst_port,
                edge_id=self._edge_id,
            )
        finally:
            _leave_undo(self._page)


# =============================================================================
# DisconnectAllEdgesCmd —— 节点右键 "Disconnect All Edges" 一次操作
# =============================================================================

class DisconnectAllEdgesCmd(QUndoCommand):
    """把节点上挂的所有边在一条命令里全断；undo 全部按原 id 重建."""

    def __init__(self, page, node_inst_id: str, edges: Sequence[Dict[str, Any]]) -> None:
        super().__init__("Disconnect all edges")
        self._page = page
        self._node = node_inst_id
        self._edges: List[Dict[str, Any]] = [dict(e) for e in edges]
        if not self._edges:
            self.setObsolete(True)

    def redo(self) -> None:
        _enter_undo(self._page)
        try:
            for e in self._edges:
                self._page._disconnect_edge_by_id_raw(e["id"])
        finally:
            _leave_undo(self._page)

    def undo(self) -> None:
        _enter_undo(self._page)
        try:
            for e in self._edges:
                self._page._reconnect_edge_raw(e)
        finally:
            _leave_undo(self._page)


# =============================================================================
# ParamChangeCmd —— ParamRow.set_value 提交
# =============================================================================

class ParamChangeCmd(QUndoCommand):
    def __init__(
        self,
        page,
        inst_id: str,
        key: str,
        old: Any,
        new: Any,
    ) -> None:
        super().__init__(f"Change {key}")
        self._page = page
        self._inst = inst_id
        self._key = key
        self._old = old
        self._new = new

    def redo(self) -> None:
        self._apply(self._new)

    def undo(self) -> None:
        self._apply(self._old)

    def _apply(self, value: Any) -> None:
        _enter_undo(self._page)
        try:
            self._page._set_param_raw(self._inst, self._key, value)
        finally:
            _leave_undo(self._page)


# =============================================================================
# PasteCmd —— 粘贴 / 复制选中（_import_snapshot 一次操作）
# =============================================================================

class PasteCmd(QUndoCommand):
    """把一份 ``_serialize_selection`` 快照重新落到画布。

    首次 redo 时 ``_apply_snapshot_raw`` 自动生成新 instance_id + 新 edge_id，
    并把这两组 id 锁存到命令上。后续 redo / undo 都用同一组 id 重做或回滚，
    保证多次往返画布状态稳定。
    """

    def __init__(
        self,
        page,
        snapshot: Dict[str, Any],
        offset: QPointF,
        text: str = "Paste",
    ) -> None:
        super().__init__(text)
        self._page = page
        self._snapshot = snapshot
        self._offset = QPointF(offset)
        self._node_ids: Optional[List[str]] = None
        self._edge_ids: Optional[List[str]] = None

    def redo(self) -> None:
        _enter_undo(self._page)
        try:
            node_ids, edge_ids = self._page._apply_snapshot_raw(
                self._snapshot,
                offset=self._offset,
                fixed_node_ids=self._node_ids,
                fixed_edge_ids=self._edge_ids,
            )
            self._node_ids = node_ids
            self._edge_ids = edge_ids
        finally:
            _leave_undo(self._page)

    def undo(self) -> None:
        if not self._node_ids:
            return
        _enter_undo(self._page)
        try:
            for eid in self._edge_ids or []:
                self._page._disconnect_edge_by_id_raw(eid)
            for nid in self._node_ids:
                self._page._delete_node_raw(nid)
        finally:
            _leave_undo(self._page)


# =============================================================================
# CreateGroupCmd —— 把若干 node 包进一个新 GroupItem
# =============================================================================

class CreateGroupCmd(QUndoCommand):
    """新建分组框。

    ``snapshot`` 形如::
        {
            "id": "<group_uuid12>",
            "title": "Group",
            "members": ["<node_id>", ...],
            "rect": {"x": .., "y": .., "w": .., "h": ..},
            "evicted": [                   # 来自旧 group 的成员 → 旧 group id
                {"old_group": "<gid>", "node": "<nid>"}, ...
            ],
            "removed_old_groups": [        # 旧 group 因被掏空而销毁的快照（用于 undo 重建）
                {"id": "<gid>", "title": ..., "members": [...], "rect": {...}},
            ],
        }
    """

    def __init__(self, page, snapshot: Dict[str, Any]) -> None:
        super().__init__("Create group")
        self._page = page
        self._snapshot = snapshot

    def redo(self) -> None:
        _enter_undo(self._page)
        try:
            self._page._create_group_raw(self._snapshot)
        finally:
            _leave_undo(self._page)

    def undo(self) -> None:
        _enter_undo(self._page)
        try:
            self._page._undo_create_group_raw(self._snapshot)
        finally:
            _leave_undo(self._page)


# =============================================================================
# UngroupCmd —— 解散一个 GroupItem，成员留在原地
# =============================================================================

class UngroupCmd(QUndoCommand):
    """解散一个 GroupItem。``snapshot`` 是 ``GroupItem.serialize()`` 的结果。"""

    def __init__(self, page, snapshot: Dict[str, Any]) -> None:
        super().__init__("Ungroup")
        self._page = page
        self._snapshot = snapshot

    def redo(self) -> None:
        _enter_undo(self._page)
        try:
            self._page._delete_group_raw(self._snapshot.get("id"))
        finally:
            _leave_undo(self._page)

    def undo(self) -> None:
        _enter_undo(self._page)
        try:
            self._page._restore_group_raw(self._snapshot)
        finally:
            _leave_undo(self._page)


# =============================================================================
# ToggleNodeFlagCmd —— 批量切换 NodeItem._collapsed / _disabled
# =============================================================================

class ToggleNodeFlagCmd(QUndoCommand):
    """批量切换 NodeItem._collapsed / _disabled (ComfyUI-style bypass + fold).

    ``flag`` ∈ {"_collapsed", "_disabled"};
    ``changes`` 是 [(node_id, old_bool, new_bool), ...] 列表 —— 一次工具栏
    点击或 chevron 单击产生一条命令,无论改了几个节点都只占一个 undo 步。
    """

    def __init__(self, page, flag: str, changes) -> None:
        super().__init__(f"Toggle {flag.strip('_')}")
        self._page = page
        self._flag = "_collapsed" if flag == "_collapsed" else "_disabled"
        self._changes = [(str(i), bool(o), bool(n)) for i, o, n in changes]
        if not self._changes:
            self.setObsolete(True)

    def redo(self) -> None:
        self._apply(use_old=False)

    def undo(self) -> None:
        self._apply(use_old=True)

    def _apply(self, use_old: bool) -> None:
        _enter_undo(self._page)
        try:
            setter = "set_collapsed" if self._flag == "_collapsed" else "set_disabled"
            for nid, old, new in self._changes:
                it = self._page._instances.get(nid)
                if it is None:
                    continue
                target = old if use_old else new
                fn = getattr(it, setter, None)
                if callable(fn):
                    fn(target)
        finally:
            _leave_undo(self._page)


# =============================================================================
# ResizeGroupCmd —— 鼠标 resize group 框边界(self.pos 可能也变;成员重算)
# =============================================================================

class ResizeGroupCmd(QUndoCommand):
    """GroupItem 边界 resize 一次操作。带 old/new 的 pos+rect+members 三元组."""

    def __init__(
        self,
        page,
        group_id: str,
        *,
        old_pos: QPointF,
        old_rect: Any,
        old_members: List[str],
        new_pos: QPointF,
        new_rect: Any,
        new_members: List[str],
    ) -> None:
        super().__init__("Resize group")
        self._page = page
        self._gid = str(group_id)
        self._old_pos = QPointF(old_pos)
        self._old_rect = old_rect  # QRectF
        self._old_members = list(old_members)
        self._new_pos = QPointF(new_pos)
        self._new_rect = new_rect
        self._new_members = list(new_members)

    def redo(self) -> None:
        self._apply(self._new_pos, self._new_rect, self._new_members)

    def undo(self) -> None:
        self._apply(self._old_pos, self._old_rect, self._old_members)

    def _apply(self, pos, rect, members) -> None:
        _enter_undo(self._page)
        try:
            grp = getattr(self._page, "_groups", {}).get(self._gid)
            if grp is None:
                return
            grp._suspend_member_sync = True
            try:
                grp.prepareGeometryChange()
                grp.setPos(pos)
                grp._last_pos = QPointF(pos)
                from PyQt6.QtCore import QRectF as _QRectF
                grp._rect = _QRectF(0, 0, rect.width(), rect.height())
                grp.member_ids = list(members)
                if hasattr(grp, "_reposition_title_proxy"):
                    grp._reposition_title_proxy()
                grp.update()
            finally:
                grp._suspend_member_sync = False
            # 跨 group 一致性:resize 的目标 group 已被还原,其它 group 可能
            # 因这次 resize 失去过成员,需要再扫一遍把他们正确归位。
            if hasattr(self._page, "_recompute_all_group_memberships"):
                # 把目标 group 的 members 锁回我们刚还原的值;只让其它 group
                # 重新认领"不在 members 中"的节点。
                self._page._recompute_all_group_memberships()
                grp.member_ids = list(members)  # restore exact snapshot
        finally:
            _leave_undo(self._page)


# =============================================================================
# ResizeNoteCmd —— NoteItem 鼠标 resize 边界
# =============================================================================

class ResizeNoteCmd(QUndoCommand):
    """NoteItem 鼠标 resize 一次操作 —— 带 old/new 的 pos + size 二元组.

    GroupItem 走 ResizeGroupCmd(还要带成员快照);Note 没有成员概念,只需
    pos + (width, height) 两组数。
    """

    def __init__(
        self,
        page,
        note_id: str,
        *,
        old_pos: QPointF,
        old_size: Tuple[float, float],
        new_pos: QPointF,
        new_size: Tuple[float, float],
    ) -> None:
        super().__init__("Resize note")
        self._page = page
        self._note_id = str(note_id)
        self._old_pos = QPointF(old_pos)
        self._old_size = (float(old_size[0]), float(old_size[1]))
        self._new_pos = QPointF(new_pos)
        self._new_size = (float(new_size[0]), float(new_size[1]))
        if self._old_pos == self._new_pos and self._old_size == self._new_size:
            self.setObsolete(True)

    def redo(self) -> None:
        self._apply(self._new_pos, self._new_size)

    def undo(self) -> None:
        self._apply(self._old_pos, self._old_size)

    def _apply(self, pos: QPointF, size: Tuple[float, float]) -> None:
        _enter_undo(self._page)
        try:
            it = self._page._instances.get(self._note_id)
            if it is None:
                return
            if QPointF(it.pos()) != pos:
                it.setPos(pos)
            setter = getattr(it, "set_size", None)
            if callable(setter):
                setter(size[0], size[1])
        finally:
            _leave_undo(self._page)


# =============================================================================
# RenameCmd —— Node / Group title 改名
# =============================================================================

class RenameCmd(QUndoCommand):
    """单条改名(NodeItem._display_name_override 或 GroupItem.title)。"""

    def __init__(self, page, kind: str, target_id: str, old: str, new: str) -> None:
        super().__init__(f"Rename {kind}")
        self._page = page
        self._kind = "group" if kind == "group" else "node"
        self._target_id = str(target_id)
        self._old = "" if old is None else str(old)
        self._new = "" if new is None else str(new)
        if self._old == self._new:
            self.setObsolete(True)

    def redo(self) -> None:
        self._apply(self._new)

    def undo(self) -> None:
        self._apply(self._old)

    def _apply(self, value: str) -> None:
        _enter_undo(self._page)
        try:
            if self._kind == "group":
                grp = getattr(self._page, "_groups", {}).get(self._target_id)
                if grp is None:
                    return
                grp.title = value
                grp.update()
            else:
                it = self._page._instances.get(self._target_id)
                if it is None:
                    return
                setter = getattr(it, "set_display_name_override", None)
                if callable(setter):
                    setter(value)
        finally:
            _leave_undo(self._page)


# =============================================================================
# ColorOverrideCmd —— SelectionToolbar 色盘按钮设置 / 清空 per-canvas 颜色覆写
# =============================================================================

class ColorOverrideCmd(QUndoCommand):
    """批量颜色覆写。``changes`` = ``[(target_id, old_hex, new_hex), ...]``。

    target_id 既可以是 NodeItem / NoteItem 的 ``node_id``,也可以是 GroupItem 的
    ``group_id``。redo / undo 时按 id 在 ``page._instances`` 和 ``page._groups``
    两边各找一次,谁找到就调谁的 ``set_color_override``。

    空 hex (``""``) 表示清空覆写、回到默认主题色。
    """

    def __init__(
        self,
        page,
        changes: Sequence[Tuple[str, str, str]],
    ) -> None:
        super().__init__("Override color")
        self._page = page
        self._changes: List[Tuple[str, str, str]] = [
            (str(tid), str(old or ""), str(new or "")) for tid, old, new in changes
            if (old or "") != (new or "")
        ]
        if not self._changes:
            self.setObsolete(True)

    def redo(self) -> None:
        self._apply(use_new=True)

    def undo(self) -> None:
        self._apply(use_new=False)

    def _apply(self, *, use_new: bool) -> None:
        _enter_undo(self._page)
        try:
            for tid, old, new in self._changes:
                target = (
                    self._page._instances.get(tid)
                    or getattr(self._page, "_groups", {}).get(tid)
                )
                if target is None:
                    continue
                setter = getattr(target, "set_color_override", None)
                if callable(setter):
                    setter(new if use_new else old)
        finally:
            _leave_undo(self._page)


__all__ = [
    "SpawnNodeCmd",
    "DeleteSelectionCmd",
    "MoveNodeCmd",
    "ConnectCmd",
    "DisconnectCmd",
    "DisconnectAllEdgesCmd",
    "ParamChangeCmd",
    "PasteCmd",
    "CreateGroupCmd",
    "UngroupCmd",
    "ToggleNodeFlagCmd",
    "ResizeGroupCmd",
    "ResizeNoteCmd",
    "RenameCmd",
    "ColorOverrideCmd",
]
