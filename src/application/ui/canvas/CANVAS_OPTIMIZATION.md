# Canvas Optimization Plan

> PyQt6 Canvas 性能优化方案 — 4 层 LOD + 视口剔除 + 选择性 pixmap 缓存。
> 来源：DEMO→RELEASE NODE 迁移规划期间（2026-05-02 ~ 05-03）的对话记录。
> 目标：50 节点 / 1000 边规模下，pan/zoom 全程稳定 60 FPS。

---

## 1. 4-Tier LOD 体系

按 zoom factor（`view.transform().m11()` 或 `painter.worldTransform().m11()`）分四档。
所有 item 在 `paint()` 起始处用 `lod.tier_for_painter(painter)` 取 tier，按档分支渲染。

| Tier | Zoom | Node | Edge | Grid | 说明 |
|---|---|---|---|---|---|
| **T0 Overview** | 0.10 – 0.30 | 实心矩形剪影（layer 配色，无描边）；不画端口、参数行、标题、阴影 | 完全 cull，除非两端节点都在视口内 | 仅 100px 大网格 | 缩略图档；目标是"能看见拓扑分布"即可 |
| **T1 Minimap** | 0.30 – 0.80 | 标题 + 类型 + 输入/输出端口数徽章；端口本体用小圆点；无参数行/阴影 | 直线段（跳过 cubic Bézier） | 100px 大网格 + 20px 小网格 | 概览档；标题可读，但不需要参数细节 |
| **T2 Working** | 0.80 – 2.50 | 完整节点体：标题、所有端口、参数行、图标；端口标签仅 hover 显示；阴影简化为单线描边 | 完整 cubic Bézier，按类型上色 | 大 + 小网格 | 主要工作档；80% 时间用户停留在这里 |
| **T3 Detail** | 2.50 – 4.00 | 全细节：每端口固定标签、参数注释、阴影、debug overlay | Bézier + 选中光晕 | 大 + 小网格；zoom > 3.5x 加 1px 微格 | 精修档；用户在审视单个节点的连线 |

**Tier 切换阈值的理由：**
- 0.30：节点剪影 → 标题可读的临界点
- 0.80：参数行高度（26px）开始有意义
- 2.50：端口名（~10px 字号）变得清晰
- 4.00：再放大就该用 Inspector 面板了，不该靠 canvas

**关键收益估算：**
- T0 边 cull：50 节点典型连接密度 ~1000 边，全 cull 省 **~40% 帧时间**
- T2/T3 端口标签延迟到 hover：~20% 减少 paint
- 视口剔除（独立于 LOD）：再省 **~30% paint 调用**

---

## 2. Qt API 决策

| 机制 | 决策 | 理由 |
|---|---|---|
| `QStyleOptionGraphicsItem.levelOfDetailFromTransform(painter.worldTransform())` | **采用** | Qt 设计的标准 LOD 钩子；返回标准化 zoom 浮点。本仓库画布只均匀缩放，等价于 `painter.worldTransform().m11()`，所以 `lod.py` 直接读 m11。 |
| `setCacheMode(DeviceCoordinateCache)` | **避免**（节点上） | 每次 pan/zoom 都失效，重新光栅化代价 > 直接重画。仅适合真静态 item（如背景网格 — 已经在 `grid.py` 里手动缓存）。 |
| `setOptimizationFlag(DontAdjustForAntialiasing)` | **采用**（已在 view.py:40） | 减少非抗锯齿描边的 paint 开销 |
| `setItemIndexMethod(BspTreeIndex)` | **采用**（已设置） | ≤50 节点时 BSP 重建开销低；O(log n) 相交查询。NoIndex 需要纯手动剔除，不如 BSP + 手动视口剔除组合。 |
| `SmartViewportUpdate` | **采用**（已在 view.py:43） | 只重绘脏区域，远好于 FullUpdate |
| 视口剔除 `_visible_rect` | **手动实现** | BSP 自带剔除偏保守；手动剔除在小节点数下最快 |
| `QPixmapCache` | **选择性使用** | 仅缓存 T0 剪影 + T1 缩略图，不缓存 T2/T3（端口/参数变化频繁，缓存命中率低，得不偿失） |

---

## 3. 缓存策略 / Cache Strategy

### 3.1 范围

只缓存 **T0 + T1** 节点产物：
- T0 剪影：~40×30 px 实心矩形，`(node_id, layer_color)` → pixmap
- T1 缩略图：标题 + 端口数徽章，~120×60 px，`(node_id, title, port_counts)` → pixmap

**不缓存** T2/T3：
- 端口连接状态、参数值、hover 高亮变化频繁
- 缓存命中率 < 30%，QPixmapCache 抖动反而更慢

### 3.2 Key 设计

```python
key = f"node_{node_id}_tier_{tier_name}_{state_hash}"
# state_hash 只编码影响外观的字段：
#   T0: layer_color
#   T1: title + input_count + output_count + layer_color
```

**关键：key 不包含 zoom**。同一 tier 内（如 T1 跨 0.30~0.80）共享同一 pixmap，Qt 缩放时本身就快。

### 3.3 失效

- 节点属性变更（标题、layer、端口数）→ 调 `QPixmapCache.remove(key)`
- 跨 tier（zoom 越过 T0_MAX / T1_MAX）→ **不失效**，下次 paint 自然换 key
- 全局：QPixmapCache 默认 ~64 MB，Qt 自管 LRU，无需手动清

### 3.4 内存预算

100 节点 × 2 tier × ~120×60 × 4B ≈ **5.5 MB**，远低于 64 MB 默认。

---

## 4. 视口剔除 / Viewport Culling

### 4.1 状态

`CanvasView` 持有 `self._visible_rect: QRectF`，等于当前视口在 scene 坐标下的 bounding rect。

```python
def _update_visible_rect(self) -> None:
    self._visible_rect = self.mapToScene(self.viewport().rect()).boundingRect()
```

### 4.2 更新时机

- `wheelEvent` 后（zoom 改变）
- `mouseMoveEvent` 中拖拽 pan 时
- `resizeEvent` 中

**批处理：** 不按像素更新，改为 deferred — 标记 dirty，在下次 paint 前一次性算。

### 4.3 应用点

```python
# 节点 paint() 起始：
def paint(self, painter, option, widget):
    visible = self.scene().views()[0]._visible_rect
    if not visible.intersects(self.sceneBoundingRect()):
        return
    # ...正常渲染
```

边的剔除更激进 — 两端节点都在视口外 → 整条 cull（T0 时这是主要收益）。

---

## 5. 实施清单 / Implementation Checklist

目标目录：`RELEASE/src/application/ui/canvas/`

| 文件 | 状态 | 改动 |
|---|---|---|
| `lod.py` | ✅ 已实现 | 阈值 + tier 工具（`tier_for_zoom`, `tier_for_painter`, `lod_for_painter`） |
| `scene.py` | ⚠ 待校验 | `drawBackground` 内取 tier；跨 tier 时发 `zoomChanged(int)` 信号 |
| `view.py` | ⚠ 待校验 | `_visible_rect` 跟踪 + 三处 event 钩子；`zoom_tier` 缓存 |
| `items.py` | ⚠ 待校验 | `NodeItem.paint` / `PortItem.paint` / `ConnectionItem.paint` 按 tier 分支 |
| `grid.py` | ✅ 已实现 | 大/小网格 LOD 已就绪（profiling >5% 帧才需进一步优化） |

每次推进任一文件前，先读现状，再增量改 — 不重写。

---

## 6. 不做的事 / Non-Goals

- **不引入 OpenGL viewport**：`QGraphicsView` + `QPainter` 在本规模够用；OpenGL 会复杂化文本/SVG 渲染
- **不做 mipmap**：节点不是连续纹理，4 个离散 tier 已经覆盖
- **不优化 grid.py**：现状（缓存 pixmap + tile）已经够；profiling 显示 >5% 帧预算才动
- **不抽象通用 LOD 框架**：当前只服务 canvas，不为 minimap / 其他 view 提前抽象（YAGNI）

---

## 7. 验收 / Acceptance

50 节点 + 1000 边的合成场景下：
- 0.1x ~ 4.0x 任意 zoom，pan 全程 ≥ 60 FPS（16.7ms/frame）
- 跨 tier 切换无可见跳变（pixmap key 切换在一帧内完成）
- QPixmapCache 命中率 T1 ≥ 80%（profiling 验证）
- DEMO 同场景下基线 ~25 FPS（参考用，非硬指标）
