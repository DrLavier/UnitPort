# Runtime Layer

`system/runtime` orchestrates execution, monitoring, interception, and safety.

## Modules

| Module | Role |
|--------|------|
| `runtime_engine.py` | Single execution entry — dispatches to either WorkflowRunner or NodeExecutor |
| `workflow_runner.py` | Graph control-flow runner used by `execution_graph` path |
| `node_executor.py` | Flow-aware DAG executor used by `workflow_ir` path |
| `contracts.py` | **Unified result contract** — `RuntimeResult` dataclass (Phase 1 + Phase 2 tracing fields) |
| `behavior_invoker.py` | **Phase 2 behavior dispatch** — `BehaviorSubgraphInvoker.invoke()` |
| `migration.py` | **Phase 1 migration switch** — `MigrationFlags` with env-var override |
| `interception/*` | Compile / execute guards |
| `safety/*` | Policy, checker, emergency handling, audit |

---

## Phase 1 — Runtime Path Unification (完成)

Phase 1 统一了 WorkflowIR 与 execution graph 两条执行路径，使其在结果格式、
控制流语义、策略参数和审计可比性上达到同等可用级别。

### 执行路径判定

```
mission_ir 类型判定（RuntimeEngine._is_execution_graph）：
  isinstance(mission_ir, dict) and isinstance(mission_ir.get("nodes"), dict)
    → True  → Path A: Execution Graph（WorkflowRunner）
    → False → Path B: WorkflowIR（NodeExecutor）
```

---

## 统一运行结果契约（contracts.py）

`RuntimeEngine.execute()` 唯一返回类型，两条路径输出同构。

```python
@dataclass
class RuntimeResult:
    status: str        # "success" | "failed" | "blocked"
    reason: str        # 失败/拦截原因码（成功时为 ""）
    task_id: str       # Scheduler 分配的 ID（blocked 阶段为 ""）
    node_count: int    # 工作流节点总数（blocked 阶段为 0）
    results: Dict[str, Any]      # node_id → 节点输出 dict
    metrics: Dict[str, Any]      # Monitor 指标
    diagnostics: Dict[str, Any]  # 附加诊断：path / executed_count / failed_nodes / has_action / emergency
```

### diagnostics 子字段

Phase 1 字段（两条路径、所有 status 均存在）:

| Key (`DiagnosticsKey.*`) | 类型 | 说明 |
|--------------------------|------|------|
| `path` | str | `"execution_graph"` / `"workflow_ir"` / `"blocked_compile"` / … |
| `executed_count` | int | 实际走过的节点数 |
| `failed_nodes` | List[str] | results 中含 `"error"` 字段的节点 ID |
| `has_action` | bool | 是否包含 action/stop 节点 |
| `emergency` | Any | 安全拦截时 emergency_handler 输出（仅 blocked_safety） |

Phase 2 字段（STAGE-05，仅 success / failed 结果；blocked 无此字段）:

| Key (`DiagnosticsKey.*`) | 类型 | 说明 |
|--------------------------|------|------|
| `mission_trace_id` | str | 本次 execute() 调用的 UUID；同一次执行所有 behavior 节点共享此 ID |
| `behavior_trace_ids` | Dict[str, str] | `{node_id: trace_id}` — 每个 behavior 节点的 trace_id |
| `behavior_diagnostics` | List[dict] | 所有 behavior 节点的 `BehaviorDiagnostic.to_dict()` 汇总（成功时为 `[]`） |

### 兼容策略

`RuntimeResult.to_dict()` **保留**现有全部关键字段名，既有调用方零修改可读：
- `status` / `reason` / `task_id` / `node_count` / `results` / `metrics`
- blocked 路径额外保留 `phase`（`"compile"` / `"execute"` / `"safety"`）和 `emergency`

新增字段 `diagnostics` 为附加字段，老调用方安全忽略。

---

## NodeExecutor 执行模式

### 主路径：Flow-Aware DFS（STEP-04）

- 从 FLOW in-degree=0 的节点出发（排除 `comparison`/`end` 种类）
- DATA-edge 依赖在消费节点执行前按需解析（对齐 WorkflowRunner._evaluate_condition 策略）
- 条件节点（if/switch/gate）：仅跟随输出值非 None 的端口
- 循环节点（while/for）：由 `_handle_loop` 驱动迭代，`max_loop_iterations` 封顶
- AbortNode：`RuntimeError("AbortWorkflow: …")` → `_abort=True`，立即停止遍历
- 非条件节点：无条件跟随所有 FLOW 出边（与 WorkflowRunner 行为对齐）

### 回退路径：Topological Sort（保留，可控 fallback）

当 flow 图无入度为 0 的节点时自动降级，或通过迁移开关强制启用。

---

## 迁移开关（migration.py — STEP-05）

控制 WorkflowIR 路径使用新（flow-aware）还是旧（topological）执行策略。

```python
from system.runtime.migration import MigrationFlags
flags = MigrationFlags.from_env()
# flags.use_flow_aware_execution: bool  (默认 True)
```

| 环境变量 | 值 | 效果 |
|----------|----|------|
| `UNITPORT_FLOW_AWARE_EXECUTION` | 未设置 / `"1"` | 新路径（flow-aware DFS） |
| `UNITPORT_FLOW_AWARE_EXECUTION` | `"0"` / `"false"` | 旧路径（topological sort） |

迁移开关状态在每次 WorkflowIR 执行时写入审计日志 `"migration_flags"` 事件。

---

## 审计日志事件

| 事件 | 触发时机 | 关键字段 |
|------|---------|---------|
| `compile_blocked` | compile guard 拦截 | `reason` |
| `execute_blocked` | execute guard 拦截 | `reason` |
| `safety_blocked` | safety checker 拦截 | `check`, `emergency` |
| `migration_flags` | WorkflowIR 路径执行前 | `use_flow_aware_execution`, `source_env_key` |
| `execution_completed` | 执行完成（两路径共用） | `task_id`, `mission_trace_id`, `node_count`, `path`, `executed_count`, `failed_nodes`, `reason`, `loop_limit_exceeded`, `behavior_trace_ids`, `behavior_diagnostics_count` |

---

## 两路径差异对比（Phase 1 审计快照）

| 维度 | Path A: Execution Graph | Path B: WorkflowIR |
|------|-------------------------|---------------------|
| **输入格式** | `dict{"nodes": Dict[id→data], "outgoing", "incoming", "entry_nodes"}` | `WorkflowIR` 对象（nodes: List[IRNode]，edges: List[IREdge]） |
| **调度方式** | 事件驱动递归 DFS，从 `entry_nodes` 出发 | Flow-Aware DFS（主）/ Kahn 拓扑排序（备） |
| **控制流支持** | 完整：If/else、for loop、while loop | 完整（STEP-04）：If/else、for loop、while loop、abort |
| **robot_model 注入** | 有 | 有（STEP-03）：`_ROBOT_AWARE_TYPES` 集合控制 |
| **节点实际执行** | 真实执行 | 真实执行（STEP-03）：registry lookup + `execute(inputs)` |
| **单节点失败处理** | `results[id] = {"error": ...}`，继续遍历 | 同左（abort 节点除外：立即停止） |
| **max_loop_iterations** | `WorkflowRunner.max_loop_iterations` | `NodeExecutor.max_loop_iterations`（STEP-04） |
| **executed_count** | `runner_out["executed_count"]` | `executor._last_executed_count`（STEP-04） |
| **has_action** | `runner_out["has_action"]` | `NodeKind.ACTION/STOP` 扫描（STEP-02 fix） |

---

## Phase 2 — Behavior Dispatch (STAGE-03 / STAGE-05)

### Behavior Node Detection

`NodeExecutor._execute_node()` routes to `_execute_behavior_node()` when:

```python
node_type == "behavior"
or node_data.get("schema_id") == "behavior"
or node_data.get("external_kind") == "behavior"   # Canvas-origin nodes
```

Non-behavior nodes are completely unaffected by this check.

### Dispatch Flow

```
RuntimeEngine.execute(WorkflowIR, scenario):
    mission_trace_id = uuid4()
    executor.execute(context={
        "scenario": scenario,
        "mission_trace_id": mission_trace_id   ← all behavior nodes share this
    })

NodeExecutor._execute_behavior_node(node, inputs, context):
    trace_id     = context["mission_trace_id"]
    behavior_ref = _extract_behavior_ref(node_data)   ← WorkflowIR + Canvas formats
    invoke_input = BehaviorInvokeInput(behavior_ref, inputs, context, trace_id)
    sub_executor = NodeExecutor()   ← fresh, isolated
    invoke_result = invoker.invoke(invoke_input, bridge, sub_executor, policy)
    → {"status": "success", "behavior_ref": ..., "trace_id": ..., ...}
      or {"error": reason, "status": "blocked/failed", "diagnostics": [...], ...}

After all nodes execute:
    _collect_behavior_tracing(results) →
        behavior_trace_ids   = {node_id: trace_id, ...}
        behavior_diagnostics = flat list of all diagnostic dicts from behavior nodes
    → stored in RuntimeResult.diagnostics[MISSION_TRACE_ID / BEHAVIOR_TRACE_IDS / BEHAVIOR_DIAGNOSTICS]
```

### Backward Compatibility

When `RuntimeEngine.behavior_bridge` is `None` (default), behavior nodes return:

```python
{"status": "skipped", "reason": "no_behavior_invoker:<behavior_ref>"}
```

No `"error"` key → not added to `failed_nodes` → mission succeeds.
Existing non-behavior workflows require zero configuration changes.

### behavior_ref Extraction

`NodeExecutor._extract_behavior_ref(node_data)` handles two formats:

| Format | Source | Lookup |
|--------|--------|--------|
| WorkflowIR | `params["behavior_ref"]["value"]` | IRParam dict with `"value"` key |
| Canvas dict | `node_data["behavior_ref"]` | Top-level string |

`params` takes precedence over top-level when both are present.

---

## Phase 1 验收报告

### 变更文件列表

| 文件 | 变更类型 | 对应步骤 |
|------|---------|---------|
| `system/runtime/contracts.py` | 新建 | STEP-01/02 |
| `system/runtime/migration.py` | 新建 | STEP-05 |
| `system/runtime/runtime_engine.py` | 修改 | STEP-02/04/05 |
| `system/runtime/node_executor.py` | 主要重写 | STEP-03/04/05 |
| `system/runtime/README.md` | 更新 | STEP-01/07 |
| `tests/unit/test_runtime_contracts.py` | 新建 | STEP-06 |
| `tests/unit/test_node_executor.py` | 新建 | STEP-06 |
| `tests/integration/test_runtime_engine.py` | 新建 | STEP-06 |
| `tests/regression/test_runtime_legacy.py` | 新建 | STEP-06 |

### 设计决策与权衡

1. **`_BLOCKED_PATH_TO_PHASE` 常量映射**（contracts.py）：避免字符串切片/拼接漂移；
   `to_dict()` 用 dict.get() 查表，不再做 `startswith("blocked_")` + `[len("blocked_"):]`。

2. **`_resolve_registry_type` 三级解析**（node_executor.py）：
   直接匹配 → 剥离 `"builtin."` 前缀（Canvas IR 格式）→ `node_data["schema_id"]` 回退；
   保证与现有 Canvas 导出的 WorkflowIR 兼容。

3. **Condition 端口 `{"value": X}` 解包**（node_executor.py `_collect_inputs`）：
   对齐 WorkflowRunner._evaluate_condition 的布尔值提取语义，
   避免 IfNode/WhileLoopNode 收到非空 dict 被误判为 truthy。

4. **非条件节点无条件跟随 FLOW 出边**（node_executor.py `visit()`）：
   与 WorkflowRunner `flow_out` 行为对齐；端口输出值无效性检查仅用于条件分支节点。

5. **迁移开关优先级**：env var `UNITPORT_FLOW_AWARE_EXECUTION` 覆盖默认值；
   默认启用新路径（`True`）；旧路径（topological sort）保留为可控 fallback，
   代码未删除，可通过设置环境变量一键回退。

6. **Abort vs 普通失败**：`workflow_aborted` 优先于 `node_execution_failed`；
   Abort 节点之后的节点不执行；普通异常节点之后（同层级）的兄弟节点继续执行。

7. **测试隔离策略**（STEP-06/07 收尾修复）：
   `node_executor.py` / `workflow_runner.py` 在无 Qt 环境下通过 try/except 回退到空实现，
   避免测试文件须在模块级注入 `sys.modules["bin"]` stub。
   所有运行时测试文件的 `sys.modules` 变更限定在 `setUpClass`/`tearDownClass` 内，
   不对同进程其他测试模块产生污染。

### 测试证据

以下数字均来自实际执行输出，可通过相同命令独立复验。

#### 全量 pytest（`python -m pytest tests/ -q`）

```
259 passed, 14 subtests passed in 0.91s
```

#### 按套件 `unittest discover`

| 命令 | 结果 |
|------|------|
| `python -m unittest discover -s tests/regression -p "test_*.py"` | `Ran 11 tests … OK` |
| `python -m unittest discover -s tests/unit -p "test_*.py"` | `Ran 156 tests … OK` |
| `python -m unittest discover -s tests/integration -p "test_*.py"` | `Ran 65 tests … OK` |

#### 新增测试分布（STEP-06，共 61 个）

| 文件 | 位置 | 数量 | 覆盖内容 |
|------|------|------|---------|
| `test_runtime_contracts.py` | `tests/unit/` | 22 | 结果契约、blocked compat、diagnostics 结构 |
| `test_node_executor.py` | `tests/unit/` | 23 + 4 subtests | 节点失败传播、loop 上限、abort、分支、迁移开关 |
| `test_runtime_legacy.py` | `tests/unit/` | 10 + 4 subtests | compat 路径回归验证（原在 regression/，已迁移） |
| `test_runtime_engine.py` | `tests/integration/` | 14 | RuntimeEngine 两路径统一输出验证 |

- 198 条原有测试全部通过（无回归）

### 已知缺口与下一步建议

| 缺口 | 严重性 | 建议（Phase 2+） |
|------|--------|----------------|
| WorkflowIR 路径无端到端 robot_model 集成测试 | 中 | Phase 2 引入 mock RobotModel |
| SwitchNode / GateNode 的 `case_*` 端口分支尚未有完整测试覆盖 | 低 | STEP-06 补充 |
| `_handle_loop` 嵌套深度硬编码为 8 | 低 | 可将 `_MAX_NESTING` 接入 SafetyPolicy |
| execution_graph 路径不受迁移开关控制（WorkflowRunner 未变动） | 设计意图 | Phase 2 若统一 runner 再考虑 |

### 声明：未进入 Phase 2+ 内容

以下内容**未**在本阶段实施：
- Behavior 子图执行与调度
- Adapter 会话生命周期管理
- 多品牌适配（非 Unitree 机器人）
- 设置 schema 重构
- UI 修改

---

*Phase 1 于 2026-02-22 完成，收尾修复于 2026-02-22，评审者可基于本报告独立复验。*
