"""Hermes Agent 工具系统的中央注册表

每个工具文件在模块级别调用 ``registry.register()`` 来声明其
模式(schema)、处理器(handler)、工具集(toolset)成员关系和可用性检查。
``model_tools.py`` 查询此注册表，而不是维护自己的并行数据结构。

导入链（避免循环导入）：
    tools/registry.py  （不导入 model_tools 或任何工具文件）
           ^
    tools/*.py  （在模块级别从 tools.registry 导入）
           ^
    model_tools.py  （导入 tools.registry + 所有工具模块）
           ^
    run_agent.py, cli.py, batch_runner.py 等
"""

import ast
import importlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


def _is_registry_register_call(node: ast.AST) -> bool:
    """判断节点是否为 ``registry.register(...)`` 调用表达式。
    
    Args:
        node: AST 抽象语法树节点
        
    Returns:
        True 如果节点是 registry.register() 调用，否则 False
    """
    # 必须是表达式节点且值是调用(Call)类型
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    func = node.value.func
    # 检查是否是属性访问形式：registry.register
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "register"  # 方法名是 register
        and isinstance(func.value, ast.Name)  # 对象部分是名称
        and func.value.id == "registry"  # 对象名是 registry
    )


def _module_registers_tools(module_path: Path) -> bool:
    """判断模块是否包含顶层的 ``registry.register(...)`` 调用。
    
    只检查模块主体语句，这样那些在函数内部调用 ``registry.register()`` 
    的辅助模块不会被误识别为工具模块。
    
    Args:
        module_path: 模块文件路径
        
    Returns:
        True 如果模块在顶层调用了 registry.register()，否则 False
    """
    try:
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(module_path))
    except (OSError, SyntaxError):
        # 文件读取失败或语法错误时返回 False
        return False

    # 只检查模块体(body)中的语句是否有 registry.register() 调用
    return any(_is_registry_register_call(stmt) for stmt in tree.body)


def discover_builtin_tools(tools_dir: Optional[Path] = None) -> List[str]:
    """导入内置的自注册工具模块并返回它们的模块名称列表。
    
    该函数会扫描 tools 目录下的所有 .py 文件，通过 AST 分析判断哪些文件
    是真正的工具模块（包含顶层 registry.register() 调用），然后动态导入它们。
    
    Args:
        tools_dir: 工具目录路径，默认为当前文件所在目录（即 tools/）
        
    Returns:
        成功导入的工具模块名称列表，格式如 ["tools.browser_tool", "tools.file_tool"]
    """
    # 确定工具目录路径
    tools_path = Path(tools_dir) if tools_dir is not None else Path(__file__).resolve().parent
    
    # 筛选出符合条件的工具模块文件：
    # 1. 排除 __init__.py、registry.py、mcp_tool.py
    # 2. 必须包含顶层 registry.register() 调用
    module_names = [
        f"tools.{path.stem}"
        for path in sorted(tools_path.glob("*.py"))
        if path.name not in {"__init__.py", "registry.py", "mcp_tool.py"}
        and _module_registers_tools(path)
    ]

    imported: List[str] = []
    for mod_name in module_names:
        try:
            # 动态导入模块，触发模块级别的 registry.register() 调用
            importlib.import_module(mod_name)
            imported.append(mod_name)
        except Exception as e:
            logger.warning("Could not import tool module %s: %s", mod_name, e)
    return imported


class ToolEntry:
    """单个已注册工具的元数据容器。
    
    使用 __slots__ 优化内存占用，因为可能有数百个工具条目。
    每个工具的所有关键信息都存储在这个对象中。
    """

    __slots__ = (
        "name", "toolset", "schema", "handler", "check_fn",
        "requires_env", "is_async", "description", "emoji",
        "max_result_size_chars", "dynamic_schema_overrides",
    )

    def __init__(self, name, toolset, schema, handler, check_fn,
                 requires_env, is_async, description, emoji,
                 max_result_size_chars=None, dynamic_schema_overrides=None):
        self.name = name  # 工具名称
        self.toolset = toolset  # 工具集名称（如 "browser", "filesystem"）
        self.schema = schema  # OpenAI 格式的 JSON Schema
        self.handler = handler  # 工具执行函数
        self.check_fn = check_fn  # 可用性检查函数（可选）
        self.requires_env = requires_env  # 需要的环境变量列表
        self.is_async = is_async  # 是否为异步工具
        self.description = description  # 工具描述
        self.emoji = emoji  # 工具对应的 emoji 图标
        self.max_result_size_chars = max_result_size_chars  # 最大结果字符数限制
        # 可选的零参数可调用对象，返回 schema 覆盖字典
        # 在 get_definitions() 时被调用。用于依赖运行时配置的字段
        # （例如 delegate_task 的描述必须反映用户当前的 delegation.max_concurrent_children / 
        # max_spawn_depth，这样模型不会被误导）。每次调用 get_definitions() 时都会调用此函数，
        # 结果会被浅合并到基础 schema 上，然后再包装成 {"type": "function", ...} 格式。
        self.dynamic_schema_overrides = dynamic_schema_overrides


# ---------------------------------------------------------------------------
# check_fn TTL 缓存机制
#
# check_fn 可调用对象（如 tools/terminal_tool.check_terminal_requirements）
# 会探测外部状态（Docker 守护进程、Modal SDK 安装、playwright 二进制文件
# 可用性等）。对于长期运行的 CLI 或网关进程，每次调用 get_definitions() 
# 都执行这些检查是纯粹的浪费——外部状态的变化是人类时间尺度的。
# 将结果缓存约 30 秒，这样通过 ``hermes tools`` 修改环境变量或实时凭证
# 文件更改可以在一两轮内生效，而无需任何显式失效操作。
# ---------------------------------------------------------------------------

_CHECK_FN_TTL_SECONDS = 30.0  # 缓存有效期（秒）
_check_fn_cache: Dict[Callable, tuple[float, bool]] = {}  # 缓存字典：{函数: (时间戳, 结果)}
_check_fn_cache_lock = threading.Lock()  # 线程锁，保证缓存操作的线程安全


def _check_fn_cached(fn: Callable) -> bool:
    """返回 bool(fn()) 的结果，带 TTL 缓存。
    
    该函数会对 check_fn 的调用结果进行缓存，避免重复执行耗时的外部状态检查。
    如果函数执行抛出异常，会被捕获并返回 False。
    
    Args:
        fn: 需要缓存结果的检查函数
        
    Returns:
        检查函数的布尔结果，失败时返回 False
    """
    now = time.monotonic()
    with _check_fn_cache_lock:
        cached = _check_fn_cache.get(fn)
        if cached is not None:
            ts, value = cached
            # 如果缓存未过期，直接返回缓存值
            if now - ts < _CHECK_FN_TTL_SECONDS:
                return value
    try:
        # 执行检查函数并转换为布尔值
        value = bool(fn())
    except Exception:
        # 异常情况下视为不可用
        value = False
    with _check_fn_cache_lock:
        # 更新缓存
        _check_fn_cache[fn] = (now, value)
    return value


def invalidate_check_fn_cache() -> None:
    """清除所有缓存的 ``check_fn`` 结果。
    
    在配置更改影响工具可用性后调用此函数（例如 ``hermes tools enable``）。
    这会强制下次检查时重新执行实际的检查函数。
    """
    with _check_fn_cache_lock:
        _check_fn_cache.clear()


class ToolRegistry:
    """单例注册表，从工具文件中收集工具模式(schema)和处理器(handler)。
    
    这是整个工具系统的核心，负责：
    1. 工具的注册和管理
    2. 工具集的可用性检查
    3. 工具模式的动态生成
    4. 工具调用的分发执行
    
    使用 RLock 保证多线程环境下的安全性，因为 MCP 动态刷新可能会在
    其他线程读取工具元数据时修改注册表。
    """

    def __init__(self):
        self._tools: Dict[str, ToolEntry] = {}  # 工具名称 -> ToolEntry 映射
        self._toolset_checks: Dict[str, Callable] = {}  # 工具集名称 -> 检查函数映射
        self._toolset_aliases: Dict[str, str] = {}  # 工具集别名 -> 规范名称映射
        # MCP 动态刷新可以在其他线程读取工具元数据时修改注册表，所以保持
        # 修改串行化，读者使用稳定快照。
        self._lock = threading.RLock()
        # 单调递增的代计数器。每次突变（register / deregister / 
        # register_toolset_alias / MCP 刷新）都会增加。外部调用者（如 
        # get_tool_definitions）可以据此进行记忆化：以代为键的缓存条目
        # 在代未改变时一直有效。
        self._generation: int = 0

    def _snapshot_state(self) -> tuple[List[ToolEntry], Dict[str, Callable]]:
        """返回注册表条目和工具集检查的一致快照。
        
        使用锁保证在多线程环境下获取的数据是原子性的，避免读到不一致的状态。
        
        Returns:
            (工具条目列表, 工具集检查函数字典) 的元组
        """
        with self._lock:
            return list(self._tools.values()), dict(self._toolset_checks)

    def _snapshot_entries(self) -> List[ToolEntry]:
        """返回已注册工具条目的稳定快照。"""
        return self._snapshot_state()[0]

    def _snapshot_toolset_checks(self) -> Dict[str, Callable]:
        """返回工具集可用性检查的稳定快照。"""
        return self._snapshot_state()[1]

    def _evaluate_toolset_check(self, toolset: str, check: Callable | None) -> bool:
        """执行工具集检查，将缺失或失败的检查视为不可用/可用。
        
        Args:
            toolset: 工具集名称
            check: 检查函数，如果为 None 则视为可用
            
        Returns:
            True 如果工具集可用，False 如果不可用
        """
        if not check:
            # 没有检查函数，默认认为可用
            return True
        try:
            return bool(check())
        except Exception:
            # 检查函数抛出异常时，记录调试日志并标记为不可用
            logger.debug("Toolset %s check raised; marking unavailable", toolset)
            return False

    def get_entry(self, name: str) -> Optional[ToolEntry]:
        """按名称返回已注册的工具条目，不存在则返回 None。
        
        Args:
            name: 工具名称
            
        Returns:
            ToolEntry 对象或 None
        """
        with self._lock:
            return self._tools.get(name)

    def get_registered_toolset_names(self) -> List[str]:
        """返回注册表中存在的排序后的唯一工具集名称列表。
        
        Returns:
            排序后的工具集名称列表，如 ["browser", "filesystem", "web_search"]
        """
        return sorted({entry.toolset for entry in self._snapshot_entries()})

    def get_tool_names_for_toolset(self, toolset: str) -> List[str]:
        """返回给定工具集下注册的排序后工具名称列表。
        
        Args:
            toolset: 工具集名称
            
        Returns:
            该工具集下的工具名称列表，已排序
        """
        return sorted(
            entry.name for entry in self._snapshot_entries()
            if entry.toolset == toolset
        )

    def register_toolset_alias(self, alias: str, toolset: str) -> None:
        """为规范工具集名称注册显式别名。
        
        允许用户使用简短或替代名称引用工具集。如果别名已存在且指向不同的
        工具集，会发出警告并覆盖。
        
        Args:
            alias: 别名，如 "fs" 代表 "filesystem"
            toolset: 规范的工具集名称
        """
        with self._lock:
            existing = self._toolset_aliases.get(alias)
            if existing and existing != toolset:
                logger.warning(
                    "Toolset alias collision: '%s' (%s) overwritten by %s",
                    alias, existing, toolset,
                )
            self._toolset_aliases[alias] = toolset
            self._generation += 1  # 增加代计数器，使缓存失效

    def get_registered_toolset_aliases(self) -> Dict[str, str]:
        """返回 ``{别名: 规范工具集}`` 映射的快照。"""
        with self._lock:
            return dict(self._toolset_aliases)

    def get_toolset_alias_target(self, alias: str) -> Optional[str]:
        """返回别名的规范工具集名称，不存在则返回 None。
        
        Args:
            alias: 别名
            
        Returns:
            规范工具集名称或 None
        """
        with self._lock:
            return self._toolset_aliases.get(alias)

    # ------------------------------------------------------------------
    # 注册功能
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        toolset: str,
        schema: dict,
        handler: Callable,
        check_fn: Callable = None,
        requires_env: list = None,
        is_async: bool = False,
        description: str = "",
        emoji: str = "",
        max_result_size_chars: int | float | None = None,
        dynamic_schema_overrides: Callable = None,
        override: bool = False,
    ):
        """注册一个工具。由每个工具文件在模块导入时调用。
        
        ``override=True`` 是插件替换现有内置工具实现的显式选择（例如，
        将默认浏览器工具替换为有头 Chrome CDP 后端）。没有它时，会拒绝
        来自不同工具集的、可能覆盖现有工具的注册，防止意外覆盖。
        
        Args:
            name: 工具名称，必须唯一
            toolset: 工具集名称
            schema: OpenAI 格式的 JSON Schema
            handler: 工具执行函数
            check_fn: 可用性检查函数（可选）
            requires_env: 需要的环境变量列表
            is_async: 是否为异步工具
            description: 工具描述
            emoji: 工具对应的 emoji
            max_result_size_chars: 最大结果字符数限制
            dynamic_schema_overrides: 动态 schema 覆盖函数
            override: 是否允许覆盖同名工具（默认 False）
        """
        with self._lock:
            existing = self._tools.get(name)
            if existing and existing.toolset != toolset:
                # 允许 MCP 到 MCP 的覆盖（合法情况：服务器刷新，
                # 或两个 MCP 服务器有重叠的工具名称）。
                both_mcp = (
                    existing.toolset.startswith("mcp-")
                    and toolset.startswith("mcp-")
                )
                if both_mcp:
                    logger.debug(
                        "Tool '%s': MCP toolset '%s' overwriting MCP toolset '%s'",
                        name, toolset, existing.toolset,
                    )
                elif override:
                    # 显式插件选择：替换现有工具。
                    # 记录在 INFO 级别，以便在 agent.log 中审计覆盖操作。
                    logger.info(
                        "Tool '%s': toolset '%s' overriding existing toolset '%s' "
                        "(override=True opt-in)",
                        name, toolset, existing.toolset,
                    )
                else:
                    # 拒绝覆盖——防止插件/MCP 覆盖内置工具或反之。
                    logger.error(
                        "Tool registration REJECTED: '%s' (toolset '%s') would "
                        "shadow existing tool from toolset '%s'. Pass "
                        "override=True to register() if the replacement is "
                        "intentional, or deregister the existing tool first.",
                        name, toolset, existing.toolset,
                    )
                    return
            # 创建并存储工具条目
            self._tools[name] = ToolEntry(
                name=name,
                toolset=toolset,
                schema=schema,
                handler=handler,
                check_fn=check_fn,
                requires_env=requires_env or [],
                is_async=is_async,
                description=description or schema.get("description", ""),
                emoji=emoji,
                max_result_size_chars=max_result_size_chars,
                dynamic_schema_overrides=dynamic_schema_overrides,
            )
            # 如果这是该工具集的第一个工具，注册其检查函数
            if check_fn and toolset not in self._toolset_checks:
                self._toolset_checks[toolset] = check_fn
            self._generation += 1  # 增加代计数器

    def deregister(self, name: str) -> None:
        """从注册表中移除工具。
        
        如果同一工具集中没有其他工具剩余，还会清理工具集检查。
        MCP 动态工具发现使用此功能在服务器发送 
        ``notifications/tools/list_changed`` 时进行核弹式重建。
        
        Args:
            name: 要移除的工具名称
        """
        with self._lock:
            entry = self._tools.pop(name, None)
            if entry is None:
                return
            # 如果这是该工具集的最后一个工具，删除工具集检查和别名
            toolset_still_exists = any(
                e.toolset == entry.toolset for e in self._tools.values()
            )
            if not toolset_still_exists:
                self._toolset_checks.pop(entry.toolset, None)
                # 清除此工具集的所有别名
                self._toolset_aliases = {
                    alias: target
                    for alias, target in self._toolset_aliases.items()
                    if target != entry.toolset
                }
            self._generation += 1  # 增加代计数器
        logger.debug("Deregistered tool: %s", name)

    # ------------------------------------------------------------------
    # Schema 检索功能
    # ------------------------------------------------------------------

    def get_definitions(self, tool_names: Set[str], quiet: bool = False) -> List[dict]:
        """返回请求的工具名称的 OpenAI 格式工具模式列表。
        
        只包含 ``check_fn()`` 返回 True（或没有 check_fn）的工具。
        ``check_fn()`` 结果通过 :func:`_check_fn_cached` 缓存约 30 秒，
        以分摊重复探测的开销（check_terminal_requirements 探测 modal/docker，
        browser 检查探测 playwright 等）；TTL 选择使得环境变量更改
        （``hermes tools enable foo``）仍然能在近实时生效，而无需在每次
        调用时强制完全刷新缓存。
        
        Args:
            tool_names: 需要获取定义的工具名称集合
            quiet: 如果为 True，不记录调试日志
            
        Returns:
            OpenAI 格式的工具模式列表，每个元素为 {"type": "function", "function": {...}}
        """
        result = []
        # 单次调用缓存，建立在 30 秒 TTL 之上——处理一次定义传递中对同一
        # check_fn 的重复探测，而无需重新读取 TTL 时钟。
        check_results: Dict[Callable, bool] = {}
        entries_by_name = {entry.name: entry for entry in self._snapshot_entries()}
        for name in sorted(tool_names):
            entry = entries_by_name.get(name)
            if not entry:
                continue
            # 执行可用性检查（带缓存）
            if entry.check_fn:
                if entry.check_fn not in check_results:
                    check_results[entry.check_fn] = _check_fn_cached(entry.check_fn)
                if not check_results[entry.check_fn]:
                    if not quiet:
                        logger.debug("Tool %s unavailable (check failed)", name)
                    continue
            # 确保 schema 始终有 "name" 字段——使用 entry.name 作为后备
            schema_with_name = {**entry.schema, "name": entry.name}
            # 应用运行时动态覆盖（例如 delegate_task 描述依赖于当前的
            # delegation.max_concurrent_children / max_spawn_depth）。调用方侧
            # （model_tools.get_tool_definitions）已经以其 config.yaml mtime + size
            # 为键进行记忆化，所以 config 中 delegation.* 的更改会自动使缓存失效。
            if entry.dynamic_schema_overrides is not None:
                try:
                    overrides = entry.dynamic_schema_overrides()
                    if isinstance(overrides, dict):
                        schema_with_name.update(overrides)
                except Exception as exc:
                    logger.warning(
                        "dynamic_schema_overrides for tool %s raised %s; "
                        "using static schema",
                        name, exc,
                    )
            # 包装成 OpenAI 格式
            result.append({"type": "function", "function": schema_with_name})
        return result

    # ------------------------------------------------------------------
    # 分发执行功能
    # ------------------------------------------------------------------

    def dispatch(self, name: str, args: dict, **kwargs) -> str:
        """按名称执行工具处理器。
        
        * 异步处理器通过 ``_run_async()`` 自动桥接。
        * 所有异常都被捕获并作为 ``{"error": "..."}`` 返回，
          以保证一致的错误格式。
        
        Args:
            name: 工具名称
            args: 工具参数字典
            **kwargs: 额外关键字参数（如 task_id）
            
        Returns:
            JSON 字符串形式的执行结果或错误信息
        """
        entry = self.get_entry(name)
        if not entry:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            # 如果是异步工具，使用 _run_async 桥接
            if entry.is_async:
                from model_tools import _run_async
                return _run_async(entry.handler(args, **kwargs))
            # 同步工具直接调用
            return entry.handler(args, **kwargs)
        except Exception as e:
            logger.exception("Tool %s dispatch error: %s", name, e)
            # 通过清理器路由，这样异常字符串中的框架令牌 / CDATA / 围栏
            # 不会作为结构性噪声到达模型。参见 model_tools._sanitize_tool_error
            # 了解原理。
            raw = f"Tool execution failed: {type(e).__name__}: {e}"
            try:
                from model_tools import _sanitize_tool_error
                sanitized = _sanitize_tool_error(raw)
            except Exception:
                sanitized = raw  # 防御性：永远不要让清理器阻止错误传播
            return json.dumps({"error": sanitized})

    # ------------------------------------------------------------------
    # 查询辅助函数（替换 model_tools.py 中的冗余字典）
    # ------------------------------------------------------------------

    def get_max_result_size(self, name: str, default: int | float | None = None) -> int | float:
        """返回每个工具的最大结果大小，或 *default*（或全局默认值）。
        
        Args:
            name: 工具名称
            default: 默认值，如果未指定则使用全局默认值
            
        Returns:
            最大结果字符数限制
        """
        entry = self.get_entry(name)
        if entry and entry.max_result_size_chars is not None:
            return entry.max_result_size_chars
        if default is not None:
            return default
        from tools.budget_config import DEFAULT_RESULT_SIZE_CHARS
        return DEFAULT_RESULT_SIZE_CHARS

    def get_all_tool_names(self) -> List[str]:
        """返回所有已注册工具名称的排序列表。"""
        return sorted(entry.name for entry in self._snapshot_entries())

    def get_schema(self, name: str) -> Optional[dict]:
        """返回工具的原始 schema 字典，绕过 check_fn 过滤。
        
        用于令牌估算和内省，此时可用性不重要——只有 schema 内容重要。
        
        Args:
            name: 工具名称
            
        Returns:
            工具的 schema 字典，不存在则返回 None
        """
        entry = self.get_entry(name)
        return entry.schema if entry else None

    def get_toolset_for_tool(self, name: str) -> Optional[str]:
        """返回工具所属的工具集，不存在则返回 None。
        
        Args:
            name: 工具名称
            
        Returns:
            工具集名称或 None
        """
        entry = self.get_entry(name)
        return entry.toolset if entry else None

    def get_emoji(self, name: str, default: str = "⚡") -> str:
        """返回工具的 emoji，未设置则返回 *default*。
        
        Args:
            name: 工具名称
            default: 默认 emoji，默认为 "⚡"
            
        Returns:
            工具的 emoji 字符串
        """
        entry = self.get_entry(name)
        return (entry.emoji if entry and entry.emoji else default)

    def get_tool_to_toolset_map(self) -> Dict[str, str]:
        """返回 ``{工具名称: 工具集名称}`` 映射，包含所有已注册工具。"""
        return {entry.name: entry.toolset for entry in self._snapshot_entries()}

    def is_toolset_available(self, toolset: str) -> bool:
        """检查工具集的要求是否满足。
        
        当检查函数抛出意外异常（例如网络错误、缺少导入、配置错误）时，
        返回 False（而不是崩溃）。
        
        Args:
            toolset: 工具集名称
            
        Returns:
            True 如果工具集可用，False 如果不可用
        """
        with self._lock:
            check = self._toolset_checks.get(toolset)
        return self._evaluate_toolset_check(toolset, check)

    def check_toolset_requirements(self) -> Dict[str, bool]:
        """返回 ``{工具集: 可用布尔值}`` 映射，包含所有工具集。"""
        entries, toolset_checks = self._snapshot_state()
        toolsets = sorted({entry.toolset for entry in entries})
        return {
            toolset: self._evaluate_toolset_check(toolset, toolset_checks.get(toolset))
            for toolset in toolsets
        }

    def get_available_toolsets(self) -> Dict[str, dict]:
        """返回工具集元数据，用于 UI 显示。
        
        Returns:
            字典，键为工具集名称，值为包含 available/tools/description/
            requirements 等信息的字典
        """
        toolsets: Dict[str, dict] = {}
        entries, toolset_checks = self._snapshot_state()
        for entry in entries:
            ts = entry.toolset
            if ts not in toolsets:
                toolsets[ts] = {
                    "available": self._evaluate_toolset_check(
                        ts, toolset_checks.get(ts)
                    ),
                    "tools": [],
                    "description": "",
                    "requirements": [],
                }
            toolsets[ts]["tools"].append(entry.name)
            # 收集环境变量要求
            if entry.requires_env:
                for env in entry.requires_env:
                    if env not in toolsets[ts]["requirements"]:
                        toolsets[ts]["requirements"].append(env)
        return toolsets

    def get_toolset_requirements(self) -> Dict[str, dict]:
        """构建与 TOOLSET_REQUIREMENTS 兼容的字典，用于向后兼容。"""
        result: Dict[str, dict] = {}
        entries, toolset_checks = self._snapshot_state()
        for entry in entries:
            ts = entry.toolset
            if ts not in result:
                result[ts] = {
                    "name": ts,
                    "env_vars": [],
                    "check_fn": toolset_checks.get(ts),
                    "setup_url": None,
                    "tools": [],
                }
            if entry.name not in result[ts]["tools"]:
                result[ts]["tools"].append(entry.name)
            # 收集环境变量
            for env in entry.requires_env:
                if env not in result[ts]["env_vars"]:
                    result[ts]["env_vars"].append(env)
        return result

    def check_tool_availability(self, quiet: bool = False):
        """返回 (available_toolsets, unavailable_info)，类似旧函数。
        
        Args:
            quiet: 如果为 True，不记录调试日志
            
        Returns:
            (可用工具集列表, 不可用工具集信息列表) 的元组
        """
        available = []
        unavailable = []
        seen = set()
        entries, toolset_checks = self._snapshot_state()
        for entry in entries:
            ts = entry.toolset
            if ts in seen:
                continue
            seen.add(ts)
            if self._evaluate_toolset_check(ts, toolset_checks.get(ts)):
                available.append(ts)
            else:
                unavailable.append({
                    "name": ts,
                    "env_vars": entry.requires_env,
                    "tools": [e.name for e in entries if e.toolset == ts],
                })
        return available, unavailable


# 模块级单例
registry = ToolRegistry()


# ---------------------------------------------------------------------------
# 工具响应序列化辅助函数
# ---------------------------------------------------------------------------
# 每个工具处理器必须返回 JSON 字符串。这些辅助函数消除了出现在
# 数百个工具文件中的样板代码 ``json.dumps({"error": msg}, ensure_ascii=False)``。
#
# 用法：
#   from tools.registry import registry, tool_error, tool_result
#
#   return tool_error("something went wrong")
#   return tool_error("not found", code=404)
#   return tool_result(success=True, data=payload)
#   return tool_result(items)            # 直接传递字典


def tool_error(message, **extra) -> str:
    """为工具处理器返回 JSON 错误字符串。
    
    Args:
        message: 错误消息
        **extra: 额外的键值对，会合并到结果中
        
    Returns:
        JSON 格式的错误字符串
        
    Examples:
        >>> tool_error("file not found")
        '{"error": "file not found"}'
        >>> tool_error("bad input", success=False)
        '{"error": "bad input", "success": false}'
    """
    result = {"error": str(message)}
    if extra:
        result.update(extra)
    return json.dumps(result, ensure_ascii=False)


def tool_result(data=None, **kwargs) -> str:
    """为工具处理器返回 JSON 结果字符串。
    
    接受位置参数字典 *或* 关键字参数（不能同时使用）：
    
    Args:
        data: 结果数据字典（可选）
        **kwargs: 结果键值对（如果 data 为 None 时使用）
        
    Returns:
        JSON 格式的结果字符串
        
    Examples:
        >>> tool_result(success=True, count=42)
        '{"success": true, "count": 42}'
        >>> tool_result({"key": "value"})
        '{"key": "value"}'
    """
    if data is not None:
        # 直接序列化传入的字典
        return json.dumps(data, ensure_ascii=False)
    # 序列化关键字参数
    return json.dumps(kwargs, ensure_ascii=False)
