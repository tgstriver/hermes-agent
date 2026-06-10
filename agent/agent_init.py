"""
AI Agent 初始化模块 — 将 AIAgent.__init__ 的核心逻辑提取为独立的模块函数。

设计背景：
    AIAgent.__init__ 是整个代码库中最长的方法之一（60+ 参数，约 1400 行代码），
    包含属性初始化、AI 提供商自动检测、凭据解析、上下文引擎引导启动等大量逻辑。
    如果继续留在 run_agent.py 中，会让该文件充满"初始化一次就不再管"的冗余代码。

提取后的结构：
    - 原来 __init__ 的主体逻辑变成了本模块的 init_agent(agent, ...) 函数
    - AIAgent.__init__ 变成一个薄封装层，内部直接调用 init_agent(self, ...)
    - 本模块列出了所有在模块加载时就需要导入的依赖
    - 函数体内还有大量懒加载导入（lazy import），这些保持不变

关于测试兼容性：
    测试代码中经常通过 mock 替换 run_agent 模块上的符号（如 run_agent.OpenAI、
    run_agent.cleanup_vm 等），为了保证这些 mock 依然生效，本模块通过 _ra() 函数
    动态引用 run_agent 模块，而不是直接 import，从而保证测试的 patch 机制正常工作。
"""

# ============================================================================
# 标准库导入
# ============================================================================
from __future__ import annotations

import logging          # 日志记录
import os               # 操作系统接口（环境变量、路径等）
import re               # 正则表达式（URL 解析、模型名匹配等）
import sys              # 系统相关（stderr 输出等）
import threading        # 多线程支持（中断机制、并发工具执行等）
import time             # 时间戳（活动追踪、超时检测等）
import uuid             # 唯一标识符生成（会话 ID）
from datetime import datetime   # 日期时间（会话开始时间记录）
from pathlib import Path        # 路径操作（日志目录、配置文件路径等）
from typing import Any, Dict, List, Optional   # 类型注解
from urllib.parse import urlparse, parse_qs, urlunparse  # URL 解析（提取查询参数等）

# ============================================================================
# 项目内部模块导入
# ============================================================================
from agent.context_compressor import ContextCompressor       # 上下文压缩器：对话接近模型上下文窗口限制时自动压缩
from agent.iteration_budget import IterationBudget           # 迭代预算：主 Agent + 子 Agent 共享的 LLM 调用次数上限
from agent.memory_manager import StreamingContextScrubber    # 流式上下文清洗器：处理跨流式分块的 <memory-context> 标签
from agent.model_metadata import (
    MINIMUM_CONTEXT_LENGTH,     # 模型允许的最小上下文长度常量（64K tokens）
    fetch_model_metadata,       # 从 API 获取模型元数据（价格、上下文长度等）
    get_model_context_length,   # 获取指定模型的上下文窗口大小
    is_local_endpoint,          # 判断 base_url 是否为本地端点（如 Ollama）
    query_ollama_num_ctx,       # 查询 Ollama 服务器支持的最大上下文长度
)
from agent.process_bootstrap import _install_safe_stdio      # 安装安全的标准输入输出（防止子进程继承问题）
from agent.subdirectory_hints import SubdirectoryHintTracker # 子目录提示追踪器：为 LLM 提供项目目录结构信息
from agent.think_scrubber import StreamingThinkScrubber      # 流式思考内容清洗器：清理模型输出的 <think> 标签
from agent.tool_guardrails import (
    ToolCallGuardrailConfig,    # 工具调用护栏配置（限制工具调用频率、检测异常模式等）
    ToolCallGuardrailController,# 工具调用护栏控制器
    ToolGuardrailDecision,      # 护栏决策结果类型
)
from hermes_cli.config import cfg_get                        # 配置读取工具（支持嵌套路径访问）
from hermes_cli.timeouts import get_provider_request_timeout # 获取各提供商的 API 请求超时时间
from hermes_constants import get_hermes_home                 # 获取 Hermes 主目录路径（~/.hermes/）
from model_tools import check_toolset_requirements, get_tool_definitions  # 工具集定义和依赖检查
from utils import base_url_host_matches                      # URL 主机名匹配工具（安全比较，防止 DNS 重绑定）

# 使用与 run_agent 相同的 logger 名称，这样测试代码中对 run_agent.logger 的 mock
# 也能捕获本模块发出的警告日志。（run_agent.py 中 logger = logging.getLogger(__name__)
# 在该模块内解析为 "run_agent"）
logger = logging.getLogger("run_agent")


def _ra():
    """
    懒加载引用 run_agent 模块。

    为什么不直接在顶部 import run_agent？
    因为测试代码会通过 mock.patch 替换 run_agent 上的属性（如 run_agent.OpenAI、
    run_agent.cleanup_vm 等）。如果在模块加载时就 import，那么后续对 run_agent 的
    patch 就无法影响到已经绑定的引用。通过 _ra() 动态获取，每次调用时都拿到最新的
    run_agent 模块引用，保证测试 patch 能正确生效。
    """
    import run_agent
    return run_agent


def _normalized_custom_base_url(value: Any) -> str:
    """
    标准化自定义 base URL：去除首尾空格和末尾斜杠。

    用途：在比较两个 base URL 是否指向同一服务时，需要先统一格式。
    例如 "https://api.example.com/" 和 "https://api.example.com" 应视为相同。

    参数:
        value: 待标准化的 URL 值，可能不是字符串类型
    返回:
        标准化后的 URL 字符串；如果输入不是字符串则返回空字符串
    """
    if not isinstance(value, str):
        return ""
    return value.strip().rstrip("/")


def _custom_provider_model_matches(agent_model: str, entry: Dict[str, Any]) -> bool:
    """
    判断自定义提供商配置条目中的模型名是否与 Agent 使用的模型匹配。

    匹配规则：
    - 如果配置条目中没有指定 model 字段（或为空），则视为通配符，匹配任何模型（返回 True）
    - 如果指定了 model，则进行大小写不敏感的精确比较

    参数:
        agent_model: Agent 当前使用的模型名称（如 "gpt-4o"）
        entry: 自定义提供商配置条目（字典），其中可能包含 "model" 字段
    返回:
        是否匹配
    """
    provider_model = str(entry.get("model", "") or "").strip().lower()
    if not provider_model:
        return True  # 未指定模型名 = 通配符，匹配任何模型
    return provider_model == str(agent_model or "").strip().lower()


def _custom_provider_extra_body_for_agent(
    *,
    provider: str,
    model: str,
    base_url: str,
    custom_providers: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    为当前 Agent 查找匹配的自定义提供商 extra_body 配置。

    什么是 extra_body？
    某些自定义 API 端点需要在请求体中附加额外字段（如特殊的路由参数、
    安全令牌等），这些字段通过 extra_body 注入到每次 API 请求中。

    查找逻辑：
    1. 只处理 provider == "custom" 的情况
    2. 在 custom_providers 列表中找到 base_url 匹配的条目
    3. 如果条目指定了 model，还需模型名也匹配才使用该条目的 extra_body
    4. 如果条目没有指定 model（通配），则作为 fallback 使用
    5. 优先返回精确匹配模型的条目，其次返回通配条目

    参数:
        provider: 当前提供商标识（如 "custom", "openai" 等）
        model: 当前模型名称
        base_url: 当前 API base URL
        custom_providers: 用户配置的自定义提供商列表
    返回:
        匹配到的 extra_body 字典，或 None（无匹配）
    """
    # 只对 "custom" 类型的提供商生效
    if (provider or "").strip().lower() != "custom":
        return None

    # 标准化当前 base URL 以便比较
    target_url = _normalized_custom_base_url(base_url)
    if not target_url:
        return None

    fallback: Optional[Dict[str, Any]] = None  # 通配条目（未指定 model 的）作为后备
    for entry in custom_providers or []:
        if not isinstance(entry, dict):
            continue
        # 比较 base_url 是否匹配
        if _normalized_custom_base_url(entry.get("base_url")) != target_url:
            continue
        # 检查是否有有效的 extra_body
        extra_body = entry.get("extra_body")
        if not isinstance(extra_body, dict) or not extra_body:
            continue
        # 检查模型名是否匹配
        provider_model = str(entry.get("model", "") or "").strip()
        if provider_model:
            # 条目指定了具体模型——只有模型名匹配才使用
            if _custom_provider_model_matches(model, entry):
                return dict(extra_body)  # 精确匹配，立即返回（拷贝一份防止修改原配置）
        elif fallback is None:
            # 条目没有指定模型（通配）——记录为 fallback，但继续查找精确匹配
            fallback = dict(extra_body)

    return fallback


def _merge_custom_provider_extra_body(agent, custom_providers: List[Dict[str, Any]]) -> None:
    """
    将匹配到的自定义提供商 extra_body 合并到 Agent 的 request_overrides 中。

    合并策略：
    - extra_body 中的字段作为底层默认值
    - 如果 request_overrides 中已有 extra_body，用户显式设置的字段优先（覆盖默认值）
    - 即：用户设置 > 自定义提供商配置 > 无

    这允许用户在 config.yaml 中为自定义 API 端点注入默认请求参数，
    同时保留在运行时通过 request_overrides 覆盖的能力。

    参数:
        agent: Agent 实例（会修改其 request_overrides 属性）
        custom_providers: 自定义提供商配置列表
    """
    # 查找当前 Agent 匹配的 extra_body 配置
    extra_body = _custom_provider_extra_body_for_agent(
        provider=agent.provider,
        model=agent.model,
        base_url=agent.base_url,
        custom_providers=custom_providers,
    )
    if not extra_body:
        return

    # 获取已有的 request_overrides（如果有的话）
    overrides = dict(getattr(agent, "request_overrides", {}) or {})
    merged_extra_body = dict(extra_body)  # 以 extra_body 为基础
    existing_extra_body = overrides.get("extra_body")
    if isinstance(existing_extra_body, dict):
        # 用户已有的设置覆盖默认值（update 后者的键值对覆盖前者）
        merged_extra_body.update(existing_extra_body)
    overrides["extra_body"] = merged_extra_body
    agent.request_overrides = overrides


def init_agent(
    agent,
    base_url: str = None,            # API 基础 URL（可选，不提供时使用提供商默认值）
    api_key: str = None,             # API 密钥（可选，不提供时从环境变量或配置文件读取）
    provider: str = None,            # 提供商标识（如 "openai", "anthropic", "openrouter" 等）
    api_mode: str = None,            # API 模式覆盖（"chat_completions" / "codex_responses" 等）
    acp_command: str = None,         # ACP（Agent Communication Protocol）运行时命令
    acp_args: list[str] | None = None,  # ACP 运行时参数
    command: str = None,             # 同 acp_command 的旧版参数名（向后兼容）
    args: list[str] | None = None,   # 同 acp_args 的旧版参数名（向后兼容）
    model: str = "",                 # 模型名称（如 "gpt-4o", "claude-sonnet-4-20250514"）
    max_iterations: int = 90,        # 最大工具调用迭代次数（主 Agent 和子 Agent 共享）
    tool_delay: float = 1.0,         # 工具调用之间的延迟秒数（防止触发速率限制）
    enabled_toolsets: List[str] = None,   # 只启用这些工具集（白名单模式）
    disabled_toolsets: List[str] = None,  # 禁用这些工具集（黑名单模式）
    save_trajectories: bool = False,      # 是否保存对话轨迹到 JSONL 文件
    verbose_logging: bool = False,        # 是否启用详细日志（调试用）
    quiet_mode: bool = False,             # 安静模式（CLI 默认开启，抑制进度输出）
    ephemeral_system_prompt: str = None,  # 临时系统提示（执行时使用但不保存到轨迹文件）
    log_prefix_chars: int = 100,          # 日志预览显示的字符数
    log_prefix: str = "",                 # 日志消息前缀（并行处理时区分不同 Agent）
    providers_allowed: List[str] = None,  # 允许使用的 OpenRouter 上游提供商列表
    providers_ignored: List[str] = None,  # 要忽略的 OpenRouter 上游提供商列表
    providers_order: List[str] = None,    # OpenRouter 提供商尝试顺序
    provider_sort: str = None,            # 提供商排序方式（按价格/吞吐量/延迟）
    provider_require_parameters: bool = False,  # 是否要求提供商返回参数信息
    provider_data_collection: str = None,       # 数据收集策略
    openrouter_min_coding_score: Optional[float] = None,  # 编程能力评分下限（0.0-1.0）
    session_id: str = None,                     # 预生成的会话 ID
    tool_progress_callback: callable = None,    # 工具进度通知回调
    tool_start_callback: callable = None,       # 工具开始执行回调
    tool_complete_callback: callable = None,    # 工具执行完成回调
    thinking_callback: callable = None,         # 思考过程回调
    reasoning_callback: callable = None,        # 推理过程回调
    clarify_callback: callable = None,          # 交互式用户提问回调
    step_callback: callable = None,             # 步骤回调
    stream_delta_callback: callable = None,     # 流式文本增量回调
    interim_assistant_callback: callable = None,# 中间助手消息回调
    tool_gen_callback: callable = None,         # 工具生成回调
    status_callback: callable = None,           # 状态更新回调
    max_tokens: int = None,                     # 模型响应最大 token 数（None = 使用模型默认值）
    reasoning_config: Dict[str, Any] = None,    # 推理配置（如 {"effort": "none"} 禁用思考）
    service_tier: str = None,                   # 服务层级（如 "flex" 弹性计算资源）
    request_overrides: Dict[str, Any] = None,   # 请求参数覆盖
    prefill_messages: List[Dict[str, Any]] = None,  # 预填充消息（注入到对话历史开头）
    platform: str = None,                       # 用户所在平台（"cli"/"telegram"/"discord" 等）
    user_id: str = None,                        # 平台用户标识
    user_id_alt: str = None,                    # 可选的稳定备用用户标识
    user_name: str = None,                      # 用户显示名称
    chat_id: str = None,                        # 聊天/频道 ID
    chat_name: str = None,                      # 聊天名称
    chat_type: str = None,                      # 聊天类型（如 "private", "group"）
    thread_id: str = None,                      # 线程/话题 ID
    gateway_session_key: str = None,            # 稳定的网关会话键
    skip_context_files: bool = False,           # 跳过自动注入上下文文件（SOUL.md 等）
    load_soul_identity: bool = False,           # 即使 skip_context_files=True 也加载 SOUL.md
    skip_memory: bool = False,                  # 跳过记忆系统初始化
    session_db=None,                            # SQLite 会话存储（由 CLI 或网关提供）
    parent_session_id: str = None,              # 父会话 ID（子 Agent 使用）
    iteration_budget: "IterationBudget" = None, # 共享的迭代预算对象
    fallback_model: Dict[str, Any] = None,      # 备用模型配置（单个字典或字典列表）
    credential_pool=None,                       # 凭据池（多 API 密钥轮换）
    checkpoints_enabled: bool = False,          # 是否启用文件系统检查点
    checkpoint_max_snapshots: int = 20,         # 最大快照数
    checkpoint_max_total_size_mb: int = 500,    # 检查点总大小上限（MB）
    checkpoint_max_file_size_mb: int = 10,      # 单个检查点文件大小上限（MB）
    pass_session_id: bool = False,              # 是否将会话 ID 传递给模型
):
    """
    初始化 AI Agent —— 这是 Agent 启动的核心函数。

    本函数完成以下主要工作：
    1. 基础属性赋值（模型、迭代次数、工具延迟等）
    2. API 模式自动检测（chat_completions / codex_responses / anthropic_messages / bedrock_converse）
    3. LLM 客户端构建（根据提供商选择正确的认证方式和 API 适配器）
    4. 工具集加载和过滤
    5. 上下文压缩器/引擎初始化
    6. 持久化记忆系统初始化
    7. 会话日志和追踪系统初始化
    8. Fallback（备用模型）链配置

    参数按类别分组说明：

    ── 连接参数 ──
    base_url: API 基础 URL（可选，不提供时使用提供商默认值）
    api_key: API 密钥（可选，不提供时从环境变量或配置文件读取）
    provider: 提供商标识（如 "openai", "anthropic", "openrouter" 等）
    api_mode: API 模式覆盖（"chat_completions" / "codex_responses" 等）

    ── 模型参数 ──
    model: 模型名称（如 "gpt-4o", "claude-sonnet-4-20250514"）
    max_iterations: 最大工具调用迭代次数（默认 90，主 Agent 和子 Agent 共享）
    tool_delay: 工具调用之间的延迟秒数（默认 1.0，防止触发速率限制）
    max_tokens: 模型响应的最大 token 数（None 表示使用模型默认值）
    reasoning_config: 推理配置覆盖（如 {"effort": "none"} 禁用思考模式）
    prefill_messages: 预填充消息（注入到对话历史开头的 few-shot 示例等）
        注意: Anthropic Sonnet 4.6+ 和 Opus 4.6+ 会拒绝以 assistant 消息结尾的
        对话（400 错误），对这些模型请使用结构化输出代替尾部 assistant 预填充。

    ── 工具集控制 ──
    enabled_toolsets: 只启用这些工具集（白名单模式）
    disabled_toolsets: 禁用这些工具集（黑名单模式）

    ── 日志和输出控制 ──
    save_trajectories: 是否保存对话轨迹到 JSONL 文件
    verbose_logging: 是否启用详细日志（调试用）
    quiet_mode: 安静模式（CLI 默认开启，抑制进度输出）
    ephemeral_system_prompt: 临时系统提示（执行时使用但不保存到轨迹文件）
    log_prefix_chars: 日志预览显示的字符数（工具调用/响应的前 N 个字符）
    log_prefix: 日志消息前缀（并行处理时用于区分不同 Agent 的输出）

    ── OpenRouter 特定参数 ──
    providers_allowed: 允许使用的 OpenRouter 上游提供商列表
    providers_ignored: 要忽略的 OpenRouter 上游提供商列表
    providers_order: OpenRouter 提供商尝试顺序
    provider_sort: 提供商排序方式（按价格/吞吐量/延迟）
    openrouter_min_coding_score: 编程能力评分下限（0.0-1.0），仅对
        model == "openrouter/pareto-code" 生效。None = 让 OpenRouter 选择最强编码器。

    ── 回调函数 ──
    tool_progress_callback: 工具进度通知回调 (tool_name, args_preview)
    clarify_callback: 交互式用户提问回调 (question, choices) -> str
        由平台层（CLI 或网关）提供。如果为 None，clarify 工具返回错误。

    ── 平台/用户信息（网关模式） ──
    platform: 用户所在平台（"cli", "telegram", "discord", "whatsapp" 等）
        用于向系统提示注入平台特定的格式化提示。
    skip_context_files: 跳过自动注入 SOUL.md, AGENTS.md, .cursorrules 等上下文文件
        用于批处理和数据生成，避免用户特定的身份/项目指令污染轨迹。
    load_soul_identity: 即使 skip_context_files=True 也加载 ~/.hermes/SOUL.md
        作为主身份。当前工作目录的项目上下文文件仍然被跳过。
    """
    # ========================================================================
    # 第 0 步：安装安全的标准输入输出
    # 防止子进程（如终端工具执行的命令）继承父进程的 stdin/stdout 文件描述符，
    # 避免子进程意外读取用户输入或写入混乱的输出。
    # ========================================================================
    _install_safe_stdio()

    # ========================================================================
    # 第 1 节：基础属性赋值
    # 将传入的参数直接赋值到 agent 实例上，这些属性在后续的对话循环中被频繁读取。
    # ========================================================================
    agent.model = model                          # 模型名称
    agent.max_iterations = max_iterations        # 最大工具调用迭代次数
    # 共享迭代预算：由父 Agent 创建，子 Agent 继承。
    # 每次 LLM 调用（无论是主 Agent 还是子 Agent）都会消耗预算。
    # 如果没有传入外部预算对象，则根据 max_iterations 创建一个新的。
    agent.iteration_budget = iteration_budget or IterationBudget(max_iterations)
    agent.tool_delay = tool_delay                # 工具调用间的延迟（防止触发 API 速率限制）
    agent.save_trajectories = save_trajectories  # 是否保存对话轨迹
    agent.verbose_logging = verbose_logging      # 详细日志开关
    agent.quiet_mode = quiet_mode                # 安静模式（抑制进度输出）
    agent.ephemeral_system_prompt = ephemeral_system_prompt  # 临时系统提示（不持久化）
    agent.platform = platform                    # 平台标识："cli", "telegram", "discord" 等

    # ── 平台用户/聊天标识（网关模式下使用）──
    agent._user_id = user_id                     # 平台用户标识（网关会话）
    agent._user_id_alt = user_id_alt             # 可选的稳定备用用户标识
    agent._user_name = user_name                 # 用户显示名称
    agent._chat_id = chat_id                     # 聊天/频道 ID
    agent._chat_name = chat_name                 # 聊天名称
    agent._chat_type = chat_type                 # 聊天类型（如 "private", "group"）
    agent._thread_id = thread_id                 # 线程/话题 ID
    agent._gateway_session_key = gateway_session_key  # 稳定的会话键（如 agent:main:telegram:dm:123）

    # 可插拔的 print 函数：CLI 模式下会替换为 _cprint，将 ANSI 状态行
    # 通过 prompt_toolkit 的渲染器输出，而不是直接写到 stdout（直接写会被
    # patch_stdout 的 StdoutProxy 破坏 ANSI 转义序列）。None 表示使用内置 print。
    agent._print_fn = None
    agent.background_review_callback = None      # 可选的同步回调，用于网关投递后台审查结果
    agent.skip_context_files = skip_context_files    # 是否跳过上下文文件注入
    agent.load_soul_identity = load_soul_identity    # 是否加载 SOUL.md 身份文件
    agent.pass_session_id = pass_session_id      # 是否将会话 ID 传递给模型
    agent._credential_pool = credential_pool     # 凭据池（多 API 密钥轮换机制）
    agent.log_prefix_chars = log_prefix_chars    # 日志预览字符数
    agent.log_prefix = f"{log_prefix} " if log_prefix else ""  # 日志前缀（并行时区分不同 Agent）

    # ========================================================================
    # 第 2 节：存储有效的 base URL 和提供商信息
    # ========================================================================
    agent.base_url = base_url or ""
    # 标准化提供商名称：去除空格并转为小写
    provider_name = provider.strip().lower() if isinstance(provider, str) and provider.strip() else None
    agent.provider = provider_name or ""
    agent.acp_command = acp_command or command   # ACP 运行时命令
    agent.acp_args = list(acp_args or args or [])  # ACP 运行时参数
    # ========================================================================
    # 第 3 节：API 模式自动检测
    # 根据提供商类型和 base URL 自动选择正确的 API 通信协议。
    # Hermes 支持多种 API 模式：
    # - chat_completions: OpenAI 兼容的聊天补全 API（最通用）
    # - codex_responses: OpenAI Responses API（GPT-5.x 等较新模型使用）
    # - anthropic_messages: Anthropic 原生消息 API（Claude 系列模型）
    # - bedrock_converse: AWS Bedrock Converse API
    # - codex_app_server: Codex 应用服务器模式
    # ========================================================================
    if api_mode in {"chat_completions", "codex_responses", "anthropic_messages", "bedrock_converse", "codex_app_server"}:
        agent.api_mode = api_mode  # 用户显式指定了 API 模式，直接使用
    elif agent.provider == "openai-codex":
        agent.api_mode = "codex_responses"  # OpenAI Codex 提供商使用 Responses API
    elif agent.provider in {"xai", "xai-oauth"}:
        agent.api_mode = "codex_responses"  # xAI（Grok）使用 Responses API 模式
    elif (provider_name is None) and (
        agent._base_url_hostname == "chatgpt.com"
        and "/backend-api/codex" in agent._base_url_lower
    ):
        # 自动检测：ChatGPT 的 Codex 后端 API
        agent.api_mode = "codex_responses"
        agent.provider = "openai-codex"
    elif (provider_name is None) and agent._base_url_hostname == "api.x.ai":
        # 自动检测：xAI 官方 API 端点
        agent.api_mode = "codex_responses"
        agent.provider = "xai"
    elif agent.provider == "anthropic" or (provider_name is None and agent._base_url_hostname == "api.anthropic.com"):
        # Anthropic 原生 API 或直连 api.anthropic.com
        agent.api_mode = "anthropic_messages"
        agent.provider = "anthropic"
    elif agent._base_url_lower.rstrip("/").endswith("/anthropic"):
        # 第三方 Anthropic 兼容端点（如 MiniMax、DashScope 等）
        # 它们的 URL 约定以 /anthropic 结尾，自动检测并使用 Anthropic Messages API 适配器
        agent.api_mode = "anthropic_messages"
    elif agent.provider == "bedrock" or (
        agent._base_url_hostname.startswith("bedrock-runtime.")
        and base_url_host_matches(agent._base_url_lower, "amazonaws.com")
    ):
        # AWS Bedrock：通过提供商名称或 base URL 自动检测
        # URL 格式为 bedrock-runtime.<region>.amazonaws.com
        agent.api_mode = "bedrock_converse"
    else:
        agent.api_mode = "chat_completions"  # 默认使用 OpenAI 兼容的聊天补全 API

    # ========================================================================
    # 第 4 节：预热传输层缓存
    # 提前调用 _get_transport() 来验证 api_mode 是否已注册对应的传输层实现。
    # 这样可以在初始化阶段就发现导入错误，而不是在对话进行到一半时才崩溃。
    # ========================================================================
    try:
        agent._get_transport()
    except Exception:
        pass  # 非致命——传输层可能尚未实现

    # ========================================================================
    # 第 5 节：模型名称标准化
    # 对于非聚合器提供商（非 OpenRouter 等），将模型名称标准化为该提供商的标准格式。
    # 例如某些提供商可能需要 "gpt-4o-2024-08-06" 而不是 "gpt-4o"。
    # ========================================================================
    try:
        from hermes_cli.model_normalize import (
            _AGGREGATOR_PROVIDERS,       # 聚合器提供商列表（如 "openrouter"）
            normalize_model_for_provider, # 根据提供商标准化模型名
        )

        if agent.provider not in _AGGREGATOR_PROVIDERS:
            agent.model = normalize_model_for_provider(agent.model, agent.provider)
    except Exception:
        pass  # 标准化失败不阻断初始化

    # ========================================================================
    # 第 6 节：GPT-5.x 自动升级到 Responses API
    # GPT-5.x 系列模型通常需要 Responses API 路径，但有些例外：
    # - Copilot 的 gpt-5-mini 仍然使用 chat completions
    # - 直连 OpenAI（api.openai.com）时，所有较新的工具调用模型都推荐 Responses
    # - ACP 运行时被排除：CopilotACPClient 自己处理路由，不实现 Responses API
    # - 当 api_mode 被显式指定时，尊重用户的选择（用户知道自己的端点支持什么）
    # - Azure OpenAI 例外：Azure 的 gpt-5.x 运行在 /chat/completions 上，不支持 Responses API
    # ========================================================================
    if (
        api_mode is None                              # 用户没有显式指定 api_mode
        and agent.api_mode == "chat_completions"       # 当前是 chat completions 模式
        and agent.provider != "copilot-acp"            # 不是 Copilot ACP
        and not str(agent.base_url or "").lower().startswith("acp://copilot")  # 不是 ACP 协议
        and not str(agent.base_url or "").lower().startswith("acp+tcp://")     # 不是 ACP+TCP
        and not agent._is_azure_openai_url()           # 不是 Azure OpenAI
        and (
            agent._is_direct_openai_url()              # 直连 OpenAI
            or agent._provider_model_requires_responses_api(  # 或模型需要 Responses API
                agent.model,
                provider=agent.provider,
            )
        )
    ):
        agent.api_mode = "codex_responses"  # 升级到 Responses API
        # 清除之前预热阶段缓存的传输层——因为 api_mode 从 chat_completions
        # 变成了 codex_responses，缓存的传输层不再适用
        if hasattr(agent, "_transport_cache"):
            agent._transport_cache.clear()

    # ========================================================================
    # 第 7 节：预 warm OpenRouter 模型元数据缓存
    # fetch_model_metadata() 有 1 小时的缓存；在后台线程中提前调用可以避免
    # 第一次 API 响应时因需要获取价格信息而产生阻塞的 HTTP 请求。
    #
    # 使用进程级别的 Event 标志来确保这个预热线程只启动一次——
    # 网关模式下每条消息都会创建新的 AIAgent 实例，如果没有这个守卫，
    # 每条消息都会泄漏一个线程，最终导致进程耗尽系统线程限制
    # （RuntimeError: can't start new thread）。
    # ========================================================================
    if (agent.provider == "openrouter" or agent._is_openrouter_url()) and \
            not _ra()._openrouter_prewarm_done.is_set():
        _ra()._openrouter_prewarm_done.set()  # 标记预热已启动，防止重复
        threading.Thread(
            target=fetch_model_metadata,  # 在后台线程中获取模型元数据
            daemon=True,                  # 守护线程——主进程退出时自动结束
            name="openrouter-prewarm",    # 线程名称（调试时可见）
        ).start()

    # ========================================================================
    # 第 8 节：回调函数注册
    # 将各种回调函数保存到 agent 实例上。这些回调在对话循环的不同阶段被调用，
    # 用于向前端（CLI 或网关）报告进度、流式文本、工具执行状态等。
    # ========================================================================
    agent.tool_progress_callback = tool_progress_callback      # 工具进度通知
    agent.tool_start_callback = tool_start_callback            # 工具开始执行
    agent.tool_complete_callback = tool_complete_callback      # 工具执行完成
    agent.suppress_status_output = False                       # 状态输出抑制标志
    agent.thinking_callback = thinking_callback                # 思考过程回调
    agent.reasoning_callback = reasoning_callback              # 推理过程回调
    agent.clarify_callback = clarify_callback                  # 交互式提问回调
    agent.step_callback = step_callback                        # 步骤回调
    agent.stream_delta_callback = stream_delta_callback        # 流式文本增量回调
    agent.interim_assistant_callback = interim_assistant_callback  # 中间助手消息回调
    agent.status_callback = status_callback                    # 状态更新回调
    agent.tool_gen_callback = tool_gen_callback                # 工具生成回调

    # ========================================================================
    # 第 9 节：工具执行状态和护栏
    # ========================================================================

    # 工具执行标志：当正在执行工具时设为 True，允许 _vprint 在工具执行期间
    # 输出信息（即使已注册了流式文本消费者——因为工具执行期间没有 token 流）
    agent._executing_tools = False
    # 工具调用护栏：检测工具调用中的异常模式（如死循环、重复调用等）
    agent._tool_guardrails = ToolCallGuardrailController()
    agent._tool_guardrail_halt_decision: ToolGuardrailDecision | None = None  # 护栏暂停决策

    # ========================================================================
    # 第 10 节：中断机制
    # 用于在工具循环执行过程中中断 Agent（如用户按下 Ctrl+C）。
    # ========================================================================
    agent._interrupt_requested = False          # 是否请求中断
    agent._interrupt_message = None             # 触发中断的可选消息
    agent._execution_thread_id: int | None = None  # 执行线程 ID（在 run_conversation() 开始时设置）
    agent._interrupt_thread_signal_pending = False  # 中断信号挂起标志
    agent._client_lock = threading.RLock()      # 客户端操作的可重入锁

    # ========================================================================
    # 第 11 节：/steer 机制（方向引导）
    # /steer 允许用户在不中断 Agent 的情况下注入一条备注到下一个工具结果中。
    # 与 interrupt() 不同，steer() 不会设置 _interrupt_requested；
    # 它等待当前工具批次自然完成，然后由 drain 钩子将文本追加到最后一个
    # 工具结果的 content 中，这样模型在下次迭代时能看到它。
    # 消息角色的交替得以保持（我们修改已有的 tool 消息而不是插入新的 user 轮次）。
    # ========================================================================
    agent._pending_steer: Optional[str] = None  # 待注入的引导文本
    agent._pending_steer_lock = threading.Lock()  # 引导文本的线程锁

    # ========================================================================
    # 第 12 节：并发工具执行的线程追踪
    # _execute_tool_calls_concurrent 在 ThreadPoolExecutor 的工作线程上运行每个工具——
    # 这些工作线程的 tid 与 _execution_thread_id 不同，所以 _set_interrupt() 单独
    # 使用不会使工作线程中的 is_interrupted() 返回 True。
    # 这里追踪所有工作线程的 ID，以便 interrupt() / clear_interrupt() 可以向它们
    # 广播中断信号。
    # ========================================================================
    agent._tool_worker_threads: set[int] = set()         # 活跃的工具工作线程 ID 集合
    agent._tool_worker_threads_lock = threading.Lock()    # 线程集合的锁

    # ========================================================================
    # 第 13 节：子 Agent 委派状态
    # ========================================================================
    agent._delegate_depth = 0        # 委派深度：0 = 顶级 Agent，子 Agent 递增
    agent._active_children = []      # 正在运行的子 AIAgent 列表（用于中断传播）
    agent._active_children_lock = threading.Lock()  # 子 Agent 列表的锁

    # ========================================================================
    # 第 14 节：OpenRouter 提供商偏好配置
    # 这些参数控制 OpenRouter 如何选择上游模型提供商。
    # ========================================================================
    agent.providers_allowed = providers_allowed          # 允许的提供商白名单
    agent.providers_ignored = providers_ignored          # 要忽略的提供商黑名单
    agent.providers_order = providers_order              # 提供商尝试顺序
    agent.provider_sort = provider_sort                  # 排序方式（价格/吞吐量/延迟）
    agent.provider_require_parameters = provider_require_parameters  # 是否要求返回参数
    agent.provider_data_collection = provider_data_collection       # 数据收集策略
    agent.openrouter_min_coding_score = openrouter_min_coding_score  # 编程能力评分下限

    # ========================================================================
    # 第 15 节：工具集过滤选项
    # ========================================================================
    agent.enabled_toolsets = enabled_toolsets    # 启用的工具集白名单
    agent.disabled_toolsets = disabled_toolsets  # 禁用的工具集黑名单

    # ========================================================================
    # 第 16 节：模型响应配置
    # ========================================================================
    agent.max_tokens = max_tokens                # 最大输出 token 数（None = 使用模型默认值）
    agent.reasoning_config = reasoning_config    # 推理配置（None = 使用默认值，OpenRouter 默认为 medium）
    agent.service_tier = service_tier            # 服务层级（如 "flex"）
    agent.request_overrides = dict(request_overrides or {})  # 请求参数覆盖（拷贝一份防止修改原对象）
    agent.prefill_messages = prefill_messages or []  # 预填充消息（注入到对话历史开头）
    agent._force_ascii_payload = False          # 是否强制 ASCII 编码请求体

    # ========================================================================
    # 第 17 节：Anthropic 提示缓存配置
    # Anthropic 提示缓存：对于在原生 Anthropic、OpenRouter 以及使用 Anthropic 协议
    # 的第三方网关上运行的 Claude 模型自动启用。
    # 通过缓存重复的输入 token，在多轮对话中可降低约 75% 的输入成本。
    # 使用 system_and_3 策略（4 个断点），详见 _anthropic_prompt_cache_policy。
    # ========================================================================
    agent._use_prompt_caching, agent._use_native_cache_layout = (
        agent._anthropic_prompt_cache_policy()
    )
    # Anthropic 支持 "5m"（默认）和 "1h" 两种缓存 TTL 层级。
    # 从 config.yaml 的 prompt_caching.cache_ttl 读取；未知值保持 "5m"。
    # 1h 层级的写入成本是 2x，而 5m 是 1.25x，但 1h 在长会话中
    # （两次交互间隔超过 5 分钟时）有更好的摊销效果。
    agent._cache_ttl = "5m"  # 默认 5 分钟缓存 TTL
    try:
        from hermes_cli.config import load_config as _load_pc_cfg

        _pc_cfg = _load_pc_cfg().get("prompt_caching", {}) or {}
        _ttl = _pc_cfg.get("cache_ttl", "5m")
        if _ttl in {"5m", "1h"}:  # 只接受已知的 TTL 值
            agent._cache_ttl = _ttl
    except Exception:
        pass  # 配置读取失败不影响初始化

    # ========================================================================
    # 第 18 节：迭代预算通知策略
    # 迭代预算：LLM 只在真正耗尽迭代预算时才被通知（api_call_count >= max_iterations）。
    # 此时注入一条消息，允许最后一次 API 调用，如果模型没有产生文本响应，
    # 则强制发送一条 user 消息要求它总结。
    # 不发送中间压力警告——之前的实验表明中间警告会导致模型在复杂任务上"过早放弃"。
    # ========================================================================
    agent._budget_exhausted_injected = False  # 是否已注入预算耗尽消息
    agent._budget_grace_call = False          # 是否在宽限调用中

    # ========================================================================
    # 第 19 节：活动追踪
    # 每次 API 调用、工具执行和流式分块时都会更新这些字段。
    # 用途：
    # - 网关超时处理器用它来报告 Agent 被终止时正在做什么
    # - "仍在工作"通知用它来显示进度信息
    # ========================================================================
    agent._last_activity_ts: float = time.time()    # 最后活动时间戳
    agent._last_activity_desc: str = "initializing"  # 最后活动描述
    agent._current_tool: str | None = None           # 当前正在执行的工具名
    agent._api_call_count: int = 0                   # API 调用计数器

    # ========================================================================
    # 第 20 节：速率限制追踪
    # 每次 API 调用后从 x-ratelimit-* 响应头中更新。
    # 被 /usage 斜杠命令访问以显示当前速率限制状态。
    # ========================================================================
    agent._rate_limit_state: Optional["RateLimitState"] = None

    # OpenRouter 响应缓存命中计数器——当在流式响应头中看到
    # X-OpenRouter-Cache-Status: HIT 时递增。
    agent._or_cache_hits: int = 0

    # ========================================================================
    # 第 21 节：集中式日志设置
    # agent.log（INFO+ 级别）和 errors.log（WARNING+ 级别）都存放在 ~/.hermes/logs/ 下。
    # 幂等设计——网关模式（每条消息创建新 AIAgent）不会重复添加处理器。
    # ========================================================================
    from hermes_logging import setup_logging, setup_verbose_logging
    setup_logging(hermes_home=_ra()._hermes_home)

    if agent.verbose_logging:
        # 详细日志模式：启用第三方库的日志输出
        setup_verbose_logging()
        _ra().logger.info("Verbose logging enabled (third-party library logs suppressed)")
    elif agent.quiet_mode:
        # 安静模式（CLI 默认）：
        # 注意：不要在这里提高单个 logger 的级别。这样做会阻止根 logger 的
        # 文件处理器（agent.log, errors.log）收到日志记录，因为 Python 在
        # 处理器传播之前会检查 logger.isEnabledFor()。
        # 我们依赖 hermes_logging.setup_logging() 在安静模式下不安装控制台
        # StreamHandler 的事实——所以 INFO 记录会流向文件处理器但永远不会
        # 到达控制台。任何未来的降噪逻辑应该在 hermes_logging.py 的处理器层面处理。
        pass

    # ========================================================================
    # 第 22 节：流式输出相关状态
    # ========================================================================

    # 内部流回调（在流式 TTS 期间设置）。
    # 在此初始化，以便 _vprint 在 run_conversation 之前就能引用它。
    agent._stream_callback = None

    # 延迟段落换行标志：在工具迭代完成后设为 True，
    # 这样下一个真正的文本增量前会预置一个 "\n\n"（段落分隔）。
    agent._stream_needs_break = False

    # 有状态的上下文清洗器：处理跨流式分块的 <memory-context> 标签。
    # 单独的 sanitize_context() 无法处理跨块边界的情况，因为块正则表达式
    # 需要在同一个字符串中同时看到开始和结束标签。
    agent._stream_context_scrubber = StreamingContextScrubber()

    # 有状态的思考标签清洗器：处理流式增量中的 <think>/reasoning 标签。
    # 替代了之前的逐增量 _strip_think_blocks 正则表达式，
    # 那个方案会破坏下游状态（例如 MiniMax-M2.7 流式传输 '<think>' 作为 delta1，
    # 'Let me check' 作为 delta2——正则表达式擦除了 delta1，导致下游状态机
    # 永远不知道有一个块被打开了，从而将 delta2 错误地泄漏为内容）。
    agent._stream_think_scrubber = StreamingThinkScrubber()

    # 当前模型响应期间通过实时 token 回调已传递的可见助手文本。
    # 用于避免当提供商稍后将其作为已完成的中间助手消息返回时重复发送相同的评论。
    agent._current_streamed_assistant_text = ""

    # 可选的当前轮次用户消息覆盖：当 API 层面的用户消息需要与持久化的
    # 对话记录不同时使用（例如 CLI 语音模式只在实时调用时添加临时前缀）。
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = None

    # 缓存 Anthropic 图像转文本的回退结果：按图像 payload/URL 缓存，
    # 这样单个工具循环不会对同一张图像历史重复运行辅助视觉处理。
    agent._anthropic_image_fallback_cache: Dict[str, str] = {}

    # ========================================================================
    # 第 23 节：LLM 客户端初始化
    # 通过集中式提供商路由器初始化 LLM 客户端。
    # 路由器负责认证解析、base URL、请求头、以及 Codex/Anthropic 封装。
    # raw_codex=True 因为主 Agent 需要直接访问 responses.stream()
    # 来进行 Codex Responses API 的流式传输。
    # ========================================================================
    agent._anthropic_client = None       # Anthropic 原生客户端（仅 anthropic_messages 模式使用）
    agent._is_anthropic_oauth = False    # 是否使用 Anthropic OAuth 认证

    # 一次性解析每个提供商/每个模型的请求超时时间，
    # 以便下面的所有客户端构建路径（Anthropic 原生、OpenAI 协议、
    # 基于路由器的隐式认证）都能一致地应用它。
    # Bedrock Claude 有自己的超时路径，不在此处处理。
    _provider_timeout = get_provider_request_timeout(agent.provider, agent.model)

    # ────────────────────────────────────────────────────────────────────────
    # 分支 A：Anthropic Messages API 模式
    # ────────────────────────────────────────────────────────────────────────
    if agent.api_mode == "anthropic_messages":
        from agent.anthropic_adapter import build_anthropic_client, resolve_anthropic_token

        # ── 子分支 A1：AWS Bedrock + Claude → 使用 AnthropicBedrock SDK ──
        # AnthropicBedrock SDK 提供完整的功能支持（提示缓存、思考预算、自适应思考）
        _is_bedrock_anthropic = agent.provider == "bedrock"
        if _is_bedrock_anthropic:
            from agent.anthropic_adapter import build_anthropic_bedrock_client
            # 从 base URL 中提取 AWS 区域（如 us-east-1, eu-west-1 等）
            _region_match = re.search(r"bedrock-runtime\.([a-z0-9-]+)\.", base_url or "")
            _br_region = _region_match.group(1) if _region_match else "us-east-1"
            agent._bedrock_region = _br_region
            # 构建 Bedrock 专用的 Anthropic 客户端
            agent._anthropic_client = build_anthropic_bedrock_client(_br_region)
            agent._anthropic_api_key = "aws-sdk"  # 使用 AWS SDK 认证，不需要单独的 API 密钥
            agent._anthropic_base_url = base_url
            agent._is_anthropic_oauth = False
            agent.api_key = "aws-sdk"
            agent.client = None          # Anthropic 模式下不需要 OpenAI 客户端
            agent._client_kwargs = {}
            if not agent.quiet_mode:
                print(f"🤖 AI Agent initialized with model: {agent.model} (AWS Bedrock + AnthropicBedrock SDK, {_br_region})")

        # ── 子分支 A2：原生 Anthropic 或第三方 Anthropic 兼容端点 ──
        else:
            # 只有当提供商确实是 "anthropic" 时，才回退到 ANTHROPIC_TOKEN 环境变量。
            # 其他使用 anthropic_messages 的提供商（MiniMax、Alibaba 等）必须使用自己的 API 密钥。
            # 如果回退，会将 Anthropic 的凭据发送到第三方端点，导致认证错误（修复 #1739, #minimax-401）。
            _is_native_anthropic = agent.provider == "anthropic"
            effective_key = (api_key or resolve_anthropic_token() or "") if _is_native_anthropic else (api_key or "")

            # MiniMax OAuth 特殊处理：
            # MiniMax OAuth 颁发的是短期（约 15 分钟）的访问令牌。
            # Anthropic SDK 在客户端构建时将 api_key 缓存为静态字符串，
            # 所以在启动时解析一次 bearer token 的会话会一直发送同一个令牌，
            # 直到 MiniMax 在会话中途返回 401。
            # 解决方案：将静态字符串替换为可调用的令牌提供器——
            # build_anthropic_client 识别 callable 并安装 httpx 事件钩子，
            # 在每个出站请求时铸造新的 bearer token（重新读取 auth.json，
            # 这样另一个进程保存的刷新令牌也能立即生效）。
            if agent.provider == "minimax-oauth" and isinstance(effective_key, str) and effective_key:
                try:
                    from hermes_cli.auth import build_minimax_oauth_token_provider
                    effective_key = build_minimax_oauth_token_provider()
                except Exception as _mm_exc:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "MiniMax OAuth: failed to install per-request token provider "
                        "(%s); falling back to static bearer that will expire ~15min in.",
                        _mm_exc,
                    )

            agent.api_key = effective_key
            agent._anthropic_api_key = effective_key
            agent._anthropic_base_url = base_url

            # OAuth 身份标记：只有当令牌确实属于原生 Anthropic 时才标记为 OAuth。
            # 使用 Anthropic 协议的第三方提供商（MiniMax、Kimi、GLM、LiteLLM 代理）
            # 绝不能触发 OAuth 代码路径——否则会注入 Claude-Code 身份头和系统提示，
            # 导致 401/403 错误。
            from agent.anthropic_adapter import _is_oauth_token as _is_oat
            agent._is_anthropic_oauth = _is_oat(effective_key) if (_is_native_anthropic and isinstance(effective_key, str)) else False

            # 构建 Anthropic 客户端（包含超时配置）
            agent._anthropic_client = build_anthropic_client(effective_key, base_url, timeout=_provider_timeout)
            agent.client = None          # Anthropic 模式下不需要 OpenAI 客户端
            agent._client_kwargs = {}
            if not agent.quiet_mode:
                print(f"🤖 AI Agent initialized with model: {agent.model} (Anthropic native)")
                # 检查是否使用 Microsoft Entra ID 凭据（Azure Foundry 场景）
                from agent.azure_identity_adapter import is_token_provider

                if is_token_provider(effective_key):
                    print("🔑 Using credentials: Microsoft Entra ID")
                elif isinstance(effective_key, str) and len(effective_key) > 12:
                    # 显示脱敏的令牌信息（只显示前 8 位和后 4 位）
                    print(f"🔑 Using token: {effective_key[:8]}...{effective_key[-4:]}")

    # ────────────────────────────────────────────────────────────────────────
    # 分支 B：AWS Bedrock Converse API 模式
    # ────────────────────────────────────────────────────────────────────────
    elif agent.api_mode == "bedrock_converse":
        # AWS Bedrock 直接使用 boto3，不需要 OpenAI 客户端。
        # 区域从 base_url 提取或默认为 us-east-1。
        _region_match = re.search(r"bedrock-runtime\.([a-z0-9-]+)\.", base_url or "")
        agent._bedrock_region = _region_match.group(1) if _region_match else "us-east-1"

        # Bedrock Guardrail 配置：从 config.yaml 在初始化时读取。
        # Guardrail 可以过滤输入/输出中的有害内容、PII 等。
        agent._bedrock_guardrail_config = None
        try:
            from hermes_cli.config import load_config as _load_br_cfg
            _gr = _load_br_cfg().get("bedrock", {}).get("guardrail", {})
            if _gr.get("guardrail_identifier") and _gr.get("guardrail_version"):
                agent._bedrock_guardrail_config = {
                    "guardrailIdentifier": _gr["guardrail_identifier"],
                    "guardrailVersion": _gr["guardrail_version"],
                }
                # 可选的流处理模式和追踪配置
                if _gr.get("stream_processing_mode"):
                    agent._bedrock_guardrail_config["streamProcessingMode"] = _gr["stream_processing_mode"]
                if _gr.get("trace"):
                    agent._bedrock_guardrail_config["trace"] = _gr["trace"]
        except Exception:
            pass  # Guardrail 配置可选，读取失败不阻断

        agent.client = None
        agent._client_kwargs = {}
        if not agent.quiet_mode:
            _gr_label = " + Guardrails" if agent._bedrock_guardrail_config else ""
            print(f"🤖 AI Agent initialized with model: {agent.model} (AWS Bedrock, {agent._bedrock_region}{_gr_label})")

    # ────────────────────────────────────────────────────────────────────────
    # 分支 C：OpenAI 兼容的 Chat Completions / Responses API 模式
    # ────────────────────────────────────────────────────────────────────────
    else:
        if api_key and base_url:
            # ================================================================
            # 子分支 C1：显式凭据（从 CLI/网关直接传入）
            # ================================================================
            # 从 base_url 中提取查询参数（如 Azure 的 api-version），
            # 并通过 default_query 传递，防止 SDK URL 拼接时丢失
            # （httpx 在拼接路径时会丢弃查询字符串）。
            _parsed_url = urlparse(base_url)
            if _parsed_url.query:
                # 将查询参数从 URL 中分离出来
                _clean_url = urlunparse(_parsed_url._replace(query=""))
                _query_params = {
                    k: v[0] for k, v in parse_qs(_parsed_url.query).items()
                }
                client_kwargs = {
                    "api_key": api_key,
                    "base_url": _clean_url,
                    "default_query": _query_params,  # 查询参数通过 default_query 传递
                }
            else:
                client_kwargs = {"api_key": api_key, "base_url": base_url}

            # 应用提供商级别的请求超时
            if _provider_timeout is not None:
                client_kwargs["timeout"] = _provider_timeout

            # Copilot ACP 模式需要额外的命令和参数
            if agent.provider == "copilot-acp":
                client_kwargs["command"] = agent.acp_command
                client_kwargs["args"] = agent.acp_args

            effective_base = base_url

            # ================================================================
            # 为不同提供商添加特定的请求头
            # ================================================================
            if base_url_host_matches(effective_base, "openrouter.ai"):
                # OpenRouter：添加自定义请求头（如 HTTP-Referer、X-Title 等）
                from agent.auxiliary_client import build_or_headers
                client_kwargs["default_headers"] = build_or_headers()
            elif base_url_host_matches(effective_base, "integrate.api.nvidia.com"):
                # NVIDIA NIM：添加 NVIDIA 特定的请求头
                from agent.auxiliary_client import build_nvidia_nim_headers
                client_kwargs["default_headers"] = build_nvidia_nim_headers(effective_base)
            elif base_url_host_matches(effective_base, "api.routermint.com"):
                # RouterMint：添加 RouterMint 特定的请求头
                client_kwargs["default_headers"] = _ra()._routermint_headers()
            elif base_url_host_matches(effective_base, "api.githubcopilot.com"):
                # GitHub Copilot：添加 Copilot 特定的请求头
                from hermes_cli.models import copilot_default_headers

                client_kwargs["default_headers"] = copilot_default_headers()
            elif base_url_host_matches(effective_base, "api.kimi.com"):
                # Kimi（Moonshot AI）：伪装为 claude-code 客户端
                client_kwargs["default_headers"] = {
                    "User-Agent": "claude-code/0.1.0",
                }
            elif base_url_host_matches(effective_base, "portal.qwen.ai"):
                # Qwen Portal（通义千问）：添加 Qwen 特定的请求头
                client_kwargs["default_headers"] = _ra()._qwen_portal_headers()
            elif base_url_host_matches(effective_base, "chatgpt.com"):
                # ChatGPT：添加 Cloudflare 绕过请求头
                from agent.auxiliary_client import _codex_cloudflare_headers
                client_kwargs["default_headers"] = _codex_cloudflare_headers(api_key)
            elif "default_headers" not in client_kwargs:
                # 兜底：从提供商配置文件中读取默认请求头
                # 某些提供商在配置文件中声明了自定义请求头（如 Kimi User-Agent）
                try:
                    from providers import get_provider_profile as _gpf
                    _ph = _gpf(agent.provider)
                    if _ph and _ph.default_headers:
                        client_kwargs["default_headers"] = dict(_ph.default_headers)
                except Exception:
                    pass  # 提供商配置文件可选

        else:
            # ================================================================
            # 子分支 C2：无显式凭据 — 使用集中式提供商路由器
            # ================================================================
            # resolve_provider_client 会自动查找已配置的凭据（环境变量、auth.json 等），
            # 并返回一个预配置好的客户端。
            from agent.auxiliary_client import resolve_provider_client
            _routed_client, _ = resolve_provider_client(
                agent.provider or "auto", model=agent.model, raw_codex=True)

            if _routed_client is not None:
                # 路由器成功找到了凭据和客户端
                client_kwargs = {
                    "api_key": _routed_client.api_key,
                    "base_url": str(_routed_client.base_url),
                }
                if _provider_timeout is not None:
                    client_kwargs["timeout"] = _provider_timeout
                # 保留路由器设置的提供商特定请求头。
                # OpenAI SDK 将调用者提供的 default_headers 存储在 _custom_headers 中；
                # 较旧/模拟的客户端可能使用 _default_headers。
                _routed_headers = getattr(_routed_client, "_custom_headers", None)
                if not _routed_headers:
                    _routed_headers = getattr(_routed_client, "_default_headers", None)
                if _routed_headers:
                    client_kwargs["default_headers"] = dict(_routed_headers)

            else:
                # ============================================================
                # 路由器也找不到凭据 — 进行 fallback 或报错
                # ============================================================
                # 当用户显式选择了非 OpenRouter 的提供商但找不到凭据时，
                # 快速失败并给出清晰的提示，而不是默默回退到 OpenRouter。
                _explicit = (agent.provider or "").strip().lower()
                if _explicit and _explicit not in {"auto", "openrouter", "custom"}:
                    # 从提供商配置中查找正确的环境变量名
                    # 某些提供商使用非标准的变量名（如 alibaba → DASHSCOPE_API_KEY）
                    _env_hint = f"{_explicit.upper()}_API_KEY"
                    try:
                        from hermes_cli.auth import PROVIDER_REGISTRY
                        _pcfg = PROVIDER_REGISTRY.get(_explicit)
                        if _pcfg and _pcfg.api_key_env_vars:
                            _env_hint = _pcfg.api_key_env_vars[0]
                    except Exception:
                        pass

                    # ── 初始化时的 fallback 尝试（#17929）──
                    # 在报错之前，先尝试使用备用模型配置是否能成功初始化
                    _fb_entries = []
                    if isinstance(fallback_model, list):
                        _fb_entries = [
                            f for f in fallback_model
                            if isinstance(f, dict) and f.get("provider") and f.get("model")
                        ]
                    elif isinstance(fallback_model, dict) and fallback_model.get("provider") and fallback_model.get("model"):
                        _fb_entries = [fallback_model]

                    _fb_resolved = False
                    for _fb in _fb_entries:
                        # 尝试解析 fallback 模型的凭据
                        _fb_explicit_key = (_fb.get("api_key") or "").strip() or None
                        if not _fb_explicit_key:
                            _fb_key_env = (_fb.get("key_env") or _fb.get("api_key_env") or "").strip()
                            if _fb_key_env:
                                _fb_explicit_key = os.getenv(_fb_key_env, "").strip() or None
                        _fb_client, _fb_model = resolve_provider_client(
                            _fb["provider"], model=_fb["model"], raw_codex=True,
                            explicit_base_url=_fb.get("base_url"),
                            explicit_api_key=_fb_explicit_key,
                        )
                        if _fb_client is not None:
                            # Fallback 成功！切换到备用模型
                            agent.provider = _fb["provider"]
                            agent.model = _fb_model or _fb["model"]
                            agent._fallback_activated = True  # 标记 fallback 已激活
                            client_kwargs = {
                                "api_key": _fb_client.api_key,
                                "base_url": str(_fb_client.base_url),
                            }
                            if _provider_timeout is not None:
                                client_kwargs["timeout"] = _provider_timeout
                            _fb_headers = getattr(_fb_client, "_custom_headers", None)
                            if not _fb_headers:
                                _fb_headers = getattr(_fb_client, "_default_headers", None)
                            if _fb_headers:
                                client_kwargs["default_headers"] = dict(_fb_headers)
                            _fb_resolved = True
                            break  # 找到可用的 fallback，停止遍历

                    if not _fb_resolved:
                        # 所有 fallback 都失败，给出清晰的错误提示
                        raise RuntimeError(
                            f"Provider '{_explicit}' is set in config.yaml but no API key "
                            f"was found. Set the {_env_hint} environment "
                            f"variable, or switch to a different provider with `hermes model`."
                        )

                if not getattr(agent, "_fallback_activated", False):
                    # 没有配置任何提供商——给出明确的错误提示
                    raise RuntimeError(
                        "No LLM provider configured. Run `hermes model` to "
                        "select a provider, or run `hermes setup` for first-time "
                        "configuration."
                    )

        # 保存客户端构建参数（用于在中断后重建客户端）
        agent._client_kwargs = client_kwargs

        # ================================================================
        # 为 OpenRouter 上的 Claude 模型启用细粒度工具流式传输
        # 没有这个设置时，Anthropic 会缓冲整个工具调用并在思考期间保持沉默——
        # OpenRouter 的上游代理会在沉默期间超时。
        # beta 请求头让 Anthropic 逐 token 流式传输工具调用参数，保持连接活跃。
        # ================================================================
        _effective_base = str(client_kwargs.get("base_url", "")).lower()
        if base_url_host_matches(_effective_base, "openrouter.ai") and "claude" in (agent.model or "").lower():
            headers = client_kwargs.get("default_headers") or {}
            existing_beta = headers.get("x-anthropic-beta", "")
            _FINE_GRAINED = "fine-grained-tool-streaming-2025-05-14"
            if _FINE_GRAINED not in existing_beta:
                if existing_beta:
                    # 追加到已有的 beta 头
                    headers["x-anthropic-beta"] = f"{existing_beta},{_FINE_GRAINED}"
                else:
                    headers["x-anthropic-beta"] = _FINE_GRAINED
                client_kwargs["default_headers"] = headers

        # ================================================================
        # 创建 OpenAI 兼容的客户端
        # ================================================================
        agent.api_key = client_kwargs.get("api_key", "")
        agent.base_url = client_kwargs.get("base_url", agent.base_url)
        try:
            # 使用 _create_openai_client 创建客户端（支持各种 OpenAI 兼容端点）
            agent.client = agent._create_openai_client(client_kwargs, reason="agent_init", shared=True)
            if not agent.quiet_mode:
                print(f"🤖 AI Agent initialized with model: {agent.model}")
                if base_url:
                    print(f"🔗 Using custom base URL: {base_url}")
                # 检查凭据类型并显示脱敏信息
                from agent.azure_identity_adapter import is_token_provider

                key_used = client_kwargs.get("api_key", "none")
                if is_token_provider(key_used):
                    print("🔑 Using credentials: Microsoft Entra ID")
                elif isinstance(key_used, str) and key_used and key_used != "dummy-key" and len(key_used) > 12:
                    # 显示脱敏的 API 密钥（前 8 位...后 4 位）
                    print(f"🔑 Using API key: {key_used[:8]}...{key_used[-4:]}")
                else:
                    print("⚠️  Warning: API key appears invalid or missing")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

    # ========================================================================
    # 第 24 节：提供商 Fallback 链配置
    # 当主提供商不可用时（速率限制、过载、连接失败），按顺序尝试备用提供商。
    # 支持两种格式：
    # - 旧版单个字典格式：fallback_model = {"provider": "...", "model": "..."}
    # - 新版列表格式：fallback_model = [{"provider": "...", "model": "..."}, ...]
    # ========================================================================
    if isinstance(fallback_model, list):
        # 新版列表格式：过滤出有效的条目（必须同时有 provider 和 model）
        agent._fallback_chain = [
            f for f in fallback_model
            if isinstance(f, dict) and f.get("provider") and f.get("model")
        ]
    elif isinstance(fallback_model, dict) and fallback_model.get("provider") and fallback_model.get("model"):
        # 旧版单个字典格式：包装为单元素列表
        agent._fallback_chain = [fallback_model]
    else:
        agent._fallback_chain = []  # 没有配置 fallback
    agent._fallback_index = 0       # 当前尝试的 fallback 索引
    agent._fallback_activated = getattr(agent, "_fallback_activated", False)  # fallback 是否已被激活
    # 保留旧版属性用于向后兼容（测试、外部调用者）
    agent._fallback_model = agent._fallback_chain[0] if agent._fallback_chain else None
    if agent._fallback_chain and not agent.quiet_mode:
        if len(agent._fallback_chain) == 1:
            fb = agent._fallback_chain[0]
            print(f"🔄 Fallback model: {fb['model']} ({fb['provider']})")
        else:
            # 显示完整的 fallback 链
            print(f"🔄 Fallback chain ({len(agent._fallback_chain)} providers): " +
                  " → ".join(f"{f['model']} ({f['provider']})" for f in agent._fallback_chain))

    # ========================================================================
    # 第 25 节：工具集加载和过滤
    # 获取可用工具列表，并根据 enabled_toolsets / disabled_toolsets 进行过滤。
    # ========================================================================
    agent.tools = _ra().get_tool_definitions(
        enabled_toolsets=enabled_toolsets,
        disabled_toolsets=disabled_toolsets,
        quiet_mode=agent.quiet_mode,
    )

    # 构建有效工具名称集合（用于后续验证模型的工具调用是否合法）
    agent.valid_tool_names = set()
    if agent.tools:
        agent.valid_tool_names = {tool["function"]["name"] for tool in agent.tools}
        tool_names = sorted(agent.valid_tool_names)
        if not agent.quiet_mode:
            print(f"🛠️  Loaded {len(agent.tools)} tools: {', '.join(tool_names)}")
            # 显示过滤器应用信息
            if enabled_toolsets:
                print(f"   ✅ Enabled toolsets: {', '.join(enabled_toolsets)}")
            if disabled_toolsets:
                print(f"   ❌ Disabled toolsets: {', '.join(disabled_toolsets)}")
    elif not agent.quiet_mode:
        print("🛠️  No tools loaded (all tools filtered out or unavailable)")

    # ========================================================================
    # 第 26 节：Kanban 工作/编排者生命周期引导
    # Kanban 工作者/编排者的生命周期引导是会话静态的：
    # 调度器在生成时决定当前进程是否是 kanban 工作者
    # （kanban_show 工具存在当且仅当 HERMES_KANBAN_TASK 环境变量被设置）。
    # 在初始化时解析一次（约 835 token 的块），避免在每次系统提示重建时
    # （初始化 + 每次上下文压缩）重新运行成员测试 + 引用。
    # ========================================================================
    from agent.prompt_builder import KANBAN_GUIDANCE
    agent._kanban_worker_guidance = (
        KANBAN_GUIDANCE if "kanban_show" in agent.valid_tool_names else ""
    )

    # ========================================================================
    # 第 27 节：工具依赖检查
    # ========================================================================
    if agent.tools and not agent.quiet_mode:
        # 检查工具集运行所需的外部依赖（如 ripgrep、docker 等）
        requirements = _ra().check_toolset_requirements()
        missing_reqs = [name for name, available in requirements.items() if not available]
        if missing_reqs:
            print(f"⚠️  Some tools may not work due to missing requirements: {missing_reqs}")

    # ========================================================================
    # 第 28 节：轨迹保存和临时系统提示状态显示
    # ========================================================================
    if agent.save_trajectories and not agent.quiet_mode:
        print("📝 Trajectory saving enabled")

    if agent.ephemeral_system_prompt and not agent.quiet_mode:
        # 显示临时系统提示的前 60 个字符预览
        prompt_preview = agent.ephemeral_system_prompt[:60] + "..." if len(agent.ephemeral_system_prompt) > 60 else agent.ephemeral_system_prompt
        print(f"🔒 Ephemeral system prompt: '{prompt_preview}' (not saved to trajectories)")

    # ========================================================================
    # 第 29 节：提示缓存状态显示
    # ========================================================================
    if agent._use_prompt_caching and not agent.quiet_mode:
        if agent._use_native_cache_layout and agent.provider == "anthropic":
            source = "native Anthropic"        # 原生 Anthropic API
        elif agent._use_native_cache_layout:
            source = "Anthropic-compatible endpoint"  # Anthropic 兼容端点
        else:
            source = "Claude via OpenRouter"   # 通过 OpenRouter 使用 Claude
        print(f"💾 Prompt caching: ENABLED ({source}, {agent._cache_ttl} TTL)")

    # ========================================================================
    # 第 30 节：会话 ID 生成和注册
    # ========================================================================
    agent.session_start = datetime.now()  # 记录会话开始时间
    if session_id:
        # 使用外部提供的会话 ID（如来自 CLI）
        agent.session_id = session_id
    else:
        # 生成新的会话 ID：时间戳 + 短 UUID（确保唯一性同时便于排序和调试）
        timestamp_str = agent.session_start.strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:6]
        agent.session_id = f"{timestamp_str}_{short_uuid}"

    # 将会话 ID 暴露给工具（如终端工具、execute_code），使 Agent 可以
    # 引用自己的会话来执行 --resume 命令、跨会话协调和日志记录。
    # 保持 ContextVar 和 os.environ 后备同步，因为不同的工具路径仍然读取两者。
    try:
        from gateway.session_context import set_current_session_id

        set_current_session_id(agent.session_id)
    except Exception:
        # 如果网关模块不可用（如 CLI 模式），回退到环境变量
        os.environ["HERMES_SESSION_ID"] = agent.session_id

    # ========================================================================
    # 第 31 节：会话日志目录设置
    # 会话日志存放在 ~/.hermes/sessions/ 下，与网关会话共享目录。
    # ========================================================================
    hermes_home = get_hermes_home()
    agent.logs_dir = hermes_home / "sessions"
    agent.logs_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在

    # 每会话 JSON 快照写入器（~/.hermes/sessions/session_{sid}.json）
    # 默认关闭（opt-in），通过 sessions.write_json_snapshots 配置开启。
    # state.db 是权威数据源——快照仅用于直接读取 JSON 文件的外部工具。
    agent._session_json_enabled = False
    try:
        from hermes_cli.config import load_config as _load_sess_cfg
        _sess_cfg = (_load_sess_cfg().get("sessions") or {})
        agent._session_json_enabled = bool(_sess_cfg.get("write_json_snapshots", False))
    except Exception:
        pass
    # logs_dir 无条件保留，用于 request_dump_*.json（由
    # agent_runtime_helpers.dump_api_request_debug 写入的调试面包屑路径）。

    # ========================================================================
    # 第 32 节：对话消息追踪和推理重放状态
    # ========================================================================
    # 追踪对话消息用于会话日志记录
    agent._session_messages: List[Dict[str, Any]] = []

    # Responses 加密推理重放状态。
    # 某些 OpenAI 兼容路由接受 GPT-5 Responses 请求，但稍后会拒绝重放的
    # 加密推理块（HTTP 400 invalid_encrypted_content）。
    # 当这种情况发生时，我们在会话剩余时间内禁用重放，回退到无状态连续性。
    agent._codex_reasoning_replay_enabled = True

    # 记忆写入来源和上下文标记
    agent._memory_write_origin = "assistant_tool"  # 记忆写入来源标识
    agent._memory_write_context = "foreground"     # 记忆写入上下文（前台/后台）

    # ========================================================================
    # 第 33 节：缓存的系统提示
    # 缓存的系统提示：每个会话只构建一次，仅在上下文压缩时重建。
    # 这避免了每次 API 调用都重新构建完整的系统提示（包含身份、工具描述等）。
    # ========================================================================
    agent._cached_system_prompt: Optional[str] = None

    # ========================================================================
    # 第 34 节：文件系统检查点管理器
    # 透明的检查点机制——不是工具，而是在后台自动工作。
    # 可以在对话过程中创建文件快照，支持回滚到之前的状态。
    # ========================================================================
    from tools.checkpoint_manager import CheckpointManager
    agent._checkpoint_mgr = CheckpointManager(
        enabled=checkpoints_enabled,
        max_snapshots=checkpoint_max_snapshots,
        max_total_size_mb=checkpoint_max_total_size_mb,
        max_file_size_mb=checkpoint_max_file_size_mb,
    )

    # ========================================================================
    # 第 35 节：SQLite 会话存储
    # ========================================================================
    agent._session_db = session_db              # SQLite 会话存储（可选，由 CLI 或网关提供）
    agent._parent_session_id = parent_session_id  # 父会话 ID（子 Agent 使用）
    agent._last_flushed_db_idx = 0              # DB 写入游标（防止重复写入）
    agent._session_db_created = False           # DB 行创建延迟到 run_conversation()
    agent._session_init_model_config = {
        "max_iterations": agent.max_iterations,
        "reasoning_config": reasoning_config,
        "max_tokens": max_tokens,
    }

    # ========================================================================
    # 第 36 节：内存中的 TODO 列表
    # 每个 Agent/会话一个 TODO 存储，用于任务规划。
    # ========================================================================
    from tools.todo_tool import TodoStore
    agent._todo_store = TodoStore()

    # ========================================================================
    # 第 37 节：加载配置文件（用于记忆、技能和压缩配置）
    # ========================================================================
    try:
        from hermes_cli.config import load_config as _load_agent_config
        _agent_cfg = _load_agent_config()
    except Exception:
        _agent_cfg = {}  # 配置加载失败不影响初始化

    # ========================================================================
    # 第 38 节：工具调用护栏配置
    # 从 config.yaml 的 tool_loop_guardrails 部分加载护栏配置。
    # 护栏可以检测工具调用中的异常模式（如死循环、重复调用相同参数等），
    # 并在必要时中断执行。
    # ========================================================================
    try:
        agent._tool_guardrails = ToolCallGuardrailController(
            ToolCallGuardrailConfig.from_mapping(
                _agent_cfg.get("tool_loop_guardrails", {})
            )
        )
    except Exception as _tlg_err:
        _ra().logger.warning("Tool loop guardrail config ignored: %s", _tlg_err)

    # 缓存辅助压缩模型的上下文长度配置覆盖。
    # 自定义端点通常无法通过 /models 报告此信息，所以启动可行性检查需要配置提示。
    agent._aux_compression_context_length_config = None

    # ========================================================================
    # 第 39 节：持久化记忆系统初始化
    # 持久化记忆（MEMORY.md + USER.md）从磁盘加载。
    # MEMORY.md 存储 Agent 积累的知识和经验。
    # USER.md 存储用户偏好和个性化信息。
    # ========================================================================
    agent._memory_store = None
    agent._memory_enabled = False
    agent._user_profile_enabled = False
    agent._memory_nudge_interval = 10    # 每隔多少轮提醒 Agent 使用记忆
    agent._turns_since_memory = 0        # 自上次记忆操作以来的轮次数
    agent._iters_since_skill = 0         # 自上次技能操作以来的迭代次数

    if not skip_memory:
        try:
            mem_config = _agent_cfg.get("memory", {})
            agent._memory_enabled = mem_config.get("memory_enabled", False)
            agent._user_profile_enabled = mem_config.get("user_profile_enabled", False)
            agent._memory_nudge_interval = int(mem_config.get("nudge_interval", 10))
            if agent._memory_enabled or agent._user_profile_enabled:
                from tools.memory_tool import MemoryStore
                agent._memory_store = MemoryStore(
                    memory_char_limit=mem_config.get("memory_char_limit", 2200),  # 记忆字符限制
                    user_char_limit=mem_config.get("user_char_limit", 1375),      # 用户资料字符限制
                )
                agent._memory_store.load_from_disk()  # 从磁盘加载已有记忆
        except Exception:
            pass  # 记忆系统是可选的——不应阻断 Agent 初始化

    # ========================================================================
    # 第 40 节：记忆提供商插件初始化
    # 记忆提供商插件（外部插件，与内置记忆并存）。
    # 从 config 的 memory.provider 字段读取选择哪个插件。
    # 支持的插件如 Honcho（提供跨会话的对话记忆和用户画像）。
    # ========================================================================
    agent._memory_manager = None
    if not skip_memory:
        try:
            _mem_provider_name = mem_config.get("provider", "") if mem_config else ""

            if _mem_provider_name and _mem_provider_name.strip():
                from agent.memory_manager import MemoryManager as _MemoryManager
                from plugins.memory import load_memory_provider as _load_mem
                agent._memory_manager = _MemoryManager()
                # 加载指定的记忆提供商插件
                _mp = _load_mem(_mem_provider_name)
                if _mp and _mp.is_available():
                    agent._memory_manager.add_provider(_mp)
                if agent._memory_manager.providers:
                    # 构建记忆提供商的初始化参数
                    _init_kwargs = {
                        "session_id": agent.session_id,          # 会话 ID
                        "platform": platform or "cli",            # 平台标识
                        "hermes_home": str(get_hermes_home()),    # Hermes 主目录
                        "agent_context": "primary",               # Agent 上下文（主/子）
                    }
                    # 传递会话标题用于记忆提供商的会话作用域
                    # （如 Honcho 用它来推导聊天级别的会话键）
                    if agent._session_db:
                        try:
                            _st = agent._session_db.get_session_title(agent.session_id)
                            if _st:
                                _init_kwargs["session_title"] = _st
                        except Exception:
                            pass
                    # 传递网关用户身份用于每用户记忆作用域
                    if agent._user_id:
                        _init_kwargs["user_id"] = agent._user_id
                    if agent._user_id_alt:
                        _init_kwargs["user_id_alt"] = agent._user_id_alt
                    if agent._user_name:
                        _init_kwargs["user_name"] = agent._user_name
                    if agent._chat_id:
                        _init_kwargs["chat_id"] = agent._chat_id
                    if agent._chat_name:
                        _init_kwargs["chat_name"] = agent._chat_name
                    if agent._chat_type:
                        _init_kwargs["chat_type"] = agent._chat_type
                    if agent._thread_id:
                        _init_kwargs["thread_id"] = agent._thread_id
                    # 传递网关会话键用于稳定的每聊天 Honcho 会话隔离
                    if agent._gateway_session_key:
                        _init_kwargs["gateway_session_key"] = agent._gateway_session_key
                    # 传递配置文件身份用于每配置文件提供商作用域
                    try:
                        from hermes_cli.profiles import get_active_profile_name
                        _profile = get_active_profile_name()
                        _init_kwargs["agent_identity"] = _profile
                        _init_kwargs["agent_workspace"] = "hermes"
                    except Exception:
                        pass
                    # 用收集到的参数初始化所有记忆提供商
                    agent._memory_manager.initialize_all(**_init_kwargs)
                    _ra().logger.info("Memory provider '%s' activated", _mem_provider_name)
                else:
                    _ra().logger.debug("Memory provider '%s' not found or not available", _mem_provider_name)
                    agent._memory_manager = None
        except Exception as _mpe:
            _ra().logger.warning("Memory provider plugin init failed: %s", _mpe)
            agent._memory_manager = None

    # ========================================================================
    # 第 41 节：注入记忆提供商工具定义到工具集
    # 将记忆提供商的工具定义（如 fact_store 等）注入到 Agent 的工具列表中。
    # 跳过名称已存在的工具（插件可能通过 ctx.register_tool() 注册了同名工具，
    # 这些工具已经通过 _ra().get_tool_definitions() 进入了 agent.tools）。
    # 重复的函数名会导致某些提供商（如小米 MiMo via Nous Portal）返回 400 错误。
    #
    # 同时尊重平台的 enabled_toolsets 配置（#5544）：
    #   enabled_toolsets 为 None → 不过滤，注入（向后兼容）
    #   "memory" 在 enabled_toolsets 中 → 用户选择了启用，注入
    #   否则（包括 []）→ 用户排除了记忆，跳过注入
    #
    # 没有这个守卫时，`platform_toolsets: telegram: []` 仍然会将记忆提供商工具
    # （如 fact_store）泄漏到工具集中——在本地模型上造成 10 倍延迟惩罚，
    # 并频繁触发工具调用循环。
    # ========================================================================
    if agent._memory_manager and agent.tools is not None and (
        agent.enabled_toolsets is None or "memory" in agent.enabled_toolsets
    ):
        _existing_tool_names = {
            t.get("function", {}).get("name")
            for t in agent.tools
            if isinstance(t, dict)
        }
        for _schema in agent._memory_manager.get_all_tool_schemas():
            _tname = _schema.get("name", "")
            if _tname and _tname in _existing_tool_names:
                continue  # 已通过插件路径注册，跳过
            _wrapped = {"type": "function", "function": _schema}
            agent.tools.append(_wrapped)
            if _tname:
                agent.valid_tool_names.add(_tname)
                _existing_tool_names.add(_tname)

    # ========================================================================
    # 第 42 节：技能（Skills）配置
    # 技能创建提醒间隔：每隔多少次迭代提醒 Agent 创建新技能。
    # ========================================================================
    agent._skill_nudge_interval = 10
    try:
        skills_config = _agent_cfg.get("skills", {})
        agent._skill_nudge_interval = int(skills_config.get("creation_nudge_interval", 10))
    except Exception:
        pass

    # ========================================================================
    # 第 43 节：工具使用强制配置
    # 控制是否强制模型使用工具（而不是直接回答）。
    # "auto"（默认）：根据内置模型列表自动决定
    # true：总是强制使用工具
    # false：从不强制
    # 字符串列表：模型名包含列表中任何子串时强制使用
    # ========================================================================
    _agent_section = _agent_cfg.get("agent", {})
    if not isinstance(_agent_section, dict):
        _agent_section = {}
    agent._tool_use_enforcement = _agent_section.get("tool_use_enforcement", "auto")

    # ========================================================================
    # 第 44 节：API 重试配置
    # 应用层 API 重试次数（包装每次模型 API 调用）。默认 3 次。
    # 可通过 config.yaml 的 agent.api_max_retries 覆盖。
    # ========================================================================
    try:
        _raw_api_retries = _agent_section.get("api_max_retries", 3)
        _api_retries = int(_raw_api_retries)
        _api_retries = max(_api_retries, 1)  # 最小值为 1（1 = 不重试，单次尝试）
    except (TypeError, ValueError):
        _api_retries = 3
    agent._api_max_retries = _api_retries

    # ========================================================================
    # 第 45 节：上下文压缩器初始化
    # 上下文压缩器在对话接近模型上下文窗口限制时自动压缩旧消息。
    # 通过 config.yaml 的 compression 部分进行配置。
    # ========================================================================
    _compression_cfg = _agent_cfg.get("compression", {})
    if not isinstance(_compression_cfg, dict):
        _compression_cfg = {}

    # 压缩触发阈值：上下文使用率达到多少时触发压缩（默认 50%）
    compression_threshold = float(_compression_cfg.get("threshold", 0.50))
    try:
        from agent.auxiliary_client import _compression_threshold_for_model as _cthresh_fn
        _model_cthresh = _cthresh_fn(agent.model)
        if _model_cthresh is not None:
            compression_threshold = _model_cthresh
    except Exception:
        pass
    compression_enabled = str(_compression_cfg.get("enabled", True)).lower() in {"true", "1", "yes"}
    compression_target_ratio = float(_compression_cfg.get("target_ratio", 0.20))
    compression_protect_last = int(_compression_cfg.get("protect_last_n", 20))
    # protect_first_n is the number of non-system messages to protect at
    # the head, in addition to the system prompt (which is always
    # implicitly protected by the compressor).  Floor at 0 — a value of
    # 0 means "preserve only the system prompt + summary + tail", which
    # is a legitimate (and common) configuration for long-running
    # rolling-compaction sessions.
    compression_protect_first = max(
        0, int(_compression_cfg.get("protect_first_n", 3))
    )
    compression_abort_on_summary_failure = str(
        _compression_cfg.get("abort_on_summary_failure", False)
    ).lower() in {"true", "1", "yes"}

    # Read optional explicit context_length override for the auxiliary
    # compression model. Custom endpoints often cannot report this via
    # /models, so the startup feasibility check needs the config hint.
    try:
        _aux_cfg = cfg_get(_agent_cfg, "auxiliary", "compression", default={})
    except Exception:
        _aux_cfg = {}
    if isinstance(_aux_cfg, dict):
        _aux_context_config = _aux_cfg.get("context_length")
    else:
        _aux_context_config = None
    if _aux_context_config is not None:
        try:
            _aux_context_config = int(_aux_context_config)
        except (TypeError, ValueError):
            _aux_context_config = None
    agent._aux_compression_context_length_config = _aux_context_config

    # Read explicit model output-token override from config when the
    # caller did not pass one directly.
    _model_cfg = _agent_cfg.get("model", {})
    if agent.max_tokens is None and isinstance(_model_cfg, dict):
        _config_max_tokens = _model_cfg.get("max_tokens")
        if _config_max_tokens is not None:
            try:
                if isinstance(_config_max_tokens, bool):
                    raise ValueError
                _parsed_max_tokens = int(_config_max_tokens)
                if _parsed_max_tokens <= 0:
                    raise ValueError
                agent.max_tokens = _parsed_max_tokens
            except (TypeError, ValueError):
                _ra().logger.warning(
                    "Invalid model.max_tokens in config.yaml: %r — "
                    "must be a positive integer (e.g. 4096). "
                    "Falling back to provider default.",
                    _config_max_tokens,
                )
                print(
                    f"\n⚠ Invalid model.max_tokens in config.yaml: {_config_max_tokens!r}\n"
                    f"  Must be a positive integer (e.g. 4096).\n"
                    f"  Falling back to provider default.\n",
                    file=sys.stderr,
                )
    agent._session_init_model_config["max_tokens"] = agent.max_tokens

    # Read explicit context_length override from model config
    if isinstance(_model_cfg, dict):
        _config_context_length = _model_cfg.get("context_length")
    else:
        _config_context_length = None
    if _config_context_length is not None:
        try:
            _config_context_length = int(_config_context_length)
        except (TypeError, ValueError):
            _ra().logger.warning(
                "Invalid model.context_length in config.yaml: %r — "
                "must be a plain integer (e.g. 256000, not '256K'). "
                "Falling back to auto-detection.",
                _config_context_length,
            )
            print(
                f"\n⚠ Invalid model.context_length in config.yaml: {_config_context_length!r}\n"
                f"  Must be a plain integer (e.g. 256000, not '256K').\n"
                f"  Falling back to auto-detected context window.\n",
                file=sys.stderr,
            )
            _config_context_length = None

    # Resolve custom_providers list once for reuse below (startup
    # context-length override and plugin context-engine init).
    try:
        from hermes_cli.config import get_compatible_custom_providers
        _custom_providers = get_compatible_custom_providers(_agent_cfg)
    except Exception:
        _custom_providers = _agent_cfg.get("custom_providers")
        if not isinstance(_custom_providers, list):
            _custom_providers = []

    # Store for reuse by _check_compression_model_feasibility (auxiliary
    # compression model context-length detection needs the same list).
    agent._custom_providers = _custom_providers
    _merge_custom_provider_extra_body(agent, _custom_providers)

    # Check custom_providers per-model context_length
    if _config_context_length is None and _custom_providers:
        try:
            from hermes_cli.config import get_custom_provider_context_length
            _cp_ctx_resolved = get_custom_provider_context_length(
                model=agent.model,
                base_url=agent.base_url,
                custom_providers=_custom_providers,
            )
            if _cp_ctx_resolved:
                _config_context_length = int(_cp_ctx_resolved)
        except Exception:
            _cp_ctx_resolved = None

        # Surface a clear warning if the user set a context_length but it
        # wasn't a valid positive int — the helper silently skips those.
        if _config_context_length is None:
            _target = agent.base_url.rstrip("/") if agent.base_url else ""
            for _cp_entry in _custom_providers:
                if not isinstance(_cp_entry, dict):
                    continue
                _cp_url = (_cp_entry.get("base_url") or "").rstrip("/")
                if _target and _cp_url == _target:
                    _cp_models = _cp_entry.get("models", {})
                    if isinstance(_cp_models, dict):
                        _cp_model_cfg = _cp_models.get(agent.model, {})
                        if isinstance(_cp_model_cfg, dict):
                            _cp_ctx = _cp_model_cfg.get("context_length")
                            if _cp_ctx is not None:
                                try:
                                    _parsed = int(_cp_ctx)
                                    if _parsed <= 0:
                                        raise ValueError
                                except (TypeError, ValueError):
                                    _ra().logger.warning(
                                        "Invalid context_length for model %r in "
                                        "custom_providers: %r — must be a positive "
                                        "integer (e.g. 256000, not '256K'). "
                                        "Falling back to auto-detection.",
                                        agent.model, _cp_ctx,
                                    )
                                    print(
                                        f"\n⚠ Invalid context_length for model {agent.model!r} in custom_providers: {_cp_ctx!r}\n"
                                        f"  Must be a positive integer (e.g. 256000, not '256K').\n"
                                        f"  Falling back to auto-detected context window.\n",
                                        file=sys.stderr,
                                    )
                    break

    # Persist for reuse on switch_model / fallback activation. Must come
    # AFTER the custom_providers branch so per-model overrides aren't lost.
    agent._config_context_length = _config_context_length

    agent._ensure_lmstudio_runtime_loaded(_config_context_length)



    # Select context engine: config-driven (like memory providers).
    # 1. Check config.yaml context.engine setting
    # 2. Check plugins/context_engine/<name>/ directory (repo-shipped)
    # 3. Check general plugin system (user-installed plugins)
    # 4. Fall back to built-in ContextCompressor
    _selected_engine = None
    _engine_name = "compressor"  # default
    try:
        _ctx_cfg = _agent_cfg.get("context", {}) if isinstance(_agent_cfg, dict) else {}
        _engine_name = _ctx_cfg.get("engine", "compressor") or "compressor"
    except Exception:
        pass

    if _engine_name != "compressor":
        # Try loading from plugins/context_engine/<name>/
        try:
            from plugins.context_engine import load_context_engine
            _selected_engine = load_context_engine(_engine_name)
        except Exception as _ce_load_err:
            _ra().logger.debug("Context engine load from plugins/context_engine/: %s", _ce_load_err)

        # Try general plugin system as fallback
        if _selected_engine is None:
            try:
                from hermes_cli.plugins import get_plugin_context_engine
                _candidate = get_plugin_context_engine()
                if _candidate and _candidate.name == _engine_name:
                    _selected_engine = _candidate
            except Exception:
                pass

        if _selected_engine is None:
            _ra().logger.warning(
                "Context engine '%s' not found — falling back to built-in compressor",
                _engine_name,
            )
    # else: config says "compressor" — use built-in, don't auto-activate plugins

    if _selected_engine is not None:
        agent.context_compressor = _selected_engine
        # Resolve context_length for plugin engines — mirrors switch_model() path
        from agent.model_metadata import get_model_context_length
        _plugin_ctx_len = get_model_context_length(
            agent.model,
            base_url=agent.base_url,
            api_key=getattr(agent, "api_key", ""),
            config_context_length=_config_context_length,
            provider=agent.provider,
            custom_providers=_custom_providers,
        )
        agent.context_compressor.update_model(
            model=agent.model,
            context_length=_plugin_ctx_len,
            base_url=agent.base_url,
            api_key=getattr(agent, "api_key", ""),
            provider=agent.provider,
            api_mode=agent.api_mode,
        )
        if not agent.quiet_mode:
            _ra().logger.info("Using context engine: %s", _selected_engine.name)
    else:
        agent.context_compressor = ContextCompressor(
            model=agent.model,
            threshold_percent=compression_threshold,
            protect_first_n=compression_protect_first,
            protect_last_n=compression_protect_last,
            summary_target_ratio=compression_target_ratio,
            summary_model_override=None,
            quiet_mode=agent.quiet_mode,
            base_url=agent.base_url,
            api_key=getattr(agent, "api_key", ""),
            config_context_length=_config_context_length,
            provider=agent.provider,
            api_mode=agent.api_mode,
            abort_on_summary_failure=compression_abort_on_summary_failure,
        )
    agent.compression_enabled = compression_enabled

    # Reject models whose context window is below the minimum required
    # for reliable tool-calling workflows (64K tokens).
    from agent.model_metadata import MINIMUM_CONTEXT_LENGTH
    _ctx = getattr(agent.context_compressor, "context_length", 0)
    if _ctx and _ctx < MINIMUM_CONTEXT_LENGTH:
        raise ValueError(
            f"Model {agent.model} has a context window of {_ctx:,} tokens, "
            f"which is below the minimum {MINIMUM_CONTEXT_LENGTH:,} required "
            f"by Hermes Agent.  Choose a model with at least "
            f"{MINIMUM_CONTEXT_LENGTH // 1000}K context, or set "
            f"model.context_length in config.yaml to override."
        )

    # Inject context engine tool schemas (e.g. lcm_grep, lcm_describe, lcm_expand).
    # Skip names that are already present — the _ra().get_tool_definitions()
    # quiet_mode cache returned a shared list pre-#17335, so a stray
    # mutation here would poison subsequent agent inits in the same
    # Gateway process and trip provider-side 'duplicate tool name'
    # errors. Even with the cache fix, dedup is the right defense
    # against plugin paths that may register the same schemas via
    # ctx.register_tool(). Mirrors the memory tools dedup above.
    #
    # Respect the platform's enabled_toolsets configuration (#5544):
    # context engine tools follow the same gating pattern as memory
    # provider tools — without the gate, `platform_toolsets: telegram: []`
    # would still leak lcm_* tools into the tool surface and incur the
    # same local-model latency penalty.
    agent._context_engine_tool_names: set = set()
    if (
        hasattr(agent, "context_compressor")
        and agent.context_compressor
        and agent.tools is not None
        and (
            agent.enabled_toolsets is None
            or "context_engine" in agent.enabled_toolsets
        )
    ):
        _existing_tool_names = {
            t.get("function", {}).get("name")
            for t in agent.tools
            if isinstance(t, dict)
        }
        for _schema in agent.context_compressor.get_tool_schemas():
            _tname = _schema.get("name", "")
            if _tname and _tname in _existing_tool_names:
                continue  # already registered via plugin/cache path
            _wrapped = {"type": "function", "function": _schema}
            agent.tools.append(_wrapped)
            if _tname:
                agent.valid_tool_names.add(_tname)
                agent._context_engine_tool_names.add(_tname)
                _existing_tool_names.add(_tname)

    # Notify context engine of session start
    if hasattr(agent, "context_compressor") and agent.context_compressor:
        try:
            agent.context_compressor.on_session_start(
                agent.session_id,
                hermes_home=str(get_hermes_home()),
                platform=agent.platform or "cli",
                model=agent.model,
                context_length=getattr(agent.context_compressor, "context_length", 0),
                conversation_id=getattr(agent, "_gateway_session_key", None),
            )
        except Exception as _ce_err:
            _ra().logger.debug("Context engine on_session_start: %s", _ce_err)

    agent._subdirectory_hints = SubdirectoryHintTracker(
        working_dir=os.getenv("TERMINAL_CWD") or None,
    )
    agent._user_turn_count = 0

    # Cumulative token usage for the session
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    agent.session_api_calls = 0
    agent.session_input_tokens = 0
    agent.session_output_tokens = 0
    agent.session_cache_read_tokens = 0
    agent.session_cache_write_tokens = 0
    agent.session_reasoning_tokens = 0
    agent.session_estimated_cost_usd = 0.0
    agent.session_cost_status = "unknown"
    agent.session_cost_source = "none"
    
    # ── Ollama num_ctx injection ──
    # Ollama defaults to 2048 context regardless of the model's capabilities.
    # When running against an Ollama server, detect the model's max context
    # and pass num_ctx on every chat request so the full window is used.
    # User override: set model.ollama_num_ctx in config.yaml to cap VRAM use.
    # If model.context_length is set, it caps num_ctx so the user's VRAM
    # budget is respected even when GGUF metadata advertises a larger window.
    agent._ollama_num_ctx: int | None = None
    _ollama_num_ctx_override = None
    if isinstance(_model_cfg, dict):
        _ollama_num_ctx_override = _model_cfg.get("ollama_num_ctx")
    if _ollama_num_ctx_override is not None:
        try:
            agent._ollama_num_ctx = int(_ollama_num_ctx_override)
        except (TypeError, ValueError):
            _ra().logger.debug("Invalid ollama_num_ctx config value: %r", _ollama_num_ctx_override)
    if agent._ollama_num_ctx is None and agent.base_url and is_local_endpoint(agent.base_url):
        try:
            # ``agent.api_key`` may be a callable (Entra token provider).
            # Ollama detection makes a manual HTTP request and expects a
            # string — Azure Foundry isn't a local endpoint so this branch
            # never fires for Entra, but guard defensively.
            _key_for_ollama = agent.api_key if isinstance(agent.api_key, str) else ""
            _detected = query_ollama_num_ctx(agent.model, agent.base_url, api_key=_key_for_ollama or "")
            if _detected and _detected > 0:
                agent._ollama_num_ctx = _detected
        except Exception as exc:
            _ra().logger.debug("Ollama num_ctx detection failed: %s", exc)
    # Cap auto-detected ollama_num_ctx to the user's explicit context_length.
    # Without this, GGUF metadata can advertise 256K+ which Ollama honours
    # by allocating that much VRAM — blowing up small GPUs even though the
    # user explicitly set a smaller context_length in config.yaml.
    if (
        agent._ollama_num_ctx
        and _config_context_length
        and _ollama_num_ctx_override is None  # don't override explicit ollama_num_ctx
        and agent._ollama_num_ctx > _config_context_length
    ):
        _ra().logger.info(
            "Ollama num_ctx capped: %d -> %d (model.context_length override)",
            agent._ollama_num_ctx, _config_context_length,
        )
        agent._ollama_num_ctx = _config_context_length
    if agent._ollama_num_ctx and not agent.quiet_mode:
        _ra().logger.info(
            "Ollama num_ctx: will request %d tokens (model max from /api/show)",
            agent._ollama_num_ctx,
        )

    if not agent.quiet_mode:
        if compression_enabled:
            print(f"📊 Context limit: {agent.context_compressor.context_length:,} tokens (compress at {int(compression_threshold*100)}% = {agent.context_compressor.threshold_tokens:,})")
        else:
            print(f"📊 Context limit: {agent.context_compressor.context_length:,} tokens (auto-compression disabled)")

    # Check immediately so CLI users see the warning at startup.
    # Gateway status_callback is not yet wired, so any warning is stored
    # in _compression_warning and replayed in the first run_conversation().
    agent._compression_warning = None
    # Lazy feasibility check: deferred to the first turn that approaches the
    # compression threshold. Running it eagerly here costs ~400ms cold (network
    # probe of the auxiliary provider chain + /models lookup) on every agent
    # init, including short ``chat -q`` runs that never reach the threshold.
    # ``ensure_compression_feasibility_checked`` (called from
    # ``run_conversation``'s preflight) runs it at most once per agent.
    agent._compression_feasibility_checked = False

    # Snapshot primary runtime for per-turn restoration.  When fallback
    # activates during a turn, the next turn restores these values so the
    # preferred model gets a fresh attempt each time.  Uses a single dict
    # so new state fields are easy to add without N individual attributes.
    _cc = agent.context_compressor
    agent._primary_runtime = {
        "model": agent.model,
        "provider": agent.provider,
        "base_url": agent.base_url,
        "api_mode": agent.api_mode,
        "api_key": getattr(agent, "api_key", ""),
        "client_kwargs": dict(agent._client_kwargs),
        "use_prompt_caching": agent._use_prompt_caching,
        "use_native_cache_layout": agent._use_native_cache_layout,
        # Context engine state that _try_activate_fallback() overwrites.
        # Use getattr for model/base_url/api_key/provider since plugin
        # engines may not have these (they're ContextCompressor-specific).
        "compressor_model": getattr(_cc, "model", agent.model),
        "compressor_base_url": getattr(_cc, "base_url", agent.base_url),
        "compressor_api_key": getattr(_cc, "api_key", ""),
        "compressor_provider": getattr(_cc, "provider", agent.provider),
        "compressor_context_length": _cc.context_length,
        "compressor_threshold_tokens": _cc.threshold_tokens,
    }
    if agent.api_mode == "anthropic_messages":
        agent._primary_runtime.update({
            "anthropic_api_key": agent._anthropic_api_key,
            "anthropic_base_url": agent._anthropic_base_url,
            "is_anthropic_oauth": agent._is_anthropic_oauth,
        })



__all__ = ["init_agent"]
