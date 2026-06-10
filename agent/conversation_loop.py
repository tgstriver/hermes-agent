"""智能体对话主循环 —— 从 ``run_agent.AIAgent`` 中提取。

这是从 ``run_agent.py`` 中抽出的最大一块代码：原先约 3900 行的
:func:`run_conversation` 函数体，负责驱动一轮用户交互的完整流程，
包括：模型调用、工具分发、重试机制、故障回退、上下文压缩、
轮次结束后的钩子，以及后台记忆/技能审查提醒。

该函数接收父级 ``AIAgent`` 实例作为第一个参数（``agent``），
通过属性访问来读取其状态。``_ra().AIAgent.run_conversation``
现在只是一个薄薄的转发器。

对于生产代码或测试中直接在 ``run_agent`` 上打补丁的符号
（``handle_function_call``、``_set_interrupt``、``OpenAI`` 等），
通过 :func:`_ra` 进行解析，以确保这些补丁在本代码路径中依然生效。
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import ssl
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from agent.anthropic_adapter import _is_oauth_token
from agent.auxiliary_client import set_runtime_main
from agent.codex_responses_adapter import _summarize_user_message_for_log
from agent.display import KawaiiSpinner
from agent.error_classifier import FailoverReason, classify_api_error
from agent.iteration_budget import IterationBudget
from agent.memory_manager import build_memory_context_block
from agent.message_sanitization import (
    _repair_tool_call_arguments,
    _sanitize_messages_non_ascii,
    _sanitize_messages_surrogates,
    _sanitize_structure_non_ascii,
    _sanitize_structure_surrogates,
    _sanitize_surrogates,
    _sanitize_tools_non_ascii,
    _strip_images_from_messages,
    _strip_non_ascii,
)
from agent.model_metadata import (
    MINIMUM_CONTEXT_LENGTH,
    estimate_messages_tokens_rough,
    estimate_request_tokens_rough,
    get_context_length_from_provider_error,
    parse_available_output_tokens_from_error,
    save_context_length,
)
from agent.nous_rate_guard import (
    clear_nous_rate_limit,
    is_genuine_nous_rate_limit,
    nous_rate_limit_remaining,
    record_nous_rate_limit,
)
from agent.process_bootstrap import _install_safe_stdio
from agent.prompt_caching import apply_anthropic_cache_control
from agent.retry_utils import jittered_backoff
from agent.trajectory import has_incomplete_scratchpad
from agent.usage_pricing import estimate_usage_cost, normalize_usage
from hermes_constants import display_hermes_home as _dhh_fn, PARTIAL_STREAM_STUB_ID
from hermes_logging import set_session_context
from tools.schema_sanitizer import strip_pattern_and_format
from tools.skill_provenance import set_current_write_origin
from utils import base_url_host_matches, env_var_enabled

logger = logging.getLogger(__name__)


def _ollama_context_limit_error(agent: Any, request_tokens: int) -> Optional[str]:
    """当 Ollama 加载的上下文窗口过小时，返回面向用户的错误信息。"""
    # 如果智能体没有配置工具，则不需要检查上下文限制
    if not getattr(agent, "tools", None):
        return None

    # 获取运行时实际的上下文窗口大小配置
    runtime_ctx = getattr(agent, "_ollama_num_ctx", None)
    if not isinstance(runtime_ctx, int) or runtime_ctx <= 0:
        return None
    # 如果运行时上下文已经满足最低要求，无需报错
    if runtime_ctx >= MINIMUM_CONTEXT_LENGTH:
        return None

    # 收集诊断信息，用于生成详细的错误提示
    model = getattr(agent, "model", "") or "the selected model"
    base_url = getattr(agent, "base_url", "") or "unknown base URL"
    provider = getattr(agent, "provider", "") or "unknown"
    tool_count = len(getattr(agent, "tools", None) or [])

    # 记录警告日志，便于调试 Ollama 上下文不足的问题
    logger.warning(
        "Ollama runtime context too small for Hermes tool use: "
        "model=%s provider=%s base_url=%s runtime_context=%d "
        "minimum_context=%d estimated_request_tokens=%d tool_count=%d "
        "session=%s",
        model,
        provider,
        base_url,
        runtime_ctx,
        MINIMUM_CONTEXT_LENGTH,
        request_tokens,
        tool_count,
        getattr(agent, "session_id", None) or "none",
    )

    # 返回详细的中文/英文混合错误提示，指导用户如何修复
    return (
        f"Ollama loaded `{model}` with only {runtime_ctx:,} tokens of runtime "
        f"context, but Hermes needs at least {MINIMUM_CONTEXT_LENGTH:,} tokens "
        "for reliable tool use.\n\n"
        "Increase the Ollama context for this model and restart/reload the "
        "model before trying again. A known-good starting point is 65,536 "
        "tokens. In Hermes config, set `model.ollama_num_ctx: 65536` "
        "(and `model.context_length: 65536` if you also override the displayed "
        "model context). If you manage the model through an Ollama Modelfile, "
        "set `PARAMETER num_ctx 65536` there instead."
    )


def _ra():
    """延迟引用 ``run_agent`` 模块，使调用方可以补丁
    ``run_agent.handle_function_call`` / ``run_agent._set_interrupt`` /
    ``run_agent.OpenAI`` 等符号，并确保这些补丁在本代码路径中生效。
    """
    import run_agent
    return run_agent


def _nous_entitlement_message(capability: str) -> str:
    """获取 Nous Portal 账户的权益提示信息。"""
    try:
        from hermes_cli.nous_account import (
            format_nous_portal_entitlement_message,
            get_nous_portal_account_info,
        )

        # 强制刷新账户信息，确保获取最新的权益状态
        account_info = get_nous_portal_account_info(force_fresh=True)
        message = format_nous_portal_entitlement_message(
            account_info,
            capability=capability,
        )
        return message or ""
    except Exception:
        return ""


def _print_nous_entitlement_guidance(agent, capability: str) -> bool:
    """向用户打印 Nous Portal 权益指引信息。"""
    message = _nous_entitlement_message(capability)
    if not message:
        return False
    # 逐行打印，每行前加灯泡图标以突出提示
    for line in message.splitlines():
        agent._vprint(f"{agent.log_prefix}   💡 {line}", force=True)
    return True


def _is_nous_inference_route(provider: str, base_url: str) -> bool:
    """判断当前请求是否路由到 Nous 推理服务。"""
    provider = (provider or "").strip().lower()
    if provider == "nous":
        return True
    base = str(base_url or "")
    # 检查 base_url 是否指向 Nous 推理 API 的域名
    return (
        base_url_host_matches(base, "inference-api.nousresearch.com")
        or base_url_host_matches(base, "inference.nousresearch.com")
    )


def _billing_or_entitlement_message(
    *,
    capability: str,
    provider: str,
    base_url: str,
    model: str,
) -> str:
    """生成账单/权益相关的提示信息。"""
    # 如果是 Nous 推理路由，使用 Nous 专用的权益提示
    if _is_nous_inference_route(provider, base_url):
        return _nous_entitlement_message(capability)

    # 为通用提供商生成账单耗尽/权益不足的提示
    provider_label = (provider or "").strip() or "the selected provider"
    model_label = (model or "").strip() or "the selected model"
    lines = [
        (
            f"{provider_label} reported that billing, credits, or account "
            f"entitlement is exhausted for {model_label}."
        ),
        "Add credits or update billing with that provider, then retry.",
    ]
    # 如果使用的是 OpenRouter，附加充值链接
    if base_url_host_matches(str(base_url or ""), "openrouter.ai"):
        lines.append("OpenRouter credits: https://openrouter.ai/settings/credits")
    lines.append("You can switch providers temporarily with /model <model> --provider <provider>.")
    return "\n".join(lines)


def _print_billing_or_entitlement_guidance(
    agent,
    *,
    capability: str,
    provider: str,
    base_url: str,
    model: str,
) -> bool:
    """打印账单/权益指引信息到用户界面。"""
    message = _billing_or_entitlement_message(
        capability=capability,
        provider=provider,
        base_url=base_url,
        model=model,
    )
    if not message:
        return False
    for line in message.splitlines():
        agent._vprint(f"{agent.log_prefix}   💡 {line}", force=True)
    return True


def _try_refresh_nous_paid_entitlement_credentials(agent) -> bool:
    """在验证付费权益后刷新 Nous 运行时凭证。"""
    try:
        from hermes_cli.auth import NOUS_INFERENCE_AUTH_MODE_LEGACY
        from hermes_cli.nous_account import get_nous_portal_account_info

        # 强制刷新账户信息以检查付费服务访问权限
        account_info = get_nous_portal_account_info(force_fresh=True)
        if account_info.paid_service_access is not True:
            return False
        # 刷新 Nous 客户端凭证（使用 legacy 认证模式）
        return agent._try_refresh_nous_client_credentials(
            force=False,
            inference_auth_mode=NOUS_INFERENCE_AUTH_MODE_LEGACY,
        )
    except Exception:
        return False


def _restore_or_build_system_prompt(agent, system_message, conversation_history):
    """从会话数据库恢复缓存的系统提示词，或从头构建新的系统提示词。

    修改 ``agent._cached_system_prompt``，并在首次构建时将新构建的
    提示词持久化到会话数据库。从 ``run_conversation`` 中提取出来，
    以便前缀缓存恢复路径可以独立测试。

    存储行的三种状态区分（通过日志可见，方便排查静默的前缀缓存未命中）：

      * ``missing`` — 尚无会话行（合法的首轮情况）。
      * ``null``   — 行存在，但 ``system_prompt`` 列为 NULL。
        通常是早于系统提示词持久化功能的遗留会话，或迁移残留。
        当 ``conversation_history`` 非空时打印警告。
      * ``empty``  — 行存在，但 ``system_prompt`` 列为空字符串。
        表示上一轮写入执行了但存储了空内容（静默持久化 bug）。
        总是打印警告。
      * ``present`` — 行存在且包含可用的提示词 → 原样复用。

    对会话数据库的读/写失败会以 WARNING 级别（而非 DEBUG）记录，
    以便持久性问题（磁盘满、schema 变更、锁争用）无需开启 verbose
    模式也能被发现。此处曾经是 debug 级别的日志，导致在网关路径
    （每轮都新建 ``AIAgent``，依赖此数据库往返）上静默破坏了
    前缀缓存复用。
    """
    stored_prompt = None
    stored_state = "missing"
    # 仅在存在对话历史且有会话数据库时尝试恢复
    if conversation_history and agent._session_db:
        try:
            session_row = agent._session_db.get_session(agent.session_id)
            if session_row is not None:
                raw_prompt = session_row.get("system_prompt")
                if raw_prompt is None:
                    stored_state = "null"
                elif raw_prompt == "":
                    stored_state = "empty"
                else:
                    stored_prompt = raw_prompt
                    stored_state = "present"
        except Exception as exc:
            # 以 WARNING 级别记录数据库读取失败，便于排查前缀缓存问题
            logger.warning(
                "Session DB get_session failed for system-prompt restore "
                "(session=%s): %s. Falling back to fresh build — prefix "
                "cache will miss for this turn.",
                agent.session_id, exc,
            )

    if stored_prompt:
        # 继续会话 —— 复用上一轮完全相同的系统提示词，
        # 以便 Anthropic 前缀缓存能够命中
        agent._cached_system_prompt = stored_prompt
        return

    if conversation_history and stored_state in ("null", "empty"):
        # 继续会话但存储的提示词不可用。上一轮的写入要么从未发生，
        # 要么写入了空字符串 —— 无论哪种情况，每轮都会重新构建，
        # 导致前缀缓存每次都未命中
        logger.warning(
            "Stored system prompt for session %s is %s; rebuilding "
            "from scratch this turn. Prefix cache will miss until "
            "the rebuild persists. Investigate the previous turn's "
            "update_system_prompt write path.",
            agent.session_id, stored_state,
        )

    # 新会话的首轮（或从损坏的存储提示词中恢复）—— 从头构建
    agent._cached_system_prompt = agent._build_system_prompt(system_message)

    # 插件钩子：on_session_start —— 仅在全新会话创建时触发一次
    # （不在继续会话时触发）。插件可使用此钩子初始化会话级状态
    # （例如预热记忆缓存）
    try:
        from hermes_cli.plugins import invoke_hook as _invoke_hook
        _invoke_hook(
            "on_session_start",
            session_id=agent.session_id,
            model=agent.model,
            platform=getattr(agent, "platform", None) or "",
        )
    except Exception as exc:
        logger.warning("on_session_start hook failed: %s", exc)

    # 将系统提示词快照持久化到 SQLite。此处失败曾经以 DEBUG 级别
    # 记录，导致在网关路径上静默破坏了前缀缓存复用（每轮新建
    # AIAgent → 后续每轮都从此行读取）
    if agent._session_db:
        try:
            agent._session_db.update_system_prompt(agent.session_id, agent._cached_system_prompt)
        except Exception as exc:
            logger.warning(
                "Session DB update_system_prompt failed for session %s: "
                "%s. Subsequent turns will rebuild the system prompt and "
                "miss the prefix cache.",
                agent.session_id, exc,
            )


def _get_continuation_prompt(is_partial_stub: bool, dropped_tools: Optional[List[str]] = None) -> str:
    """根据中断类型生成对应的续传提示词。

    Args:
        is_partial_stub: 是否为部分流式传输的存根响应
        dropped_tools: 被丢弃的工具调用名称列表

    Returns:
        用于让模型继续未完成响应的系统提示词
    """
    if is_partial_stub and dropped_tools:
        # 流式传输超时导致工具调用过大被丢弃 —— 提示模型拆分内容
        tool_list = ", ".join(dropped_tools[:3])
        return (
            "[System: Your previous tool call "
            f"({tool_list}) was too large and "
            "the stream timed out before it "
            "could be delivered. Do NOT retry "
            "the same tool call with the same "
            "large content. Instead, break the "
            "content into multiple smaller tool "
            "calls (e.g. use multiple patch calls "
            "or write smaller files). Each tool "
            "call's arguments must be under ~8K "
            "tokens to avoid stream timeouts.]"
        )
    elif is_partial_stub:
        # 网络错误导致流式传输中途断开 —— 提示模型从断点继续
        return (
            "[System: The previous response was cut off by a "
            "network error mid-stream. Continue exactly where "
            "you left off. Do not restart or repeat prior text. "
            "Finish the answer directly.]"
        )
    else:
        # 输出长度限制导致截断 —— 提示模型从截断处继续
        return (
            "[System: Your previous response was truncated by the output "
            "length limit. Continue exactly where you left off. Do not "
            "restart or repeat prior text. Finish the answer directly.]"
        )


def run_conversation(
    agent,
    user_message: str,
    system_message: str = None,
    conversation_history: List[Dict[str, Any]] = None,
    task_id: str = None,
    stream_callback: Optional[callable] = None,
    persist_user_message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    运行一轮完整的对话，包含工具调用循环直到完成。

    Args:
        user_message (str): 用户的消息/问题
        system_message (str): 自定义系统消息（可选，覆盖 ephemeral_system_prompt）
        conversation_history (List[Dict]): 之前的对话消息历史（可选）
        task_id (str): 此任务的唯一标识符，用于在并发任务间隔离虚拟机（可选，不提供则自动生成）
        stream_callback: 可选的回调函数，在流式传输期间每个文本增量时被调用。
            TTS 管线使用此回调在完整响应生成前启动音频生成。
            为 None（默认值）时，API 调用使用标准的非流式路径。
        persist_user_message: 当 user_message 包含仅用于 API 的合成前缀时，
            提供干净的用户消息用于存储到转录/历史记录中。

    Returns:
        Dict: 完整的对话结果，包含最终响应和消息历史
    """
    # 保护 stdio 防止来自损坏管道的 OSError（systemd/无头/守护进程模式）。
    # 安装一次，流健康时透明无感，防止写入时崩溃。
    _install_safe_stdio()

    # 确保数据库会话已初始化
    agent._ensure_db_session()

    # 告知 auxiliary_client 本轮次实际的主提供商/模型。
    # 供依赖主模型行为的工具使用（如 vision_analyze 的原生快速路径），
    # 使其看到 CLI/网关的覆盖值而非陈旧的 config.yaml 默认值。
    # 幂等操作 —— 每轮调用无副作用。
    try:
        from agent.auxiliary_client import set_runtime_main
        set_runtime_main(
            getattr(agent, "provider", "") or "",
            getattr(agent, "model", "") or "",
        )
    except Exception:
        pass

    # 在此线程上标记所有日志记录为当前会话 ID，
    # 使 ``hermes logs --session <id>`` 能过滤单个对话的日志
    from hermes_logging import set_session_context
    set_session_context(agent.session_id)

    # 绑定技能写入来源的 ContextVar（每线程独立），供工具处理器
    # （如 skill_manage create）判断是否运行在后台智能体改进审查分支中，
    # 还是前台用户交互轮次中。在每次调用开头设置；
    # 审查分支运行在独立线程上，拥有独立的上下文，
    # 因此此处的前台值不会泄漏到审查分支中。
    from tools.skill_provenance import set_current_write_origin
    set_current_write_origin(getattr(agent, "_memory_write_origin", "assistant_tool"))

    # 如果上一轮激活了回退机制，本轮恢复主运行时，以便使用首选模型重新尝试。
    # 当 _fallback_activated 为 False 时是空操作（网关、首轮等）。
    agent._restore_primary_runtime()

    # 清理用户输入中的代理字符（surrogate characters）。
    # 从富文本编辑器（Google Docs、Word 等）粘贴可能引入孤立的代理字符
    # （U+D800-U+DFFF），这是无效的 UTF-8，会导致 OpenAI SDK 中的
    # JSON 序列化崩溃。
    if isinstance(user_message, str):
        user_message = _sanitize_surrogates(user_message)
    if isinstance(persist_user_message, str):
        persist_user_message = _sanitize_surrogates(persist_user_message)

    # 存储流式回调，供 _interruptible_api_call 获取使用
    agent._stream_callback = stream_callback
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = persist_user_message
    # 如果未提供 task_id，生成唯一标识符以隔离并发任务间的虚拟机
    effective_task_id = task_id or str(uuid.uuid4())
    # 暴露当前活跃的 task_id，使轮次中运行的工具
    # （如 delegate_tool.py 中的 delegate_task）能识别此智能体，
    # 用于跨智能体文件状态注册表。
    # 在任何工具分发之前设置，确保子任务启动时快照看到的是
    # 父任务的真实 id，而非 None。
    agent._current_task_id = effective_task_id

    # 在每轮开始时重置重试计数器和迭代预算，
    # 避免上一轮的子智能体使用量影响下一轮。
    agent._invalid_tool_retries = 0
    agent._invalid_json_retries = 0
    agent._empty_content_retries = 0
    agent._incomplete_scratchpad_retries = 0
    agent._codex_incomplete_retries = 0
    agent._thinking_prefill_retries = 0
    agent._post_tool_empty_retried = False
    agent._last_content_with_tools = None
    agent._last_content_tools_all_housekeeping = False
    agent._mute_post_response = False
    agent._unicode_sanitization_passes = 0
    agent._tool_guardrails.reset_for_turn()
    agent._tool_guardrail_halt_decision = None
    # 在服务器以 "Only 'text' content type is supported." 等错误
    # 拒绝 image_url 内容部分之前，此值一直为 True。
    # 首次被拒绝后设为 False，并在会话剩余时间内保持，
    # 避免再向纯文本端点发送图像。
    # 作用域为每次 ``_run()`` 调用，而非实例级别。
    agent._vision_supported = True

    # 轮次前连接健康检查：检测并清理由提供商故障或断开流遗留的
    # 死 TCP 连接。这可以防止下一次 API 调用悬挂在僵尸套接字上。
    if agent.api_mode != "anthropic_messages":
        try:
            if agent._cleanup_dead_connections():
                agent._emit_status(
                    "🔌 Detected stale connections from a previous provider "
                    "issue — cleaned up automatically. Proceeding with fresh "
                    "connection."
                )
        except Exception:
            pass
    # 通过 status_callback 重放压缩警告（网关平台在 __init__ 时
    # 未连接此回调）
    if agent._compression_warning:
        agent._replay_compression_warning()
        agent._compression_warning = None  # 仅发送一次

    # 注意：_turns_since_memory 和 _iters_since_skill 不在此处重置。
    # 它们在 __init__ 中初始化，必须在多次 run_conversation 调用间持续累积，
    # 使 CLI 模式下的提醒逻辑正确工作。
    agent.iteration_budget = IterationBudget(agent.max_iterations)

    # 记录对话轮次开始信息，用于调试和可观测性
    _preview_text = _summarize_user_message_for_log(user_message)
    _msg_preview = (_preview_text[:80] + "...") if len(_preview_text) > 80 else _preview_text
    _msg_preview = _msg_preview.replace("\n", " ")
    logger.info(
        "conversation turn: session=%s model=%s provider=%s platform=%s history=%d msg=%r",
        agent.session_id or "none", agent.model, agent.provider or "unknown",
        agent.platform or "unknown", len(conversation_history or []),
        _msg_preview,
    )

    # 初始化对话消息列表（复制以避免修改调用方的列表）
    messages = list(conversation_history) if conversation_history else []

    # 从对话历史中水合 todo 存储（网关每条消息都新建 AIAgent，
    # 内存中的存储为空 —— 需要从历史中最近一条 todo 工具响应恢复状态）
    if conversation_history and not agent._todo_store.has_items():
        agent._hydrate_todo_store(conversation_history)

    # 从持久化历史中水合每轮次的提醒计数器。
    # 网关每条入站消息都新建 AIAgent（缓存未命中 / 1小时空闲驱逐 /
    # 配置签名不匹配 / 进程重启），因此 _turns_since_memory 和
    # _user_turn_count 每轮都从 0 开始，memory.nudge_interval 触发器
    # 可能永远无法达到。从 conversation_history 中之前的用户轮次重建
    # 有效计数。幂等操作：已累积计数的缓存智能体保留原值；
    # 仅新构建的内存状态为空的智能体执行水合。参见 issue #22357。
    if conversation_history and agent._user_turn_count == 0:
        prior_user_turns = sum(
            1 for m in conversation_history if m.get("role") == "user"
        )
        if prior_user_turns > 0:
            agent._user_turn_count = prior_user_turns
            if agent._memory_nudge_interval > 0 and agent._turns_since_memory == 0:
                # 取模保留原始的每 N 轮触发节奏，而非在恢复时立即触发审查
                # （否则会在会话恰好落在 N 的倍数之后时让用户感到意外）
                agent._turns_since_memory = prior_user_turns % agent._memory_nudge_interval


    # 预填充消息（few-shot 引导）仅在 API 调用时注入，
    # 永不存储到消息列表中。这样保持它们是临时的：不会保存到
    # 会话数据库、会话日志或批处理轨迹中，但在每次 API 调用时
    # （包括会话继续）会自动重新应用。

    # 跟踪用户轮次，用于记忆刷新和定期提醒逻辑
    agent._user_turn_count += 1

    # 在每轮开始时重置流式上下文清洗器，防止上一轮被中断的流
    # 遗留的悬挂 span 污染本轮输出。
    scrubber = getattr(agent, "_stream_context_scrubber", None)
    if scrubber is not None:
        scrubber.reset()
    # 同样重置思考清洗器 —— 上一轮被中断的流可能留下了
    # 未终止的思考块。
    think_scrubber = getattr(agent, "_stream_think_scrubber", None)
    if think_scrubber is not None:
        think_scrubber.reset()

    # 保存原始用户消息（不注入任何提醒内容）
    original_user_message = persist_user_message if persist_user_message is not None else user_message

    # 跟踪记忆提醒触发（基于轮次，在此处检查）。
    # 技能触发在智能体循环完成后检查，基于本轮使用了多少工具迭代。
    _should_review_memory = False
    if (agent._memory_nudge_interval > 0
            and "memory" in agent.valid_tool_names
            and agent._memory_store):
        agent._turns_since_memory += 1
        if agent._turns_since_memory >= agent._memory_nudge_interval:
            _should_review_memory = True
            agent._turns_since_memory = 0

    # 添加用户消息到消息列表
    user_msg = {"role": "user", "content": user_message}
    messages.append(user_msg)
    current_turn_user_idx = len(messages) - 1
    agent._persist_user_message_idx = current_turn_user_idx

    if not agent.quiet_mode:
        _print_preview = _summarize_user_message_for_log(user_message)
        agent._safe_print(f"💬 Starting conversation: '{_print_preview[:60]}{'...' if len(_print_preview) > 60 else ''}'")

    # ── 系统提示词（每轮缓存以实现前缀缓存）──
    # 首次调用时构建一次，后续所有调用复用。
    # 仅在上下文压缩事件后重新构建（压缩会使缓存失效
    # 并从磁盘重新加载记忆）。
    #
    # 对于继续的会话（网关每条消息都新建 AIAgent），
    # 从会话数据库加载已存储的系统提示词，而非重新构建。
    # 重新构建会从磁盘中拾取模型已经知道（它自己写的！）的
    # 记忆变更，产生不同的系统提示词并破坏 Anthropic 前缀缓存。
    if agent._cached_system_prompt is None:
        _restore_or_build_system_prompt(agent, system_message, conversation_history)

    active_system_prompt = agent._cached_system_prompt

    # ── 飞行前上下文压缩 ──
    # 在进入主循环之前，检查已加载的对话历史是否已超过
    # 模型的上下文阈值。处理以下场景：用户在拥有较大现有会话时
    # 切换到上下文窗口更小的模型 —— 主动压缩，而非等待 API 错误
    # （后者可能被识别为不可重试的 4xx 错误而直接中止请求）。
    if (
        agent.compression_enabled
        and len(messages) > agent.context_compressor.protect_first_n
                            + agent.context_compressor.protect_last_n + 1
    ):
        # 包含工具 schema 的 token 数 —— 当工具很多时，这部分可能
        # 增加 20-30K+ token，旧的系统+消息估算完全遗漏了这部分。
        _preflight_tokens = estimate_request_tokens_rough(
            messages,
            system_prompt=active_system_prompt or "",
            tools=agent.tools or None,
        )

        if agent.context_compressor.should_compress(_preflight_tokens):
            logger.info(
                "Preflight compression: ~%s tokens >= %s threshold (model %s, ctx %s)",
                f"{_preflight_tokens:,}",
                f"{agent.context_compressor.threshold_tokens:,}",
                agent.model,
                f"{agent.context_compressor.context_length:,}",
            )
            agent._emit_status(
                f"📦 Preflight compression: ~{_preflight_tokens:,} tokens "
                f">= {agent.context_compressor.threshold_tokens:,} threshold. "
                "This may take a moment."
            )
            # 对于上下文窗口很小而会话很大的情况，可能需要多轮压缩
            # （每轮压缩中间 N 条消息）
            for _pass in range(3):
                _orig_len = len(messages)
                messages, active_system_prompt = agent._compress_context(
                    messages, system_message, approx_tokens=_preflight_tokens,
                    task_id=effective_task_id,
                )
                if len(messages) >= _orig_len:
                    break  # 无法进一步压缩
                # 压缩创建了新会话 —— 清除历史引用，使
                # _flush_messages_to_session_db 将所有压缩后的消息
                # 写入新会话的 SQLite，而非因为 conversation_history
                # 仍指向压缩前的长度而跳过写入。
                conversation_history = None
                # 修复：压缩后重置重试计数器，使模型在压缩后的上下文上
                # 获得全新的重试预算。否则压缩前的重试会延续过来，
                # 模型在压缩导致的上下文丢失后立即遇到 "(empty)" 错误。
                agent._empty_content_retries = 0
                agent._thinking_prefill_retries = 0
                agent._last_content_with_tools = None
                agent._last_content_tools_all_housekeeping = False
                agent._mute_post_response = False
                # 压缩后重新估算 token 数
                _preflight_tokens = estimate_request_tokens_rough(
                    messages,
                    system_prompt=active_system_prompt or "",
                    tools=agent.tools or None,
                )
                if _preflight_tokens < agent.context_compressor.threshold_tokens:
                    break  # 已降至阈值以下

    # 插件钩子：pre_llm_call
    # 每轮在工具调用循环之前触发一次。插件可返回包含 ``context``
    # 键的字典（或纯字符串），其值会追加到当前轮次的用户消息中。
    #
    # 上下文始终注入到用户消息中，永不修改系统提示词。
    # 这是为了保护提示缓存前缀 —— 系统提示词在轮次间保持不变，
    # 使缓存的 token 能被复用。系统提示词是 Hermes 的专属区域；
    # 插件的贡献内容放在用户输入旁边。
    #
    # 所有注入的上下文都是临时的（不持久化到会话数据库）。
    _plugin_user_context = ""
    try:
        from hermes_cli.plugins import invoke_hook as _invoke_hook
        _pre_results = _invoke_hook(
            "pre_llm_call",
            session_id=agent.session_id,
            user_message=original_user_message,
            conversation_history=list(messages),
            is_first_turn=(not bool(conversation_history)),
            model=agent.model,
            platform=getattr(agent, "platform", None) or "",
            sender_id=getattr(agent, "_user_id", None) or "",
        )
        _ctx_parts: list[str] = []
        for r in _pre_results:
            if isinstance(r, dict) and r.get("context"):
                _ctx_parts.append(str(r["context"]))
            elif isinstance(r, str) and r.strip():
                _ctx_parts.append(r)
        if _ctx_parts:
            _plugin_user_context = "\n\n".join(_ctx_parts)
    except Exception as exc:
        logger.warning("pre_llm_call hook failed: %s", exc)

    # ── 主对话循环 ──
    # 循环变量初始化
    api_call_count = 0          # API 调用计数
    final_response = None       # 最终响应内容
    interrupted = False         # 是否被用户中断
    failed = False              # 是否发生致命失败
    codex_ack_continuations = 0  # Codex 确认续传次数
    length_continue_retries = 0  # 长度截断续传重试次数
    truncated_tool_call_retries = 0  # 截断工具调用重试次数
    truncated_response_parts: List[str] = []  # 截断响应的已接收片段
    compression_attempts = 0    # 压缩尝试次数
    _turn_exit_reason = "unknown"  # 诊断信息：循环结束原因

    # 每轮文件变更验证器状态。以解析后的路径为键；
    # 每次失败的 ``write_file`` / ``patch`` 调用记录错误预览。
    # 后续对同一路径的成功写入会移除该条目（模型已恢复）。
    # 在轮次结束时，仍存在的条目会作为建议性页脚追加到助手响应中，
    # 防止模型在文件实际未修改的情况下过度宣称成功。
    agent._turn_failed_file_mutations: Dict[str, Dict[str, Any]] = {}

    # 记录执行线程的 ID，使 interrupt()/clear_interrupt() 可以将
    # 工具级的中断信号限定在仅影响此智能体的线程。
    # 必须在任何线程级中断同步之前设置。
    agent._execution_thread_id = threading.current_thread().ident

    # 始终清除上一轮遗留的每线程中断状态。如果中断在启动完成前
    # 到达，保留它并绑定到当前执行线程，而非丢弃。
    _ra()._set_interrupt(False, agent._execution_thread_id)
    if agent._interrupt_requested:
        _ra()._set_interrupt(True, agent._execution_thread_id)
        agent._interrupt_thread_signal_pending = False
    else:
        agent._interrupt_message = None
        agent._interrupt_thread_signal_pending = False

    # 通知记忆提供者新轮次的开始，使节奏跟踪正常工作。
    # 必须在 prefetch_all() 之前调用，让提供者知道当前是哪一轮，
    # 并可通过 contextCadence/dialecticCadence 控制上下文/辩证刷新。
    if agent._memory_manager:
        try:
            _turn_msg = original_user_message if isinstance(original_user_message, str) else ""
            agent._memory_manager.on_turn_start(agent._user_turn_count, _turn_msg)
        except Exception:
            pass

    # 外部记忆提供者：在工具循环之前预取一次。
    # 在每次工具调用中复用缓存结果，避免重复调用 prefetch_all()
    # （10 次工具调用 = 10 倍延迟 + 成本）。
    # 使用 original_user_message（干净输入）—— user_message 可能
    # 包含注入的技能内容，会膨胀/破坏提供者查询。
    _ext_prefetch_cache = ""
    if agent._memory_manager:
        try:
            _query = original_user_message if isinstance(original_user_message, str) else ""
            _ext_prefetch_cache = agent._memory_manager.prefetch_all(_query) or ""
        except Exception:
            pass

    # 可选的运行时模式：如果 api_mode == codex_app_server，
    # 将轮次交给 codex app-server 子进程处理（终端/文件操作/补丁
    # 全部在 Codex 内部运行）。完全绕过 Hermes 默认路径。
    # 参见 agent/transports/codex_app_server_session.py 中的适配器
    # 和 references/codex-app-server-runtime.md 中的设计说明。
    if agent.api_mode == "codex_app_server":
        return agent._run_codex_app_server_turn(
            user_message=user_message,
            original_user_message=original_user_message,
            messages=messages,
            effective_task_id=effective_task_id,
            should_review_memory=_should_review_memory,
        )

    # ── 主工具调用循环 ──
    # 在 API 调用次数限制和迭代预算内持续循环，直到获得最终响应
    while (api_call_count < agent.max_iterations and agent.iteration_budget.remaining > 0) or agent._budget_grace_call:
        # 重置每轮检查点去重，使每次迭代可以拍摄一个快照
        agent._checkpoint_mgr.new_turn()

        # 检查中断请求（如用户发送了新消息）
        if agent._interrupt_requested:
            interrupted = True
            _turn_exit_reason = "interrupted_by_user"
            if not agent.quiet_mode:
                agent._safe_print("\n⚡ Breaking out of tool loop due to interrupt...")
            break

        api_call_count += 1
        agent._api_call_count = api_call_count
        agent._touch_activity(f"starting API call #{api_call_count}")

        # 宽限调用：预算已耗尽但给了模型最后一次机会。
        # 消费此标志，使循环在本次迭代后退出。
        if agent._budget_grace_call:
            agent._budget_grace_call = False
        elif not agent.iteration_budget.consume():
            _turn_exit_reason = "budget_exhausted"
            if not agent.quiet_mode:
                agent._safe_print(f"\n⚠️  Iteration budget exhausted ({agent.iteration_budget.used}/{agent.iteration_budget.max_total} iterations used)")
            break

        # 为网关钩子触发 step_callback（agent:step 事件）
        if agent.step_callback is not None:
            try:
                # 从消息末尾向前搜索最近的带 tool_calls 的助手消息，
                # 提取上一轮工具调用的信息用于回调
                prev_tools = []
                for _idx, _m in enumerate(reversed(messages)):
                    if _m.get("role") == "assistant" and _m.get("tool_calls"):
                        _fwd_start = len(messages) - _idx
                        _results_by_id = {}
                        for _tm in messages[_fwd_start:]:
                            if _tm.get("role") != "tool":
                                break
                            _tcid = _tm.get("tool_call_id")
                            if _tcid:
                                _results_by_id[_tcid] = _tm.get("content", "")
                        prev_tools = [
                            {
                                "name": tc["function"]["name"],
                                "result": _results_by_id.get(tc.get("id")),
                                "arguments": tc["function"].get("arguments"),
                            }
                            for tc in _m["tool_calls"]
                            if isinstance(tc, dict)
                        ]
                        break
                agent.step_callback(api_call_count, prev_tools)
            except Exception as _step_err:
                logger.debug("step_callback error (iteration %s): %s", api_call_count, _step_err)

        # 跟踪工具调用迭代次数，用于技能提醒。
        # 每当 skill_manage 实际被使用时计数器重置。
        if (agent._skill_nudge_interval > 0
                and "skill_manage" in agent.valid_tool_names):
            agent._iters_since_skill += 1

        # ── API 调用前的 /steer 指令排放 ──────────────────────────────
        # 如果在上一轮 API 调用期间（模型思考时）收到了 /steer 指令，
        # 在此处排放 —— 在构建 api_messages 之前 —— 使模型在本次迭代
        # 就能看到 steer 文本。否则，在 API 调用期间发送的 steer 只有在
        # 下一批工具调用之后才能生效（而如果模型返回最终响应，
        # 可能不会有下一批工具调用了）。
        #
        # 从消息列表末尾向前扫描最后一条 tool 角色的消息。
        # 如果找到，steer 注入到该消息中。
        # 如果未找到（首次迭代，还没有工具调用），steer 保持待定，
        # 等后续工具批次排放 —— 注入到 user 消息会破坏角色交替规则。
        _pre_api_steer = agent._drain_pending_steer()
        if _pre_api_steer:
            _injected = False
            for _si in range(len(messages) - 1, -1, -1):
                _sm = messages[_si]
                if isinstance(_sm, dict) and _sm.get("role") == "tool":
                    marker = f"\n\nUser guidance: {_pre_api_steer}"
                    existing = _sm.get("content", "")
                    if isinstance(existing, str):
                        _sm["content"] = existing + marker
                    else:
                        # 多模态内容块 —— 追加文本块
                        try:
                            blocks = list(existing) if existing else []
                            blocks.append({"type": "text", "text": marker})
                            _sm["content"] = blocks
                        except Exception:
                            pass
                    _injected = True
                    logger.debug(
                        "Pre-API-call steer drain: injected into tool msg at index %d",
                        _si,
                    )
                    break
            if not _injected:
                # 没有工具消息可以注入 —— 放回待处理队列，
                # 等后续工具执行后的排放处理
                _lock = getattr(agent, "_pending_steer_lock", None)
                if _lock is not None:
                    with _lock:
                        if agent._pending_steer:
                            agent._pending_steer = agent._pending_steer + "\n" + _pre_api_steer
                        else:
                            agent._pending_steer = _pre_api_steer
                else:
                    existing = getattr(agent, "_pending_steer", None)
                    agent._pending_steer = (existing + "\n" + _pre_api_steer) if existing else _pre_api_steer

        # ── 为 API 调用准备消息 ──
        # 如果有临时系统提示词，将其前置到消息中
        # 注意：推理内容通过 <think> 标签嵌入到 content 中，用于轨迹存储。
        # 但某些提供商（如 Moonshot AI）要求在带 tool_calls 的助手消息上
        # 使用独立的 'reasoning_content' 字段。此处同时处理两种情况。
        request_logger = getattr(agent, "logger", None) or logging.getLogger(__name__)
        # 修复损坏的 tool_call 参数（JSON 格式错误等）
        repaired_tool_calls = agent._sanitize_tool_call_arguments(
            messages,
            logger=request_logger,
            session_id=agent.session_id,
        )
        if repaired_tool_calls > 0:
            request_logger.info(
                "Sanitized %s corrupted tool_call arguments before request (session=%s)",
                repaired_tool_calls,
                agent.session_id or "-",
            )

        # 防御性修复：修复 malformed 的角色交替序列。
        # 捕获历史消息被卡住为 ``tool → user`` 或 ``user → user``
        # 尾部的情况（例如在空响应脚手架被剥离后、新的用户消息
        # 落在孤立的工具结果之后）。大多数提供商在 malformed 序列上
        # 返回空内容，否则会导致空重试循环无限触发。
        repaired_seq = agent._repair_message_sequence(messages)
        if repaired_seq > 0:
            request_logger.info(
                "Repaired %s message-alternation violations before request (session=%s)",
                repaired_seq,
                agent.session_id or "-",
            )

        # 构建 API 消息副本（不修改原始 messages 列表）
        api_messages = []
        for idx, msg in enumerate(messages):
            api_msg = msg.copy()

            # 将临时上下文注入到当前轮次的用户消息中。
            # 来源：记忆管理器预取 + 插件 pre_llm_call 钩子
            # （target="user_message"，为默认值）。
            # 两者都是 API 调用时才生效 —— `messages` 中的原始消息
            # 不会被修改，因此不会泄漏到会话持久化中。
            if idx == current_turn_user_idx and msg.get("role") == "user":
                _injections = []
                if _ext_prefetch_cache:
                    _fenced = build_memory_context_block(_ext_prefetch_cache)
                    if _fenced:
                        _injections.append(_fenced)
                if _plugin_user_context:
                    _injections.append(_plugin_user_context)
                if _injections:
                    _base = api_msg.get("content", "")
                    if isinstance(_base, str):
                        api_msg["content"] = _base + "\n\n" + "\n\n".join(_injections)

            # 对所有助手消息，将推理内容传回 API。
            # 确保多轮推理上下文被保留。
            agent._copy_reasoning_content_for_api(msg, api_msg)

            # 移除 'reasoning' 字段 —— 它仅用于轨迹存储
            # 上面已通过 _copy_reasoning_content_for_api 复制到 'reasoning_content'
            if "reasoning" in api_msg:
                api_msg.pop("reasoning")
            # 移除 finish_reason —— 严格的 API（如 Mistral）不接受此字段
            if "finish_reason" in api_msg:
                api_msg.pop("finish_reason")
            # 剥离内部思考预填充标记
            api_msg.pop("_thinking_prefill", None)
            # 剥离 Codex Responses API 字段（call_id, response_item_id），
            # 严格提供商（Mistral, Fireworks 等）会拒绝未知字段。
            # 使用新的 dict，使内部 messages 列表保留这些字段
            # 以兼容 Codex Responses。
            if agent._should_sanitize_tool_calls():
                agent._sanitize_tool_calls_for_strict_api(api_msg)
            # 保留 'reasoning_details' —— OpenRouter 使用此字段进行
            # 多轮推理上下文传递。签名字段有助于维护推理连续性。
            api_messages.append(api_msg)

        # 构建最终的系统消息：缓存的提示词 + 临时系统提示词。
        # 临时添加内容仅在 API 调用时生效（不持久化到会话数据库）。
        # 外部回忆上下文注入到用户消息中，而非系统提示词中，
        # 以保持稳定的缓存前缀不变。
        #
        # 注意：来自 pre_llm_call 钩子的插件上下文注入到用户消息中
        # （见上方注入块），而非系统提示词。这是有意为之 ——
        # 系统提示词的修改会破坏提示缓存前缀。
        # 系统提示词保留给 Hermes 内部使用。
        #
        # Hermes 不变式：系统提示词在每轮会话中构建一次
        # （缓存在 ``_cached_system_prompt`` 上），并在每轮逐字重放。
        # 我们将其作为单个内容字符串发送，使字节在轮次间保持稳定，
        # 上游提示缓存保持温热。
        effective_system = active_system_prompt or ""
        if agent.ephemeral_system_prompt:
            effective_system = (effective_system + "\n\n" + agent.ephemeral_system_prompt).strip()
        if effective_system:
            api_messages = [{"role": "system", "content": effective_system}] + api_messages

        # 在系统提示词之后、对话历史之前注入临时预填充消息。
        # 同样是 API 调用时才生效的模式。
        if agent.prefill_messages:
            sys_offset = 1 if (api_messages and api_messages[0].get("role") == "system") else 0
            for idx, pfm in enumerate(agent.prefill_messages):
                api_messages.insert(sys_offset + idx, pfm.copy())

        # 为 Claude 模型应用 Anthropic 提示缓存。
        # 在原生 Anthropic、OpenRouter 以及第三方 Anthropic 兼容网关上，
        # 自动检测并注入 cache_control 断点（系统 + 最后 3 条消息），
        # 在多轮对话中减少约 75% 的输入 token 成本。
        if agent._use_prompt_caching:
            api_messages = apply_anthropic_cache_control(
                api_messages,
                cache_ttl=agent._cache_ttl,
                native_anthropic=agent._use_native_cache_layout,
            )

        # 安全网：在发送到 API 之前剥离孤立的工具结果 / 为缺失结果添加存根。
        # 无条件运行 —— 不受 context_compressor 开关控制 ——
        # 确保来自会话加载或手动消息操作的孤立消息始终被捕获。
        api_messages = agent._sanitize_api_messages(api_messages)

        # 剥离仅包含思考内容的助手轮次（有推理但没有可见输出且没有 tool_calls），
        # 并合并遗留的相邻用户消息。防止 Anthropic 400 错误
        # （"The final block in an assistant message cannot be `thinking`."）
        # 以及第三方 Anthropic 兼容网关的等效错误（无法重放仅思考的轮次）。
        # 仅在每调用副本上运行 —— 存储的对话历史保留推理块，
        # 用于 UI 转录和会话持久化。
        api_messages = agent._drop_thinking_only_and_merge_users(api_messages)

        # 规范化消息中的空白和工具调用 JSON，以实现一致的前缀匹配。
        # 确保跨轮次的比特级精确前缀，启用本地推理服务器
        # （llama.cpp, vLLM, Ollama）上的 KV 缓存复用，
        # 并提高云提供商的缓存命中率。
        # 操作对象是 api_messages（API 副本），因此原始对话历史
        # `messages` 不受影响。
        for am in api_messages:
            if isinstance(am.get("content"), str):
                am["content"] = am["content"].strip()
        # 规范化工具调用的参数 JSON（紧凑格式 + 键排序）
        for am in api_messages:
            tcs = am.get("tool_calls")
            if not tcs:
                continue
            new_tcs = []
            for tc in tcs:
                if isinstance(tc, dict) and "function" in tc:
                    try:
                        args_obj = json.loads(tc["function"]["arguments"])
                        tc = {**tc, "function": {
                            **tc["function"],
                            "arguments": json.dumps(
                                args_obj, separators=(",", ":"),
                                sort_keys=True,
                            ),
                        }}
                    except Exception:
                        tc["function"]["arguments"] = _repair_tool_call_arguments(
                            tc["function"]["arguments"],
                            tc["function"].get("name", "?"),
                        )
                new_tcs.append(tc)
            am["tool_calls"] = new_tcs

        # 主动剥离代理字符（surrogate characters），防止 API 调用崩溃。
        # 通过 Ollama 提供的模型（Kimi K2.5, GLM-5, Qwen）可能返回
        # 孤立的代理字符（U+D800-U+DFFF），导致 OpenAI SDK 内部的
        # json.dumps() 崩溃。此处清理可防止 3 次重试循环。
        _sanitize_messages_surrogates(api_messages)

        # 计算请求的大致大小，用于日志记录
        total_chars = sum(len(str(msg)) for msg in api_messages)
        approx_tokens = estimate_messages_tokens_rough(api_messages)
        approx_request_tokens = estimate_request_tokens_rough(
            api_messages, tools=agent.tools or None
        )

        # 检查 Ollama 运行时上下文限制错误
        _runtime_context_error = _ollama_context_limit_error(
            agent, approx_request_tokens
        )
        if _runtime_context_error:
            final_response = _runtime_context_error
            failed = True
            _turn_exit_reason = "ollama_runtime_context_too_small"
            messages.append({"role": "assistant", "content": final_response})
            agent._emit_status("❌ Ollama runtime context is too small for Hermes tool use")
            api_call_count -= 1
            agent._api_call_count = api_call_count
            try:
                agent.iteration_budget.refund()
            except Exception:
                pass
            break

        # 静默模式下的思考动画器（API 调用期间动画展示）
        thinking_spinner = None

        if not agent.quiet_mode:
            agent._vprint(f"\n{agent.log_prefix}🔄 Making API call #{api_call_count}/{agent.max_iterations}...")
            agent._vprint(f"{agent.log_prefix}   📊 Request size: {len(api_messages)} messages, ~{approx_tokens:,} tokens (~{total_chars:,} chars)")
            agent._vprint(f"{agent.log_prefix}   🔧 Available tools: {len(agent.tools) if agent.tools else 0}")
        else:
            # 静默模式下的动画思考提示器
            face = random.choice(KawaiiSpinner.get_thinking_faces())
            verb = random.choice(KawaiiSpinner.get_thinking_verbs())
            if agent.thinking_callback:
                # CLI TUI 模式：使用 prompt_toolkit 控件而非原始 spinner
                # （流式和非流式模式均可工作）
                agent.thinking_callback(f"{face} {verb}...")
            elif not agent._has_stream_consumers() and agent._should_start_quiet_spinner():
                # 仅在没有流式消费者且 spinner 输出有安全接收器时使用原始 KawaiiSpinner
                spinner_type = random.choice(['brain', 'sparkle', 'pulse', 'moon', 'star'])
                thinking_spinner = KawaiiSpinner(f"{face} {verb}...", spinner_type=spinner_type, print_fn=agent._print_fn)
                thinking_spinner.start()

        # 如果启用了详细日志，记录请求详情
        if agent.verbose_logging:
            logging.debug(f"API Request - Model: {agent.model}, Messages: {len(messages)}, Tools: {len(agent.tools) if agent.tools else 0}")
            logging.debug(f"Last message role: {messages[-1]['role'] if messages else 'none'}")
            logging.debug(f"Total message size: ~{approx_tokens:,} tokens")
        
        api_start_time = time.time()
        retry_count = 0                       # 当前重试次数
        max_retries = agent._api_max_retries   # 最大重试次数
        primary_recovery_attempted = False     # 是否已尝试主通道恢复
        max_compression_attempts = 3           # 最大压缩尝试次数
        codex_auth_retry_attempted=False       # Codex 认证重试标志
        anthropic_auth_retry_attempted=False   # Anthropic 认证重试标志
        nous_auth_retry_attempted=False        # Nous 认证重试标志
        nous_paid_entitlement_refresh_attempted=False  # Nous 付费权益刷新标志
        copilot_auth_retry_attempted=False     # Copilot 认证重试标志
        thinking_sig_retry_attempted = False   # 思考块签名恢复重试标志
        invalid_encrypted_content_retry_attempted = False  # 无效加密内容重试标志
        image_shrink_retry_attempted = False   # 图片缩小重试标志
        multimodal_tool_content_retry_attempted = False  # 多模态工具内容重试标志
        oauth_1m_beta_retry_attempted = False  # OAuth 1M 上下文 beta 重试标志
        llama_cpp_grammar_retry_attempted = False  # llama.cpp 语法恢复重试标志
        has_retried_429 = False               # 是否已重试过 429 错误
        restart_with_compressed_messages = False  # 是否以压缩消息重新开始
        restart_with_length_continuation = False  # 是否以长度续传重新开始

        finish_reason = "stop"
        response = None   # 防止所有重试都失败时的 UnboundLocalError
        api_kwargs = None  # 防止异常处理器中的 UnboundLocalError

        # ── 重试循环 ──
        while retry_count < max_retries:
            # ── Nous Portal 速率限制守卫 ──────────────────────
            # 如果另一个会话已经记录了 Nous 的速率限制，
            # 完全跳过 API 调用。每次尝试（包括 SDK 级别的重试）
            # 都会消耗 RPH 配额并加深速率限制的困境。
            if agent.provider == "nous":
                try:
                    from agent.nous_rate_guard import (
                        nous_rate_limit_remaining,
                        format_remaining as _fmt_nous_remaining,
                    )
                    _nous_remaining = nous_rate_limit_remaining()
                    if _nous_remaining is not None and _nous_remaining > 0:
                        _nous_msg = (
                            f"Nous Portal rate limit active — "
                            f"resets in {_fmt_nous_remaining(_nous_remaining)}."
                        )
                        agent._buffer_vprint(
                            f"⏳ {_nous_msg} Trying fallback..."
                        )
                        agent._buffer_status(f"⏳ {_nous_msg}")
                        if agent._try_activate_fallback():
                            retry_count = 0
                            compression_attempts = 0
                            primary_recovery_attempted = False
                            continue
                        # 没有回退可用 —— 显示缓冲的上下文
                        # 使用户看到导致此处的速率限制消息。
                        agent._flush_status_buffer()
                        agent._persist_session(messages, conversation_history)
                        return {
                            "final_response": (
                                f"⏳ {_nous_msg}\n\n"
                                "No fallback provider available. "
                                "Try again after the reset, or add a "
                                "fallback provider in config.yaml."
                            ),
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "failed": True,
                            "error": _nous_msg,
                        }
                except ImportError:
                    pass
                except Exception:
                    pass  # Never let rate guard break the agent loop

            try:
                agent._reset_stream_delivery_tracking()
                # api_messages 在此重试循环之前构建一次，此时主提供商处于活跃状态。
                # 会话中途的回退可能切换到需要推理端的提供商（DeepSeek / Kimi / MiMo），
                # 它们会拒绝缺少 reasoning_content 的助手轮次。
                # 在此处为*当前*提供商重新应用回显填充
                # （幂等操作，除非当前提供商确实需要，否则为空操作），
                # 使回退请求不会带着陈旧的、主提供商形状的推理字段发出。
                agent._reapply_reasoning_echo_for_provider(api_messages)
                api_kwargs = agent._build_api_kwargs(api_messages)
                # 如果启用了强制 ASCII 负载，清理所有非 ASCII 字符
                if agent._force_ascii_payload:
                    _sanitize_structure_non_ascii(api_kwargs)
                # Codex Responses API 需要特殊的预飞行参数处理
                if agent.api_mode == "codex_responses":
                    api_kwargs = agent._get_transport().preflight_kwargs(api_kwargs, allow_stream=False)

                try:
                    # 插件钩子：pre_api_request —— 在每次 API 请求前触发
                    from hermes_cli.plugins import invoke_hook as _invoke_hook
                    # 获取请求中的消息列表（兼容不同的 API 模式）
                    request_messages = api_kwargs.get("messages")
                    if not isinstance(request_messages, list):
                        request_messages = api_kwargs.get("input")
                    if not isinstance(request_messages, list):
                        request_messages = api_messages
                    # 浅拷贝外层列表，防止保留引用的插件在异步快照时
                    # 观察到 api_messages 的后续变更。
                    # 内部 dict 不会被智能体循环修改，浅拷贝足够；
                    # 深拷贝会遍历每个工具结果和 base64 图像，代价太高。
                    _invoke_hook(
                        "pre_api_request",
                        task_id=effective_task_id,
                        session_id=agent.session_id or "",
                        user_message=original_user_message,
                        conversation_history=list(messages),
                        platform=agent.platform or "",
                        model=agent.model,
                        provider=agent.provider,
                        base_url=agent.base_url,
                        api_mode=agent.api_mode,
                        api_call_count=api_call_count,
                        request_messages=list(request_messages) if isinstance(request_messages, list) else [],
                        message_count=len(api_messages),
                        tool_count=len(agent.tools or []),
                        approx_input_tokens=approx_tokens,
                        request_char_count=total_chars,
                        max_tokens=agent.max_tokens,
                    )
                except Exception:
                    pass

                # 如果设置了调试环境变量，导出 API 请求内容
                if env_var_enabled("HERMES_DUMP_REQUESTS"):
                    agent._dump_api_request_debug(api_kwargs, reason="preflight")

                # 始终优先使用流式路径 —— 即使没有流式消费者。
                # 流式路径提供细粒度的健康检查（90 秒过期流检测、
                # 60 秒读取超时），而非流式路径缺乏这些能力。
                # 没有此功能时，子智能体和其他静默模式调用方
                # 可能在提供商通过 SSE ping 保持连接存活但从不
                # 交付响应时无限期挂起。
                # 没有消费者时，流式路径对回调是空操作，
                # 如果提供商不支持流式，会自动回退到非流式。
                def _stop_spinner():
                    """停止思考动画器的辅助函数"""
                    nonlocal thinking_spinner
                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    if agent.thinking_callback:
                        agent.thinking_callback("")

                _use_streaming = True
                # 提供商在之前的尝试中发出了 "不支持流式" 的信号 ——
                # 在会话剩余时间内切换到非流式，避免每次重试都失败。
                if getattr(agent, "_disable_streaming", False):
                    _use_streaming = False
                # CopilotACPClient 通过子进程 stdio 通信，
                # 返回纯 SimpleNamespace —— 不是可迭代的流。
                # 镜像 Responses API 升级中使用的 ACP 排除逻辑。
                elif (
                    agent.provider == "copilot-acp"
                    or str(agent.base_url or "").lower().startswith("acp://copilot")
                    or str(agent.base_url or "").lower().startswith("acp+tcp://")
                ):
                    _use_streaming = False
                elif not agent._has_stream_consumers():
                    # 没有显示/TTS 消费者。仍优先使用流式以进行健康检查，
                    # 但对测试中的 Mock 客户端跳过
                    # （mock 返回 SimpleNamespace，不是流迭代器）。
                    from unittest.mock import Mock
                    if isinstance(getattr(agent, "client", None), Mock):
                        _use_streaming = False

                # 根据判断结果选择流式或非流式 API 调用
                if _use_streaming:
                    response = agent._interruptible_streaming_api_call(
                        api_kwargs, on_first_delta=_stop_spinner
                    )
                else:
                    response = agent._interruptible_api_call(api_kwargs)
                
                api_duration = time.time() - api_start_time

                # 静默停止思考动画器 —— 后续的响应框或工具执行消息更具信息量
                if thinking_spinner:
                    thinking_spinner.stop("")
                    thinking_spinner = None
                if agent.thinking_callback:
                    agent.thinking_callback("")

                if not agent.quiet_mode:
                    agent._vprint(f"{agent.log_prefix}⏱️  API call completed in {api_duration:.2f}s")

                if agent.verbose_logging:
                    # 记录响应模型和使用情况
                    resp_model = getattr(response, 'model', 'N/A') if response else 'N/A'
                    logging.debug(f"API Response received - Model: {resp_model}, Usage: {response.usage if hasattr(response, 'usage') else 'N/A'}")

                # ── 验证响应结构 ──
                # 根据不同的 API 模式验证响应是否有效
                response_invalid = False
                error_details = []
                if agent.api_mode == "codex_responses":
                    # Codex Responses API 模式的响应验证
                    _ct_v = agent._get_transport()
                    if not _ct_v.validate_response(response):
                        if response is None:
                            response_invalid = True
                            error_details.append("response is None")
                        else:
                            # 提供商返回了终端失败（如配额耗尽）。
                            # 视为无效，触发回退链而非让错误冒泡到重试/回退循环之外。
                            _codex_resp_status = str(getattr(response, "status", "") or "").strip().lower()
                            if _codex_resp_status in {"failed", "cancelled"}:
                                _codex_error_obj = getattr(response, "error", None)
                                _codex_error_msg = (
                                    _codex_error_obj.get("message") if isinstance(_codex_error_obj, dict)
                                    else str(_codex_error_obj) if _codex_error_obj
                                    else f"Responses API returned status '{_codex_resp_status}'"
                                )
                                logger.warning(
                                    "Codex response status='%s' (error=%s). Routing to fallback. %s",
                                    _codex_resp_status, _codex_error_msg,
                                    agent._client_log_context(),
                                )
                                response_invalid = True
                                error_details.append(f"response.status={_codex_resp_status}: {_codex_error_msg}")
                            else:
                                # output_text 回退：流式回填可能失败，
                                # 但 normalize 仍可从 output_text 恢复
                                _out_text = getattr(response, "output_text", None)
                                _out_text_stripped = _out_text.strip() if isinstance(_out_text, str) else ""
                                if _out_text_stripped:
                                    logger.debug(
                                        "Codex response.output is empty but output_text is present "
                                        "(%d chars); deferring to normalization.",
                                        len(_out_text_stripped),
                                    )
                                else:
                                    # output 和 output_text 都为空 —— 视为无效
                                    _resp_status = getattr(response, "status", None)
                                    _resp_incomplete = getattr(response, "incomplete_details", None)
                                    logger.warning(
                                        "Codex response.output is empty after stream backfill "
                                        "(status=%s, incomplete_details=%s, model=%s). %s",
                                        _resp_status, _resp_incomplete,
                                        getattr(response, "model", None),
                                        f"api_mode={agent.api_mode} provider={agent.provider}",
                                    )
                                    response_invalid = True
                                    error_details.append("response.output is empty")
                elif agent.api_mode == "anthropic_messages":
                    # Anthropic Messages API 模式的响应验证
                    _tv = agent._get_transport()
                    if not _tv.validate_response(response):
                        response_invalid = True
                        if response is None:
                            error_details.append("response is None")
                        else:
                            error_details.append("response.content invalid (not a non-empty list)")
                elif agent.api_mode == "bedrock_converse":
                    # Bedrock Converse API 模式的响应验证
                    _btv = agent._get_transport()
                    if not _btv.validate_response(response):
                        response_invalid = True
                        if response is None:
                            error_details.append("response is None")
                        else:
                            error_details.append("Bedrock response invalid (no output or choices)")
                else:
                    # 标准 OpenAI Chat Completions 兼容模式的响应验证
                    _ctv = agent._get_transport()
                    if not _ctv.validate_response(response):
                        response_invalid = True
                        if response is None:
                            error_details.append("response is None")
                        elif not hasattr(response, 'choices'):
                            error_details.append("response has no 'choices' attribute")
                        elif response.choices is None:
                            error_details.append("response.choices is None")
                        else:
                            error_details.append("response.choices is empty")

                if response_invalid:
                    # 停止动画器 —— 重试状态现在缓冲，仅在每次重试+回退都耗尽后才显示
                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    if agent.thinking_callback:
                        agent.thinking_callback("")

                    # 无效响应 —— 可能是速率限制、提供商超时、
                    # 上游服务器错误或格式错误的响应。
                    retry_count += 1

                    # 快速回退：空/格式错误的响应是常见的速率限制症状。
                    # 立即切换到回退，而非用指数退避重试。
                    if agent._fallback_index < len(agent._fallback_chain):
                        agent._buffer_status("⚠️ Empty/malformed response — switching to fallback...")
                    if agent._try_activate_fallback():
                        retry_count = 0
                        compression_attempts = 0
                        primary_recovery_attempted = False
                        continue

                    # 检查响应中的错误字段（某些提供商会包含此字段）
                    error_msg = "Unknown"
                    provider_name = "Unknown"
                    if response and hasattr(response, 'error') and response.error:
                        error_msg = str(response.error)
                        # 尝试从错误元数据中提取提供商信息
                        if hasattr(response.error, 'metadata') and response.error.metadata:
                            provider_name = response.error.metadata.get('provider_name', 'Unknown')
                    elif response and hasattr(response, 'message') and response.message:
                        error_msg = str(response.message)
                    
                    # 尝试从 model 字段获取提供商（OpenRouter 通常返回实际使用的模型）
                    if provider_name == "Unknown" and response and hasattr(response, 'model') and response.model:
                        provider_name = f"model={response.model}"
                    
                    # 检查 x-openrouter-provider 或类似元数据
                    if provider_name == "Unknown" and response:
                        # 记录所有响应属性用于调试
                        resp_attrs = {k: str(v)[:100] for k, v in vars(response).items() if not k.startswith('_')}
                        if agent.verbose_logging:
                            logging.debug(f"Response attributes for invalid response: {resp_attrs}")
                    
                    # 从响应中提取错误代码用于上下文诊断
                    _resp_error_code = None
                    if response and hasattr(response, 'error') and response.error:
                        _code_raw = getattr(response.error, 'code', None)
                        if _code_raw is None and isinstance(response.error, dict):
                            _code_raw = response.error.get('code')
                        if _code_raw is not None:
                            try:
                                _resp_error_code = int(_code_raw)
                            except (TypeError, ValueError):
                                pass

                    # 根据错误代码和响应时间构建人类可读的失败提示，
                    # 而非总是假设速率限制。
                    if _resp_error_code == 524:
                        _failure_hint = f"upstream provider timed out (Cloudflare 524, {api_duration:.0f}s)"
                    elif _resp_error_code == 504:
                        _failure_hint = f"upstream gateway timeout (504, {api_duration:.0f}s)"
                    elif _resp_error_code == 429:
                        _failure_hint = f"rate limited by upstream provider (429)"
                    elif _resp_error_code in {500, 502}:
                        _failure_hint = f"upstream server error ({_resp_error_code}, {api_duration:.0f}s)"
                    elif _resp_error_code in {503, 529}:
                        _failure_hint = f"upstream provider overloaded ({_resp_error_code})"
                    elif _resp_error_code is not None:
                        _failure_hint = f"upstream error (code {_resp_error_code}, {api_duration:.0f}s)"
                    elif api_duration < 10:
                        _failure_hint = f"fast response ({api_duration:.1f}s) — likely rate limited"
                    elif api_duration > 60:
                        _failure_hint = f"slow response ({api_duration:.0f}s) — likely upstream timeout"
                    else:
                        _failure_hint = f"response time {api_duration:.1f}s"

                    agent._buffer_vprint(f"⚠️  Invalid API response (attempt {retry_count}/{max_retries}): {', '.join(error_details)}")
                    agent._buffer_vprint(f"   🏢 Provider: {provider_name}")
                    cleaned_provider_error = agent._clean_error_message(error_msg)
                    agent._buffer_vprint(f"   📝 Provider message: {cleaned_provider_error}")
                    agent._buffer_vprint(f"   ⏱️  {_failure_hint}")
                    
                    if retry_count >= max_retries:
                        # 在放弃之前尝试回退
                        agent._buffer_status(f"⚠️ Max retries ({max_retries}) for invalid responses — trying fallback...")
                        if agent._try_activate_fallback():
                            retry_count = 0
                            compression_attempts = 0
                            primary_recovery_attempted = False
                            continue
                        # 终端 —— 刷新缓冲的重试跟踪，使用户看到发生了什么。
                        agent._flush_status_buffer()
                        agent._emit_status(f"❌ Max retries ({max_retries}) exceeded for invalid responses. Giving up.")
                        logger.error(f"{agent.log_prefix}Invalid API response after {max_retries} retries.")
                        agent._persist_session(messages, conversation_history)
                        return {
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": f"Invalid API response after {max_retries} retries: {_failure_hint}",
                            "failed": True  # Mark as failure for filtering
                        }
                    
                    # 重试前退避 —— 抖动指数退避：5 秒基数，120 秒上限
                    wait_time = jittered_backoff(retry_count, base_delay=5.0, max_delay=120.0)
                    agent._buffer_vprint(f"⏳ Retrying in {wait_time:.1f}s ({_failure_hint})...")
                    logger.warning(f"Invalid API response (retry {retry_count}/{max_retries}): {', '.join(error_details)} | Provider: {provider_name}")
                    
                    # 以小增量睡眠以保持对中断的响应能力
                    sleep_end = time.time() + wait_time
                    _backoff_touch_counter = 0
                    while time.time() < sleep_end:
                        if agent._interrupt_requested:
                            agent._vprint(f"{agent.log_prefix}⚡ Interrupt detected during retry wait, aborting.", force=True)
                            agent._persist_session(messages, conversation_history)
                            agent.clear_interrupt()
                            return {
                                "final_response": f"Operation interrupted during retry ({_failure_hint}, attempt {retry_count}/{max_retries}).",
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "interrupted": True,
                            }
                        time.sleep(0.2)
                        # 每约 30 秒触碰活动计数器，使网关的不活动监控
                        # 知道我们在退避等待期间仍然存活。
                        _backoff_touch_counter += 1
                        if _backoff_touch_counter % 150 == 0:  # 150 × 0.2s = 30s
                            agent._touch_activity(
                                f"retry backoff ({retry_count}/{max_retries}), "
                                f"{int(sleep_end - time.time())}s remaining"
                            )
                    continue  # Retry the API call

                # ── 检查响应的 finish_reason ──
                if agent.api_mode == "codex_responses":
                    # Codex Responses API 的完成原因通过 status 和 incomplete_details 判断
                    status = getattr(response, "status", None)
                    incomplete_details = getattr(response, "incomplete_details", None)
                    incomplete_reason = None
                    if isinstance(incomplete_details, dict):
                        incomplete_reason = incomplete_details.get("reason")
                    else:
                        incomplete_reason = getattr(incomplete_details, "reason", None)
                    if status == "incomplete" and incomplete_reason in {"max_output_tokens", "length"}:
                        finish_reason = "length"
                    else:
                        finish_reason = "stop"
                elif agent.api_mode == "anthropic_messages":
                    # Anthropic API 的完成原因通过 stop_reason 映射
                    _tfr = agent._get_transport()
                    finish_reason = _tfr.map_finish_reason(response.stop_reason)
                elif agent.api_mode == "bedrock_converse":
                    # Bedrock 响应已在分发时规范化 —— 使用 transport 获取
                    _bt_fr = agent._get_transport()
                    _bedrock_result = _bt_fr.normalize_response(response)
                    finish_reason = _bedrock_result.finish_reason
                else:
                    # 标准 Chat Completions 兼容模式
                    _cc_fr = agent._get_transport()
                    _finish_result = _cc_fr.normalize_response(response)
                    finish_reason = _finish_result.finish_reason
                    assistant_message = _finish_result
                    # 检查 Ollama/GLM 的可疑 stop 响应 —— 可能是截断伪装成正常停止
                    if agent._should_treat_stop_as_truncated(
                        finish_reason,
                        assistant_message,
                        messages,
                    ):
                        agent._vprint(
                            f"{agent.log_prefix}⚠️  Treating suspicious Ollama/GLM stop response as truncated",
                            force=True,
                        )
                        finish_reason = "length"

                if finish_reason == "length":
                    # 响应被截断（finish_reason='length'）—— 模型达到最大输出 token 数
                    if getattr(response, "id", "") == PARTIAL_STREAM_STUB_ID:
                        agent._vprint(
                            f"{agent.log_prefix}⚠️  Stream interrupted by network error "
                            f"(finish_reason='length' on partial-stream-stub)",
                            force=True,
                        )
                    else:
                        agent._vprint(
                            f"{agent.log_prefix}⚠️  Response truncated "
                            f"(finish_reason='length') - model hit max output tokens",
                            force=True,
                        )

                    # 将截断的响应规范化为统一的 OpenAI 风格消息格式，
                    # 使文本续传和工具调用重试在不同 API 模式
                    # （chat_completions, bedrock_converse, anthropic_messages）下
                    # 能统一工作。对于 Anthropic，使用智能体循环已依赖的
                    # 同一适配器，使重建的临时助手消息与
                    # 非截断路径中的字节完全一致。
                    _trunc_msg = None
                    _trunc_transport = agent._get_transport()
                    if agent.api_mode == "anthropic_messages":
                        _trunc_result = _trunc_transport.normalize_response(
                            response, strip_tool_prefix=agent._is_anthropic_oauth
                        )
                    else:
                        _trunc_result = _trunc_transport.normalize_response(response)
                    _trunc_msg = _trunc_result

                    _trunc_content = getattr(_trunc_msg, "content", None) if _trunc_msg else None
                    _trunc_has_tool_calls = bool(getattr(_trunc_msg, "tool_calls", None)) if _trunc_msg else False

                    # ── 检测思考预算耗尽 ──────────────
                    # 当模型将所有输出 token 用于推理而没有剩余给响应时，
                    # 续传重试毫无意义。尽早检测此情况并给出针对性错误，
                    # 而非浪费 3 次 API 调用。
                    # 仅当模型实际产生了推理块但之后没有可见文本时，
                    # 才算"思考耗尽"。不使用 <think> 标签的模型
                    # （如 NVIDIA Build 上的 GLM-4.7, minimax）可能因
                    # 其他原因返回 content=None 或空字符串 ——
                    # 将这些视为正常截断进行续传重试，而非思考预算耗尽。
                    _has_think_tags = bool(
                        _trunc_content and re.search(
                            r'<(?:think|thinking|reasoning|REASONING_SCRATCHPAD)[^>]*>',
                            _trunc_content,
                            re.IGNORECASE,
                        )
                    )
                    _thinking_exhausted = (
                        not _trunc_has_tool_calls
                        and _has_think_tags
                        and (
                            (_trunc_content is not None and not agent._has_content_after_think_block(_trunc_content))
                            or _trunc_content is None
                        )
                    )

                    if _thinking_exhausted:
                        _exhaust_error = (
                            "Model used all output tokens on reasoning with none left "
                            "for the response. Try lowering reasoning effort or "
                            "increasing max_tokens."
                        )
                        agent._vprint(
                            f"{agent.log_prefix}💭 Reasoning exhausted the output token budget — "
                            f"no visible response was produced.",
                            force=True,
                        )
                        # 返回用户友好的消息作为响应，使
                        # CLI（响应框）和网关（聊天消息）都能
                        # 自然地显示它，而非被抑制的错误。
                        _exhaust_response = (
                            "⚠️ **Thinking Budget Exhausted**\n\n"
                            "The model used all its output tokens on reasoning "
                            "and had none left for the actual response.\n\n"
                            "To fix this:\n"
                            "→ Lower reasoning effort: `/thinkon low` or `/thinkon minimal`\n"
                            "→ Or switch to a larger/non-reasoning model with `/model`"
                        )
                        agent._cleanup_task_resources(effective_task_id)
                        agent._persist_session(messages, conversation_history)
                        return {
                            "final_response": _exhaust_response,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "partial": True,
                            "error": _exhaust_error,
                        }

                    if agent.api_mode in {"chat_completions", "bedrock_converse", "anthropic_messages"}:
                        assistant_message = _trunc_msg
                        if assistant_message is not None and not _trunc_has_tool_calls:
                            length_continue_retries += 1
                            interim_msg = agent._build_assistant_message(assistant_message, finish_reason)
                            messages.append(interim_msg)
                            if assistant_message.content:
                                truncated_response_parts.append(assistant_message.content)

                            if length_continue_retries < 3:
                                _is_partial_stream_stub = (
                                    getattr(response, "id", "") == PARTIAL_STREAM_STUB_ID
                                )
                                _dropped_tools = getattr(
                                    response, "_dropped_tool_names", None
                                )

                                if _is_partial_stream_stub and _dropped_tools:
                                    _tool_list = ", ".join(_dropped_tools[:3])
                                    agent._vprint(
                                        f"{agent.log_prefix}↻ Stream interrupted mid "
                                        f"tool-call ({_tool_list}) — requesting "
                                        f"chunked retry "
                                        f"({length_continue_retries}/3)..."
                                    )
                                elif _is_partial_stream_stub:
                                    agent._vprint(
                                        f"{agent.log_prefix}↻ Stream interrupted — "
                                        f"requesting continuation "
                                        f"({length_continue_retries}/3)..."
                                    )
                                else:
                                    agent._vprint(
                                        f"{agent.log_prefix}↻ Requesting continuation "
                                        f"({length_continue_retries}/3)..."
                                    )

                                _continue_content = _get_continuation_prompt(
                                    _is_partial_stream_stub, _dropped_tools
                                )
                                continue_msg = {
                                    "role": "user",
                                    "content": _continue_content,
                                }
                                messages.append(continue_msg)
                                agent._session_messages = messages
                                restart_with_length_continuation = True
                                break

                            partial_response = agent._strip_think_blocks("".join(truncated_response_parts)).strip()
                            agent._cleanup_task_resources(effective_task_id)
                            agent._persist_session(messages, conversation_history)
                            return {
                                "final_response": partial_response or None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": "Response remained truncated after 3 continuation attempts",
                            }

                    if agent.api_mode in {"chat_completions", "bedrock_converse", "anthropic_messages"}:
                        assistant_message = _trunc_msg
                        if assistant_message is not None and _trunc_has_tool_calls:
                            if truncated_tool_call_retries < 1:
                                truncated_tool_call_retries += 1
                                agent._buffer_vprint(
                                    f"⚠️  Truncated tool call detected — retrying API call..."
                                )
                                # 不要将损坏的响应追加到消息中；
                                # 仅重新运行相同的 API 调用，给模型另一次机会。
                                continue
                            agent._flush_status_buffer()
                            agent._vprint(
                                f"{agent.log_prefix}⚠️  Truncated tool call response detected again — refusing to execute incomplete tool arguments.",
                                force=True,
                            )
                            agent._cleanup_task_resources(effective_task_id)
                            agent._persist_session(messages, conversation_history)
                            return {
                                "final_response": None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": "Response truncated due to output length limit",
                            }

                    # 如果有先前的消息，回退到上一个完整状态
                    if len(messages) > 1:
                        agent._vprint(f"{agent.log_prefix}   ⏪ Rolling back to last complete assistant turn")
                        rolled_back_messages = agent._get_messages_up_to_last_assistant(messages)

                        agent._cleanup_task_resources(effective_task_id)
                        agent._persist_session(messages, conversation_history)

                        return {
                            "final_response": None,
                            "messages": rolled_back_messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "partial": True,
                            "error": "Response truncated due to output length limit"
                        }
                    else:
                        # 第一条消息被截断 - 标记为失败
                        agent._flush_status_buffer()
                        agent._vprint(f"{agent.log_prefix}❌ First response truncated - cannot recover", force=True)
                        agent._persist_session(messages, conversation_history)
                        return {
                            "final_response": None,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "failed": True,
                            "error": "First response truncated due to output length limit"
                        }
                
                # ── 跟踪响应中的实际 token 使用量，用于上下文管理 ──
                if hasattr(response, 'usage') and response.usage:
                    # 规范化 token 使用数据（适配不同提供商的格式差异）
                    canonical_usage = normalize_usage(
                        response.usage,
                        provider=agent.provider,
                        api_mode=agent.api_mode,
                    )
                    prompt_tokens = canonical_usage.prompt_tokens
                    completion_tokens = canonical_usage.output_tokens
                    total_tokens = canonical_usage.total_tokens
                    # 转发规范化的 token + 缓存桶数据，使上下文引擎
                    # 能基于缓存命中率/推理成本做决策，而非仅依赖
                    # 传统的汇总 token 数。传统键保留用于兼容
                    # 只读取 prompt/completion/total 的引擎。
                    usage_dict = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "input_tokens": canonical_usage.input_tokens,
                        "output_tokens": canonical_usage.output_tokens,
                        "cache_read_tokens": canonical_usage.cache_read_tokens,
                        "cache_write_tokens": canonical_usage.cache_write_tokens,
                        "reasoning_tokens": canonical_usage.reasoning_tokens,
                    }
                    # 更新上下文压缩器的使用统计
                    agent.context_compressor.update_from_response(usage_dict)

                    # 在成功调用后缓存已发现的上下文长度。
                    # 仅持久化提供商确认的限制（从错误消息中解析），
                    # 而非猜测的探测层级。
                    if getattr(agent.context_compressor, "_context_probed", False):
                        ctx = agent.context_compressor.context_length
                        if getattr(agent.context_compressor, "_context_probe_persistable", False):
                            save_context_length(agent.model, agent.base_url, ctx)
                            agent._safe_print(f"{agent.log_prefix}💾 Cached context length: {ctx:,} tokens for {agent.model}")
                        agent.context_compressor._context_probed = False
                        agent.context_compressor._context_probe_persistable = False

                    # 累加会话级别的 token 统计
                    agent.session_prompt_tokens += prompt_tokens
                    agent.session_completion_tokens += completion_tokens
                    agent.session_total_tokens += total_tokens
                    agent.session_api_calls += 1
                    agent.session_input_tokens += canonical_usage.input_tokens
                    agent.session_output_tokens += canonical_usage.output_tokens
                    agent.session_cache_read_tokens += canonical_usage.cache_read_tokens
                    agent.session_cache_write_tokens += canonical_usage.cache_write_tokens
                    agent.session_reasoning_tokens += canonical_usage.reasoning_tokens

                    # 记录 API 调用详情用于调试/可观测性
                    _cache_pct = ""
                    if canonical_usage.cache_read_tokens and prompt_tokens:
                        _cache_pct = f" cache={canonical_usage.cache_read_tokens}/{prompt_tokens} ({100*canonical_usage.cache_read_tokens/prompt_tokens:.0f}%)"
                    logger.info(
                        "API call #%d: model=%s provider=%s in=%d out=%d total=%d latency=%.1fs%s",
                        agent.session_api_calls, agent.model, agent.provider or "unknown",
                        prompt_tokens, completion_tokens, total_tokens,
                        api_duration, _cache_pct,
                    )

                    # 估算本次调用的费用
                    cost_result = estimate_usage_cost(
                        agent.model,
                        canonical_usage,
                        provider=agent.provider,
                        base_url=agent.base_url,
                        api_key=getattr(agent, "api_key", ""),
                    )
                    if cost_result.amount_usd is not None:
                        agent.session_estimated_cost_usd += float(cost_result.amount_usd)
                    agent.session_cost_status = cost_result.status
                    agent.session_cost_source = cost_result.source

                    # 将 token 计数持久化到会话数据库，供 /insights 使用。
                    # 对每个有 session_id 的平台都执行此操作，使非 CLI
                    # 会话（网关、定时任务、委托运行）不会因上层持久化路径
                    # 被跳过或失败而丢失 token/记账数据。
                    # 网关/会话存储写入使用绝对总数，因此安全地覆盖
                    # 这些每调用增量而非重复计数。
                    if agent._session_db and agent.session_id:
                        try:
                            # 确保在尝试 UPDATE 之前会话行存在。
                            # 在并发负载下（定时任务/看板），初始的
                            # _ensure_db_session() 可能因 SQLite 锁定而失败。
                            # 此处重试，使每调用的 token 增量不会被静默丢失
                            # （对不存在行的 UPDATE 影响 0 行但不报错）。
                            if not agent._session_db_created:
                                agent._ensure_db_session()
                            agent._session_db.update_token_counts(
                                agent.session_id,
                                input_tokens=canonical_usage.input_tokens,
                                output_tokens=canonical_usage.output_tokens,
                                cache_read_tokens=canonical_usage.cache_read_tokens,
                                cache_write_tokens=canonical_usage.cache_write_tokens,
                                reasoning_tokens=canonical_usage.reasoning_tokens,
                                estimated_cost_usd=float(cost_result.amount_usd)
                                if cost_result.amount_usd is not None else None,
                                cost_status=cost_result.status,
                                cost_source=cost_result.source,
                                billing_provider=agent.provider,
                                billing_base_url=agent.base_url,
                                billing_mode="subscription_included"
                                if cost_result.status == "included" else None,
                                model=agent.model,
                                api_call_count=1,
                            )
                        except Exception as e:
                            # 记录 token 持久化失败日志，使其在 agent.log 中可见 ——
                            # 此处的静默丢失是分析数据不足的根因。
                            logger.debug(
                                "Token persistence failed (session=%s, tokens=%d): %s",
                                agent.session_id, total_tokens, e,
                            )

                    if agent.verbose_logging:
                        logging.debug(f"Token usage: prompt={usage_dict['prompt_tokens']:,}, completion={usage_dict['completion_tokens']:,}, total={usage_dict['total_tokens']:,}")

                    # 显示缓存命中统计 —— 适用于任何报告此数据的提供商，
                    # 不仅限于我们注入 cache_control 标记的提供商。
                    # OpenAI/Kimi/DeepSeek/Qwen 都执行自动的
                    # 服务端前缀缓存，并返回
                    # ``prompt_tokens_details.cached_tokens``；
                    # 用户之前无法看到缓存百分比，因为此行受
                    # ``_use_prompt_caching`` 门控，后者仅在
                    # Anthropic 风格标记注入时为 True。
                    # ``canonical_usage`` 已从三种 API 格式
                    # （Anthropic / Codex / OpenAI-chat）规范化，
                    # 因此可以直接依赖其值。
                    cached = canonical_usage.cache_read_tokens
                    written = canonical_usage.cache_write_tokens
                    prompt = usage_dict["prompt_tokens"]
                    if (cached or written) and not agent.quiet_mode:
                        hit_pct = (cached / prompt * 100) if prompt > 0 else 0
                        agent._vprint(
                            f"{agent.log_prefix}   💾 Cache: "
                            f"{cached:,}/{prompt:,} tokens "
                            f"({hit_pct:.0f}% hit, {written:,} written)"
                        )
                
                has_retried_429 = False  # 成功时重置 429 重试标志
                # 注意：不在此处清除重试缓冲区 —— "API 调用成功"
                # 仅表示收到了字节，不代表获得了可用内容。
                # 空响应仍然会循环通过下方的空重试路径；
                # 缓冲区在后续检测到真正成功的内容时清除（约 L4127）。
                # 在成功请求时清除 Nous 速率限制状态 ——
                # 证明限制已重置，其他会话可以恢复对 Nous 的调用。
                if agent.provider == "nous":
                    try:
                        from agent.nous_rate_guard import clear_nous_rate_limit
                        clear_nous_rate_limit()
                    except Exception:
                        pass
                agent._touch_activity(f"API call #{api_call_count} completed")
                break  # 成功，退出重试循环

            except InterruptedError:
                # API 调用被中断 —— 清理并返回
                if thinking_spinner:
                    thinking_spinner.stop("")
                    thinking_spinner = None
                if agent.thinking_callback:
                    agent.thinking_callback("")
                api_elapsed = time.time() - api_start_time
                agent._vprint(f"{agent.log_prefix}⚡ Interrupted during API call.", force=True)
                agent._persist_session(messages, conversation_history)
                interrupted = True
                final_response = f"Operation interrupted: waiting for model response ({api_elapsed:.1f}s elapsed)."
                break

            except Exception as api_error:
                # 停止动画器 —— 重试状态缓冲，仅在每次重试+回退都耗尽后显示
                if thinking_spinner:
                    thinking_spinner.stop("")
                    thinking_spinner = None
                if agent.thinking_callback:
                    agent.thinking_callback("")

                # ── UnicodeEncodeError 恢复 ───────────────────────
                # 两种常见原因：
                #   1. 来自剪贴板粘贴的孤立代理字符（U+D800..U+DFFF）
                #      （Google Docs、富文本编辑器）—— 清理后重试。
                #   2. 使用 LANG=C 或非 UTF-8 locale 的系统上的 ASCII 编解码器
                #      （例如 Chromebook）—— 任何非 ASCII 字符都会失败。
                #      通过错误消息中提到 'ascii' 编解码器来检测。
                # 我们就地清理消息，可能需要重试两次：
                # 第一次清理代理字符，如果需要则第二次执行纯 ASCII-only
                # locale 清理。
                if isinstance(api_error, UnicodeEncodeError) and getattr(agent, '_unicode_sanitization_passes', 0) < 2:
                    _err_str = str(api_error).lower()
                    _is_ascii_codec = "'ascii'" in _err_str or "ascii" in _err_str
                    # 检测代理字符错误 —— utf-8 编解码器拒绝
                    # 编码 U+D800..U+DFFF。错误文本为：
                    #   "'utf-8' codec can't encode characters in position
                    #    N-M: surrogates not allowed"（U+D800..U+DFFF 不允许）
                    _is_surrogate_error = (
                        "surrogate" in _err_str
                        or ("'utf-8'" in _err_str and not _is_ascii_codec)
                    )
                    # 同时从规范的 `messages` 列表和 `api_messages` 中清理代理字符
                    # （API 副本，可能携带从 `reasoning` 转换而来的
                    # `reasoning_content`/`reasoning_details` —— 规范列表
                    # 不直接拥有的字段）。同时清理已构建的 `api_kwargs` 和
                    # `prefill_messages`（如果存在）。镜像下方的 ASCII
                    # 编解码器恢复逻辑。
                    _surrogates_found = _sanitize_messages_surrogates(messages)
                    if isinstance(api_messages, list):
                        if _sanitize_messages_surrogates(api_messages):
                            _surrogates_found = True
                    if isinstance(api_kwargs, dict):
                        if _sanitize_structure_surrogates(api_kwargs):
                            _surrogates_found = True
                    if isinstance(getattr(agent, "prefill_messages", None), list):
                        if _sanitize_messages_surrogates(agent.prefill_messages):
                            _surrogates_found = True
                    # 根据错误类型而非是否找到代理字符来门控重试 ——
                    # _force_ascii_payload / 上方的扩展代理字符遍历器
                    # 覆盖了所有已知路径，但新的转换字段仍可能溜过。
                    # 如果错误是代理字符编码失败，始终让重试运行；
                    # 约 L8781 处的主动清理器会在下一次迭代中再次运行。
                    # 受 _unicode_sanitization_passes < 2（外层守卫）限制。
                    if _surrogates_found or _is_surrogate_error:
                        agent._unicode_sanitization_passes += 1
                        if _surrogates_found:
                            agent._buffer_vprint(
                                f"⚠️  Stripped invalid surrogate characters from messages. Retrying..."
                            )
                        else:
                            agent._buffer_vprint(
                                f"⚠️  Surrogate encoding error — retrying after full-payload sanitization..."
                            )
                        continue
                    if _is_ascii_codec:
                        agent._force_ascii_payload = True
                        # ASCII 编解码器：系统编码完全无法处理非 ASCII 字符。
                        # 从消息/工具 schema 中清理所有非 ASCII 内容后重试。
                        # 同时清理规范的 `messages` 列表和 `api_messages`
                        # （重试循环前构建的 API 副本，可能包含
                        # `messages` 中没有的额外字段，如 reasoning_content）。
                        _messages_sanitized = _sanitize_messages_non_ascii(messages)
                        if isinstance(api_messages, list):
                            _sanitize_messages_non_ascii(api_messages)
                        # 同时清理已构建的 api_kwargs，使转换字段中
                        # 残留的非 ASCII 值（如 extra_body, reasoning_content）
                        # 不会通过 _build_api_kwargs 缓存路径存活到下次尝试。
                        if isinstance(api_kwargs, dict):
                            _sanitize_structure_non_ascii(api_kwargs)
                        _prefill_sanitized = False
                        if isinstance(getattr(agent, "prefill_messages", None), list):
                            _prefill_sanitized = _sanitize_messages_non_ascii(agent.prefill_messages)

                        _tools_sanitized = False
                        if isinstance(getattr(agent, "tools", None), list):
                            _tools_sanitized = _sanitize_tools_non_ascii(agent.tools)

                        _system_sanitized = False
                        if isinstance(active_system_prompt, str):
                            _sanitized_system = _strip_non_ascii(active_system_prompt)
                            if _sanitized_system != active_system_prompt:
                                active_system_prompt = _sanitized_system
                                agent._cached_system_prompt = _sanitized_system
                                _system_sanitized = True
                        if isinstance(getattr(agent, "ephemeral_system_prompt", None), str):
                            _sanitized_ephemeral = _strip_non_ascii(agent.ephemeral_system_prompt)
                            if _sanitized_ephemeral != agent.ephemeral_system_prompt:
                                agent.ephemeral_system_prompt = _sanitized_ephemeral
                                _system_sanitized = True

                        _headers_sanitized = False
                        _default_headers = (
                            agent._client_kwargs.get("default_headers")
                            if isinstance(getattr(agent, "_client_kwargs", None), dict)
                            else None
                        )
                        if isinstance(_default_headers, dict):
                            _headers_sanitized = _sanitize_structure_non_ascii(_default_headers)

                        # 清理 API 密钥 —— 凭证中的非 ASCII 字符
                        # （例如因复制粘贴错误导致的 ʋ 代替 v）
                        # 会导致 httpx 在将 Authorization 头编码为 ASCII 时失败。
                        # 这是在消息/工具清理后仍然存活的持久性
                        # UnicodeEncodeError 最常见原因 (#6843)。
                        _credential_sanitized = False
                        _raw_key = getattr(agent, "api_key", None) or ""
                        # Entra ID bearer 提供者是 callable —— 它们
                        # 生成的 JWT 始终是 ASCII，无需清理
                        # （且 ``_strip_non_ascii`` 在 callable 输入上会崩溃）。
                        if _raw_key and isinstance(_raw_key, str):
                            _clean_key = _strip_non_ascii(_raw_key)
                            if _clean_key != _raw_key:
                                agent.api_key = _clean_key
                                if isinstance(getattr(agent, "_client_kwargs", None), dict):
                                    agent._client_kwargs["api_key"] = _clean_key
                                # 同时更新活跃的客户端 —— 它持有自己的
                                # api_key 副本，auth_headers 在每次请求时动态读取。
                                if getattr(agent, "client", None) is not None and hasattr(agent.client, "api_key"):
                                    agent.client.api_key = _clean_key
                                _credential_sanitized = True
                                agent._vprint(
                                    f"{agent.log_prefix}⚠️  API key contained non-ASCII characters "
                                    f"(bad copy-paste?) — stripped them. If auth fails, "
                                    f"re-copy the key from your provider's dashboard.",
                                    force=True,
                                )

                        # 始终在检测到 ASCII 编解码器时重试 ——
                        # _force_ascii_payload 保证在下一次迭代中
                        # 完整的 api_kwargs 负载会被清理。
                        # 即使上面的逐组件检查未发现任何问题
                        # （例如非 ASCII 仅在 api_messages 的
                        # reasoning_content 中），此标志也能捕获。
                        # 受 _unicode_sanitization_passes < 2 限制。
                        agent._unicode_sanitization_passes += 1
                        _any_sanitized = (
                            _messages_sanitized
                            or _prefill_sanitized
                            or _tools_sanitized
                            or _system_sanitized
                            or _headers_sanitized
                            or _credential_sanitized
                        )
                        if _any_sanitized:
                            agent._vprint(
                                f"{agent.log_prefix}⚠️  System encoding is ASCII — stripped non-ASCII characters from request payload. Retrying...",
                                force=True,
                            )
                        else:
                            agent._vprint(
                                f"{agent.log_prefix}⚠️  System encoding is ASCII — enabling full-payload sanitization for retry...",
                                force=True,
                            )
                        continue

                # ── 图像拒绝恢复 ──────────────────────────────
                # 某些提供商（mlx-lm、纯文本端点、多模态模型上的
                # 纯文本回退）拒绝任何包含 image_url 内容的消息，
                # 返回 4xx 错误如 "Only 'text' content type is supported."
                # 首次遇到时，从消息列表中剥离所有图像，
                # 标记会话为不支持视觉，然后以纯文本重试。
                #
                # 检测采用尽力而为的英文短语匹配 ——
                # 经过本地化翻译或大幅改写的上游错误会绕过此守卫，
                # 落入正常错误处理器。观察到新的提供商措辞时扩展短语列表。
                _err_body = ""
                try:
                    _err_body = str(getattr(api_error, "body", None) or
                                    getattr(api_error, "message", None) or
                                    str(api_error))
                except Exception:
                    pass
                _err_status = getattr(api_error, "status_code", None)
                _IMAGE_REJECTION_PHRASES = (
                    "only 'text' content type is supported",
                    "only text content type is supported",
                    "image_url is not supported",
                    "image content is not supported",
                    "multimodal is not supported",
                    "multimodal content is not supported",
                    "multimodal input is not supported",
                    "vision is not supported",
                    "vision input is not supported",
                    "does not support images",
                    "does not support image input",
                    "does not support multimodal",
                    "does not support vision",
                    "model does not support image",
                    # ChatGPT 账号的 Codex 后端
                    # (https://chatgpt.com/backend-api/codex) 以
                    # HTTP 400 拒绝 input_image 字段中的
                    # data:image/...base64 URL。公共端点上的
                    # OpenAI Responses API 接受 data URL，但
                    # ChatGPT 账号变体不接受。没有此短语时，
                    # 智能体会级联进入压缩/上下文过大的恢复流程，
                    # 而非简单地剥离图像。匹配故意设计得较窄 ——
                    # 以字段路径的撇号为键，避免在其他 URL 验证
                    # 错误上误触发。(issue #23570)
                    "image_url'. expected",
                    # DeepSeek 的 OpenAI 兼容 API 将纯文本请求体变体报告为：
                    # "unknown variant `image_url`, expected `text`"
                    "unknown variant `image_url`, expected `text`",
                    "unknown variant image_url, expected text",
                )
                _err_lower = _err_body.lower()
                _looks_like_image_rejection = any(
                    p in _err_lower for p in _IMAGE_REJECTION_PHRASES
                )
                # 仅 4xx 门控：永远不要将 5xx/超时解释为
                # "服务器拒绝了图像" —— 那些是瞬态错误，
                # 必须路由到正常重试路径。
                _status_ok = _err_status is None or (400 <= int(_err_status) < 500)
                if (
                    getattr(agent, "_vision_supported", True)
                    and _looks_like_image_rejection
                    and _status_ok
                ):
                    agent._vision_supported = False
                    _imgs_removed = _strip_images_from_messages(messages)
                    if isinstance(api_messages, list):
                        _strip_images_from_messages(api_messages)
                    agent._vprint(
                        f"{agent.log_prefix}⚠️  Server rejected image content — "
                        f"switching to text-only mode for this session"
                        + (". Stripped images from history and retrying." if _imgs_removed else "."),
                        force=True,
                    )
                    continue

                status_code = getattr(api_error, "status_code", None)
                error_context = agent._extract_api_error_context(api_error)

                # ── 对错误进行分类，以做出结构化的恢复决策 ──
                _compressor = getattr(agent, "context_compressor", None)
                _ctx_len = getattr(_compressor, "context_length", 200000) if _compressor else 200000
                classified = classify_api_error(
                    api_error,
                    provider=getattr(agent, "provider", "") or "",
                    model=getattr(agent, "model", "") or "",
                    approx_tokens=approx_tokens,
                    context_length=_ctx_len,
                    num_messages=len(api_messages) if api_messages else 0,
                )
                logger.debug(
                    "Error classified: reason=%s status=%s retryable=%s compress=%s rotate=%s fallback=%s",
                    classified.reason.value, classified.status_code,
                    classified.retryable, classified.should_compress,
                    classified.should_rotate_credential, classified.should_fallback,
                )

                if (
                    classified.reason == FailoverReason.billing
                    and _is_nous_inference_route(
                        getattr(agent, "provider", "") or "",
                        getattr(agent, "base_url", "") or "",
                    )
                    and not nous_paid_entitlement_refresh_attempted
                ):
                    nous_paid_entitlement_refresh_attempted = True
                    if _try_refresh_nous_paid_entitlement_credentials(agent):
                        agent._vprint(
                            f"{agent.log_prefix}🔐 Nous paid access verified — "
                            "refreshed runtime credentials and retrying request...",
                            force=True,
                        )
                        continue

                recovered_with_pool, has_retried_429 = agent._recover_with_credential_pool(
                    status_code=status_code,
                    has_retried_429=has_retried_429,
                    classified_reason=classified.reason,
                    error_context=error_context,
                )
                if recovered_with_pool:
                    continue

                # 图像过大恢复：就地缩小超大的原生图像部分并重试一次。
                # 由 Anthropic 的每张图像 5 MB 上限触发
                # （400 错误 + "image exceeds 5 MB maximum"），
                # 或其他抱怨图像大小的提供商。
                # 如果缩小失败或第二次尝试仍然失败，落入正常错误处理。
                if (
                    classified.reason == FailoverReason.image_too_large
                    and not image_shrink_retry_attempted
                ):
                    image_shrink_retry_attempted = True
                    if agent._try_shrink_image_parts_in_messages(api_messages):
                        agent._vprint(
                            f"{agent.log_prefix}📐 Image(s) exceeded provider size limit — "
                            f"shrank and retrying...",
                            force=True,
                        )
                        continue
                    else:
                        logger.info(
                            "image-shrink recovery: no data-URL image parts found "
                            "or shrink didn't reduce size; surfacing original error."
                        )

                # 多模态工具内容恢复：严格遵循 OpenAI 规范的提供商
                # （工具消息内容必须是字符串）会以 400 错误拒绝
                # 我们的列表类型内容。从列表类型的工具消息中剥离
                # 图像部分，在会话剩余时间内将 (provider, model) 标记为
                # 不支持列表类型工具内容，使未来的工具结果预防性地降级，
                # 然后重试一次。参见 issue #27344。
                if (
                    classified.reason == FailoverReason.multimodal_tool_content_unsupported
                    and not multimodal_tool_content_retry_attempted
                ):
                    multimodal_tool_content_retry_attempted = True
                    if agent._try_strip_image_parts_from_tool_messages(api_messages):
                        agent._vprint(
                            f"{agent.log_prefix}📐 Provider rejected list-type tool content — "
                            f"downgraded screenshots to text and retrying...",
                            force=True,
                        )
                        continue
                    else:
                        logger.info(
                            "multimodal-tool-content recovery: no list-type tool "
                            "messages with image parts found; surfacing original error."
                        )

                # Anthropic OAuth 订阅拒绝了 1M 上下文 beta 头
                # （"long context beta is not yet available for this subscription"）。
                # 在会话剩余时间内禁用 beta，重建客户端，并重试一次。
                # 支持 1M 的订阅永远不会进入此分支 —— 它们接受 beta
                # 并保持完整的 1M 上下文。参见 PR #17680 的原始报告
                # （我们选择了反应性恢复而非提议的无条件省略，
                # 使支持的订阅不会静默丢失此能力）。
                if (
                    classified.reason == FailoverReason.oauth_long_context_beta_forbidden
                    and agent.api_mode == "anthropic_messages"
                    and agent._is_anthropic_oauth
                    and not oauth_1m_beta_retry_attempted
                ):
                    oauth_1m_beta_retry_attempted = True
                    if not getattr(agent, "_oauth_1m_beta_disabled", False):
                        agent._oauth_1m_beta_disabled = True
                        try:
                            agent._anthropic_client.close()
                        except Exception:
                            pass
                        agent._rebuild_anthropic_client()
                        agent._vprint(
                            f"{agent.log_prefix}🔕 OAuth subscription doesn't support "
                            f"the 1M-context beta — disabled for this session and retrying...",
                            force=True,
                        )
                        continue

                if (
                    agent.api_mode == "codex_responses"
                    and agent.provider in {"openai-codex", "xai-oauth"}
                    and status_code == 401
                    and not codex_auth_retry_attempted
                ):
                    codex_auth_retry_attempted = True
                    if agent._try_refresh_codex_client_credentials(force=True):
                        _label = "xAI OAuth" if agent.provider == "xai-oauth" else "Codex"
                        agent._buffer_vprint(f"🔐 {_label} auth refreshed after 401. Retrying request...")
                        continue
                if (
                    agent.api_mode == "chat_completions"
                    and agent.provider == "nous"
                    and status_code == 401
                    and not nous_auth_retry_attempted
                ):
                    nous_auth_retry_attempted = True
                    if agent._try_refresh_nous_client_credentials(force=True):
                        print(f"{agent.log_prefix}🔐 Nous agent key refreshed after 401. Retrying request...")
                        continue
                    # 凭证刷新没有效果 —— 显示诊断信息。
                    # 最常见原因：Portal OAuth 过期/撤销、
                    # 账户额度耗尽，或智能体密钥被封禁。
                    from hermes_constants import display_hermes_home as _dhh_fn
                    _dhh = _dhh_fn()
                    _body_text = ""
                    try:
                        _body = getattr(api_error, "body", None) or getattr(api_error, "response", None)
                        if _body is not None:
                            _body_text = str(_body)[:200]
                    except Exception:
                        pass
                    print(f"{agent.log_prefix}🔐 Nous 401 — Portal authentication failed.")
                    if _body_text:
                        print(f"{agent.log_prefix}   Response: {_body_text}")
                    if not _print_nous_entitlement_guidance(agent, "Nous model access"):
                        print(f"{agent.log_prefix}   Most likely: Portal OAuth expired, account out of credits, or agent key revoked.")
                    print(f"{agent.log_prefix}   Troubleshooting:")
                    print(f"{agent.log_prefix}     • Re-authenticate: hermes auth add nous")
                    print(f"{agent.log_prefix}     • Check credits / billing: https://portal.nousresearch.com")
                    print(f"{agent.log_prefix}     • Verify stored credentials: {_dhh}/auth.json")
                    print(f"{agent.log_prefix}     • Switch providers temporarily: /model <model> --provider openrouter")
                if (
                    agent.provider == "copilot"
                    and status_code == 401
                    and not copilot_auth_retry_attempted
                ):
                    copilot_auth_retry_attempted = True
                    if agent._try_refresh_copilot_client_credentials():
                        agent._buffer_vprint(f"🔐 Copilot credentials refreshed after 401. Retrying request...")
                        continue
                if (
                    agent.api_mode == "anthropic_messages"
                    and status_code == 401
                    and hasattr(agent, '_anthropic_api_key')
                    and not anthropic_auth_retry_attempted
                ):
                    anthropic_auth_retry_attempted = True
                    from agent.anthropic_adapter import _is_oauth_token
                    from agent.azure_identity_adapter import is_token_provider
                    if agent._try_refresh_anthropic_client_credentials():
                        print(f"{agent.log_prefix}🔐 Anthropic credentials refreshed after 401. Retrying request...")
                        continue
                    # 凭证刷新没有效果 —— 显示诊断信息
                    key = agent._anthropic_api_key
                    print(f"{agent.log_prefix}🔐 Anthropic 401 — authentication failed.")
                    if is_token_provider(key):
                        # Azure Foundry Entra ID —— bearer token 由 httpx 事件钩子
                        # 在传递给 SDK 的自定义 http_client 上按需生成。
                        # 401 表示 Azure 拒绝了 JWT（RBAC 角色缺失、
                        # az login 过期、IMDS 不可达等）。
                        print(f"{agent.log_prefix}   Auth method: Microsoft Entra ID (httpx event hook)")
                        print(f"{agent.log_prefix}   Run `hermes doctor` for credential-chain diagnostics, or")
                        print(f"{agent.log_prefix}   `az login` if your developer session expired.")
                    else:
                        auth_method = "Bearer (OAuth/setup-token)" if _is_oauth_token(key) else "x-api-key (API key)"
                        print(f"{agent.log_prefix}   Auth method: {auth_method}")
                        print(f"{agent.log_prefix}   Token prefix: {key[:12]}..." if isinstance(key, str) and len(key) > 12 else f"{agent.log_prefix}   Token: (empty or short)")
                    print(f"{agent.log_prefix}   Troubleshooting:")
                    from hermes_constants import display_hermes_home as _dhh_fn
                    _dhh = _dhh_fn()
                    print(f"{agent.log_prefix}     • Check ANTHROPIC_TOKEN in {_dhh}/.env for Hermes-managed OAuth/setup tokens")
                    print(f"{agent.log_prefix}     • Check ANTHROPIC_API_KEY in {_dhh}/.env for API keys or legacy token values")
                    print(f"{agent.log_prefix}     • For API keys: verify at https://platform.claude.com/settings/keys")
                    print(f"{agent.log_prefix}     • For Claude Code: run 'claude /login' to refresh, then retry")
                    print(f"{agent.log_prefix}     • Legacy cleanup: hermes config set ANTHROPIC_TOKEN \"\"")
                    print(f"{agent.log_prefix}     • Clear stale keys: hermes config set ANTHROPIC_API_KEY \"\"")

                # ── 思考块签名恢复 ─────────────────
                # Anthropic 对思考块根据完整轮次内容进行签名。
                # 任何上游变更（上下文压缩、会话截断、消息合并）
                # 都会使签名失效 → HTTP 400。
                # 恢复：从所有消息中剥离 reasoning_details，
                # 使下次重试完全不发送思考块。一次性操作 ——
                # 不要无限重试。
                if (
                    classified.reason == FailoverReason.thinking_signature
                    and not thinking_sig_retry_attempted
                ):
                    thinking_sig_retry_attempted = True
                    for _m in messages:
                        if isinstance(_m, dict):
                            _m.pop("reasoning_details", None)
                    agent._vprint(
                        f"{agent.log_prefix}⚠️  Thinking block signature invalid — "
                        f"stripped all thinking blocks, retrying...",
                        force=True,
                    )
                    logger.warning(
                        "%sThinking block signature recovery: stripped "
                        "reasoning_details from %d messages",
                        agent.log_prefix, len(messages),
                    )
                    continue

                # ── 无效加密推理重放恢复 ───────
                # OpenAI Responses API 表面（和一些兼容中继）
                # 在重放的 ``codex_reasoning_items`` blob 验证失败时
                # 返回 HTTP 400 ``invalid_encrypted_content``
                # （提供商轮换加密密钥、路由实际上不持久化推理状态等）。
                # 恢复：在会话剩余时间内禁用重放，从历史中剥离缓存项，
                # 重试一次。一次性操作 —— 如果第二次 400 触发，
                # 则落入正常重试/退避路径。仅在 codex_responses 模式下
                # 且至少有一条带缓存 ``codex_reasoning_items`` 的助手消息时触发；
                # 没有重放状态时，此错误与我们的缓存无关，
                # 正常重试路径可以处理（提供商在拒绝其他东西）。
                if (
                    classified.reason == FailoverReason.invalid_encrypted_content
                    and not invalid_encrypted_content_retry_attempted
                    and agent.api_mode == "codex_responses"
                    and bool(getattr(agent, "_codex_reasoning_replay_enabled", True))
                    and any(
                        isinstance(_m, dict)
                        and _m.get("role") == "assistant"
                        and isinstance(_m.get("codex_reasoning_items"), list)
                        and _m.get("codex_reasoning_items")
                        for _m in messages
                    )
                ):
                    invalid_encrypted_content_retry_attempted = True
                    replay_stats = agent._disable_codex_reasoning_replay(messages)
                    agent._vprint(
                        f"{agent.log_prefix}⚠️  Encrypted reasoning replay was rejected by the provider — "
                        f"disabled replay and stripped {replay_stats['items']} item(s) from "
                        f"{replay_stats['messages']} message(s), retrying...",
                        force=True,
                    )
                    logger.warning(
                        "%sInvalid encrypted reasoning recovery: disabled replay and stripped %d items from %d messages",
                        agent.log_prefix,
                        replay_stats["items"],
                        replay_stats["messages"],
                    )
                    continue

                # ── llama.cpp 语法解析恢复 ──────────────────
                # llama.cpp 的 ``json-schema-to-grammar`` 转换器
                # 拒绝正则转义类（``\d``、``\w``、``\s``）和大多数
                # 工具 schema 中的 ``format`` 值。MCP 服务器
                # 经常在日期/电话/邮箱参数中生成这些。
                # 恢复：从 ``agent.tools`` 中剥离 ``pattern``/``format``
                # 并重试一次。默认保留这些关键字，使云提供商获得
                # 完整的提示引导；此分支仅在用户使用 llama.cpp 的
                # OAI 服务器时触发。
                if (
                    classified.reason == FailoverReason.llama_cpp_grammar_pattern
                    and not llama_cpp_grammar_retry_attempted
                ):
                    llama_cpp_grammar_retry_attempted = True
                    try:
                        from tools.schema_sanitizer import strip_pattern_and_format
                        _, _stripped = strip_pattern_and_format(agent.tools)
                    except Exception as _strip_exc:  # pragma: no cover — defensive
                        logger.warning(
                            "%sllama.cpp grammar recovery: strip helper failed: %s",
                            agent.log_prefix, _strip_exc,
                        )
                        _stripped = 0
                    if _stripped:
                        agent._vprint(
                            f"{agent.log_prefix}⚠️  llama.cpp rejected tool schema grammar — "
                            f"stripped {_stripped} pattern/format keyword(s), retrying...",
                            force=True,
                        )
                        logger.warning(
                            "%sllama.cpp grammar recovery: stripped %d "
                            "pattern/format keyword(s) from tool schemas",
                            agent.log_prefix, _stripped,
                        )
                        continue
                    # 没有找到可剥离的关键字 —— 落入正常重试路径，
                    # 而非在同一错误上无限循环。
                    logger.warning(
                        "%sllama.cpp grammar error but no pattern/format "
                        "keywords to strip — falling through to normal retry",
                        agent.log_prefix,
                    )

                retry_count += 1
                elapsed_time = time.time() - api_start_time
                agent._touch_activity(
                    f"API error recovery (attempt {retry_count}/{max_retries})"
                )
                
                error_type = type(api_error).__name__
                error_msg = str(api_error).lower()
                _error_summary = agent._summarize_api_error(api_error)
                logger.warning(
                    "API call failed (attempt %s/%s) error_type=%s %s summary=%s",
                    retry_count,
                    max_retries,
                    error_type,
                    agent._client_log_context(),
                    _error_summary,
                )

                _provider = getattr(agent, "provider", "unknown")
                _base = getattr(agent, "base_url", "unknown")
                _model = getattr(agent, "model", "unknown")
                _status_code_str = f" [HTTP {status_code}]" if status_code else ""
                agent._buffer_vprint(f"⚠️  API call failed (attempt {retry_count}/{max_retries}): {error_type}{_status_code_str}")
                agent._buffer_vprint(f"   🔌 Provider: {_provider}  Model: {_model}")
                agent._buffer_vprint(f"   🌐 Endpoint: {_base}")
                agent._buffer_vprint(f"   📝 Error: {_error_summary}")
                if status_code and status_code < 500:
                    _err_body = getattr(api_error, "body", None)
                    _err_body_str = str(_err_body)[:300] if _err_body else None
                    if _err_body_str:
                        agent._buffer_vprint(f"   📋 Details: {_err_body_str}")
                agent._buffer_vprint(f"   ⏱️  Elapsed: {elapsed_time:.2f}s  Context: {len(api_messages)} msgs, ~{approx_tokens:,} tokens")

                # 如果已重试过 429 且仍有限速，检查是否有可用的操作提示
                # 缓冲区的其余重试跟踪 —— 仅在每次重试+回退都耗尽后显示。
                # 避免对通过回退自动恢复的用户造成信息轰炸。
                if (
                    agent._is_openrouter_url()
                    and "support tool use" in error_msg
                ):
                    agent._buffer_vprint(
                        f"   💡 No OpenRouter providers for {_model} support tool calling with your current settings."
                    )
                    if agent.providers_allowed:
                        agent._buffer_vprint(
                            f"      Your provider_routing.only restriction is filtering out tool-capable providers."
                        )
                        agent._buffer_vprint(
                            f"      Try removing the restriction or adding providers that support tools for this model."
                        )
                    agent._buffer_vprint(
                        f"      Check which providers support tools: https://openrouter.ai/models/{_model}"
                    )

                # 在决定重试之前检查中断
                if agent._interrupt_requested:
                    agent._vprint(f"{agent.log_prefix}⚡ Interrupt detected during error handling, aborting retries.", force=True)
                    agent._persist_session(messages, conversation_history)
                    agent.clear_interrupt()
                    return {
                        "final_response": f"Operation interrupted: handling API error ({error_type}: {agent._clean_error_message(str(api_error))}).",
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "interrupted": True,
                    }
                
                # 在通用 4xx 处理器之前检查 413 负载过大错误。
                # 413 是负载大小错误 —— 正确的响应是压缩历史并重试，
                # 而非立即中止。
                status_code = getattr(api_error, "status_code", None)

                # ── Anthropic Sonnet 长上下文层级门控 ───────────
                # 当 Claude Max（或类似）订阅不包含 1M 上下文层级时，
                # Anthropic 返回 HTTP 429 "Extra usage is required for
                # long context requests"。这不是瞬态速率限制 ——
                # 重试或切换凭证无济于事。将上下文减少到 200k
                # （标准层级）并压缩。
                if classified.reason == FailoverReason.long_context_tier:
                    _reduced_ctx = 200000
                    compressor = agent.context_compressor
                    old_ctx = compressor.context_length
                    if old_ctx > _reduced_ctx:
                        compressor.update_model(
                            model=agent.model,
                            context_length=_reduced_ctx,
                            base_url=agent.base_url,
                            api_key=getattr(agent, "api_key", ""),
                            provider=agent.provider,
                            api_mode=agent.api_mode,
                        )
                        # 上下文探测标志 —— 仅在内置压缩器上设置
                        # （插件引擎管理自己的）。
                        if hasattr(compressor, "_context_probed"):
                            compressor._context_probed = True
                            # 不持久化 —— 这是订阅层级限制，不是模型能力。
                            # 如果用户后来启用了额外使用量，1M 限制应自动恢复。
                            compressor._context_probe_persistable = False
                        agent._buffer_vprint(
                            f"⚠️  Anthropic long-context tier "
                            f"requires extra usage — reducing context: "
                            f"{old_ctx:,} → {_reduced_ctx:,} tokens"
                        )

                    compression_attempts += 1
                    if compression_attempts <= max_compression_attempts:
                        original_len = len(messages)
                        messages, active_system_prompt = agent._compress_context(
                            messages, system_message,
                            approx_tokens=approx_tokens,
                            task_id=effective_task_id,
                        )
                        # 压缩创建了新会话 —— 清除历史引用
                        # 使 _flush_messages_to_session_db 将压缩后的消息
                        # 写入新会话，而非跳过它们。
                        conversation_history = None
                        if len(messages) < original_len or old_ctx > _reduced_ctx:
                            agent._buffer_status(
                                f"🗜️ Context reduced to {_reduced_ctx:,} tokens "
                                f"(was {old_ctx:,}), retrying..."
                            )
                            time.sleep(2)
                            restart_with_compressed_messages = True
                            break
                    # 如果压缩耗尽或没有帮助，落入正常错误处理。

                # 速率限制错误的快速回退（429 或配额耗尽）。
                # 当配置了回退模型时，立即切换而非用指数退避
                # 耗尽重试次数 —— 主提供商不会在重试窗口内恢复。
                is_rate_limited = classified.reason in {
                    FailoverReason.rate_limit,
                    FailoverReason.billing,
                }
                if is_rate_limited and agent._fallback_index < len(agent._fallback_chain):
                    # 如果凭证池轮换仍可能恢复，不急于回退。
                    # 参见 _pool_may_recover_from_rate_limit 了解
                    # 单凭证池和 CloudCode 配额例外情况。修复 #11314 和 #13636。
                    pool_may_recover = _ra()._pool_may_recover_from_rate_limit(
                        agent._credential_pool,
                        provider=agent.provider,
                        base_url=getattr(agent, "base_url", None),
                    )
                    if not pool_may_recover:
                        if classified.reason == FailoverReason.billing:
                            agent._buffer_status(
                                "⚠️ Billing or credits exhausted — switching to fallback provider..."
                            )
                        else:
                            agent._buffer_status("⚠️ Rate limited — switching to fallback provider...")
                        if agent._try_activate_fallback(reason=classified.reason):
                            retry_count = 0
                            compression_attempts = 0
                            primary_recovery_attempted = False
                            continue

                # ── Nous Portal：记录速率限制并跳过重试 ─────
                # 当 Nous 返回的 429 是真正的账户级速率限制时，
                # 将重置时间记录到共享文件中，使所有会话（定时任务、
                # 网关、辅助）都知道不要堆积请求 ——
                # 每次重试都会消耗另一个 RPH 请求并加深困境。
                # 重试循环的顶部迭代守卫会在下次通过时捕获
                # 并尝试回退或干净退出。
                #
                # 重要：Nous Portal 多路复用多个上游提供商
                # （DeepSeek, Kimi, MiMo, Hermes）。429 也可能意味着
                # 某个上游提供商对某个特定模型的容量不足 ——
                # 瞬态错误，几秒内清除，与调用者的配额无关。
                # 对此触发跨会话断路器会在数分钟内阻塞所有 Nous 模型。
                # 我们使用 ``is_genuine_nous_rate_limit`` 通过 429 的
                # x-ratelimit-* 头和上一次成功响应捕获的最后已知状态
                # 来区分这两种情况。
                if (
                    is_rate_limited
                    and agent.provider == "nous"
                    and classified.reason == FailoverReason.rate_limit
                    and not recovered_with_pool
                ):
                    _genuine_nous_rate_limit = False
                    try:
                        from agent.nous_rate_guard import (
                            is_genuine_nous_rate_limit,
                            record_nous_rate_limit,
                        )
                        _err_resp = getattr(api_error, "response", None)
                        _err_hdrs = (
                            getattr(_err_resp, "headers", None)
                            if _err_resp else None
                        )
                        _genuine_nous_rate_limit = is_genuine_nous_rate_limit(
                            headers=_err_hdrs,
                            last_known_state=agent._rate_limit_state,
                        )
                        if _genuine_nous_rate_limit:
                            record_nous_rate_limit(
                                headers=_err_hdrs,
                                error_context=error_context,
                            )
                        else:
                            logger.info(
                                "Nous 429 looks like upstream capacity "
                                "(no exhausted bucket in headers or "
                                "last-known state) -- not tripping "
                                "cross-session breaker."
                            )
                    except Exception:
                        pass
                    if _genuine_nous_rate_limit:
                        # 直接跳到 max_retries ——
                        # 循环顶部守卫将处理回退或
                        # 干净退出。
                        retry_count = max_retries
                        continue
                    # 上游容量 429：落入正常重试逻辑。
                    # 重试逻辑。不同的模型（或稍后相同的
                    # 模型）通常会成功。

                is_payload_too_large = (
                    classified.reason == FailoverReason.payload_too_large
                )

                # GitHub Models (Azure) 413 错误的操作提示。
                # 免费层强制执行每个请求 8K token 的硬上限，
                # 而 Hermes 的系统提示词 + 工具 schema 基线就超过了。
                # 压缩无济于事 —— 底线是系统提示词本身，而非对话 ——
                # 因此显示清晰的"不兼容"消息，而非循环进入三次无用的压缩尝试。
                if (
                    status_code == 413
                    and isinstance(agent.base_url, str)
                    and "models.inference.ai.azure.com" in agent.base_url
                ):
                    agent._vprint(
                        f"{agent.log_prefix}   💡 GitHub Models free tier (models.inference.ai.azure.com) caps every",
                        force=True,
                    )
                    agent._vprint(
                        f"{agent.log_prefix}      request at ~8K tokens. Hermes' system prompt + tool schemas baseline",
                        force=True,
                    )
                    agent._vprint(
                        f"{agent.log_prefix}      exceeds that floor, so this endpoint cannot run an agentic loop.",
                        force=True,
                    )
                    agent._vprint(
                        f"{agent.log_prefix}      Use the `copilot` provider with a Copilot subscription token (`hermes",
                        force=True,
                    )
                    agent._vprint(
                        f"{agent.log_prefix}      setup` → GitHub Copilot), or pick any other provider.",
                        force=True,
                    )

                if is_payload_too_large:
                    compression_attempts += 1
                    if compression_attempts > max_compression_attempts:
                        # 终端 —— 显示缓冲的重试跟踪。
                        agent._flush_status_buffer()
                        agent._vprint(f"{agent.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached for payload-too-large error.", force=True)
                        agent._vprint(f"{agent.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                        logger.error(f"{agent.log_prefix}413 compression failed after {max_compression_attempts} attempts.")
                        agent._persist_session(messages, conversation_history)
                        return {
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": f"Request payload too large: max compression attempts ({max_compression_attempts}) reached.",
                            "partial": True,
                            "failed": True,
                            "compression_exhausted": True,
                        }
                    agent._buffer_status(f"⚠️  Request payload too large (413) — compression attempt {compression_attempts}/{max_compression_attempts}...")

                    original_len = len(messages)
                    messages, active_system_prompt = agent._compress_context(
                        messages, system_message, approx_tokens=approx_tokens,
                        task_id=effective_task_id,
                    )
                    # 压缩创建了新会话 —— 清除历史引用
                    # 使 _flush_messages_to_session_db 将压缩后的消息
                    # 写入新会话，而非跳过它们。
                    conversation_history = None

                    if len(messages) < original_len:
                        agent._buffer_status(f"🗜️ Compressed {original_len} → {len(messages)} messages, retrying...")
                        time.sleep(2)  # 压缩重试之间的短暂暂停
                        restart_with_compressed_messages = True
                        break
                    else:
                        # 终端 —— 显示缓冲的上下文，使用户
                        # 看到进行了哪些压缩尝试。
                        agent._flush_status_buffer()
                        agent._vprint(f"{agent.log_prefix}❌ Payload too large and cannot compress further.", force=True)
                        agent._vprint(f"{agent.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                        logger.error(f"{agent.log_prefix}413 payload too large. Cannot compress further.")
                        agent._persist_session(messages, conversation_history)
                        return {
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": "Request payload too large (413). Cannot compress further.",
                            "partial": True,
                            "failed": True,
                            "compression_exhausted": True,
                        }

                # 在通用 4xx 处理器之前检查上下文长度错误。
                # 分类器从以下来源检测上下文溢出：显式错误消息、
                # 通用 400 + 大会话启发式 (#1630)，以及
                # 服务器断开 + 大会话模式 (#2153)。
                is_context_length_error = (
                    classified.reason == FailoverReason.context_overflow
                )

                if is_context_length_error:
                    compressor = agent.context_compressor
                    old_ctx = compressor.context_length

                    # ── 区分两种截然不同的错误 ───────────
                    # 1. "Prompt too long"：输入超过了上下文窗口。
                    #    修复：减少 context_length + 压缩历史。
                    # 2. "max_tokens too large"：输入没问题，但
                    #    input_tokens + 请求的 max_tokens > context_window。
                    #    修复：减少 max_tokens（输出上限）用于此次调用。
                    #    不要缩小 context_length —— 窗口未变。
                    #
                    # 注意：max_tokens = 输出 token 上限（单次响应）。
                    #       context_length = 总窗口（输入 + 输出合计）。
                    available_out = parse_available_output_tokens_from_error(error_msg)
                    if available_out is not None:
                        # 错误纯粹是关于输出上限过大。
                        # 将输出限制到可用空间并重试，不修改 context_length 或触发压缩。
                        safe_out = max(1, available_out - 64)  # 小安全裕量
                        agent._ephemeral_max_output_tokens = safe_out
                        agent._buffer_vprint(
                            f"⚠️  Output cap too large for current prompt — "
                            f"retrying with max_tokens={safe_out:,} "
                            f"(available_tokens={available_out:,}; context_length unchanged at {old_ctx:,})"
                        )
                        # 仍然计入 compression_attempts，使我们在错误持续
                        # 重现时不会无限循环。
                        compression_attempts += 1
                        if compression_attempts > max_compression_attempts:
                            agent._flush_status_buffer()
                            agent._vprint(f"{agent.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached.", force=True)
                            agent._vprint(f"{agent.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                            logger.error(f"{agent.log_prefix}Context compression failed after {max_compression_attempts} attempts.")
                            agent._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Context length exceeded: max compression attempts ({max_compression_attempts}) reached.",
                                "partial": True,
                                "failed": True,
                                "compression_exhausted": True,
                            }
                        restart_with_compressed_messages = True
                        break

                    # 错误是关于输入过大。仅当提供商明确报告
                    # context_length，仅当提供商明确报告真实的
                    # 下限值时。如果提供商仅说 "input
                    # exceeds the context window"（输入超过上下文窗口），
                    # 保持已配置的窗口并尝试压缩；猜测探测层级
                    # 可能错误地将用户配置的 1M 窗口变为 256K/128K/64K。
                    new_ctx = get_context_length_from_provider_error(error_msg, old_ctx)
                    _provider_lower = (getattr(agent, "provider", "") or "").lower()
                    _base_lower = (getattr(agent, "base_url", "") or "").rstrip("/").lower()
                    is_minimax_provider = (
                        _provider_lower in {"minimax", "minimax-cn"}
                        or _base_lower.startswith((
                            "https://api.minimax.io/anthropic",
                            "https://api.minimaxi.com/anthropic",
                        ))
                    )
                    minimax_delta_only_overflow = (
                        is_minimax_provider
                        and new_ctx is None
                        and "context window exceeds limit (" in error_msg
                    )

                    if new_ctx is not None:
                        agent._buffer_vprint(f"Context limit detected from API: {new_ctx:,} tokens (was {old_ctx:,})")
                        compressor.update_model(
                            model=agent.model,
                            context_length=new_ctx,
                            base_url=agent.base_url,
                            api_key=getattr(agent, "api_key", ""),
                            provider=agent.provider,
                            api_mode=agent.api_mode,
                        )
                        # 上下文探测标志 —— 仅在内置压缩器上设置
                        # （插件引擎管理自己的）。此值来自提供商，
                        # 因此可以安全缓存。
                        if hasattr(compressor, "_context_probed"):
                            compressor._context_probed = True
                            compressor._context_probe_persistable = True
                        agent._buffer_vprint(f"⚠️  Context length exceeded — using provider limit: {old_ctx:,} → {new_ctx:,} tokens")
                    elif minimax_delta_only_overflow:
                        agent._buffer_vprint(
                            f"Provider reported overflow amount only; "
                            f"keeping context_length at {old_ctx:,} tokens and compressing."
                        )
                    else:
                        agent._buffer_vprint(
                            f"⚠️  Context length exceeded, but provider did not report a max context length; "
                            f"keeping context_length at {old_ctx:,} tokens and compressing."
                        )

                    compression_attempts += 1
                    if compression_attempts > max_compression_attempts:
                        agent._flush_status_buffer()
                        agent._vprint(f"{agent.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached.", force=True)
                        agent._vprint(f"{agent.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                        logger.error(f"{agent.log_prefix}Context compression failed after {max_compression_attempts} attempts.")
                        agent._persist_session(messages, conversation_history)
                        return {
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": f"Context length exceeded: max compression attempts ({max_compression_attempts}) reached.",
                            "partial": True,
                            "failed": True,
                            "compression_exhausted": True,
                        }
                    agent._buffer_status(f"🗜️ Context too large (~{approx_tokens:,} tokens) — compressing ({compression_attempts}/{max_compression_attempts})...")

                    original_len = len(messages)
                    messages, active_system_prompt = agent._compress_context(
                        messages, system_message, approx_tokens=approx_tokens,
                        task_id=effective_task_id,
                    )
                    # 压缩创建了新会话 —— 清除历史引用
                    # 使 _flush_messages_to_session_db 将压缩后的消息
                    # 写入新会话，而非跳过它们。
                    conversation_history = None

                    if len(messages) < original_len or new_ctx and new_ctx < old_ctx:
                        if len(messages) < original_len:
                            agent._buffer_status(f"🗜️ Compressed {original_len} → {len(messages)} messages, retrying...")
                        time.sleep(2)  # Brief pause between compression retries
                        restart_with_compressed_messages = True
                        break
                    else:
                        # 无法进一步压缩且已在最低层级
                        agent._flush_status_buffer()
                        agent._vprint(f"{agent.log_prefix}❌ Context length exceeded and cannot compress further.", force=True)
                        agent._vprint(f"{agent.log_prefix}   💡 The conversation has accumulated too much content. Try /new to start fresh, or /compress to manually trigger compression.", force=True)
                        logger.error(f"{agent.log_prefix}Context length exceeded: {approx_tokens:,} tokens. Cannot compress further.")
                        agent._persist_session(messages, conversation_history)
                        return {
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": f"Context length exceeded ({approx_tokens:,} tokens). Cannot compress further.",
                            "partial": True,
                            "failed": True,
                            "compression_exhausted": True,
                        }

                # 检查不可重试的客户端错误。分类器已考虑了
                # 413、429、529（瞬态）、上下文溢出和通用 400 启发式。
                # 本地验证错误（ValueError, TypeError）是编程 bug。
                # 排除 UnicodeEncodeError —— 它是 ValueError 子类但
                # 由上方的代理字符清理路径单独处理。
                # 排除 json.JSONDecodeError —— 也是 ValueError 子类，
                # 但它表示瞬态的提供商/网络故障（格式错误的响应体、
                # 截断的流、路由层损坏），而非本地编程 bug，应该重试 (#14782)。
                is_local_validation_error = (
                    isinstance(api_error, (ValueError, TypeError))
                    and not isinstance(
                        api_error, (UnicodeEncodeError, json.JSONDecodeError)
                    )
                    # ssl.SSLError（及其子类 SSLCertVerificationError）
                    # 通过 Python MRO 同时继承 OSError 和 ValueError，
                    # 因此上面的 isinstance(ValueError) 检查会
                    # 将 TLS 传输故障误分类为本地编程 bug 并在不重试的情况下中止。
                    # 明确排除 ssl.SSLError，使错误分类器的
                    # retryable=True 映射生效。
                    and not isinstance(api_error, ssl.SSLError)
                    # 提供商/SDK 的 "NoneType is not iterable" 失败是
                    # 来自上游的形状不匹配（例如 chatgpt.com Codex
                    # 后端 response.completed.output=null）—— 不是本地编程 bug。
                    # 即使 #33042 使我们自己的消费者免疫，第三方垫片
                    # 和模拟客户端仍可通过 TypeError 表面化此形状。
                    # 将它们视为可重试的，使错误分类器的正常重试/回退
                    # 路径运行，而非将轮次作为不可重试终止
                    # （否则会使 Telegram 用户盯着空白的 "Non-retryable error"）。
                    and not (
                        isinstance(api_error, TypeError)
                        and "nonetype" in str(api_error).lower()
                        and "not iterable" in str(api_error).lower()
                    )
                )
                # ``FailoverReason.billing`` (HTTP 402) 不在此排除集中。
                # 到达此块时：
                #   • 凭证池轮换（约 L2031）已对计费触发并 ``continue`` 或
                #     返回 (False, ...) —— 池已耗尽或不存在。
                #   • 上方的快速回退分支（约 L2422）也对计费触发并
                #     在配置了回退提供商时 ``continue``。
                # 落入此处意味着两个恢复路径都放弃了。
                # 从此处将 402 视为可重试只会耗尽更多付费请求到
                # 已耗尽的余额而没有恢复机制 —— 参见 #31273
                # （真实案例：在 24/7 网关上 48 小时内约 $40）。
                # 中止镜像了 401/403（也是 ``should_fallback=True``）
                # 在其恢复路径失败后的行为。
                is_client_error = (
                    is_local_validation_error
                    or (
                        not classified.retryable
                        and not classified.should_compress
                        and classified.reason not in {
                            FailoverReason.rate_limit,
                            FailoverReason.overloaded,
                            FailoverReason.context_overflow,
                            FailoverReason.payload_too_large,
                            FailoverReason.long_context_tier,
                            FailoverReason.thinking_signature,
                        }
                    )
                ) and not is_context_length_error

                if is_client_error:
                    # 在中止前尝试回退 —— 不同的提供商可能没有同样的问题
                    # （速率限制、认证等）
                    if classified.reason == FailoverReason.content_policy_blocked:
                        agent._buffer_status("⚠️ Provider safety filter blocked this request — trying fallback...")
                    else:
                        agent._buffer_status(f"⚠️ Non-retryable error (HTTP {status_code}) — trying fallback...")
                    if agent._try_activate_fallback():
                        retry_count = 0
                        compression_attempts = 0
                        primary_recovery_attempted = False
                        continue
                    if api_kwargs is not None:
                        agent._dump_api_request_debug(
                            api_kwargs, reason="non_retryable_client_error", error=api_error,
                        )
                    # 终端 —— 刷新缓冲的上下文，使用户看到中止前的尝试
                    agent._flush_status_buffer()
                    if classified.reason == FailoverReason.content_policy_blocked:
                        agent._emit_status(
                            f"❌ Provider safety filter blocked this request: "
                            f"{agent._summarize_api_error(api_error)}"
                        )
                    else:
                        agent._emit_status(
                            f"❌ Non-retryable error (HTTP {status_code}): "
                            f"{agent._summarize_api_error(api_error)}"
                        )
                    agent._vprint(f"{agent.log_prefix}❌ Non-retryable client error (HTTP {status_code}). Aborting.", force=True)
                    agent._vprint(f"{agent.log_prefix}   🔌 Provider: {_provider}  Model: {_model}", force=True)
                    agent._vprint(f"{agent.log_prefix}   🌐 Endpoint: {_base}", force=True)
                    # 常见认证错误的操作指引
                    if classified.is_auth or classified.reason == FailoverReason.billing:
                        if classified.reason == FailoverReason.billing and _print_billing_or_entitlement_guidance(
                            agent,
                            capability="model access",
                            provider=_provider,
                            base_url=str(_base),
                            model=_model,
                        ):
                            pass
                        elif _provider == "nous" and _print_nous_entitlement_guidance(
                            agent,
                            "Nous model access",
                        ):
                            pass
                        elif _provider in {"openai-codex", "xai-oauth", "nous"} and status_code == 401:
                            if _provider == "openai-codex":
                                agent._vprint(f"{agent.log_prefix}   💡 Codex OAuth token was rejected (HTTP 401). Your token may have been", force=True)
                                agent._vprint(f"{agent.log_prefix}      refreshed by another client (Codex CLI, VS Code). To fix:", force=True)
                                agent._vprint(f"{agent.log_prefix}      1. Run `codex` in your terminal to generate fresh tokens.", force=True)
                                agent._vprint(f"{agent.log_prefix}      2. Then run `hermes auth` to re-authenticate.", force=True)
                            elif _provider == "xai-oauth":
                                agent._vprint(f"{agent.log_prefix}   💡 xAI OAuth token was rejected (HTTP 401). To fix:", force=True)
                                agent._vprint(f"{agent.log_prefix}      re-authenticate with xAI Grok OAuth (SuperGrok / Premium+) from `hermes model`.", force=True)
                            else:  # nous
                                agent._vprint(f"{agent.log_prefix}   💡 Nous Portal OAuth token was rejected (HTTP 401). Your token may be", force=True)
                                agent._vprint(f"{agent.log_prefix}      expired, revoked, or your account may be out of credits. To fix:", force=True)
                                agent._vprint(f"{agent.log_prefix}      1. Re-authenticate: hermes auth add nous --type oauth", force=True)
                                agent._vprint(f"{agent.log_prefix}      2. Check your portal account: https://portal.nousresearch.com", force=True)
                                # ``:free`` is OpenRouter slug syntax; Nous Portal will reject
                                # 模型名称，即使在成功重新认证之后。
                                if isinstance(_model, str) and _model.endswith(":free"):
                                    agent._vprint(f"{agent.log_prefix}      ⚠️  Note: `{_model}` looks like an OpenRouter slug (`:free` suffix).", force=True)
                                    agent._vprint(f"{agent.log_prefix}         Nous Portal won't recognize that model name. Either switch to a", force=True)
                                    agent._vprint(f"{agent.log_prefix}         Nous catalog model, or run `/model openrouter:{_model}` to use OpenRouter.", force=True)
                        else:
                            agent._vprint(f"{agent.log_prefix}   💡 Your API key was rejected by the provider. Check:", force=True)
                            agent._vprint(f"{agent.log_prefix}      • Is the key valid? Run: hermes setup", force=True)
                            agent._vprint(f"{agent.log_prefix}      • Does your account have access to {_model}?", force=True)
                            if base_url_host_matches(str(_base), "openrouter.ai"):
                                agent._vprint(f"{agent.log_prefix}      • Check credits: https://openrouter.ai/settings/credits", force=True)
                    else:
                        agent._vprint(f"{agent.log_prefix}   💡 This type of error won't be fixed by retrying.", force=True)
                    # 内容策略阻止 deserving 自己的操作指引 ——
                    # "修复你的 API key" 和 "重试无济于事" 都不能
                    # 告诉用户实际该做什么。提供商拒绝了此特定提示词，
                    # 因此恢复方案是改写或路由到不同的模型。
                    if classified.reason == FailoverReason.content_policy_blocked:
                        agent._vprint(
                            f"{agent.log_prefix}   💡 The provider's safety filter rejected this specific prompt.",
                            force=True,
                        )
                        agent._vprint(
                            f"{agent.log_prefix}      • Try rephrasing the request, narrowing the context, or splitting into smaller steps.",
                            force=True,
                        )
                        agent._vprint(
                            f"{agent.log_prefix}      • Configure a fallback provider so future blocks route automatically:",
                            force=True,
                        )
                        agent._vprint(
                            f"{agent.log_prefix}        hermes fallback add   (interactive picker — same as `hermes model`)",
                            force=True,
                        )
                    logger.error(f"{agent.log_prefix}Non-retryable client error: {api_error}")
                    # 当错误可能与上下文溢出相关时（状态 400 + 大会话），
                    # 跳过会话持久化。持久化失败的用户消息会使会话更大，
                    # 导致下次尝试出现同样的失败。(issue #1630)
                    if status_code == 400 and (approx_tokens > 50000 or len(api_messages) > 80):
                        agent._vprint(
                            f"{agent.log_prefix}⚠️  Skipping session persistence "
                            f"for large failed session to prevent growth loop.",
                            force=True,
                        )
                    else:
                        agent._persist_session(messages, conversation_history)
                    if classified.reason == FailoverReason.content_policy_blocked:
                        _summary = agent._summarize_api_error(api_error)
                        _policy_response = (
                            f"⚠️  The model provider's safety filter blocked this request "
                            f"(not a Hermes/gateway failure).\n\n"
                            f"Provider message: {_summary}\n\n"
                            f"Try rephrasing the request, narrowing the context, or "
                            f"adding a fallback provider with `hermes fallback add`."
                        )
                        return {
                            "final_response": _policy_response,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "failed": True,
                            "error": f"content_policy_blocked: {_summary}",
                        }
                    return {
                        "final_response": None,
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "failed": True,
                        "error": str(api_error),
                    }

                if retry_count >= max_retries:
                    # 在回退之前，对瞬态传输错误（陈旧的连接池、TCP reset）
                    # 尝试重建主客户端一次。每次 API 调用块仅尝试一次。
                    if not primary_recovery_attempted and agent._try_recover_primary_transport(
                        api_error, retry_count=retry_count, max_retries=max_retries,
                    ):
                        primary_recovery_attempted = True
                        retry_count = 0
                        continue
                    # 在完全放弃之前尝试回退
                    agent._buffer_status(f"⚠️ Max retries ({max_retries}) exhausted — trying fallback...")
                    if agent._try_activate_fallback():
                        retry_count = 0
                        compression_attempts = 0
                        primary_recovery_attempted = False
                        continue
                    # 终端 —— 刷新缓冲的重试/回退跟踪。
                    agent._flush_status_buffer()
                    _final_summary = agent._summarize_api_error(api_error)
                    _billing_guidance = ""
                    if classified.reason == FailoverReason.billing:
                        agent._emit_status(f"❌ Billing or credits exhausted — {_final_summary}")
                        _billing_guidance = _billing_or_entitlement_message(
                            capability="model access",
                            provider=_provider,
                            base_url=str(_base),
                            model=_model,
                        )
                        _print_billing_or_entitlement_guidance(
                            agent,
                            capability="model access",
                            provider=_provider,
                            base_url=str(_base),
                            model=_model,
                        )
                    elif is_rate_limited:
                        agent._emit_status(f"❌ Rate limited after {max_retries} retries — {_final_summary}")
                    else:
                        agent._emit_status(f"❌ API failed after {max_retries} retries — {_final_summary}")
                    agent._vprint(f"{agent.log_prefix}   💀 Final error: {_final_summary}", force=True)

                    # 检测 SSE 流中断模式（如 "Network connection lost"）
                    # 并显示可操作的指引。这通常发生在模型生成非常大的
                    # 工具调用（write_file 含大内容）时，代理/CDN 在
                    # 响应中途断开流。
                    _is_stream_drop = (
                        not getattr(api_error, "status_code", None)
                        and any(p in error_msg for p in (
                            "connection lost", "connection reset",
                            "connection closed", "network connection",
                            "network error", "terminated",
                        ))
                    )
                    if _is_stream_drop:
                        agent._vprint(
                            f"{agent.log_prefix}   💡 The provider's stream "
                            f"connection keeps dropping. This often happens "
                            f"when the model tries to write a very large "
                            f"file in a single tool call.",
                            force=True,
                        )
                        agent._vprint(
                            f"{agent.log_prefix}      Try asking the model "
                            f"to use execute_code with Python's open() for "
                            f"large files, or to write the file in smaller "
                            f"sections.",
                            force=True,
                        )

                    logger.error(
                        "%sAPI call failed after %s retries. %s | provider=%s model=%s msgs=%s tokens=~%s",
                        agent.log_prefix, max_retries, _final_summary,
                        _provider, _model, len(api_messages), f"{approx_tokens:,}",
                    )
                    if api_kwargs is not None:
                        agent._dump_api_request_debug(
                            api_kwargs, reason="max_retries_exhausted", error=api_error,
                        )
                    agent._persist_session(messages, conversation_history)
                    if classified.reason == FailoverReason.billing:
                        _final_response = f"Billing or credits exhausted: {_final_summary}"
                        if _billing_guidance:
                            _final_response += f"\n\n{_billing_guidance}"
                    else:
                        _final_response = f"API call failed after {max_retries} retries: {_final_summary}"
                    if _is_stream_drop:
                        _final_response += (
                            "\n\nThe provider's stream connection keeps "
                            "dropping — this often happens when generating "
                            "very large tool call responses (e.g. write_file "
                            "with long content). Try asking me to use "
                            "execute_code with Python's open() for large "
                            "files, or to write in smaller sections."
                        )
                    return {
                        "final_response": _final_response,
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "failed": True,
                        "error": _final_summary,
                    }

                # 对于速率限制，如果存在则遵守 Retry-After 头
                _retry_after = None
                if is_rate_limited:
                    _resp_headers = getattr(getattr(api_error, "response", None), "headers", None)
                    if _resp_headers and hasattr(_resp_headers, "get"):
                        _ra_raw = _resp_headers.get("retry-after") or _resp_headers.get("Retry-After")
                        if _ra_raw:
                            try:
                                _retry_after = min(float(_ra_raw), 120)  # Cap at 2 minutes
                            except (TypeError, ValueError):
                                pass
                wait_time = _retry_after if _retry_after else jittered_backoff(retry_count, base_delay=2.0, max_delay=60.0)
                if is_rate_limited:
                    agent._buffer_status(f"⏱️ Rate limited. Waiting {wait_time:.1f}s (attempt {retry_count + 1}/{max_retries})...")
                else:
                    agent._buffer_status(f"⏳ Retrying in {wait_time:.1f}s (attempt {retry_count}/{max_retries})...")
                logger.warning(
                    "Retrying API call in %ss (attempt %s/%s) %s error=%s",
                    wait_time,
                    retry_count,
                    max_retries,
                    agent._client_log_context(),
                    api_error,
                )
                # 以小增量睡眠，使我们可以快速响应中断，
                # 而非在一个 sleep() 调用中阻塞整个 wait_time
                sleep_end = time.time() + wait_time
                _backoff_touch_counter = 0
                while time.time() < sleep_end:
                    if agent._interrupt_requested:
                        agent._vprint(f"{agent.log_prefix}⚡ Interrupt detected during retry wait, aborting.", force=True)
                        agent._persist_session(messages, conversation_history)
                        agent.clear_interrupt()
                        return {
                            "final_response": f"Operation interrupted: retrying API call after error (retry {retry_count}/{max_retries}).",
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "interrupted": True,
                        }
                    time.sleep(0.2)  # 每 200ms 检查一次中断
                    # 每约 30 秒触碰活动计数器，使网关的不活动监控
                    # 知道我们在退避等待期间仍然存活。
                    _backoff_touch_counter += 1
                    if _backoff_touch_counter % 150 == 0:  # 150 × 0.2s = 30s
                        agent._touch_activity(
                            f"error retry backoff ({retry_count}/{max_retries}), "
                            f"{int(sleep_end - time.time())}s remaining"
                        )
        
        # 如果 API 调用被中断，跳过响应处理
        if interrupted:
            _turn_exit_reason = "interrupted_during_api_call"
            break

        # 以压缩消息重新开始 —— 回退 API 调用计数和迭代预算
        if restart_with_compressed_messages:
            api_call_count -= 1
            agent.iteration_budget.refund()
            # 将压缩重启计入重试限制，防止在压缩减少了消息
            # 但仍不足以放入上下文窗口时无限循环。
            retry_count += 1
            restart_with_compressed_messages = False
            continue

        # 以长度续传重新开始 —— 逐步提升输出 token 预算
        if restart_with_length_continuation:
            # 每次重试递增输出 token 预算。
            # 重试 1 → 2× 基础值, 重试 2 → 3× 基础值, 上限 32768。
            # 通过 _ephemeral_max_output_tokens 应用于所有提供商。
            _boost_base = agent.max_tokens if agent.max_tokens else 4096
            _boost = _boost_base * (length_continue_retries + 1)
            agent._ephemeral_max_output_tokens = min(_boost, 32768)
            continue

        # 保护：如果所有重试都耗尽但没有成功响应
        # （例如重复的上下文长度错误耗尽了 retry_count），
        # `response` 变量仍为 None。干净地跳出循环。
        if response is None:
            _turn_exit_reason = "all_retries_exhausted_no_response"
            print(f"{agent.log_prefix}❌ All API retries exhausted with no successful response.")
            agent._persist_session(messages, conversation_history)
            break

        try:
            _transport = agent._get_transport()
            _normalize_kwargs = {}
            if agent.api_mode == "anthropic_messages":
                _normalize_kwargs["strip_tool_prefix"] = agent._is_anthropic_oauth
            # 规范化响应为统一的助手消息格式
            normalized = _transport.normalize_response(response, **_normalize_kwargs)
            assistant_message = normalized
            finish_reason = normalized.finish_reason

            # 规范化内容为字符串 —— 某些 OpenAI 兼容服务器
            # （llama-server 等）返回 dict 或 list 而非纯字符串，
            # 会导致下游 .strip() 调用崩溃。
            if assistant_message.content is not None and not isinstance(assistant_message.content, str):
                raw = assistant_message.content
                if isinstance(raw, dict):
                    assistant_message.content = raw.get("text", "") or raw.get("content", "") or json.dumps(raw)
                elif isinstance(raw, list):
                    # 多模态内容列表 —— 提取文本部分
                    parts = []
                    for part in raw:
                        if isinstance(part, str):
                            parts.append(part)
                        elif isinstance(part, dict) and part.get("type") == "text":
                            parts.append(part.get("text", ""))
                        elif isinstance(part, dict) and "text" in part:
                            parts.append(str(part["text"]))
                    assistant_message.content = "\n".join(parts)
                else:
                    assistant_message.content = str(raw)

            try:
                # 插件钩子：post_api_request —— API 请求完成后触发
                from hermes_cli.plugins import invoke_hook as _invoke_hook
                _assistant_tool_calls = getattr(assistant_message, "tool_calls", None) or []
                _assistant_text = assistant_message.content or ""
                _invoke_hook(
                    "post_api_request",
                    task_id=effective_task_id,
                    session_id=agent.session_id or "",
                    platform=agent.platform or "",
                    model=agent.model,
                    provider=agent.provider,
                    base_url=agent.base_url,
                    api_mode=agent.api_mode,
                    api_call_count=api_call_count,
                    api_duration=api_duration,
                    finish_reason=finish_reason,
                    message_count=len(api_messages),
                    response_model=getattr(response, "model", None),
                    response=response,
                    usage=agent._usage_summary_for_api_request_hook(response),
                    assistant_message=assistant_message,
                    assistant_content_chars=len(_assistant_text),
                    assistant_tool_call_count=len(_assistant_tool_calls),
                )
            except Exception:
                pass

            # 处理助手响应
            if assistant_message.content and not agent.quiet_mode:
                if agent.verbose_logging:
                    agent._vprint(f"{agent.log_prefix}🤖 Assistant: {assistant_message.content}")
                else:
                    agent._vprint(f"{agent.log_prefix}🤖 Assistant: {assistant_message.content[:100]}{'...' if len(assistant_message.content) > 100 else ''}")

            # 通知进度回调模型的思考（用于子智能体委托，
            # 将子智能体的推理传递给父级显示）。
            if (assistant_message.content and agent.tool_progress_callback):
                _think_text = assistant_message.content.strip()
                # 剥离不应泄漏到父级显示的推理 XML 标签
                _think_text = re.sub(
                    r'</?(?:REASONING_SCRATCHPAD|think|reasoning)>', '', _think_text
                ).strip()
                # 对于子智能体：将第一行传递给父级显示（保留现有行为）。
                # 对于所有带结构化回调的智能体：发射 reasoning.available 事件。
                first_line = _think_text.split('\n')[0][:80] if _think_text else ""
                if first_line and getattr(agent, '_delegate_depth', 0) > 0:
                    try:
                        agent.tool_progress_callback("_thinking", first_line)
                    except Exception:
                        pass
                elif _think_text:
                    try:
                        agent.tool_progress_callback("reasoning.available", "_thinking", _think_text[:500], None)
                    except Exception:
                        pass
            
            # 检查不完整的 <REASONING_SCRATCHPAD>（已打开但未关闭）
            # 这意味着模型在推理中途用完了输出 token —— 最多重试 2 次
            if has_incomplete_scratchpad(assistant_message.content or ""):
                agent._incomplete_scratchpad_retries += 1
                
                agent._buffer_vprint(f"⚠️  Incomplete <REASONING_SCRATCHPAD> detected (opened but never closed)")
                
                if agent._incomplete_scratchpad_retries <= 2:
                    agent._buffer_vprint(f"🔄 Retrying API call ({agent._incomplete_scratchpad_retries}/2)...")
                    # 不要追加损坏的消息，仅重试
                    continue
                else:
                    # 最大重试次数 - 丢弃本轮并保存为部分结果
                    agent._flush_status_buffer()
                    agent._vprint(f"{agent.log_prefix}❌ Max retries (2) for incomplete scratchpad. Saving as partial.", force=True)
                    agent._incomplete_scratchpad_retries = 0
                    
                    rolled_back_messages = agent._get_messages_up_to_last_assistant(messages)
                    agent._cleanup_task_resources(effective_task_id)
                    agent._persist_session(messages, conversation_history)
                    
                    return {
                        "final_response": None,
                        "messages": rolled_back_messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "partial": True,
                        "error": "Incomplete REASONING_SCRATCHPAD after 2 retries"
                    }
            
            # 重置不完整 scratchpad 计数器（收到完整响应时）
            agent._incomplete_scratchpad_retries = 0

            if agent.api_mode == "codex_responses" and finish_reason == "incomplete":
                agent._codex_incomplete_retries += 1

                interim_msg = agent._build_assistant_message(assistant_message, finish_reason)
                interim_has_content = bool((interim_msg.get("content") or "").strip())
                interim_has_reasoning = bool(interim_msg.get("reasoning", "").strip()) if isinstance(interim_msg.get("reasoning"), str) else False
                interim_has_codex_reasoning = bool(interim_msg.get("codex_reasoning_items"))
                interim_has_codex_message_items = bool(interim_msg.get("codex_message_items"))

                if (
                    interim_has_content
                    or interim_has_reasoning
                    or interim_has_codex_reasoning
                    or interim_has_codex_message_items
                ):
                    last_msg = messages[-1] if messages else None
                    # 重复检测：两条连续的不完整助手消息具有相同的内容和推理
                    # 会被合并。对于仅提供商状态的变更（加密推理项
                    # 项或可重放的消息 id/phases/statuses 不同而
                    # 可见内容/推理未变），同样比较这些不透明负载，
                    # 以免静默丢弃较新的续传状态。
                    last_codex_items = last_msg.get("codex_reasoning_items") if isinstance(last_msg, dict) else None
                    interim_codex_items = interim_msg.get("codex_reasoning_items")
                    last_codex_message_items = last_msg.get("codex_message_items") if isinstance(last_msg, dict) else None
                    interim_codex_message_items = interim_msg.get("codex_message_items")
                    duplicate_interim = (
                        isinstance(last_msg, dict)
                        and last_msg.get("role") == "assistant"
                        and last_msg.get("finish_reason") == "incomplete"
                        and (last_msg.get("content") or "") == (interim_msg.get("content") or "")
                        and (last_msg.get("reasoning") or "") == (interim_msg.get("reasoning") or "")
                        and last_codex_items == interim_codex_items
                        and last_codex_message_items == interim_codex_message_items
                    )
                    if not duplicate_interim:
                        messages.append(interim_msg)
                        agent._emit_interim_assistant_message(interim_msg)

                if agent._codex_incomplete_retries < 3:
                    if not agent.quiet_mode:
                        agent._vprint(f"{agent.log_prefix}↻ Codex response incomplete; continuing turn ({agent._codex_incomplete_retries}/3)")
                    agent._session_messages = messages
                    continue

                agent._codex_incomplete_retries = 0
                agent._persist_session(messages, conversation_history)
                return {
                    "final_response": None,
                    "messages": messages,
                    "api_calls": api_call_count,
                    "completed": False,
                    "partial": True,
                    "error": "Codex response remained incomplete after 3 continuation attempts",
                }
            elif hasattr(agent, "_codex_incomplete_retries"):
                agent._codex_incomplete_retries = 0
            
            # 检查工具调用
            if assistant_message.tool_calls:
                if not agent.quiet_mode:
                    agent._vprint(f"{agent.log_prefix}🔧 Processing {len(assistant_message.tool_calls)} tool call(s)...")

                if agent.verbose_logging:
                    for tc in assistant_message.tool_calls:
                        logging.debug(f"Tool call: {tc.function.name} with args: {tc.function.arguments[:200]}...")

                # 验证工具调用名称 —— 检测模型幻觉
                # 在验证前修复不匹配的工具名称
                for tc in assistant_message.tool_calls:
                    if tc.function.name not in agent.valid_tool_names:
                        repaired = agent._repair_tool_call(tc.function.name)
                        if repaired:
                            print(f"{agent.log_prefix}🔧 Auto-repaired tool name: '{tc.function.name}' -> '{repaired}'")
                            tc.function.name = repaired

                invalid_tool_calls = [
                    tc.function.name for tc in assistant_message.tool_calls
                    if tc.function.name not in agent.valid_tool_names
                ]

                if invalid_tool_calls:
                    # 跟踪无效工具调用的重试次数
                    agent._invalid_tool_retries += 1

                    # 向模型返回有用的错误 —— 模型可在下一轮自动纠正
                    available = ", ".join(sorted(agent.valid_tool_names))
                    invalid_name = invalid_tool_calls[0]
                    invalid_preview = invalid_name[:80] + "..." if len(invalid_name) > 80 else invalid_name
                    agent._buffer_vprint(f"⚠️  Unknown tool '{invalid_preview}' — sending error to model for agent-correction ({agent._invalid_tool_retries}/3)")

                    if agent._invalid_tool_retries >= 3:
                        agent._flush_status_buffer()
                        agent._vprint(f"{agent.log_prefix}❌ Max retries (3) for invalid tool calls exceeded. Stopping as partial.", force=True)
                        agent._invalid_tool_retries = 0
                        agent._persist_session(messages, conversation_history)
                        return {
                            "final_response": None,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "partial": True,
                            "error": f"Model generated invalid tool call: {invalid_preview}"
                        }

                    assistant_msg = agent._build_assistant_message(assistant_message, finish_reason)
                    messages.append(assistant_msg)
                    for tc in assistant_message.tool_calls:
                        if tc.function.name not in agent.valid_tool_names:
                            content = f"Tool '{tc.function.name}' does not exist. Available tools: {available}"
                        else:
                            content = "Skipped: another tool call in this turn used an invalid name. Please retry this tool call."
                        messages.append({
                            "role": "tool",
                            "name": tc.function.name,
                            "tool_call_id": tc.id,
                            "content": content,
                        })
                    continue

                # 在成功工具调用验证后重置重试计数器
                agent._invalid_tool_retries = 0

                # 验证工具调用参数为有效 JSON
                # 将空字符串处理为空对象（常见的模型怪癖）
                invalid_json_args = []
                for tc in assistant_message.tool_calls:
                    args = tc.function.arguments
                    if isinstance(args, (dict, list)):
                        tc.function.arguments = json.dumps(args)
                        continue
                    if args is not None and not isinstance(args, str):
                        tc.function.arguments = str(args)
                        args = tc.function.arguments
                    # 将空/空白字符串视为空对象
                    if not args or not args.strip():
                        tc.function.arguments = "{}"
                        continue
                    try:
                        json.loads(args)
                    except json.JSONDecodeError as e:
                        invalid_json_args.append((tc.function.name, str(e)))
                
                if invalid_json_args:
                    # 检查无效 JSON 是由于截断而非模型格式错误。
                    # 路由器有时会将 finish_reason 从 "length" 改写为
                    # "tool_calls"，对上方的长度处理器隐藏截断。
                    # 检测截断：参数（去除空白后）不以 } 或 ] 结尾
                    # 表示在流式传输中途被切断。
                    _truncated = any(
                        not (tc.function.arguments or "").rstrip().endswith(("}", "]"))
                        for tc in assistant_message.tool_calls
                        if tc.function.name in {n for n, _ in invalid_json_args}
                    )
                    if _truncated:
                        agent._vprint(
                            f"{agent.log_prefix}⚠️  Truncated tool call arguments detected "
                            f"(finish_reason={finish_reason!r}) — refusing to execute.",
                            force=True,
                        )
                        agent._invalid_json_retries = 0
                        agent._cleanup_task_resources(effective_task_id)
                        agent._persist_session(messages, conversation_history)
                        return {
                            "final_response": None,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "partial": True,
                            "error": "Response truncated due to output length limit",
                        }

                    # 跟踪无效 JSON 参数的重试次数
                    agent._invalid_json_retries += 1

                    tool_name, error_msg = invalid_json_args[0]
                    agent._buffer_vprint(f"⚠️  Invalid JSON in tool call arguments for '{tool_name}': {error_msg}")

                    if agent._invalid_json_retries < 3:
                        agent._buffer_vprint(f"🔄 Retrying API call ({agent._invalid_json_retries}/3)...")
                        # 不向消息中添加任何内容，仅重试 API 调用
                        continue
                    else:
                        # 不返回部分结果，而是注入工具错误结果使模型能恢复。
                        # 使用工具结果（而非用户消息）保持角色交替。
                        agent._buffer_vprint(f"⚠️  Injecting recovery tool results for invalid JSON...")
                        agent._invalid_json_retries = 0  # Reset for next attempt
                        
                        # 追加带（损坏的）tool_calls 的助手消息
                        recovery_assistant = agent._build_assistant_message(assistant_message, finish_reason)
                        messages.append(recovery_assistant)
                        
                        # 为每个工具调用回复工具错误结果
                        invalid_names = {name for name, _ in invalid_json_args}
                        for tc in assistant_message.tool_calls:
                            if tc.function.name in invalid_names:
                                err = next(e for n, e in invalid_json_args if n == tc.function.name)
                                tool_result = (
                                    f"Error: Invalid JSON arguments. {err}. "
                                    f"For tools with no required parameters, use an empty object: {{}}. "
                                    f"Please retry with valid JSON."
                                )
                            else:
                                tool_result = "Skipped: other tool call in this response had invalid JSON."
                            messages.append({
                                "role": "tool",
                                "name": tc.function.name,
                                "tool_call_id": tc.id,
                                "content": tool_result,
                            })
                        continue
                
                # 在成功 JSON 验证后重置重试计数器
                agent._invalid_json_retries = 0

                # ── 调用后防护栏 ──────────────────────────
                # 限制 delegate_task 调用数量并去重工具调用
                assistant_message.tool_calls = agent._cap_delegate_task_calls(
                    assistant_message.tool_calls
                )
                assistant_message.tool_calls = agent._deduplicate_tool_calls(
                    assistant_message.tool_calls
                )

                assistant_msg = agent._build_assistant_message(assistant_message, finish_reason)
                
                # 如果本轮同时有content和 tool_calls，将内容捕获为
                # 备用最终响应。常见模式：模型在提供答案的同时
                # 调用 memory/skill 工具作为副作用。
                # 如果工具调用后的后续轮次为空，我们使用此内容。
                turn_content = assistant_message.content or ""
                if turn_content and agent._has_content_after_think_block(turn_content):
                    agent._last_content_with_tools = turn_content
                    # 仅当本轮中的每个工具调用都是响应后的
                    # 内务处理（memory, todo, skill_manage 等）时才静默后续输出。
                    # 如果存在任何实质性工具（search_files, read_file,
                    # write_file, terminal, ...），保持输出可见使用户看到进度。
                    _HOUSEKEEPING_TOOLS = frozenset({
                        "memory", "todo", "skill_manage", "session_search",
                    })
                    _all_housekeeping = all(
                        tc.function.name in _HOUSEKEEPING_TOOLS
                        for tc in assistant_message.tool_calls
                    )
                    agent._last_content_tools_all_housekeeping = _all_housekeeping
                    if _all_housekeeping and agent._has_stream_consumers():
                        agent._mute_post_response = True
                    elif agent._should_emit_quiet_tool_messages():
                        clean = agent._strip_think_blocks(turn_content).strip()
                        if clean:
                            agent._vprint(f"  ┊ 💬 {clean}")
                
                # 在追加之前弹出纯思考预填充消息
                # (tool-call path — same rationale as the final-response path).
                _had_prefill = False
                while (
                    messages
                    and isinstance(messages[-1], dict)
                    and messages[-1].get("_thinking_prefill")
                ):
                    messages.pop()
                    _had_prefill = True

                # 当工具调用跟在预填充恢复之后时重置预填充计数器。
                # 不重置的话，计数器会在整个对话中累积 ——
                # 间歇性空的模型（empty → prefill → tools → empty → prefill →
                # tools）会耗尽两次预填充重试，第三次空响应则完全没有恢复机会。
                # 在此处重置使每次工具调用成功都被视为全新开始。
                if _had_prefill:
                    agent._thinking_prefill_retries = 0
                    agent._empty_content_retries = 0
                # 成功工具执行 —— 重置工具后提示标志，
                # 使其在模型后续工具轮次再次变空时能再次触发。
                agent._post_tool_empty_retried = False

                messages.append(assistant_msg)
                agent._emit_interim_assistant_message(assistant_msg)

                # 在工具执行开始前关闭任何已打开的流式显示
                # （响应框、推理框）。中间轮次可能已流式传输了早期内容
                # 打开了响应框；此处刷新防止其包裹工具 feed 行。
                # 仅通知显示回调 —— TTS（_stream_callback）不应接收
                # None（它将 None 用作流结束信号）。
                if agent.stream_delta_callback:
                    try:
                        agent.stream_delta_callback(None)
                    except Exception:
                        pass

                agent._execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count)

                if agent._tool_guardrail_halt_decision is not None:
                    decision = agent._tool_guardrail_halt_decision
                    _turn_exit_reason = "guardrail_halt"
                    final_response = agent._toolguard_controlled_halt_response(decision)
                    agent._emit_status(
                        f"⚠️ Tool guardrail halted {decision.tool_name}: {decision.code}"
                    )
                    messages.append({"role": "assistant", "content": final_response})
                    # 将停止消息发送给客户端，使其不会与崩溃无法区分。
                    # 流式显示在工具执行前已刷新（callback(None)），
                    # 但回调仍然存活 —— 通过它发送文本，
                    # 使 SSE/TUI 客户端能看到解释。
                    if final_response:
                        agent._safe_print(f"\n{final_response}\n")
                        if agent.stream_delta_callback:
                            try:
                                agent.stream_delta_callback(final_response)
                                agent.stream_delta_callback(None)
                            except Exception:
                                pass
                    break

                # 在成功工具执行后重置每轮重试计数器，
                # 使单次截断不会毒害整个对话。
                truncated_tool_call_retries = 0

                # 信号表示下一段流式文本前需要段落换行。
                # 不立即发送，因为多个连续工具迭代会堆积
                # 冗余的空行。_fire_stream_delta() 会在下次
                # 真正文本到达时前置一个 "\n\n"。
                agent._stream_needs_break = True

                # 如果唯一调用的工具是 execute_code（编程式工具调用），
                # 退还迭代。这些是廉价的 RPC 式调用，不应消耗预算。
                _tc_names = {tc.function.name for tc in assistant_message.tool_calls}
                if _tc_names == {"execute_code"}:
                    agent.iteration_budget.refund()

                # 使用 API 响应的实际 token 数来决定压缩。
                # prompt_tokens + completion_tokens 是
                # 提供商报告的实际上下文大小加上助手轮次 ——
                # 下一个提示词的紧密下界。
                # 上方追加的工具结果尚未计入，但阈值（默认 50%）
                # 留有充足余量；如果工具结果推过阈值，
                # 下一次 API 调用会报告真实总数并触发压缩。
                #
                # 如果 last_prompt_tokens 为 0（API 断开后陈旧或
                # 提供商未返回使用数据），回退到粗略估算
                # 以避免错过压缩。否则会话可能在断开后无限增长，
                # 因为 should_compress(0) 永不触发。(issue #2153)
                _compressor = agent.context_compressor
                if _compressor.last_prompt_tokens > 0:
                    # 仅使用 prompt_tokens —— completion/reasoning
                    # token 不消耗上下文窗口空间。
                    # 思考模型（GLM-5.1, QwQ, DeepSeek R1）
                    # 会用推理膨胀 completion_tokens，导致过早压缩。(issue #12026)
                    _real_tokens = _compressor.last_prompt_tokens
                else:
                    # 包含工具 schema —— 启用 50+ 工具时这些会
                    # 增加 20-30K token，仅消息估算会遗漏，
                    # 可能跳过压缩超过配置的阈值 (issue #14695)。
                    _real_tokens = estimate_request_tokens_rough(
                        messages, tools=agent.tools or None
                    )

                if agent.compression_enabled and _compressor.should_compress(_real_tokens):
                    agent._safe_print("  ⟳ compacting context…")
                    messages, active_system_prompt = agent._compress_context(
                        messages, system_message,
                        approx_tokens=agent.context_compressor.last_prompt_tokens,
                        task_id=effective_task_id,
                    )
                    # 压缩创建了新会话 —— 清除历史使
                    # _flush_messages_to_session_db 将压缩后的消息
                    # 写入新会话（见飞行前压缩注释）。
                    conversation_history = None
                
                # 增量保存会话日志（即使被中断也能看到进度）
                agent._session_messages = messages

                # 继续循环等待下一个响应
                continue

            else:
                # 没有工具调用 —— 这是最终响应
                final_response = assistant_message.content or ""

                # 修复：在进入无工具调用分支时取消输出静默，
                # 使用户能看到空响应警告和恢复状态消息。
                # _mute_post_response 是在上一轮内务处理工具轮次中设置的，
                # 不应静默最终响应路径。
                agent._mute_post_response = False

                # 检查响应是否只有思考块而没有实际内容
                if not agent._has_content_after_think_block(final_response):
                    # ── 部分流恢复 ─────────────────────
                    # 如果在连接断开前已有内容流式传输给用户，
                    # 使用该内容作为最终响应，而非回退到上一轮
                    # 或浪费 API 调用进行重试。
                    _partial_streamed = (
                        getattr(agent, "_current_streamed_assistant_text", "") or ""
                    )
                    if agent._has_content_after_think_block(_partial_streamed):
                        _turn_exit_reason = "partial_stream_recovery"
                        _recovered = agent._strip_think_blocks(_partial_streamed).strip()
                        logger.info(
                            "Partial stream content delivered (%d chars) "
                            "— using as final response",
                            len(_recovered),
                        )
                        agent._emit_status(
                            "↻ Stream interrupted — using delivered content "
                            "as final response"
                        )
                        final_response = _recovered
                        agent._response_was_previewed = True
                        break

                    # 如果上一轮在内务处理工具调用时已传递了真实内容
                    # （如 "You're welcome!" + memory 保存），
                    # 模型没有更多要说的。立即使用之前的内容，
                    # 而非浪费 API 调用重试。
                    # 注意：仅当该轮中的所有工具都是内务处理
                    # （memory, todo 等）时使用此快捷方式。
                    # 当实质性工具被调用时（terminal, search_files 等），
                    # 内容可能是任务进行中的叙述
                    # （"I'll scan the directory..."），空后续意味着
                    # 模型出了问题 —— 让下方的工具后空响应提示处理。
                    fallback = getattr(agent, '_last_content_with_tools', None)
                    if fallback and getattr(agent, '_last_content_tools_all_housekeeping', False):
                        _turn_exit_reason = "fallback_prior_turn_content"
                        logger.info("Empty follow-up after tool calls — using prior turn content as final response")
                        agent._emit_status("↻ Empty response after tool calls — using earlier content as final answer")
                        agent._last_content_with_tools = None
                        agent._last_content_tools_all_housekeeping = False
                        agent._empty_content_retries = 0
                        # 不要修改助手消息内容 ——
                        # 旧代码注入了 "Calling the X tools..." 这
                        # 会毒化对话历史。仅使用
                        # 回退文本作为最终响应并跳出。
                        final_response = agent._strip_think_blocks(fallback).strip()
                        agent._response_was_previewed = True
                        break

                    # ── 工具调用后空响应提示 ───────────
                    # 模型在执行工具调用后返回空。涵盖两种情况：
                    #  (a) 完全没有上一轮内容 —— 模型沉默
                    #  (b) 上一轮有内容 + 实质性工具（上方回退被跳过，
                    #      因为内容是任务进行中的叙述，不是最终答案）
                    # 与其放弃，不如通过追加用户级提示来引导模型继续。
                    # 这是 issue #9400 的情况：较弱的模型（mimo-v2-pro,
                    # GLM-5 等）有时在工具结果后返回空而非继续下一步。
                    # 一次带提示的重试通常能修复。
                    _prior_was_tool = any(
                        m.get("role") == "tool"
                        for m in messages[-5:]  # check recent messages
                    )
                    # 检测 Qwen3/Ollama 风格的内容内思考块。
                    # Ollama 将 <think> 放在 content 字段中
                    # （不在 reasoning_content 中），因此下方的
                    # _has_structured 会遗漏。我们在此检查，使
                    # 工具调用后的纯思考响应路由到预填充而非提示。
                    _has_inline_thinking = bool(
                        re.search(
                            r'<think>|<thinking>|<reasoning>',
                            final_response or "",
                            re.IGNORECASE,
                        )
                    )
                    if (
                        _prior_was_tool
                        and not getattr(agent, "_post_tool_empty_retried", False)
                        and not _has_inline_thinking  # thinking model still working — let prefill handle
                    ):
                        agent._post_tool_empty_retried = True
                        # 清除过时的叙述，使其不会在后续空响应时重新浮现
                        # 在提示后的后续空响应时重新浮现。
                        agent._last_content_with_tools = None
                        agent._last_content_tools_all_housekeeping = False
                        logger.info(
                            "Empty response after tool calls — nudging model "
                            "to continue processing"
                        )
                        agent._buffer_status(
                            "⚠️ Model returned empty after tool calls — "
                            "nudging to continue"
                        )
                        # 先追加空助手消息以保持消息序列有效：
                        #   tool(result) → assistant("(empty)") → user(nudge)
                        # 否则会有 tool → user，大多数 API 会将此视为无效序列。
                        _nudge_msg = agent._build_assistant_message(assistant_message, finish_reason)
                        _nudge_msg["content"] = "(empty)"
                        _nudge_msg["_empty_recovery_synthetic"] = True
                        messages.append(_nudge_msg)
                        messages.append({
                            "role": "user",
                            "content": (
                                "You just executed tool calls but returned an "
                                "empty response. Please process the tool "
                                "results above and continue with the task."
                            ),
                            "_empty_recovery_synthetic": True,
                        })
                        continue

                    # ── 纯思考预填充续传 ──────────
                    # 模型产生了结构化推理（通过 API 字段）但没有可见文本内容。
                    # 与其放弃，不如原样追加助手消息并继续 ——
                    # 模型将在下一轮看到自己的推理并产生文本部分。
                    # 灵感来自 clawdbot 的 "incomplete-text" 恢复。
                    # 也涵盖 Qwen3/Ollama 的内容内 <think> 块
                    # （上方检测为 _has_inline_thinking）。
                    _has_structured = bool(
                        getattr(assistant_message, "reasoning", None)
                        or getattr(assistant_message, "reasoning_content", None)
                        or getattr(assistant_message, "reasoning_details", None)
                        or _has_inline_thinking
                    )
                    if _has_structured and agent._thinking_prefill_retries < 2:
                        agent._thinking_prefill_retries += 1
                        logger.info(
                            "Thinking-only response (no visible content) — "
                            "prefilling to continue (%d/2)",
                            agent._thinking_prefill_retries,
                        )
                        agent._buffer_status(
                            f"↻ Thinking-only response — prefilling to continue "
                            f"({agent._thinking_prefill_retries}/2)"
                        )
                        interim_msg = agent._build_assistant_message(
                            assistant_message, "incomplete"
                        )
                        interim_msg["_thinking_prefill"] = True
                        messages.append(interim_msg)
                        agent._session_messages = messages
                        continue

                    # ── 空响应重试 ──────────────────────
                    # 模型没有返回可用内容。最多重试 3 次再尝试回退。
                    # 涵盖真正空响应（无内容、无推理）和
                    # 预填充耗尽后的纯推理响应 —— 像 mimo-v2-pro
                    # 这样的模型总是通过 OpenRouter 填充推理字段，
                    # 因此旧的 `not _has_structured` 守卫在预填充后
                    # 阻止了所有推理模型的重试。
                    _truly_empty = not agent._strip_think_blocks(
                        final_response
                    ).strip()
                    _prefill_exhausted = (
                        _has_structured
                        and agent._thinking_prefill_retries >= 2
                    )
                    if _truly_empty and (not _has_structured or _prefill_exhausted) and agent._empty_content_retries < 3:
                        agent._empty_content_retries += 1
                        logger.warning(
                            "Empty response (no content or reasoning) — "
                            "retry %d/3 (model=%s)",
                            agent._empty_content_retries, agent.model,
                        )
                        agent._buffer_status(
                            f"⚠️ Empty response from model — retrying "
                            f"({agent._empty_content_retries}/3)"
                        )
                        continue

                    # ── 重试耗尽 —— 尝试回退提供商 ──
                    # 在放弃并返回 "(empty)" 之前，尝试切换到
                    # 回退链中的下一个提供商。涵盖模型
                    # （如 GLM-4.5-Air）因上下文降级或提供商问题
                    # 持续返回空的情况。
                    if _truly_empty and agent._fallback_chain:
                        logger.warning(
                            "Empty response after %d retries — "
                            "attempting fallback (model=%s, provider=%s)",
                            agent._empty_content_retries, agent.model,
                            agent.provider,
                        )
                        agent._buffer_status(
                            "⚠️ Model returning empty responses — "
                            "switching to fallback provider..."
                        )
                        if agent._try_activate_fallback():
                            agent._empty_content_retries = 0
                            agent._buffer_status(
                                f"↻ Switched to fallback: {agent.model} "
                                f"({agent.provider})"
                            )
                            logger.info(
                                "Fallback activated after empty responses: "
                                "now using %s on %s",
                                agent.model, agent.provider,
                            )
                            continue

                    # 重试和回退链都耗尽了（或未配置回退）。
                    # 落入 "(empty)" 终止。
                    # 显示缓冲的重试/回退跟踪，使用户看到在 "(empty)" 之前尝试了什么。
                    agent._flush_status_buffer()
                    _turn_exit_reason = "empty_response_exhausted"
                    reasoning_text = agent._extract_reasoning(assistant_message)
                    agent._drop_trailing_empty_response_scaffolding(messages)
                    assistant_msg = agent._build_assistant_message(assistant_message, finish_reason)
                    assistant_msg["content"] = "(empty)"
                    # 这是面向网关的失败标记，不是真正的助手内容。
                    # 持久化它会使后续的 "continue" 轮次重放
                    # assistant("(empty)") 好像它是有意义的模型响应，
                    # 这会使长工具密集会话陷入空响应循环。
                    assistant_msg["_empty_terminal_sentinel"] = True
                    messages.append(assistant_msg)

                    if reasoning_text:
                        reasoning_preview = reasoning_text[:500] + "..." if len(reasoning_text) > 500 else reasoning_text
                        logger.warning(
                            "Reasoning-only response (no visible content) "
                            "after exhausting retries and fallback. "
                            "Reasoning: %s", reasoning_preview,
                        )
                        agent._emit_status(
                            "⚠️ Model produced reasoning but no visible "
                            "response after all retries. Returning empty."
                        )
                    else:
                        logger.warning(
                            "Empty response (no content or reasoning) "
                            "after %d retries. No fallback available. "
                            "model=%s provider=%s",
                            agent._empty_content_retries, agent.model,
                            agent.provider,
                        )
                        agent._emit_status(
                            "❌ Model returned no content after all retries"
                            + (" and fallback attempts." if agent._fallback_chain else
                               ". No fallback providers configured.")
                        )

                    final_response = "(empty)"
                    break
                
                # 在成功内容到达时重置重试计数器/签名
                agent._empty_content_retries = 0
                agent._thinking_prefill_retries = 0
                # 成功内容到达 —— 清除此轮早期失败尝试缓冲的重试状态。
                agent._clear_status_buffer()

                if (
                    agent.api_mode == "codex_responses"
                    and agent.valid_tool_names
                    and codex_ack_continuations < 2
                    and agent._looks_like_codex_intermediate_ack(
                        user_message=user_message,
                        assistant_content=final_response,
                        messages=messages,
                    )
                ):
                    codex_ack_continuations += 1
                    interim_msg = agent._build_assistant_message(assistant_message, "incomplete")
                    messages.append(interim_msg)
                    agent._emit_interim_assistant_message(interim_msg)

                    continue_msg = {
                        "role": "user",
                        "content": (
                            "[System: Continue now. Execute the required tool calls and only "
                            "send your final answer after completing the task.]"
                        ),
                    }
                    messages.append(continue_msg)
                    agent._session_messages = messages
                    continue

                codex_ack_continuations = 0

                if truncated_response_parts:
                    final_response = "".join(truncated_response_parts) + final_response
                    truncated_response_parts = []
                    length_continue_retries = 0
                
                final_response = agent._strip_think_blocks(final_response).strip()
                
                final_msg = agent._build_assistant_message(assistant_message, finish_reason)

                # 在追加最终响应之前弹出纯思考预填充和空响应重试
                # 脚手架。这些内部轮次仅用于下一次 API 重试，
                # 不应成为持久转录上下文。
                while (
                    messages
                    and isinstance(messages[-1], dict)
                    and (
                        messages[-1].get("_thinking_prefill")
                        or messages[-1].get("_empty_recovery_synthetic")
                        or messages[-1].get("_empty_terminal_sentinel")
                    )
                ):
                    messages.pop()

                messages.append(final_msg)
                
                _turn_exit_reason = f"text_response(finish_reason={finish_reason})"
                if not agent.quiet_mode:
                    agent._safe_print(f"🎉 Conversation completed after {api_call_count} OpenAI-compatible API call(s)")
                break
            
        except Exception as e:
            error_msg = f"Error during OpenAI-compatible API call #{api_call_count}: {str(e)}"
            try:
                print(f"❌ {error_msg}")
            except (OSError, ValueError):
                logger.error(error_msg)

            # 以 ERROR 级别发出完整回溯，使其同时出现在
            # agent.log 和 errors.log 中。以前这是在 DEBUG 级别记录的，
            # 意味着间歇性的外循环失败无法复现 ——
            # 用户会在屏幕上看到一行摘要但无法恢复调用位置。
            # logger.exception() 自动包含回溯并以 ERROR 级别发出。
            logger.exception("Outer loop error in API call #%d", api_call_count)

            # 如果已追加了带 tool_calls 的助手消息，
            # API 期望每个 tool_call_id 都有一个 role="tool" 结果。
            # 为尚未回答的工具调用填充错误结果。
            for idx in range(len(messages) - 1, -1, -1):
                msg = messages[idx]
                if not isinstance(msg, dict):
                    break
                if msg.get("role") == "tool":
                    continue
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    answered_ids = {
                        m["tool_call_id"]
                        for m in messages[idx + 1:]
                        if isinstance(m, dict) and m.get("role") == "tool"
                    }
                    for tc in msg["tool_calls"]:
                        if not tc or not isinstance(tc, dict): continue
                        if tc["id"] not in answered_ids:
                            err_msg = {
                                "role": "tool",
                                "name": _ra().AIAgent._get_tool_call_name_static(tc),
                                "tool_call_id": tc["id"],
                                "content": f"Error executing tool: {error_msg}",
                            }
                            messages.append(err_msg)
                break
            
            # 非工具错误不需要注入合成消息。
            # 错误已打印给用户（上方行），重试循环继续。
            # 注入假的用户/助手消息会污染历史、消耗 token，
            # 并可能违反角色交替不变式。

            # 如果接近限制，跳出以避免无限循环
            if api_call_count >= agent.max_iterations - 1:
                _turn_exit_reason = f"error_near_max_iterations({error_msg[:80]})"
                final_response = f"I apologize, but I encountered repeated errors: {error_msg}"
                # 追加为助手消息以保持历史对会话恢复有效
                # （避免连续的用户消息）。
                messages.append({"role": "assistant", "content": final_response})
                break

    # 预算耗尽 —— 通过一次额外的无工具 API 调用请求模型总结。
    # _handle_max_iterations 注入一条用户消息并发起单次无工具请求。
    if final_response is None and (
        api_call_count >= agent.max_iterations
        or agent.iteration_budget.remaining <= 0
    ):
        # 预算耗尽 —— 通过一次额外的无工具 API 调用请求模型总结。
        # _handle_max_iterations 注入一条用户消息并发起单次无工具请求。
        _turn_exit_reason = f"max_iterations_reached({api_call_count}/{agent.max_iterations})"
        agent._emit_status(
            f"⚠️ Iteration budget exhausted ({api_call_count}/{agent.max_iterations}) "
            "— asking model to summarise"
        )
        if not agent.quiet_mode:
            agent._safe_print(
                f"\n⚠️  Iteration budget exhausted ({api_call_count}/{agent.max_iterations}) "
                "— requesting summary..."
            )
        final_response = agent._handle_max_iterations(messages, api_call_count)

        # 如果作为看板工作者运行，阻塞任务使调度器知道
        # 工作者无法完成（而非将其视为协议违规）。
        # 智能体循环在调用 _handle_max_iterations 前剥离了工具，
        # 因此模型无法自己调用 kanban_block —— 我们必须代为执行。
        _kanban_task = os.environ.get("HERMES_KANBAN_TASK")
        if _kanban_task:
            try:
                _ra().handle_function_call(
                    "kanban_block",
                    {
                        "task_id": _kanban_task,
                        "reason": (
                            f"Iteration budget exhausted "
                            f"({api_call_count}/{agent.max_iterations}) — "
                            "task could not complete within the allowed "
                            "iterations"
                        ),
                    },
                    task_id=effective_task_id,
                )
                logger.info(
                    "kanban_block called for task %s after iteration "
                    "exhaustion (%d/%d)",
                    _kanban_task, api_call_count, agent.max_iterations,
                )
            except Exception:
                logger.warning(
                    "Failed to call kanban_block after iteration "
                    "exhaustion for task %s",
                    _kanban_task,
                    exc_info=True,
                )

    # 确定对话是否成功完成
    completed = (
        final_response is not None
        and api_call_count < agent.max_iterations
        and not failed
    )

    # 如果启用则保存轨迹。``user_message`` 可能是多模态
    # 部分列表；轨迹格式需要纯字符串。
    agent._save_trajectory(messages, _summarize_user_message_for_log(user_message), completed)

    # 对话完成后清理此任务的虚拟机和浏览器
    agent._cleanup_task_resources(effective_task_id)

    # 仅在私有重试脚手架被移除后才将会话持久化到
    # JSON 日志和 SQLite。否则后续的 "continue" 轮次
    # 可能重放 assistant("(empty)") / 恢复提示并陷入
    # 同样的空响应循环。
    agent._drop_trailing_empty_response_scaffolding(messages)
    agent._persist_session(messages, conversation_history)

    # ── 轮次结束诊断日志 ─────────────────────────────────────
    # 始终以 INFO 级别记录，使 agent.log 捕获每轮结束的原因。
    # 当最后一条消息是工具结果（智能体正在工作中）时，
    # 以 WARNING 记录 —— 这是用户报告的"突然停止"场景。
    _last_msg_role = messages[-1].get("role") if messages else None
    _last_tool_name = None
    if _last_msg_role == "tool":
        # 回退查找带工具调用的助手消息
        for _m in reversed(messages):
            if _m.get("role") == "assistant" and _m.get("tool_calls"):
                _tcs = _m["tool_calls"]
                if _tcs and isinstance(_tcs[0], dict):
                    _last_tool_name = _tcs[-1].get("function", {}).get("name")
                break

    _turn_tool_count = sum(
        1 for m in messages
        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")
    )
    _resp_len = len(final_response) if final_response else 0
    _budget_used = agent.iteration_budget.used if agent.iteration_budget else 0
    _budget_max = agent.iteration_budget.max_total if agent.iteration_budget else 0

    _diag_msg = (
        "Turn ended: reason=%s model=%s api_calls=%d/%d budget=%d/%d "
        "tool_turns=%d last_msg_role=%s response_len=%d session=%s"
    )
    _diag_args = (
        _turn_exit_reason, agent.model, api_call_count, agent.max_iterations,
        _budget_used, _budget_max,
        _turn_tool_count, _last_msg_role, _resp_len,
        agent.session_id or "none",
    )

    if _last_msg_role == "tool" and not interrupted:
        # 智能体正在工作中 —— 这是"突然停止"的情况
        logger.warning(
            "Turn ended with pending tool result (agent may appear stuck). "
            + _diag_msg + " last_tool=%s",
            *_diag_args, _last_tool_name,
        )
    else:
        logger.info(_diag_msg, *_diag_args)

    # 文件变更验证器页脚。
    # 如果本轮中一个或多个 ``write_file`` / ``patch`` 调用失败
    # 且从未被对同一路径的成功写入取代，在助手响应后追加
    # 建议性页脚。这捕获了特定情况 —— 由 Ben Eng 报告
    # （#15524 相关）—— 模型发出一批并行补丁，其中一半
    # 以 "Could not find old_string" 失败，但模型在总结本轮时
    # 声称每个文件都已编辑。用户随后必须手动运行 ``git status``
    # 来揭穿谎言。有了此页脚，真相在每轮都浮出水面，
    # 使模型无法在结构上过度宣称。
    #
    # 门控：仅在有真实文本响应且用户未中断时应用。
    # 空/中断轮次已有其他表面文本不应被增强。
    if final_response and not interrupted:
        try:
            _failed = getattr(agent, "_turn_failed_file_mutations", None) or {}
            if _failed and agent._file_mutation_verifier_enabled():
                footer = agent._format_file_mutation_failure_footer(_failed)
                if footer:
                    final_response = final_response.rstrip() + "\n\n" + footer
        except Exception as _ver_err:
            logger.debug("file-mutation verifier footer failed: %s", _ver_err)

    _response_transformed = False

    # 插件钩子：transform_llm_output
    # 每轮在工具调用循环完成后触发一次。
    # 插件可以在返回之前转换 LLM 的输出文本。
    # 第一个返回字符串的钩子获胜；None/空返回保持文本不变。
    if final_response and not interrupted:
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            _transform_results = _invoke_hook(
                "transform_llm_output",
                response_text=final_response,
                session_id=agent.session_id or "",
                model=agent.model,
                platform=getattr(agent, "platform", None) or "",
            )
            for _hook_result in _transform_results:
                if isinstance(_hook_result, str) and _hook_result:
                    final_response = _hook_result
                    _response_transformed = True
                    break  # First non-empty string wins
        except Exception as exc:
            logger.warning("transform_llm_output hook failed: %s", exc)

    # 插件钩子：post_llm_call
    # 每轮在工具调用循环完成后触发一次。
    # 插件可使用此钩子持久化对话数据（如同步到外部记忆系统）。
    if final_response and not interrupted:
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            _invoke_hook(
                "post_llm_call",
                session_id=agent.session_id,
                user_message=original_user_message,
                assistant_response=final_response,
                conversation_history=list(messages),
                model=agent.model,
                platform=getattr(agent, "platform", None) or "",
            )
        except Exception as exc:
            logger.warning("post_llm_call hook failed: %s", exc)

    # 仅从当前轮次提取推理。向后遍历但在到达开始本轮的用户
    # 消息时停止 —— 更早的内容来自上一轮，不应泄漏到推理框中
    # （混淆的过期显示；issue #17055）。在当前轮次内我们仍需要
    # *最近的*非空推理：许多提供商（Claude thinking, DeepSeek v4,
    # Codex Responses）在工具调用步骤发出推理而将最终回答步骤的
    # 推理设为 None，因此仅选取最后一个助手会静默丢弃
    # 合法的同一轮次推理。
    last_reasoning = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            break  # 轮次边界 —— 不要跨越到上一轮
        if msg.get("role") == "assistant" and msg.get("reasoning"):
            last_reasoning = msg["reasoning"]
            break

    # 构建结果字典，如果适用则包含中断信息
    result = {
        "final_response": final_response,
        "last_reasoning": last_reasoning,
        "messages": messages,
        "api_calls": api_call_count,
        "completed": completed,
        "turn_exit_reason": _turn_exit_reason,
        "failed": failed,
        "partial": False,  # 仅在因无效工具调用停止时为 True
        "interrupted": interrupted,
        "response_transformed": _response_transformed,
        "response_previewed": getattr(agent, "_response_was_previewed", False),
        "model": agent.model,
        "provider": agent.provider,
        "base_url": agent.base_url,
        "input_tokens": agent.session_input_tokens,
        "output_tokens": agent.session_output_tokens,
        "cache_read_tokens": agent.session_cache_read_tokens,
        "cache_write_tokens": agent.session_cache_write_tokens,
        "reasoning_tokens": agent.session_reasoning_tokens,
        "prompt_tokens": agent.session_prompt_tokens,
        "completion_tokens": agent.session_completion_tokens,
        "total_tokens": agent.session_total_tokens,
        "last_prompt_tokens": getattr(agent.context_compressor, "last_prompt_tokens", 0) or 0,
        "estimated_cost_usd": agent.session_estimated_cost_usd,
        "cost_status": agent.session_cost_status,
        "cost_source": agent.session_cost_source,
        "session_id": agent.session_id,
    }
    if agent._tool_guardrail_halt_decision is not None:
        result["guardrail"] = agent._tool_guardrail_halt_decision.to_metadata()
    # 如果 /steer 指令在最终助手轮次之后到达（没有更多工具批次
    # 可以注入），将其传回调用方，以便作为下一个用户轮次传递，
    # 而非被静默丢弃。
    _leftover_steer = agent._drain_pending_steer()
    if _leftover_steer:
        result["pending_steer"] = _leftover_steer
    agent._response_was_previewed = False

    # 如果有中断消息则包含在结果中
    if interrupted and agent._interrupt_message:
        result["interrupt_message"] = agent._interrupt_message

    # 处理完成后清除中断状态
    agent.clear_interrupt()

    # 清除流式回调，防止泄漏到后续调用
    agent._stream_callback = None

    # 在此处检查技能触发 —— 基于本轮使用了多少工具迭代。
    _should_review_skills = False
    if (agent._skill_nudge_interval > 0
            and agent._iters_since_skill >= agent._skill_nudge_interval
            and "skill_manage" in agent.valid_tool_names):
        _should_review_skills = True
        agent._iters_since_skill = 0

    # 外部记忆提供者：同步已完成的轮次 + 排队下一次预取。
    agent._sync_external_memory_for_turn(
        original_user_message=original_user_message,
        final_response=final_response,
        interrupted=interrupted,
        messages=messages,
    )

    # 后台记忆/技能审查 —— 在响应交付之后运行，
    # 使其永远不会与用户的任务竞争模型注意力。
    if final_response and not interrupted and (_should_review_memory or _should_review_skills):
        try:
            agent._spawn_background_review(
                messages_snapshot=list(messages),
                review_memory=_should_review_memory,
                review_skills=_should_review_skills,
            )
        except Exception:
            pass  # 后台审查是尽力而为的

    # 注意：记忆提供者的 on_session_end() + shutdown_all() 不在这里调用 ——
    # run_conversation() 在多轮会话中每条用户消息调用一次。
    # 每轮后关闭会在第二条消息之前杀死提供者。
    # 实际的会话结束清理由 CLI（atexit / /reset）和
    # 网关（会话过期 / _reset_session）处理。

    # 插件钩子：on_session_end
    # 在每次 run_conversation 调用的最末尾触发。
    # 插件可使用此钩子进行清理、刷新缓冲区等。
    try:
        from hermes_cli.plugins import invoke_hook as _invoke_hook
        _invoke_hook(
            "on_session_end",
            session_id=agent.session_id,
            completed=completed,
            interrupted=interrupted,
            model=agent.model,
            platform=getattr(agent, "platform", None) or "",
        )
    except Exception as exc:
        logger.warning("on_session_end hook failed: %s", exc)

    return result



__all__ = ["run_conversation"]
