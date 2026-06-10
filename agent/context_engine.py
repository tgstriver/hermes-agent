"""可插拔上下文引擎的抽象基类。

上下文引擎控制着在接近模型 token 限制时如何管理对话上下文。
内置的 ContextCompressor 是默认实现。第三方引擎（如 LCM）可以通过
插件系统替换它，或者放置在 ``plugins/context_engine/<name>/`` 目录下。

选择由配置驱动：config.yaml 中的 ``context.engine`` 字段。默认值为 ``"compressor"``（内置）。同一时间仅激活一个引擎。

引擎负责：
  - 决定何时触发上下文压缩（compaction）
  - 执行压缩操作（摘要生成、DAG 构建等）
  - 可选地暴露 Agent 可调用的工具（如 lcm_grep）
  - 追踪来自 API 响应的 token 使用情况

生命周期：
  1. 引擎实例化并注册（通过插件的 register() 方法或默认注册）
  2. 会话开始时调用 on_session_start()
  3. 每次 API 响应后调用 update_from_response() 传入使用量数据
  4. 每轮对话后检查 should_compress()
  5. 当 should_compress() 返回 True 时调用 compress()
  6. 在真实会话边界处调用 on_session_end()（CLI 退出、/reset 命令、
     网关会话过期）—— 不是每轮对话都调用
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ContextEngine(ABC):
    """所有上下文引擎必须继承的基类。"""

    # -- 身份标识 ----------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """简短标识符（例如 'compressor'、'lcm'）。"""

    # -- Token 状态（run_agent.py 读取用于显示/日志） ------------
    #
    # 引擎必须维护以下字段。run_agent.py 会直接读取这些值。

    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    last_total_tokens: int = 0
    threshold_tokens: int = 0
    context_length: int = 0
    compression_count: int = 0

    # -- 压缩参数（run_agent.py 读取用于预检） --------
    #
    # 这些参数控制预检压缩检查。子类可通过 __init__ 或属性重写，
    # 默认值对大多数引擎都是合理的。
    #
    # protect_first_n 语义（自 PR #13754 起）：始终原样保留的非系统头部
    # 消息数量，此外系统提示词也始终被隐式保护。默认值 3 保留了
    # 历史常见的 "系统提示 + 前 3 条非系统消息" 的头部结构。

    threshold_percent: float = 0.75
    protect_first_n: int = 3
    protect_last_n: int = 6

    # -- 核心接口 ----------------------------------------------------

    @abstractmethod
    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """从 API 响应中更新跟踪的 token 使用量。

        在每次 LLM 调用后调用，传入一个标准化的 usage 字典。传统键名
        ``prompt_tokens``、``completion_tokens`` 和 ``total_tokens``
        始终存在。较新的宿主还会包含标准桶字段：
        ``input_tokens``、``output_tokens``、``cache_read_tokens``、
        ``cache_write_tokens`` 和 ``reasoning_tokens``。引擎应
        将这些新字段视为可选字段，以兼容旧版宿主。
        """

    @abstractmethod
    def should_compress(self, prompt_tokens: int = None) -> bool:
        """返回 True 表示本轮对话应当触发压缩。"""

    @abstractmethod
    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,
    ) -> List[Dict[str, Any]]:
        """压缩消息列表并返回新的消息列表。

        这是主入口方法。引擎接收完整的消息列表，并返回一个（可能更短的）
        符合上下文预算的消息列表。实现可以是摘要生成、构建 DAG 或
        其他任何方式 —— 只要返回的列表是合法的 OpenAI 格式消息序列即可。

        Args:
            focus_topic: 可选的主题字符串，来自手动 ``/compress <focus>`` 命令。
                支持引导压缩的引擎应优先保留与该主题相关的信息。
                不支持该特性的引擎可忽略此参数。
        """

    # -- 可选：预检检查 ----------------------------------------

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        """在 API 调用之前进行快速的粗略检查（此时还没有真实的 token 计数）。

        默认返回 False（跳过预检）。如果引擎能够实现低成本估算，请重写此方法。
        """
        return False

    # -- 可选：手动 /compress 预检 ----------------------------------------------

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        """快速检查：``messages`` 中是否有可以被压缩的内容？

        网关的 ``/compress`` 命令用作预检保护 ——
        返回 False 可以让网关报告 "暂无内容可压缩"，
        而无需发起 LLM 调用。

        默认返回 True（总是尝试）。如果引擎能够以低成本方式
        检查自身的首尾边界，应重写此方法，
        当对话记录完全处于保护范围内时返回 False。
        """
        return True

    # -- 可选：会话生命周期管理 ---------------------------------------

    def on_session_start(self, session_id: str, **kwargs) -> None:
        """在新对话会话开始时调用。

        用于加载会话的持久化状态（DAG、存储等）。
        kwargs 可能包含 hermes_home、platform、model 等信息。
        """

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """在真实会话边界处调用（CLI 退出、/reset 命令、网关过期）。

        用于刷新状态、关闭数据库连接等。
        不在每轮对话中调用 —— 仅在会话真正结束时调用。
        """

    def on_session_reset(self) -> None:
        """在 /new 或 /reset 时调用，重置每会话状态。

        默认操作是重置 compression_count 和 token 追踪。
        """
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0

    # -- 可选：工具接口 ---------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """返回本引擎提供给 Agent 的工具 schema。

        默认返回空列表（无工具）。LCM 会在此返回
        lcm_grep、lcm_describe、lcm_expand 等的 schema。
        """
        return []

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        """处理来自 Agent 的工具调用。

        仅当工具名由 get_tool_schemas() 返回时才会调用。
        必须返回 JSON 格式的字符串。

        kwargs 可能包含：
          messages: 当前内存中的消息列表（用于实时摄入）
        """
        import json
        return json.dumps({"error": f"Unknown context engine tool: {name}"})

    # -- 可选：状态/显示 ----------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """返回用于显示/日志的状态字典。

        默认返回 run_agent.py 期望的标准字段。
        """
        return {
            "last_prompt_tokens": self.last_prompt_tokens,
            "threshold_tokens": self.threshold_tokens,
            "context_length": self.context_length,
            "usage_percent": (
                min(100, self.last_prompt_tokens / self.context_length * 100)
                if self.context_length else 0
            ),
            "compression_count": self.compression_count,
        }

    # -- 可选：模型切换支持 ------------------------------------

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        """当用户切换模型或触发回退机制时调用。

        默认操作是更新 context_length 并根据 threshold_percent
        重新计算 threshold_tokens。如果引擎需要更多自定义逻辑
        （例如重新计算 DAG 预算、切换摘要模型），请重写此方法。
        """
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.threshold_percent)
