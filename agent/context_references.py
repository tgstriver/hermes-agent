from __future__ import annotations

import asyncio
import inspect
import json
import mimetypes
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from agent.model_metadata import estimate_tokens_rough

# 匹配带引号包裹的引用值，支持反引号、双引号、单引号
_QUOTED_REFERENCE_VALUE = r'(?:`[^`\n]+`|"[^"\n]+"|\'[^\'\n]+\')'
# 匹配 @ 引用语法：
# - @diff / @staged ：快捷引用（git diff / git diff --staged）
# - @file:path  / @file:`path:10-20` ：文件引用，支持行号范围
# - @folder:path ：文件夹引用
# - @git:N ：最近 N 次 git 提交 diff
# - @url:url ：远程 URL 内容
REFERENCE_PATTERN = re.compile(
    rf"(?<![\w/])@(?:(?P<simple>diff|staged)\b|(?P<kind>file|folder|git|url):(?P<value>{_QUOTED_REFERENCE_VALUE}(?::\d+(?:-\d+)?)?|\S+))"
)
# 需要从引用值末尾剥离的标点符号
TRAILING_PUNCTUATION = ",.;!?"
# 用户主目录下敏感目录列表（包含 SSH 密钥、云厂商凭证等）
_SENSITIVE_HOME_DIRS = (".ssh", ".aws", ".gnupg", ".kube", ".docker", ".azure", ".config/gh")
# Hermes 项目内部敏感目录
_SENSITIVE_HERMES_DIRS = (Path("skills") / ".hub",)
# 用户主目录下敏感文件列表（包含密钥、认证配置等）
_SENSITIVE_HOME_FILES = (
    Path(".ssh") / "authorized_keys",
    Path(".ssh") / "id_rsa",
    Path(".ssh") / "id_ed25519",
    Path(".ssh") / "config",
    Path(".bashrc"),
    Path(".zshrc"),
    Path(".profile"),
    Path(".bash_profile"),
    Path(".zprofile"),
    Path(".netrc"),
    Path(".pgpass"),
    Path(".npmrc"),
    Path(".pypirc"),
)


@dataclass(frozen=True)
# 上下文引用数据类，表示一个解析后的 @ 引用
class ContextReference:
    raw: str
    # 引用原始文本，例如 `@file:"agent/context_references.py":10-20`
    kind: str
    # 引用类型：file / folder / git / url / diff / staged
    target: str
    # 解析后的目标路径或参数
    start: int
    # 在原始消息中的起始字符索引
    end: int
    # 在原始消息中的结束字符索引
    line_start: int | None = None
    # 起始行号（仅 file 类型）
    line_end: int | None = None
    # 结束行号（仅 file 类型）


@dataclass
# 上下文引用预处理结果
class ContextReferenceResult:
    message: str
    # 剥离 @ 引用标记后、注入上下文前的消息文本
    original_message: str
    # 原始消息（未剥离任何内容）
    references: list[ContextReference] = field(default_factory=list)
    # 解析出的所有引用
    warnings: list[str] = field(default_factory=list)
    # 警告信息（如 token 超限）
    injected_tokens: int = 0
    # 注入的上下文总 token 数
    expanded: bool = False
    # 是否成功注入上下文
    blocked: bool = False
    # 是否被硬限制（50% token）拦截


def parse_context_references(message: str) -> list[ContextReference]:
    """
    解析消息中的 @ 引用语法，提取所有引用信息。

    支持的语法：
    - @diff ：当前工作区未暂存的变更
    - @staged ：已暂存的变更
    - @file:path ：读取整个文件
    - @file:`path:10-20` ：读取文件的第 10-20 行
    - @folder:path ：列出文件夹内容
    - @git:3 ：最近 3 次提交的 diff（默认为 1）
    - @url:https://example.com ：抓取远程 URL 内容

    Args:
        message: 包含 @ 引用语法的原始消息

    Returns:
        解析出的 ContextReference 列表
    """
    refs: list[ContextReference] = []
    if not message:
        return refs

    for match in REFERENCE_PATTERN.finditer(message):
        simple = match.group("simple")
        if simple:
            # 处理 @diff / @staged 这类无参数的快捷引用
            refs.append(
                ContextReference(
                    raw=match.group(0),
                    kind=simple,
                    target="",
                    start=match.start(),
                    end=match.end(),
                )
            )
            continue

        kind = match.group("kind")
        # 剥离末尾的标点符号（如逗号、句号），避免将句子标点误认为路径一部分
        value = _strip_trailing_punctuation(match.group("value") or "")
        line_start = None
        line_end = None
        # 剥去包裹路径的引号
        target = _strip_reference_wrappers(value)

        if kind == "file":
            # 对于文件引用，额外解析行号范围
            target, line_start, line_end = _parse_file_reference_value(value)

        refs.append(
            ContextReference(
                raw=match.group(0),
                kind=kind,
                target=target,
                start=match.start(),
                end=match.end(),
                line_start=line_start,
                line_end=line_end,
            )
        )

    return refs


def preprocess_context_references(
    message: str,
    *,
    cwd: str | Path,
    context_length: int,
    url_fetcher: Callable[[str], str | Awaitable[str]] | None = None,
    allowed_root: str | Path | None = None,
) -> ContextReferenceResult:
    """
    同步版本的上下文引用预处理入口。

    该函数会自动检测当前是否存在运行中的事件循环：
    - 如果存在正在运行的循环（如在异步服务端中），则在线程池中调度异步执行
    - 否则直接使用 asyncio.run() 执行

    Args:
        message: 包含 @ 引用语法的消息
        cwd: 当前工作目录，用于解析相对路径
        context_length: 上下文总 token 长度上限，用于计算引用注入的限制
        url_fetcher: 自定义 URL 内容获取回调，未提供则使用默认的 web_extract_tool
        allowed_root: 允许引用的根目录，默认限制在当前工作目录内

    Returns:
        ContextReferenceResult，包含注入后的消息和警告信息
    """
    coro = preprocess_context_references_async(
        message,
        cwd=cwd,
        context_length=context_length,
        url_fetcher=url_fetcher,
        allowed_root=allowed_root,
    )
    # Safe for both CLI (no loop) and gateway (loop already running).
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


async def preprocess_context_references_async(
    message: str,
    *,
    cwd: str | Path,
    context_length: int,
    url_fetcher: Callable[[str], str | Awaitable[str]] | None = None,
    allowed_root: str | Path | None = None,
) -> ContextReferenceResult:
    """
    异步版本的上下文引用预处理核心逻辑。

    处理流程：
    1. 解析消息中所有的 @ 引用
    2. 逐个展开引用（读取文件、执行 git 命令、抓取 URL 等）
    3. 检查注入 token 是否超过限制（硬限制 50%，软警告 25%）
    4. 从原始消息中剥离 @ 引用标记
    5. 将展开的上下文附加到消息末尾

    Args:
        message: 包含 @ 引用语法的消息
        cwd: 当前工作目录
        context_length: 上下文 token 上限
        url_fetcher: URL 内容获取回调
        allowed_root: 允许的引用根目录

    Returns:
        处理后的 ContextReferenceResult
    """
    refs = parse_context_references(message)
    if not refs:
        return ContextReferenceResult(message=message, original_message=message)

    cwd_path = Path(cwd).expanduser().resolve()
    # 默认限制在当前工作目录内，防止 @ 引用逃逸到工作区外部。
    # 调用方可以显式放宽 allowed_root 限制。
    allowed_root_path = (
        Path(allowed_root).expanduser().resolve() if allowed_root is not None else cwd_path
    )
    warnings: list[str] = []
    blocks: list[str] = []
    injected_tokens = 0

    for ref in refs:
        warning, block = await _expand_reference(
            ref,
            cwd_path,
            url_fetcher=url_fetcher,
            allowed_root=allowed_root_path,
        )
        if warning:
            warnings.append(warning)
        if block:
            blocks.append(block)
            injected_tokens += estimate_tokens_rough(block)

    # 硬限制：注入 token 超过上下文 50% 则全部拒绝
    hard_limit = max(1, int(context_length * 0.50))
    soft_limit = max(1, int(context_length * 0.25))
    if injected_tokens > hard_limit:
        warnings.append(
            f"@ context injection refused: {injected_tokens} tokens exceeds the 50% hard limit ({hard_limit})."
        )
        return ContextReferenceResult(
            message=message,
            original_message=message,
            references=refs,
            warnings=warnings,
            injected_tokens=injected_tokens,
            expanded=False,
            blocked=True,
        )

    # 软限制：注入 token 超过 25% 时发出警告
    if injected_tokens > soft_limit:
        warnings.append(
            f"@ context injection warning: {injected_tokens} tokens exceeds the 25% soft limit ({soft_limit})."
        )

    # 从原始消息中剥离 @ 引用标记
    stripped = _remove_reference_tokens(message, refs)
    final = stripped
    if warnings:
        final = f"{final}\n\n--- Context Warnings ---\n" + "\n".join(f"- {warning}" for warning in warnings)
    if blocks:
        final = f"{final}\n\n--- Attached Context ---\n\n" + "\n\n".join(blocks)

    return ContextReferenceResult(
        message=final.strip(),
        original_message=message,
        references=refs,
        warnings=warnings,
        injected_tokens=injected_tokens,
        expanded=bool(blocks or warnings),
        blocked=False,
    )


async def _expand_reference(
    ref: ContextReference,
    cwd: Path,
    *,
    url_fetcher: Callable[[str], str | Awaitable[str]] | None = None,
    allowed_root: Path | None = None,
) -> tuple[str | None, str | None]:
    """
    根据引用类型展开对应的内容。

    每种类型返回 (warning, block) 二元组：
    - warning: 非 None 时表示出现警告（文件不存在、路径不可访问等）
    - block: 非 None 时表示展开的内容块，将被注入到消息中

    Args:
        ref: 解析后的引用对象
        cwd: 当前工作目录
        url_fetcher: URL 内容获取回调
        allowed_root: 允许的根目录

    Returns:
        (warning, block) 二元组
    """
    try:
        if ref.kind == "file":
            return _expand_file_reference(ref, cwd, allowed_root=allowed_root)
        if ref.kind == "folder":
            return _expand_folder_reference(ref, cwd, allowed_root=allowed_root)
        if ref.kind == "diff":
            return _expand_git_reference(ref, cwd, ["diff"], "git diff")
        if ref.kind == "staged":
            return _expand_git_reference(ref, cwd, ["diff", "--staged"], "git diff --staged")
        if ref.kind == "git":
            # 最近 N 次提交，N 取值范围 [1, 10]
            count = max(1, min(int(ref.target or "1"), 10))
            return _expand_git_reference(ref, cwd, ["log", f"-{count}", "-p"], f"git log -{count} -p")
        if ref.kind == "url":
            content = await _fetch_url_content(ref.target, url_fetcher=url_fetcher)
            if not content:
                return f"{ref.raw}: no content extracted", None
            return None, f"🌐 {ref.raw} ({estimate_tokens_rough(content)} tokens)\n{content}"
    except Exception as exc:
        return f"{ref.raw}: {exc}", None

    return f"{ref.raw}: unsupported reference type", None


def _expand_file_reference(
    ref: ContextReference,
    cwd: Path,
    *,
    allowed_root: Path | None = None,
) -> tuple[str | None, str | None]:
    """
    展开文件引用，读取文件内容并以代码块格式返回。

    功能：
    - 解析文件路径并校验路径安全
    - 检测二进制文件并拒绝读取
    - 支持按行号范围截取
    - 自动检测文件语言用于语法高亮

    Args:
        ref: 文件引用对象
        cwd: 当前工作目录
        allowed_root: 允许的根目录

    Returns:
        (warning, block) 二元组
    """
    path = _resolve_path(cwd, ref.target, allowed_root=allowed_root)
    _ensure_reference_path_allowed(path)
    if not path.exists():
        return f"{ref.raw}: file not found", None
    if not path.is_file():
        return f"{ref.raw}: path is not a file", None
    if _is_binary_file(path):
        return f"{ref.raw}: binary files are not supported", None

    text = path.read_text(encoding="utf-8")
    # 如果指定了行号范围，截取对应行
    if ref.line_start is not None:
        lines = text.splitlines()
        start_idx = max(ref.line_start - 1, 0)
        end_idx = min(ref.line_end or ref.line_start, len(lines))
        text = "\n".join(lines[start_idx:end_idx])

    lang = _code_fence_language(path)
    label = ref.raw
    return None, f"📄 {label} ({estimate_tokens_rough(text)} tokens)\n```{lang}\n{text}\n```"


def _expand_folder_reference(
    ref: ContextReference,
    cwd: Path,
    *,
    allowed_root: Path | None = None,
) -> tuple[str | None, str | None]:
    """
    展开文件夹引用，列出文件夹中的文件和子目录。

    功能：
    - 优先使用 rg (ripgrep) 快速列出文件
    - 回退到 os.walk 遍历
    - 自动过滤隐藏文件和 __pycache__
    - 限制最多返回 limit 个条目

    Args:
        ref: 文件夹引用对象
        cwd: 当前工作目录
        allowed_root: 允许的根目录

    Returns:
        (warning, block) 二元组
    """
    path = _resolve_path(cwd, ref.target, allowed_root=allowed_root)
    _ensure_reference_path_allowed(path)
    if not path.exists():
        return f"{ref.raw}: folder not found", None
    if not path.is_dir():
        return f"{ref.raw}: path is not a folder", None

    listing = _build_folder_listing(path, cwd)
    return None, f"📁 {ref.raw} ({estimate_tokens_rough(listing)} tokens)\n{listing}"


def _expand_git_reference(
    ref: ContextReference,
    cwd: Path,
    args: list[str],
    label: str,
) -> tuple[str | None, str | None]:
    """
    执行 git 命令并展开结果。

    支持的命令类型：
    - git diff ：当前变更
    - git diff --staged ：暂存变更
    - git log -N -p ：最近 N 次提交 diff

    命令超时 30 秒，无输出时返回 "(no output)"。

    Args:
        ref: git 引用对象
        cwd: 当前工作目录（git 命令执行目录）
        args: git 命令的参数列表
        label: 用于显示的命令标签

    Returns:
        (warning, block) 二元组
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return f"{ref.raw}: git command timed out (30s)", None
    if result.returncode != 0:
        stderr = (result.stderr or "").strip() or "git command failed"
        return f"{ref.raw}: {stderr}", None
    content = result.stdout.strip()
    if not content:
        content = "(no output)"
    return None, f"🧾 {label} ({estimate_tokens_rough(content)} tokens)\n```diff\n{content}\n```"


async def _fetch_url_content(
    url: str,
    *,
    url_fetcher: Callable[[str], str | Awaitable[str]] | None = None,
) -> str:
    """
    获取 URL 内容。

    使用 url_fetcher 回调（或默认的 _default_url_fetcher）获取远程页面内容。
    支持同步和异步两种 fetcher。

    Args:
        url: 要抓取的 URL
        url_fetcher: 自定义内容获取回调

    Returns:
        提取后的页面文本内容
    """
    fetcher = url_fetcher or _default_url_fetcher
    content = fetcher(url)
    if inspect.isawaitable(content):
        content = await content
    return str(content or "").strip()


async def _default_url_fetcher(url: str) -> str:
    """
    默认的 URL 内容抓取器。

    调用 web_extract_tool 提取页面内容，返回 markdown 格式。
    从返回的 JSON payload 中提取第一个文档的内容。

    Args:
        url: 要抓取的 URL

    Returns:
        提取后的页面内容（优先使用 content 字段，fallback 到 raw_content）
    """
    from tools.web_tools import web_extract_tool

    raw = await web_extract_tool([url], format="markdown", use_llm_processing=True)
    payload = json.loads(raw)
    docs = payload.get("data", {}).get("documents", [])
    if not docs:
        return ""
    doc = docs[0]
    return str(doc.get("content") or doc.get("raw_content") or "").strip()


def _resolve_path(cwd: Path, target: str, *, allowed_root: Path | None = None) -> Path:
    """
    解析引用目标为绝对路径，并校验其在允许的根目录范围内。

    处理逻辑：
    1. 展开用户目录缩写（~）
    2. 相对路径拼接 cwd
    3. 解析为绝对路径
    4. 如果指定 allowed_root，确保路径在其范围内

    Args:
        cwd: 当前工作目录
        target: 引用目标路径
        allowed_root: 允许的根目录

    Returns:
        解析后的绝对路径

    Raises:
        ValueError: 路径超出允许的根目录范围
    """
    path = Path(os.path.expanduser(target))
    if not path.is_absolute():
        path = cwd / path
    resolved = path.resolve()
    if allowed_root is not None:
        try:
            resolved.relative_to(allowed_root)
        except ValueError as exc:
            raise ValueError("path is outside the allowed workspace") from exc
    return resolved


def _ensure_reference_path_allowed(path: Path) -> None:
    """
    校验引用路径是否安全，拒绝访问敏感文件和目录。

    拒绝访问：
    - 用户主目录下的敏感文件（SSH 密钥、shell 配置、认证配置等）
    - 用户主目录下的敏感目录（.ssh, .aws, .gnupg, .kube 等）
    - Hermes 项目内部敏感路径（skills/.hub, .env）

    Args:
        path: 要校验的绝对路径

    Raises:
        ValueError: 路径属于敏感文件或目录
    """
    from hermes_constants import get_hermes_home
    home = Path(os.path.expanduser("~")).resolve()
    hermes_home = get_hermes_home().resolve()

    blocked_exact = {home / rel for rel in _SENSITIVE_HOME_FILES}
    blocked_exact.add(hermes_home / ".env")
    blocked_dirs = [home / rel for rel in _SENSITIVE_HOME_DIRS]
    blocked_dirs.extend(hermes_home / rel for rel in _SENSITIVE_HERMES_DIRS)

    if path in blocked_exact:
        raise ValueError("path is a sensitive credential file and cannot be attached")

    for blocked_dir in blocked_dirs:
        try:
            path.relative_to(blocked_dir)
        except ValueError:
            continue
        raise ValueError("path is a sensitive credential or internal Hermes path and cannot be attached")


def _strip_trailing_punctuation(value: str) -> str:
    """
    从引用值末尾剥离标点符号。

    剥离逗号、句号、分号、感叹号、问号。
    同时剥离末尾的括号/方括号/花括号，但不匹配开闭括号数量时停止。
    例如：`@file:"path/to/file.py",` -> `@file:"path/to/file.py"`

    Args:
        value: 待清理的引用值

    Returns:
        去除末尾标点的值
    """
    stripped = value.rstrip(TRAILING_PUNCTUATION)
    while stripped.endswith((")", "]", "}")):
        closer = stripped[-1]
        opener = {")": "(", "]": "[", "}": "{"}[closer]
        if stripped.count(closer) > stripped.count(opener):
            stripped = stripped[:-1]
            continue
        break
    return stripped


def _strip_reference_wrappers(value: str) -> str:
    """
    剥离引用值外层的引号包裹。

    如果值的第一个和最后一个字符都是相同的引号（反引号、双引号、单引号），
    则将其去掉。例如：``path/to/file.py`` -> `path/to/file.py`

    Args:
        value: 待剥离引号的值

    Returns:
        去除外层引号后的值
    """
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "`\"'":
        return value[1:-1]
    return value


def _parse_file_reference_value(value: str) -> tuple[str, int | None, int | None]:
    """
    解析文件引用值中的行号范围。

    支持两种格式：
    1. 引号包裹 + 行号：`path/to/file.py:10-20`
    2. 直接路径 + 行号：path/to/file.py:10-20

    Args:
        value: 引用值（已剥离外层引号）

    Returns:
        (文件路径, 起始行号, 结束行号) 三元组，行号可能为 None
    """
    quoted_match = re.match(
        r'^(?P<quote>`|"|\')(?P<path>.+?)(?P=quote)(?::(?P<start>\d+)(?:-(?P<end>\d+))?)?$',
        value,
    )
    if quoted_match:
        line_start = quoted_match.group("start")
        line_end = quoted_match.group("end")
        return (
            quoted_match.group("path"),
            int(line_start) if line_start is not None else None,
            int(line_end or line_start) if line_start is not None else None,
        )

    range_match = re.match(r"^(?P<path>.+?):(?P<start>\d+)(?:-(?P<end>\d+))?$", value)
    if range_match:
        line_start = int(range_match.group("start"))
        return (
            range_match.group("path"),
            line_start,
            int(range_match.group("end") or range_match.group("start")),
        )

    return _strip_reference_wrappers(value), None, None


def _remove_reference_tokens(message: str, refs: list[ContextReference]) -> str:
    """
    从原始消息中移除 @ 引用标记文本。

    通过引用在消息中的位置区间，将引用标记从消息中删除，
    然后清理多余的空格和调整附着在引用前后的标点位置。

    Args:
        message: 原始消息
        refs: 解析出的引用列表

    Returns:
        移除引用标记后的消息文本
    """
    pieces: list[str] = []
    cursor = 0
    for ref in refs:
        pieces.append(message[cursor:ref.start])
        cursor = ref.end
    pieces.append(message[cursor:])
    text = "".join(pieces)
    # 合并多余空格
    text = re.sub(r"\s{2,}", " ", text)
    # 修正标点前导空格（如 "word , " -> "word,"）
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def _is_binary_file(path: Path) -> bool:
    """
    检测文件是否为二进制文件。

    检测方式：
    1. MIME 类型检查：非 text/ 开头且不在白名单扩展名列表中
    2. 内容检测：前 4096 字节包含空字节 (\x00)

    白名单扩展名：.py, .md, .txt, .json, .yaml, .yml, .toml, .js, .ts

    Args:
        path: 文件路径

    Returns:
        True 表示文件是二进制，False 表示文本文件
    """
    mime, _ = mimetypes.guess_type(path.name)
    if mime and not mime.startswith("text/") and not any(
        path.name.endswith(ext) for ext in (".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".js", ".ts")
    ):
        return True
    chunk = path.read_bytes()[:4096]
    return b"\x00" in chunk


def _build_folder_listing(path: Path, cwd: Path, limit: int = 200) -> str:
    """
    构建文件夹目录树文本。

    格式示例：
    ```
    agent/
      - context_engine.py (42 lines)
      - conversation_loop.py (28 lines)
      - subfolder/
        - deep_file.py (15 lines)
      - ...
    ```

    优先使用 rg (ripgrep) 快速列出文件，回退到 os.walk。

    Args:
        path: 要列出的文件夹路径
        cwd: 当前工作目录，用于计算相对路径
        limit: 最大条目数限制

    Returns:
        格式化后的文件夹列表文本
    """
    lines = [f"{path.relative_to(cwd)}/"]
    entries = _iter_visible_entries(path, cwd, limit=limit)
    for entry in entries:
        rel = entry.relative_to(cwd)
        indent = "  " * max(len(rel.parts) - len(path.relative_to(cwd).parts) - 1, 0)
        if entry.is_dir():
            lines.append(f"{indent}- {entry.name}/")
        else:
            meta = _file_metadata(entry)
            lines.append(f"{indent}- {entry.name} ({meta})")
    if len(entries) >= limit:
        lines.append("- ...")
    return "\n".join(lines)


def _iter_visible_entries(path: Path, cwd: Path, limit: int) -> list[Path]:
    """
    遍历文件夹，列出可见的文件和目录条目。

    过滤规则：
    - 隐藏文件/目录（以 . 开头）
    - __pycache__ 目录

    优先使用 rg (ripgrep) 快速列出文件。如果 rg 不可用，
    则回退到 os.walk 遍历。

    Args:
        path: 要遍历的文件夹路径
        cwd: 当前工作目录
        limit: 最大条目数

    Returns:
        可见文件/目录路径列表
    """
    rg_entries = _rg_files(path, cwd, limit=limit)
    if rg_entries is not None:
        output: list[Path] = []
        seen_dirs: set[Path] = set()
        for rel in rg_entries:
            full = cwd / rel
            for parent in full.parents:
                if parent == cwd or parent in seen_dirs or path not in {parent, *parent.parents}:
                    continue
                seen_dirs.add(parent)
                output.append(parent)
            output.append(full)
        return sorted({p for p in output if p.exists()}, key=lambda p: (not p.is_dir(), str(p)))

    output = []
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__")
        files = sorted(f for f in files if not f.startswith("."))
        root_path = Path(root)
        for d in dirs:
            output.append(root_path / d)
            if len(output) >= limit:
                return output
        for f in files:
            output.append(root_path / f)
            if len(output) >= limit:
                return output
    return output


def _rg_files(path: Path, cwd: Path, limit: int) -> list[Path] | None:
    """
    使用 rg (ripgrep) 快速列出指定文件夹下的所有文件。

    如果 rg 命令不可用或执行失败，返回 None 表示回退到其他方法。

    Args:
        path: 要搜索的文件夹路径（相对 cwd）
        cwd: 当前工作目录
        limit: 最大返回文件数

    Returns:
        文件相对路径列表，如果 rg 不可用则返回 None
    """
    try:
        result = subprocess.run(
            ["rg", "--files", str(path.relative_to(cwd))],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    files = [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    return files[:limit]


def _file_metadata(path: Path) -> str:
    """
    获取文件元数据描述。

    优先返回行数（文本文件），如果是二进制文件则返回文件大小。

    Args:
        path: 文件路径

    Returns:
        格式如 "42 lines" 或 "1024 bytes" 的元数据字符串
    """
    if _is_binary_file(path):
        return f"{path.stat().st_size} bytes"
    try:
        line_count = path.read_text(encoding="utf-8").count("\n") + 1
    except Exception:
        return f"{path.stat().st_size} bytes"
    return f"{line_count} lines"


def _code_fence_language(path: Path) -> str:
    """
    根据文件扩展名映射代码块语言标识。

    用于在 Markdown 代码块中指定语法高亮语言。

    支持的映射：
    .py -> python, .js -> javascript, .ts/.tsx -> typescript/tsx,
    .jsx -> jsx, .json -> json, .md -> markdown, .sh -> bash,
    .yml/.yaml -> yaml, .toml -> toml

    Args:
        path: 文件路径

    Returns:
        语言标识字符串，未知扩展名返回空字符串
    """
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".json": "json",
        ".md": "markdown",
        ".sh": "bash",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".toml": "toml",
    }
    return mapping.get(path.suffix.lower(), "")
