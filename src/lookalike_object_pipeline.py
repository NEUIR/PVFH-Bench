#!/usr/bin/env python3
"""
输入一张图片 -> 调用视觉推理 API -> 得到「易混淆物体对」记录（不写中间 JSON）。

批量跑图请用 src/1-batch_lookalike_pipeline.py：若 data 目录下已存在 presence_actual_object.json，
batch 会跳过 --sample 重新抽样，只根据 lookalike_dataset.json 里的 API 失败与解析失败项重试并合并。

环境变量（或 .env）：
  OPENAI_API_KEY   必填
  OPENAI_API_BASE  可选，默认 https://dashscope.aliyuncs.com/compatible-mode/v1
  VISION_MODEL     可选，默认 qwen3.5-plus
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

load_dotenv()

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen3.5-plus")

# API 调用失败时最多再重试 3 次（首次 + 3 次重试 = 最多 4 次请求）
API_MAX_RETRIES = 3

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRESENCE_ACTUAL_OBJECT_FILENAME = "presence_actual_object.json"

VISION_PROMPT = """看这张图。请找出画面里**一个看得清、从视觉上能稳定认成某一类**的物体：它在匆匆一瞥时容易被看成**另一类东西**，且那一类东西**在图中并不存在**。

硬性要求（必须同时满足）：
1) **视觉无歧义**：你选的这个实例，在画面里应能明确支持「前者」这一类；不能是那种「说它算前者也行、算后者也行」的含糊标签。
2) **对应物体不得出现在图中**：后者所指类别在**整幅画面内**不得存在任何应被如实标成该类别的独立物体（不能另有一件真正的「后者」）；只能存在「把前者错看成后者」的误判，不能是「图里本来就有后者」。
3) **互斥、非重叠（针对前者这一实例）**：就你选定的那一个实例而言，按常识与视觉证据**不应**如实算作后者。禁止同义词、别称混用；禁止上下位导致「该实例其实仍可算作后者」（例如它明明就是 wine bottle，却写 mistaken for bottle）。
4) **相似度**：前者与后者在外观上要足够像，匆匆一瞥才会混淆；但分类上必须是两个不同物体，且满足第 2 条（后者在图中无真实实例）。

请各选一个具体的英文物体名（短语）：前者=画面中真实存在且可证实的物体；后者=容易被误认成、但**在图中并不存在**该类真实物体的类别。

下列仅为「易混淆对」的风格示例，用来体会相似度与误判感；请严格依据本图内容选择，不要机械照搬图中不存在的物体：
- beer bottle - root beer bottle
- lemon - lime
- tennis ball - baseball
- seal - sea lion
- blueberry - grape
- moth - butterfly
- zucchini - cucumber
- crow - raven

最终只输出一行，格式严格为：实际物体-易被误认成的物体
只用英文物体名，不要解释，不要标点装饰，不要引号。快速思考！！！"""


def image_to_base64_data_uri(image_path: Path) -> str:
    raw = image_path.read_bytes()
    ext = image_path.suffix.lower() or ".jpg"
    mime = {
        ".jpg": "jpeg",
        ".jpeg": "jpeg",
        ".png": "png",
        ".gif": "gif",
        ".webp": "webp",
    }.get(ext, "jpeg")
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"


def get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("未设置 OPENAI_API_KEY（可在环境变量或 .env 中配置）")
    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)


def _api_error_is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        code = getattr(exc, "status_code", None)
        return code is not None and code >= 500
    return False


def call_vision_model(image_uri: str, model: str) -> Tuple[str, Dict[str, Any]]:
    """返回 (模型正文, 调试信息)。DashScope 上 Qwen 多模态默认思考时可能出现 content 为空，故关闭 thinking。"""
    client = get_client()
    create_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_uri}},
                ],
            }
        ],
        "max_tokens": 10240,
        "timeout": 120.0,
    }
    if "dashscope" in OPENAI_API_BASE.lower():
        create_kwargs["extra_body"] = {"enable_thinking": True}

    resp = None
    for attempt in range(API_MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(**create_kwargs)
            break
        except Exception as e:
            if not _api_error_is_retryable(e) or attempt >= API_MAX_RETRIES:
                raise
            time.sleep(min(2.0**attempt, 8.0))
    assert resp is not None

    ch = resp.choices[0]
    msg = ch.message
    text = (msg.content or "").strip()
    debug: Dict[str, Any] = {"finish_reason": ch.finish_reason}
    r = getattr(msg, "refusal", None)
    if r:
        debug["refusal"] = r
    if not text:
        dump = msg.model_dump(mode="json", exclude_none=True)
        debug["message_dump"] = dump
    return text, debug


def parse_pair_line(text: str) -> Optional[Tuple[str, str]]:
    """解析模型输出的一行：实际物体-易被误认成的物体（支持空格分隔的连字符）。"""
    line = text.strip().splitlines()[0] if text.strip() else ""
    line = line.strip().strip('"').strip("'")
    line = re.sub(r"^[\s\-–—:]+", "", line)
    parts = re.split(r"\s*[-–—]\s*", line, maxsplit=1)
    if len(parts) != 2:
        return None
    a, b = parts[0].strip(), parts[1].strip()
    if not a or not b:
        return None
    return a, b


def run(image_path: Path, model: str) -> Dict[str, Any]:
    image_path = image_path.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    uri = image_to_base64_data_uri(image_path)
    raw, api_debug = call_vision_model(uri, model=model)
    pair = parse_pair_line(raw)

    record: Dict[str, Any] = {
        "原物体": pair[0] if pair else None,
        "对应物体": pair[1] if pair else None,
        "source_image": image_path.name,
        "model": model,
    }
    if not pair:
        record["raw_model_output"] = raw
        if api_debug:
            record["api_debug"] = api_debug

    return record


def main() -> None:
    p = argparse.ArgumentParser(description="视觉 API：易混淆物体对 -> 打印一条 JSON 记录到 stdout")
    p.add_argument("image", type=Path, help="输入图片路径")
    p.add_argument("--model", default=VISION_MODEL, help="视觉模型名（默认读环境变量 VISION_MODEL）")
    args = p.parse_args()

    try:
        record = run(args.image, args.model)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
