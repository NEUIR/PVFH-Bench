#!/usr/bin/env python3
"""
从 lookalike_dataset.json 在输出目录写入 **4 个独立 JSON 文件**：

1) false_premise_spatial.json — 用「对应物体」填关系模板
2) presence_corresponding_object.json — Is the {对应物体} in the image?
3) presence_actual_object.json — Is the {原物体} in the image?
4) spatial_with_llm_absent_object.json — 同模板，{对应物体} 槽位由多模态模型生成「图中不存在」的物体

示例：
  .venv/bin/python src/2-generate_false_premise_questions.py \\
    -i data/lookalike_dataset.json -o data/questions_out \\
    --seed 42 --image-root /path/to/coco/images

若未设置 OPENAI_API_KEY 或使用 --no-absent-llm，第 4 个文件里 items 为占位（无 question）。

环境变量：OPENAI_API_KEY、OPENAI_API_BASE、ABSENT_GEN_MODEL（默认同 VQA_MODEL / VISION_MODEL / qwen3.5-plus）
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ABSENT_GEN_MODEL = os.getenv(
    "ABSENT_GEN_MODEL",
    os.getenv("VQA_MODEL", os.getenv("VISION_MODEL", "qwen3.5-plus")),
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_dataset_json(output_dir: Path, filename: str, meta: Dict[str, Any], items: List[Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = (output_dir / filename).resolve()
    doc = {"meta": meta, "items": items}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


@dataclass(frozen=True)
class Template:
    relation_type: str
    needs_anchor: bool
    question: str


# mistaken = 对应物体 / 生成的不存在物体, actual = 原物体（作锚点）
TEMPLATES: List[Template] = [
    Template(
        "left_right_of_anchor",
        True,
        "Is the {mistaken} to the left or to the right of the {actual}?",
    ),
    Template(
        "above_below_anchor",
        True,
        "Is the {mistaken} above or below the {actual}?",
    ),
    Template(
        "closer_farther_anchor",
        True,
        "Is the {mistaken} closer to the camera or farther from the camera than the {actual}?",
    ),
    Template(
        "larger_smaller_anchor",
        True,
        "Does the {mistaken} appear larger or smaller than the {actual}?",
    ),
    Template(
        "overlap_left_right_anchor",
        True,
        "Is the {mistaken} positioned more to the left or more to the right relative to the {actual}?",
    ),
    Template(
        "image_left_right",
        False,
        "Is the {mistaken} located more toward the left side or the right side?",
    ),
    Template(
        "image_upper_lower",
        False,
        "Is the {mistaken} mainly in the upper half or the lower half?",
    ),
    Template(
        "foreground_background",
        False,
        "Does the {mistaken} appear nearer to the foreground or nearer to the background?",
    ),
]


def _fill(tpl: Template, mistaken: str, actual: str) -> Dict[str, str]:
    ctx = {"mistaken": mistaken, "actual": actual}
    return {
        "relation_type": tpl.relation_type,
        "question": tpl.question.format(**ctx),
    }


def _pick_template(rng: random.Random, allow_no_anchor: bool) -> Template:
    pool = TEMPLATES if allow_no_anchor else [t for t in TEMPLATES if t.needs_anchor]
    return rng.choice(pool)


def load_items(path: Path) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("items") or []
    if not isinstance(items, list):
        raise ValueError("输入 JSON 缺少 list 类型的 items")
    return raw, items


def _parse_row(row: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    actual = row.get("原物体")
    mistaken = row.get("对应物体")
    src = row.get("source_image")
    if not src or not isinstance(src, str):
        return None
    if not actual or not mistaken:
        return None
    actual_s = str(actual).strip()
    mistaken_s = str(mistaken).strip()
    if not actual_s or not mistaken_s:
        return None
    return src, actual_s, mistaken_s


def _default_image_roots(extra: Sequence[Path]) -> List[Path]:
    roots: List[Path] = [REPO_ROOT, REPO_ROOT / "data", Path.cwd()]
    roots.extend(extra)
    seen: set[Path] = set()
    out: List[Path] = []
    for r in roots:
        try:
            k = r.resolve()
        except OSError:
            continue
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def resolve_image(source_image: str, roots: Sequence[Path]) -> Path:
    name = source_image.strip()
    for root in roots:
        r = root.resolve()
        for candidate in (r / name, r / "images" / name):
            if candidate.is_file():
                return candidate
    tried = ", ".join(str(x) for x in roots)
    raise FileNotFoundError(f"找不到图片 {name!r}，roots=[{tried}]")


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
        raise RuntimeError("未设置 OPENAI_API_KEY")
    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)


def call_llm_absent_object(
    image_path: Path,
    actual: str,
    mistaken: str,
    model: str,
) -> Tuple[Optional[str], Optional[str]]:
    """多模态：返回 (英文物体短语, 错误信息)。"""
    try:
        client = get_client()
    except RuntimeError as e:
        return None, str(e)

    uri = image_to_base64_data_uri(image_path)
    prompt = (
        "Look at this photograph.\n"
        f"Two labels from another pipeline for one salient object here are: "
        f"«{actual}» (intended ground-truth) and «{mistaken}» (a visually confusable but wrong label).\n\n"
        "Reply with exactly ONE short English noun phrase naming a concrete object **category** that "
        "**does not appear anywhere** in this image (no instance of that category visible). "
        f"Do not output «{actual}», «{mistaken}», or obvious synonyms or hypernyms of them. "
        "One line only: the phrase alone, no quotes, no explanation."
    )
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": uri}},
                ],
            }
        ],
        "max_tokens": 128,
        "timeout": 120.0,
    }
    if "dashscope" in OPENAI_API_BASE.lower():
        kwargs["extra_body"] = {"enable_thinking": False}

    try:
        resp = client.chat.completions.create(**kwargs)
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return None, str(e)

    line = text.splitlines()[0].strip() if text else ""
    line = line.strip('"').strip("'")
    if not line:
        return None, "empty model output"
    return line, None


def build_output_record(
    row: Dict[str, Any],
    rng: random.Random,
    allow_no_anchor: bool,
) -> Optional[Dict[str, Any]]:
    parsed = _parse_row(row)
    if not parsed:
        return None
    src, actual_s, mistaken_s = parsed
    tpl = _pick_template(rng, allow_no_anchor=allow_no_anchor)
    filled = _fill(tpl, mistaken_s, actual_s)

    return {
        "source_image": src,
        "原物体": actual_s,
        "对应物体": mistaken_s,
        "anchor_object": actual_s,
        "false_premise": (
            "The question refers to the object as "
            f"«{mistaken_s}»; the annotated ground-truth category for that instance is "
            f"«{actual_s}»."
        ),
        **filled,
    }


def build_presence_corresponding(src: str, actual_s: str, mistaken_s: str) -> Dict[str, Any]:
    return {
        "source_image": src,
        "原物体": actual_s,
        "对应物体": mistaken_s,
        "question": f"Is the {mistaken_s} in the image?",
        "dataset_kind": "presence_corresponding_object",
        "annotated_expected": "no",
    }


def build_presence_actual(src: str, actual_s: str, mistaken_s: str) -> Dict[str, Any]:
    return {
        "source_image": src,
        "原物体": actual_s,
        "对应物体": mistaken_s,
        "question": f"Is the {actual_s} in the image?",
        "dataset_kind": "presence_actual_object",
        "annotated_expected": "yes",
    }


def build_absent_spatial_record(
    src: str,
    actual_s: str,
    mistaken_s: str,
    rng: random.Random,
    allow_no_anchor: bool,
    image_roots: Sequence[Path],
    model: str,
) -> Dict[str, Any]:
    tpl = _pick_template(rng, allow_no_anchor=allow_no_anchor)
    absent: Optional[str] = None
    err: Optional[str] = None
    try:
        img_path = resolve_image(src, image_roots)
        absent, err = call_llm_absent_object(img_path, actual_s, mistaken_s, model)
    except FileNotFoundError as e:
        err = str(e)

    base = {
        "source_image": src,
        "原物体": actual_s,
        "对应物体": mistaken_s,
        "anchor_object": actual_s,
        "generated_absent_object": absent,
        "absent_generation_error": err,
        "dataset_kind": "spatial_with_llm_absent_object",
    }
    if absent:
        base.update(_fill(tpl, absent, actual_s))
        base["false_premise"] = (
            f"The question is about «{absent}», which was generated as an object category "
            "absent from the image; the spatial anchor label is "
            f"«{actual_s}»."
        )
    else:
        base["relation_type"] = tpl.relation_type
        base["question"] = None
    return base


def main() -> None:
    p = argparse.ArgumentParser(description="由 lookalike_dataset 生成多组虚假/存在性/幻觉物体问题 JSON")
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("data/lookalike_dataset.json"),
        help="输入 lookalike_dataset.json 路径",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("data"),
        dest="output_dir",
        help="输出目录（将写入 4 个 JSON 文件）",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="模板抽样的随机种子（同输入与同种子则模板分配可复现）",
    )
    p.add_argument(
        "--anchor-only",
        action="store_true",
        help="只使用带「原物体」锚点的模板（不用纯画面左/右、上/下半等）",
    )
    p.add_argument(
        "--image-root",
        type=Path,
        action="append",
        default=[],
        metavar="DIR",
        help="查找 source_image 的目录，可多次指定（含 DIR 与 DIR/images）",
    )
    p.add_argument(
        "--no-absent-llm",
        action="store_true",
        help="不调用 API 生成「不存在物体」，spatial_with_llm_absent_object 仅含元数据或空",
    )
    p.add_argument("--absent-model", default=ABSENT_GEN_MODEL, help="生成不存在物体用的多模态模型")
    args = p.parse_args()

    inp = args.input.resolve()
    if not inp.is_file():
        print(f"找不到输入文件: {inp}", file=sys.stderr)
        sys.exit(1)

    raw, items = load_items(inp)
    rng_spatial = random.Random(args.seed)
    rng_absent_template = random.Random(args.seed + 7919)
    allow_no_anchor = not args.anchor_only
    image_roots = _default_image_roots(args.image_root)

    out_spatial: List[Dict[str, Any]] = []
    presence_corr: List[Dict[str, Any]] = []
    presence_act: List[Dict[str, Any]] = []
    absent_spatial: List[Dict[str, Any]] = []

    skipped = 0
    for row in items:
        if not isinstance(row, dict):
            skipped += 1
            continue
        parsed = _parse_row(row)
        if not parsed:
            skipped += 1
            continue
        src, actual_s, mistaken_s = parsed

        rec = build_output_record(row, rng_spatial, allow_no_anchor)
        if rec is not None:
            out_spatial.append(rec)
        presence_corr.append(build_presence_corresponding(src, actual_s, mistaken_s))
        presence_act.append(build_presence_actual(src, actual_s, mistaken_s))

        if args.no_absent_llm or not OPENAI_API_KEY:
            absent_spatial.append(
                {
                    "source_image": src,
                    "原物体": actual_s,
                    "对应物体": mistaken_s,
                    "anchor_object": actual_s,
                    "generated_absent_object": None,
                    "absent_generation_error": "skipped (--no-absent-llm or no API key)"
                    if args.no_absent_llm
                    else "skipped (no OPENAI_API_KEY)",
                    "dataset_kind": "spatial_with_llm_absent_object",
                    "relation_type": None,
                    "question": None,
                }
            )
        else:
            absent_spatial.append(
                build_absent_spatial_record(
                    src,
                    actual_s,
                    mistaken_s,
                    rng_absent_template,
                    allow_no_anchor,
                    image_roots,
                    args.absent_model,
                )
            )

    out_dir = args.output_dir.resolve()
    base_meta: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(inp),
        "source_meta": raw.get("meta"),
        "random_seed": args.seed,
        "anchor_only": args.anchor_only,
        "input_item_count": len(items),
        "skipped": skipped,
        "image_roots_tried": [str(r) for r in image_roots],
        "absent_llm_enabled": bool(not args.no_absent_llm and OPENAI_API_KEY),
        "absent_model": args.absent_model if not args.no_absent_llm else None,
    }

    bundles: List[Tuple[str, str, List[Dict[str, Any]]]] = [
        ("false_premise_spatial", "false_premise_spatial.json", out_spatial),
        ("presence_corresponding_object", "presence_corresponding_object.json", presence_corr),
        ("presence_actual_object", "presence_actual_object.json", presence_act),
        ("spatial_with_llm_absent_object", "spatial_with_llm_absent_object.json", absent_spatial),
    ]
    written: List[Path] = []
    for key, fname, data in bundles:
        m = dict(base_meta)
        m["dataset"] = key
        m["output_file"] = str(out_dir / fname)
        m["item_count"] = len(data)
        written.append(_write_dataset_json(out_dir, fname, m, data))

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
