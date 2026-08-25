from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from lookalike_object_pipeline import (
    DEFAULT_DATA_DIR,
    PRESENCE_ACTUAL_OBJECT_FILENAME,
    VISION_MODEL,
    run,
)


def _collect_images(
    paths: Sequence[Path],
    globs: Sequence[str],
    input_dir: Path | None,
    recursive: bool,
    extensions: frozenset[str],
) -> List[Path]:
    collected: List[Path] = []
    for p in paths:
        if p.is_file():
            collected.append(p.resolve())
    for g in globs:
        collected.extend(q.resolve() for q in Path().glob(g) if q.is_file())
    if input_dir is not None:
        if not input_dir.is_dir():
            raise FileNotFoundError(f"--input-dir 不是目录: {input_dir}")
        it = input_dir.rglob("*") if recursive else input_dir.iterdir()
        for p in it:
            if p.is_file() and p.suffix.lower() in extensions:
                collected.append(p.resolve())

    seen: set[Path] = set()
    out: List[Path] = []
    for p in collected:
        if p not in seen:
            seen.add(p)
            out.append(p)
    # 按字符串排序，保证跨平台、抽样前顺序稳定，便于固定随机种子复现
    return sorted(out, key=lambda x: str(x))


def _process_one(
    image_path: Path,
    model: str,
) -> Dict[str, Any]:
    try:
        record = run(image_path, model)
        return {
            "status": "ok",
            "source_image": image_path.name,
            "record": record,
        }
    except Exception as e:
        return {
            "status": "error",
            "source_image": image_path.name,
            "error": str(e),
        }


def _retry_basenames_from_lookalike(doc: Dict[str, Any]) -> Set[str]:
    """API 失败 + items 中未解析出物体对的条目，按 source_image  basename 重试。"""
    out: Set[str] = set()
    for f in doc.get("failures", []):
        name = f.get("source_image")
        if isinstance(name, str) and name:
            out.add(name)
    for it in doc.get("items", []):
        name = it.get("source_image")
        if not isinstance(name, str) or not name:
            continue
        if not (it.get("原物体") and it.get("对应物体")):
            out.add(name)
    return out


def _merge_lookalike_after_retry(
    existing: Dict[str, Any],
    retry_results: List[Dict[str, Any]],
    *,
    model: str,
    workers: int,
    merge_out: Path,
) -> Dict[str, Any]:
    items_by: Dict[str, Dict[str, Any]] = {}
    for it in existing.get("items", []):
        n = it.get("source_image")
        if isinstance(n, str) and n:
            items_by[n] = dict(it)
    failures_by: Dict[str, Dict[str, Any]] = {}
    for f in existing.get("failures", []):
        n = f.get("source_image")
        if isinstance(n, str) and n:
            failures_by[n] = dict(f)

    for r in retry_results:
        name = r.get("source_image")
        if not isinstance(name, str) or not name:
            continue
        if r.get("status") == "ok" and "record" in r:
            items_by[name] = dict(r["record"])
            failures_by.pop(name, None)
        else:
            failures_by[name] = {
                "source_image": name,
                "error": str(r.get("error", "unknown error")),
            }
            items_by.pop(name, None)

    items_sorted = sorted(items_by.values(), key=lambda x: x.get("source_image", ""))
    failures_sorted = sorted(failures_by.values(), key=lambda x: x.get("source_image", ""))

    meta = dict(existing.get("meta", {}))
    meta["generated_at"] = datetime.now(timezone.utc).isoformat()
    meta["model"] = model
    meta["workers"] = workers
    meta["ok"] = len(items_sorted)
    meta["errors"] = len(failures_sorted)
    meta["output_path"] = str(merge_out)
    meta["data_dir"] = str(merge_out.parent)

    return {
        "meta": meta,
        "items": items_sorted,
        "failures": failures_sorted,
    }


def _try_resume_plan_from_presence(
    data_dir: Path,
) -> Optional[Tuple[Set[str], Path, Dict[str, Any]]]:
    presence_path = data_dir / PRESENCE_ACTUAL_OBJECT_FILENAME
    if not presence_path.is_file():
        return None
    try:
        presence_doc = json.loads(presence_path.read_text(encoding="utf-8"))
        src = presence_doc.get("meta", {}).get("source_file")
        lookalike_path = Path(src).resolve() if src else (data_dir / "lookalike_dataset.json")
        lookalike_doc = json.loads(lookalike_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as e:
        print(f"警告: 读取 {presence_path} 或 lookalike 源文件失败，将不按 presence 恢复: {e}", file=sys.stderr)
        return None
    retry = _retry_basenames_from_lookalike(lookalike_doc)
    return retry, lookalike_path, lookalike_doc


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    p = argparse.ArgumentParser(description="并行跑 lookalike 流水线并合并 JSON")
    p.add_argument("images", nargs="*", type=Path, help="图片路径（可多个）")
    p.add_argument(
        "--glob",
        action="append",
        default=[],
        metavar="PATTERN",
        help="相对于当前工作目录的 glob，可重复",
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="从目录收集图片（配合 --ext / -r）",
    )
    p.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="配合 --input-dir 递归子目录",
    )
    p.add_argument(
        "--ext",
        default=".jpg,.jpeg,.png,.webp,.gif",
        help="配合 --input-dir 的扩展名列表（逗号分隔，小写带点）",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"未指定 -o 时，汇总 JSON 写入该目录下的 lookalike_dataset.json（默认 {DEFAULT_DATA_DIR}）",
    )
    p.add_argument("--model", default=VISION_MODEL, help="视觉模型名")
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并发线程数（受 API 限流影响，过大易 429）",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help=f"合并后的总 JSON 路径（默认: <data-dir>/lookalike_dataset.json）",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="仅打印汇总统计，不逐条打印进度",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="在收集到的全部图片中无放回随机抽取 N 张（N 大于池大小时取全部）",
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="配合 --sample 使用的随机种子（默认 0，相同池与种子得到相同子集）",
    )
    p.add_argument(
        "--force-resample",
        action="store_true",
        help=f"即使存在 {PRESENCE_ACTUAL_OBJECT_FILENAME} 也照常抽样/全量处理，不进入仅重试失败项模式",
    )
    args = p.parse_args()

    if args.sample is not None and args.sample < 1:
        print("--sample 须为正整数。", file=sys.stderr)
        sys.exit(1)

    exts = frozenset(e.strip().lower() for e in args.ext.split(",") if e.strip())
    try:
        images = _collect_images(args.images, args.glob, args.input_dir, args.recursive, exts)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    if not images:
        print("未找到任何图片。", file=sys.stderr)
        sys.exit(1)

    data_dir = args.data_dir.resolve()
    pool_size = len(images)
    sample_meta: Dict[str, Any] = {}
    resume: Optional[Tuple[Set[str], Path, Dict[str, Any]]] = None
    if not args.force_resample and (data_dir / PRESENCE_ACTUAL_OBJECT_FILENAME).is_file():
        loaded = _try_resume_plan_from_presence(data_dir)
        if loaded is not None:
            retry_names, _lk, _doc = loaded
            if not retry_names:
                print(
                    f"{data_dir / PRESENCE_ACTUAL_OBJECT_FILENAME} 存在：lookalike 中无待重试项，不重新抽样。",
                    file=sys.stderr,
                )
                sys.exit(0)
            resume = loaded

    if resume is not None:
        retry_names, lookalike_path, _existing_doc = resume
        images = [p for p in images if p.name in retry_names]
        missing = retry_names - {p.name for p in images}
        if missing and not args.quiet:
            print(
                f"警告: {len(missing)} 个待重试图片在当前收集到的路径中未找到（请确认与上次相同的 --input-dir / 路径参数）。",
                file=sys.stderr,
            )
        if not images:
            print("待重试项均找不到对应文件，未发起 API 请求。", file=sys.stderr)
            sys.exit(1)
        if not args.quiet:
            print(
                f"检测到 {data_dir / PRESENCE_ACTUAL_OBJECT_FILENAME}：跳过抽样，仅重试 {len(images)} 张失败/解析失败图片。",
                file=sys.stderr,
            )
    elif args.sample is not None:
        rng = random.Random(args.random_seed)
        k = min(args.sample, pool_size)
        images = rng.sample(images, k)
        sample_meta = {
            "sample_requested": args.sample,
            "sample_actual": k,
            "random_seed": args.random_seed,
            "pool_size_before_sample": pool_size,
        }
        if not args.quiet and k < args.sample:
            print(
                f"池中仅 {pool_size} 张，--sample {args.sample} 调整为实际抽取 {k} 张。",
                file=sys.stderr,
            )

    merge_out = (
        args.output.resolve()
        if args.output is not None
        else (
            resume[1].resolve()
            if resume is not None
            else (data_dir / "lookalike_dataset.json")
        )
    )
    if resume is not None and args.output is not None and args.output.resolve() != resume[1].resolve():
        if not args.quiet:
            print(
                f"注意: 合并结果写入 -o {merge_out}，presence 中记录的 lookalike 源为 {resume[1]}。",
                file=sys.stderr,
            )

    results: List[Dict[str, Any]] = []
    workers = max(1, args.workers)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                _process_one,
                img,
                args.model,
            ): img
            for img in images
        }
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            if not args.quiet:
                tag = r["status"]
                name = r.get("source_image", "")
                if tag == "error":
                    print(f"[{tag}] {name}: {r.get('error', '')}", file=sys.stderr)
                else:
                    print(f"[{tag}] {name}", file=sys.stderr)

    results.sort(key=lambda x: x.get("source_image", ""))

    if resume is not None:
        _, _, existing_doc = resume
        summary = _merge_lookalike_after_retry(
            existing_doc,
            results,
            model=args.model,
            workers=workers,
            merge_out=merge_out,
        )
        if sample_meta := existing_doc.get("meta", {}).get("sampling"):
            summary["meta"]["sampling"] = dict(sample_meta)
    else:
        items: List[Dict[str, Any]] = []
        for r in results:
            if r["status"] == "ok" and "record" in r:
                items.append(dict(r["record"]))

        failures = [
            {"source_image": r["source_image"], "error": r["error"]}
            for r in results
            if r["status"] == "error"
        ]

        meta: Dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "data_dir": str(data_dir),
            "output_path": str(merge_out),
            "workers": workers,
            "total_images": len(images),
            "ok": sum(1 for r in results if r["status"] == "ok"),
            "errors": len(failures),
        }
        if sample_meta:
            meta["sampling"] = sample_meta

        summary = {
            "meta": meta,
            "items": items,
            "failures": failures,
        }

    _atomic_write_json(merge_out, summary)
    print(merge_out)
    m = summary["meta"]
    print(
        f"完成: ok={m['ok']} errors={m['errors']} -> {merge_out}",
        file=sys.stderr,
    )
    if summary.get("failures"):
        sys.exit(2)


if __name__ == "__main__":
    main()
