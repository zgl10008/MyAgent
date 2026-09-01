# -*- coding: utf-8 -*-
"""下载 bge-m3 模型权重（跳过 onnx/imgs，仅需 sentence-transformers 推理所需文件）。"""
import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# hf-mirror 把大文件重定向到 HF 的 Xet CDN（xet-bridge-us），
# huggingface_hub 的 Xet 后端对该 CDN 的 HEAD 请求会报 LocalEntryNotFoundError。
# 禁用 Xet 走常规 HTTP 下载（curl 已验证该 URL 可 302→200 正常拉取）。
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import snapshot_download  # noqa: E402

# sentence-transformers 只需 pytorch_model.bin + config/tokenizer 系列，
# onnx/ 与 imgs/ 是 ONNX 导出和 README 图片，可跳过以省 ~2.3GB。
IGNORE = [
    "onnx/*",
    "imgs/*",
    "*.jpg",
    "*.JPG",
    "*.png",
    "*.webp",
    "*.gif",
    ".DS_Store",
]


def main() -> str:
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            print(f"[{attempt}/3] snapshot_download(BAAI/bge-m3) ...", flush=True)
            path = snapshot_download(
                "BAAI/bge-m3",
                ignore_patterns=IGNORE,
            )
            print(f"OK -> {path}", flush=True)
            return path
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(f"[{attempt}/3] failed: {exc}", flush=True)
            time.sleep(5 * attempt)
    raise RuntimeError(f"bge-m3 下载失败: {last_exc}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
