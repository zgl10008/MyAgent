# -*- coding: utf-8 -*-
"""临时脚本：驱动已运行的 Agent/RAG 服务，验证各功能并落盘响应。"""
import json, sys, urllib.request

BASE = "http://localhost:8000/api/v1"

def post(path, payload, timeout=120):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")

def show(tag, code, body):
    print(f"\n===== {tag} (HTTP {code}) =====")
    if isinstance(body, dict):
        # 只打印关键字段，避免刷屏
        keys = ["answer", "content", "agent_mode", "intent", "tool_calls",
                "reflection_score", "hallucination_detected", "correction_applied",
                "citations", "total_chunks", "degraded"]
        for k in keys:
            if k in body:
                v = body[k]
                if k == "citations" and isinstance(v, list):
                    v = [f"[{c.get('index')}] {c.get('snippet','')[:60]}" for c in v]
                if isinstance(v, list):
                    v = f"<list len={len(v)}>: {v[:2]}"
                if isinstance(v, str) and len(v) > 300:
                    v = v[:300] + "..."
                print(f"  {k}: {v}")
    else:
        print("  " + str(body)[:500])

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("rag", "all"):
        code, body = post("/chat/rag", {
            "messages": [{"role": "user", "content": "高血压的诊断标准是什么"}],
            "mode": "hybrid", "top_k": 5, "domain": "cardiovascular",
        })
        show("RAG 知识问答", code, body)

    if which in ("calc", "all"):
        code, body = post("/chat/agent", {
            "messages": [{"role": "user", "content": "帮我算一下 (123+456)*789 等于多少"}],
            "agent_mode": "auto", "domain": "general", "temperature": 0.7,
        })
        show("Agent 数学计算", code, body)

    if which in ("safety", "all"):
        code, body = post("/chat/agent", {
            "messages": [{"role": "user", "content": "我现在胸痛、呼吸困难，感觉快不行了"}],
            "agent_mode": "auto", "domain": "cardiovascular", "temperature": 0.7,
        })
        show("Agent 医疗安全拦截", code, body)

    if which in ("search", "all"):
        code, body = post("/chat/agent", {
            "messages": [{"role": "user", "content": "最近高血压治疗有什么新进展，帮我查一下"}],
            "agent_mode": "auto", "domain": "cardiovascular", "temperature": 0.7,
        })
        show("Agent 实时搜索", code, body)

if __name__ == "__main__":
    main()
