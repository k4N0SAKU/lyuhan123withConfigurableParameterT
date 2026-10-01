"""A1-22 业务化演示（P7 任务 1；FastAPI）。

启动：python -m src.demo.app [--port 8060]
页面：http://127.0.0.1:8060/  （输入区 / 拓扑状态 / 实时指标 / 明文指示灯 /
模式 A|B 切换 / 失败与超时 UI 反馈）

口径纪律：指标统计来自 runner（经 src.common.perf 统计模块），基线数字直接
读取 benchmarks/results/p6_full_bench.json（与 docs/04 同源，陷阱 2）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from src.demo.runner import DemoRunner
from src.demo.scenarios import SIMULATED_NOTE, scenario_list

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="A1-22 面向大模型隐私保护的密码方案演示",
              description="模拟数据；同机回环（S8）；指标与 p6_full_bench.json 同源")
runner = DemoRunner()


class RunRequest(BaseModel):
    scenario: str = "gov"
    sample: int = 0
    rounds: int = 3
    mode: str = "B"                 # A|B 一键切换
    content: str | None = None      # 自定义内容（可空 → 用样例）
    name: str | None = None         # 模拟字段覆盖（仅页面展示）
    id_no: str | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "title": "A1-22 demo", "note": SIMULATED_NOTE}


@app.get("/api/scenarios")
def scenarios() -> dict:
    return {"scenarios": scenario_list(),
            "simulated_note": SIMULATED_NOTE,
            "baseline": runner.status().get("baseline"),
            "pipeline_cfg": runner.status().get("pipeline_cfg"),
            "modes": [
                {"id": "B", "label": "模式 B（混合密文，可运行）",
                 "available": True},
                {"id": "A", "label": "模式 A（全密文深链）",
                 "available": False,
                 "reason": "本栈 SEAL 校验上限 2^15，需 2^17/84 层链——"
                           "不可实例化（docs/01 §5.3），如实申报不做模拟"}]}


@app.post("/api/run")
def start_run(req: RunRequest) -> dict:
    if not (1 <= req.rounds <= 20):
        raise HTTPException(400, "rounds 须在 1..20")
    overrides = {}
    if req.content:
        overrides["content"] = req.content
    for k, v in (("name", req.name), ("id_no", req.id_no)):
        if v:
            overrides[k] = v
    out = runner.run(req.scenario, req.sample, req.rounds, req.mode,
                     overrides or None)
    if not out.get("accepted"):
        raise HTTPException(409, out.get("reason", "运行冲突"))
    return out


@app.get("/api/status")
def status() -> dict:
    return runner.status()


@app.post("/api/stop")
def stop() -> dict:
    return runner.stop()


@app.exception_handler(Exception)
def _err(_, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})


def main(argv=None) -> int:
    import uvicorn
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8060)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)
    print(f"A1-22 demo → http://{args.host}:{args.port}/  （模拟数据，S8 回环）")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
