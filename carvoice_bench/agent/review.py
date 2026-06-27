"""Local-only HTML review service for candidates, evidence, and approvals."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

from carvoice_bench.agent.storage import AgentStore


class ReviewService:
    def __init__(self, store: AgentStore, run_id: str):
        self.store = store
        self.run_id = run_id

    def export_approved_cases(self) -> Path:
        output = self.store.workspace / "approved_regression.yaml"
        payload = {"test_cases": self.store.approved_cases(self.run_id)}
        try:
            import yaml

            output.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        except ImportError:
            output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return output

    def serve(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        service = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path == "/":
                    self._send_html(_page())
                    return
                if path == "/api/queue":
                    self._send_json({"run_id": service.run_id, "items": service.store.review_queue(service.run_id)})
                    return
                if path == "/api/export":
                    self._send_json({"path": str(service.export_approved_cases())})
                    return
                self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:  # noqa: N802
                if urlparse(self.path).path != "/api/review":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                    service.store.review_candidate(
                        str(body["candidate_id"]),
                        str(body["decision"]),
                        str(body.get("reviewer", "local-reviewer")),
                        str(body.get("note", "")),
                    )
                    response: Dict[str, Any] = {"ok": True}
                    if body["decision"] == "approved":
                        response["regression_path"] = str(service.export_approved_cases())
                    self._send_json(response)
                except (KeyError, ValueError, json.JSONDecodeError) as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

            def _send_json(self, value: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
                data = json.dumps(value, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _send_html(self, value: str) -> None:
                data = value.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: Any) -> None:
                return

        server = ThreadingHTTPServer((host, port), Handler)
        print("Review console: http://" + host + ":" + str(port), flush=True)
        try:
            server.serve_forever()
        finally:
            server.server_close()


def _page() -> str:
    return """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CarVoice Agent Review</title><style>
body{margin:0;font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;background:#f4f6f8;color:#15202b}
header{padding:20px 28px;background:#12324a;color:#fff;display:flex;justify-content:space-between;align-items:center}
main{max-width:1280px;margin:0 auto;padding:24px;display:grid;grid-template-columns:minmax(270px,1fr) minmax(0,2fr);gap:20px}
section{background:#fff;border:1px solid #d9e1e8;border-radius:6px;min-height:420px}h1{font-size:20px;margin:0}h2{font-size:16px;margin:0;padding:14px 16px;border-bottom:1px solid #d9e1e8}button{border:1px solid #6d7d8b;background:#fff;padding:7px 10px;border-radius:4px;cursor:pointer}.approve{background:#176b4b;color:#fff;border-color:#176b4b}.reject{background:#a33838;color:#fff;border-color:#a33838}.revise{background:#c47a15;color:#fff;border-color:#c47a15}
#queue{list-style:none;padding:0;margin:0}.case{padding:12px 16px;border-bottom:1px solid #e7edf1;cursor:pointer}.case:hover,.case.active{background:#e9f3f5}.muted{color:#657786}pre{white-space:pre-wrap;overflow:auto;margin:0;padding:16px;max-height:470px;background:#f8fafb}.actions{display:flex;gap:8px;padding:12px 16px;border-top:1px solid #d9e1e8}.empty{padding:24px;color:#657786}
</style></head><body><header><h1>CarVoice Agent Review</h1><button onclick="exportCases()">导出已批准回归集</button></header><main><section><h2>待审核候选</h2><ul id="queue"></ul></section><section><h2 id="detail-title">选择一个候选</h2><pre id="detail" class="muted">等待加载</pre><div class="actions" id="actions" hidden><button class="approve" onclick="review('approved')">批准</button><button class="revise" onclick="review('needs_revision')">需修改</button><button class="reject" onclick="review('rejected')">驳回</button></div></section></main>
<script>let items=[],selected=null;async function load(){let r=await fetch('/api/queue');let d=await r.json();items=d.items;let q=document.getElementById('queue');q.innerHTML=items.length?'':'';if(!items.length){q.innerHTML='<li class="empty">没有待审核候选</li>';return}items.forEach((item,i)=>{let c=item.candidate;let li=document.createElement('li');li.className='case';li.textContent=c.case.id+' · '+c.case.description;li.onclick=()=>show(i);q.appendChild(li)})}function show(i){selected=items[i];document.querySelectorAll('.case').forEach((el,n)=>el.classList.toggle('active',n===i));document.getElementById('detail-title').textContent=selected.candidate.case.id;document.getElementById('detail').textContent=JSON.stringify(selected,null,2);document.getElementById('actions').hidden=false}async function review(decision){if(!selected)return;let note=window.prompt('审核备注（可为空）','')||'';let r=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({candidate_id:selected.candidate.id,decision,reviewer:'local-reviewer',note})});let d=await r.json();if(!d.ok)alert(d.error);await load();document.getElementById('detail').textContent='审核已记录';document.getElementById('actions').hidden=true}async function exportCases(){let r=await fetch('/api/export');let d=await r.json();alert('已导出: '+d.path)}load();</script></body></html>"""
