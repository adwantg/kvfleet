"""Admin dashboard — lightweight web UI for fleet monitoring.

Provides a real-time dashboard showing:
- Fleet status and model health
- Routing decision history
- Rate limit states
- Budget/spend tracking
- Cost analysis

Runs as a standalone web server using Python's built-in http.server
with no external dependencies (no starlette/fastapi required).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class DashboardState:
    """Holds the live state for the admin dashboard.

    Updated by the Router engine as events occur.
    """

    def __init__(self, max_history: int = 500) -> None:
        self.max_history = max_history
        self._lock = threading.Lock()

        # Fleet info
        self.fleet_name: str = ""
        self.strategy: str = ""
        self.model_count: int = 0
        self.start_time: float = time.time()

        # Live counters
        self.total_requests: int = 0
        self.total_errors: int = 0
        self.total_fallbacks: int = 0
        self.total_cache_hits: int = 0
        self.total_policy_blocks: int = 0

        # Model stats
        self.model_stats: dict[str, dict[str, Any]] = {}

        # Routing history (circular buffer)
        self.route_history: list[dict[str, Any]] = []

        # Rate limit states
        self.rate_limit_states: dict[str, dict[str, Any]] = {}

        # Budget states
        self.budget_states: dict[str, dict[str, Any]] = {}

        # Health states
        self.health_states: dict[str, dict[str, Any]] = {}

    def record_route(
        self,
        prompt_preview: str,
        selected_model: str,
        strategy: str,
        latency_ms: float,
        scores: dict[str, float] | None = None,
        cache_hit: bool = False,
        fallback: bool = False,
        error: str = "",
    ) -> None:
        """Record a routing decision."""
        with self._lock:
            self.total_requests += 1
            if error:
                self.total_errors += 1
            if fallback:
                self.total_fallbacks += 1
            if cache_hit:
                self.total_cache_hits += 1

            # Model stats
            if selected_model not in self.model_stats:
                self.model_stats[selected_model] = {
                    "requests": 0,
                    "errors": 0,
                    "total_latency_ms": 0,
                }
            self.model_stats[selected_model]["requests"] += 1
            self.model_stats[selected_model]["total_latency_ms"] += latency_ms
            if error:
                self.model_stats[selected_model]["errors"] += 1

            # History
            entry = {
                "timestamp": time.time(),
                "prompt": prompt_preview[:80],
                "model": selected_model,
                "strategy": strategy,
                "latency_ms": round(latency_ms, 1),
                "cache_hit": cache_hit,
                "fallback": fallback,
                "error": error,
                "scores": scores or {},
            }
            self.route_history.append(entry)
            if len(self.route_history) > self.max_history:
                self.route_history = self.route_history[-self.max_history :]

    def record_policy_block(self, rule_name: str) -> None:
        with self._lock:
            self.total_policy_blocks += 1

    def update_health(
        self, model_name: str, endpoint: str, healthy: bool, latency_ms: float = 0
    ) -> None:
        with self._lock:
            self.health_states[f"{model_name}:{endpoint}"] = {
                "healthy": healthy,
                "latency_ms": latency_ms,
                "last_check": time.time(),
            }

    def update_rate_limits(self, states: dict[str, Any]) -> None:
        with self._lock:
            self.rate_limit_states = states

    def update_budgets(self, states: dict[str, Any]) -> None:
        with self._lock:
            self.budget_states = states

    def to_dict(self) -> dict[str, Any]:
        """Serialize full state to JSON-safe dict."""
        with self._lock:
            uptime = time.time() - self.start_time
            avg_latency = 0.0
            if self.model_stats:
                total_reqs = sum(s["requests"] for s in self.model_stats.values())
                total_lat = sum(s["total_latency_ms"] for s in self.model_stats.values())
                avg_latency = total_lat / max(total_reqs, 1)

            return {
                "fleet": {
                    "name": self.fleet_name,
                    "strategy": self.strategy,
                    "models": self.model_count,
                    "uptime_seconds": round(uptime),
                },
                "counters": {
                    "total_requests": self.total_requests,
                    "total_errors": self.total_errors,
                    "total_fallbacks": self.total_fallbacks,
                    "total_cache_hits": self.total_cache_hits,
                    "total_policy_blocks": self.total_policy_blocks,
                    "avg_latency_ms": round(avg_latency, 1),
                    "error_rate": f"{(self.total_errors / max(self.total_requests, 1)) * 100:.1f}%",
                },
                "model_stats": dict(self.model_stats),
                "health": dict(self.health_states),
                "rate_limits": dict(self.rate_limit_states),
                "budgets": dict(self.budget_states),
                "recent_routes": list(self.route_history[-20:]),
            }


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>kvfleet — Admin Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',system-ui,-apple-system,sans-serif;background:#0f0f23;color:#e0e0f0;min-height:100vh}
.header{background:linear-gradient(135deg,#1a1a3e,#2d1b4e);padding:20px 32px;border-bottom:1px solid #333}
.header h1{font-size:24px;font-weight:700;background:linear-gradient(90deg,#6ee7b7,#3b82f6);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.header .subtitle{font-size:13px;color:#888;margin-top:4px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;padding:24px 32px}
.card{background:#1a1a2e;border:1px solid #2a2a4a;border-radius:12px;padding:20px}
.card .label{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#888}
.card .value{font-size:32px;font-weight:700;margin-top:4px}
.card .value.green{color:#6ee7b7}.card .value.blue{color:#60a5fa}
.card .value.red{color:#f87171}.card .value.yellow{color:#fbbf24}
.section{padding:8px 32px 24px}
.section h2{font-size:16px;color:#a0a0c0;margin-bottom:12px;border-bottom:1px solid #2a2a4a;padding-bottom:8px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;padding:8px 12px;color:#888;border-bottom:1px solid #2a2a4a;font-weight:500}
td{padding:8px 12px;border-bottom:1px solid #1f1f3a}
tr:hover{background:#1f1f3a}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600}
.badge.green{background:#064e3b;color:#6ee7b7}
.badge.red{background:#7f1d1d;color:#fca5a5}
.badge.yellow{background:#713f12;color:#fde047}
.badge.blue{background:#1e3a5f;color:#93c5fd}
.auto-refresh{font-size:11px;color:#555;text-align:center;padding:8px}
</style>
</head>
<body>
<div class="header">
  <h1>🚀 kvfleet Dashboard</h1>
  <div class="subtitle" id="fleet-info">Loading...</div>
</div>
<div class="grid" id="counters"></div>
<div class="section">
  <h2>Model Performance</h2>
  <table id="model-table"><thead><tr><th>Model</th><th>Requests</th><th>Avg Latency</th><th>Errors</th><th>Health</th></tr></thead><tbody></tbody></table>
</div>
<div class="section">
  <h2>Recent Routing Decisions</h2>
  <table id="route-table"><thead><tr><th>Time</th><th>Prompt</th><th>Model</th><th>Latency</th><th>Cache</th><th>Fallback</th></tr></thead><tbody></tbody></table>
</div>
<div class="section">
  <h2>Rate Limits</h2>
  <table id="ratelimit-table"><thead><tr><th>Endpoint</th><th>Usage</th><th>Capacity</th><th>Throttled</th></tr></thead><tbody></tbody></table>
</div>
<div class="auto-refresh">Auto-refreshes every 3 seconds</div>
<script>
async function refresh(){
  try{
    const r=await fetch('/api/state');
    const d=await r.json();
    document.getElementById('fleet-info').textContent=
      `Fleet: ${d.fleet.name} | Strategy: ${d.fleet.strategy} | Models: ${d.fleet.models} | Uptime: ${Math.floor(d.fleet.uptime_seconds/60)}m`;
    const c=d.counters;
    document.getElementById('counters').innerHTML=`
      <div class="card"><div class="label">Total Requests</div><div class="value blue">${c.total_requests}</div></div>
      <div class="card"><div class="label">Avg Latency</div><div class="value green">${c.avg_latency_ms}ms</div></div>
      <div class="card"><div class="label">Error Rate</div><div class="value ${parseFloat(c.error_rate)>5?'red':'green'}">${c.error_rate}</div></div>
      <div class="card"><div class="label">Cache Hits</div><div class="value yellow">${c.total_cache_hits}</div></div>
      <div class="card"><div class="label">Fallbacks</div><div class="value ${c.total_fallbacks>0?'yellow':'green'}">${c.total_fallbacks}</div></div>
      <div class="card"><div class="label">Policy Blocks</div><div class="value ${c.total_policy_blocks>0?'red':'green'}">${c.total_policy_blocks}</div></div>`;
    const mt=document.querySelector('#model-table tbody');
    mt.innerHTML=Object.entries(d.model_stats).map(([name,s])=>{
      const avg=Math.round(s.total_latency_ms/Math.max(s.requests,1));
      const hk=Object.keys(d.health).find(k=>k.startsWith(name));
      const h=hk?d.health[hk]:null;
      return`<tr><td><b>${name}</b></td><td>${s.requests}</td><td>${avg}ms</td><td>${s.errors}</td>
        <td><span class="badge ${h?(h.healthy?'green':'red'):'yellow'}">${h?(h.healthy?'Healthy':'Down'):'Unknown'}</span></td></tr>`;
    }).join('');
    const rt=document.querySelector('#route-table tbody');
    rt.innerHTML=d.recent_routes.reverse().map(r=>{
      const t=new Date(r.timestamp*1000).toLocaleTimeString();
      return`<tr><td>${t}</td><td>${r.prompt}</td><td><b>${r.model}</b></td><td>${r.latency_ms}ms</td>
        <td>${r.cache_hit?'<span class="badge green">HIT</span>':'-'}</td>
        <td>${r.fallback?'<span class="badge yellow">YES</span>':'-'}</td></tr>`;
    }).join('');
    const rlt=document.querySelector('#ratelimit-table tbody');
    rlt.innerHTML=Object.entries(d.rate_limits).map(([ep,s])=>
      `<tr><td>${ep}</td><td>${s.usage_pct||'0%'}</td><td>${s.capacity||'100%'}</td>
        <td>${s.throttled?'<span class="badge red">THROTTLED</span>':'<span class="badge green">OK</span>'}</td></tr>`
    ).join('')||'<tr><td colspan="4" style="color:#555">No rate limit data</td></tr>';
  }catch(e){console.error(e)}
}
refresh();setInterval(refresh,3000);
</script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the admin dashboard."""

    dashboard_state: DashboardState | None = None

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/dashboard":
            self._serve_html()
        elif parsed.path == "/api/state":
            self._serve_json()
        elif parsed.path == "/api/health":
            self._send_json({"status": "ok"})
        else:
            self.send_error(404)

    def _serve_html(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(_DASHBOARD_HTML.encode())

    def _serve_json(self) -> None:
        if self.dashboard_state:
            self._send_json(self.dashboard_state.to_dict())
        else:
            self._send_json({"error": "No state available"})

    def _send_json(self, data: dict) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def log_message(self, format: str, *args: Any) -> None:
        pass  # Suppress default logging


def start_dashboard(
    state: DashboardState,
    host: str = "0.0.0.0",
    port: int = 8501,
) -> HTTPServer:
    """Start the admin dashboard server in a background thread.

    Args:
        state: DashboardState to serve.
        host: Bind host.
        port: Bind port.

    Returns:
        HTTPServer instance.
    """
    DashboardHandler.dashboard_state = state
    server = HTTPServer((host, port), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("Admin dashboard running at http://%s:%d", host, port)
    return server
