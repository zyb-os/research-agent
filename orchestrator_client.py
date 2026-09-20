"""
orchestrator_client.py — WebSocket + HTTP client for the research-agent.

Registers with the orchestrator, handles research task_requests via the LLM
proxy, and applies settings pushed live from the dashboard.
"""
from __future__ import annotations

import asyncio
import json
import logging
import signal
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
import websockets
import websockets.exceptions

from research_engine import ResearchEngine

logger = logging.getLogger(__name__)

# ── Stable agent identity ──────────────────────────────────────────────────────

_AGENT_ID_FILE = Path(".agent_id")


def _stable_agent_id() -> str:
    if _AGENT_ID_FILE.exists():
        return _AGENT_ID_FILE.read_text().strip()
    new_id = str(uuid.uuid4())
    _AGENT_ID_FILE.write_text(new_id)
    logger.info("Generated new stable agent ID: %s", new_id)
    return new_id


# ── Registration payload ───────────────────────────────────────────────────────

AGENT_NAME        = "research-agent"
AGENT_VERSION     = "1.0.0"
AGENT_DESCRIPTION = (
    "LLM-powered research agent. Researches any topic and returns a structured "
    "response with a summary, key findings, detailed sections, key concepts, "
    "and related topics. Depth and model are configurable per request or via "
    "dashboard defaults."
)

REGISTRATION_PAYLOAD: dict = {
    "name":        AGENT_NAME,
    "description": AGENT_DESCRIPTION,
    "version":     AGENT_VERSION,
    "tags":        ["research", "llm", "analysis", "knowledge", "summary", "report"],
    "capabilities": [
        {
            "name": "research_topic",
            "description": (
                "Research any topic using an LLM and return a structured report. "
                "Covers any domain: technology, science, history, business, medicine, "
                "law, culture, current events, and more. Returns summary, key findings, "
                "detailed sections, key concepts, related topics, and confidence level."
            ),
            "tags": [
                "research", "analysis", "report", "summarise", "explain",
                "investigate", "study", "learn", "topic", "knowledge",
                "overview", "deep-dive", "breakdown", "fact-finding",
            ],
            "input_schema": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The subject or question to research.",
                    },
                    "depth": {
                        "type": "string",
                        "enum": ["brief", "standard", "detailed"],
                        "description": (
                            "Research depth. 'brief' = concise overview, "
                            "'standard' = balanced depth (default), "
                            "'detailed' = comprehensive deep-dive."
                        ),
                    },
                    "focus_areas": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional sub-topics or angles to emphasise, "
                            "e.g. ['economic impact', 'historical context']."
                        ),
                    },
                    "output_sections": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional list of section titles the response must include, "
                            "e.g. ['Overview', 'Pros and Cons', 'Future Outlook']."
                        ),
                    },
                },
                "required": ["topic"],
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "topic":          {"type": "string"},
                    "summary":        {"type": "string", "description": "2–3 sentence executive summary."},
                    "key_findings":   {"type": "array",  "items": {"type": "string"}},
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title":   {"type": "string"},
                                "content": {"type": "string"},
                            },
                        },
                    },
                    "key_concepts":   {"type": "array", "items": {"type": "string"}},
                    "related_topics": {"type": "array", "items": {"type": "string"}},
                    "confidence":     {"type": "string", "enum": ["high", "medium", "low"]},
                    "caveats":        {"type": "string"},
                    "research_depth": {"type": "string"},
                    "model_used":     {"type": "string"},
                    "provider_used":  {"type": "string"},
                },
            },
        },
    ],
    "required_settings": [
        {
            "key":         "research_model",
            "label":       "Model",
            "type":        "string",
            "required":    False,
            "description": (
                "LLM model used for research. Leave blank to use the global default. "
                "Examples: claude-sonnet-4-6, claude-opus-4-6, gpt-4o, gemini-2.5-flash"
            ),
            "default": "",
        },
        {
            "key":         "research_provider",
            "label":       "Provider",
            "type":        "string",
            "required":    False,
            "description": "LLM provider: anthropic, openai, or gemini. Leave blank for global default.",
            "default":     "",
            "options":     ["", "anthropic", "openai", "gemini"],
        },
        {
            "key":         "research_default_depth",
            "label":       "Default Depth",
            "type":        "string",
            "required":    False,
            "description": (
                "Default research depth when no depth is specified per-request. "
                "brief = concise overview, standard = balanced, detailed = comprehensive."
            ),
            "default":  "standard",
            "options":  ["brief", "standard", "detailed"],
        },
        {
            "key":         "research_max_tokens",
            "label":       "Max Output Tokens",
            "type":        "integer",
            "required":    False,
            "description": "Maximum tokens in the LLM response. Range: 512–8192. Default: 2048.",
            "default":     2048,
            "min_value":   512,
            "max_value":   8192,
        },
        {
            "key":         "research_language",
            "label":       "Response Language",
            "type":        "string",
            "required":    False,
            "description": "Language for research output. Default: English.",
            "default":     "English",
        },
    ],
}

# ── Constants ──────────────────────────────────────────────────────────────────

HEARTBEAT_INTERVAL_S: int = 15
MAX_BACKOFF_S:         int = 60
DRAIN_TIMEOUT_S:       int = 30


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _envelope(
    sender_id:      str,
    msg_type:       str,
    payload:        dict,
    recipient_id:   Optional[str] = None,
    correlation_id: Optional[str] = None,
    msg_id:         Optional[str] = None,
) -> str:
    return json.dumps({
        "id":             msg_id or str(uuid.uuid4()),
        "type":           msg_type,
        "sender_id":      sender_id,
        "recipient_id":   recipient_id,
        "payload":        payload,
        "timestamp":      _now_iso(),
        "correlation_id": correlation_id,
    })


# ── Main client ────────────────────────────────────────────────────────────────

class OrchestratorClient:
    """Registers the research-agent and dispatches incoming task_requests."""

    def __init__(self, orchestrator_url: str = "http://localhost:8000") -> None:
        self._base = orchestrator_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=30)

        self._agent_id:       str = ""
        self._ws_url:         str = ""
        self._common_settings: dict[str, Any] = {}
        self._agent_settings:  dict[str, Any] = {}

        self._status:            str   = "starting"
        self._active_tasks:      int   = 0
        self._tasks_completed:   int   = 0
        self._tasks_failed:      int   = 0
        self._total_duration_ms: float = 0.0
        self._start_time:        float = time.monotonic()

        self._shutting_down: bool = False
        self._current_ws:    Any  = None

        # Engine initialised after registration (needs proxy_url + agent_id)
        self._engine: Optional[ResearchEngine] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self._graceful_shutdown()))

        await self._register()
        await self._connect_loop()

    # ── Registration ───────────────────────────────────────────────────────────

    async def _register(self) -> None:
        url = f"{self._base}/api/v1/agents/register"
        logger.info("Registering with orchestrator at %s …", url)
        payload = {**REGISTRATION_PAYLOAD, "id": _stable_agent_id()}
        resp = await self._http.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

        self._agent_id       = data["agent_id"]
        self._ws_url         = data["ws_url"]
        self._common_settings = data.get("common_settings", {})
        self._agent_settings  = data.get("agent_settings", {})

        proxy_url = f"{self._base}/api/v1/llm/complete"
        self._engine = ResearchEngine(proxy_url=proxy_url, agent_id=self._agent_id)
        self._engine.update_settings(self._common_settings, self._agent_settings)

        logger.info("Registered — agent_id=%s  ws=%s", self._agent_id, self._ws_url)

    # ── WebSocket loop ─────────────────────────────────────────────────────────

    async def _connect_loop(self) -> None:
        backoff = 1.0
        while not self._shutting_down:
            try:
                logger.info("Connecting to %s …", self._ws_url)
                async with websockets.connect(self._ws_url) as ws:
                    backoff = 1.0
                    await self._run_session(ws)

            except websockets.exceptions.ConnectionClosed as exc:
                code = exc.rcvd.code if exc.rcvd else None
                if code == 4004:
                    logger.warning("Unknown agent_id (4004) — re-registering …")
                    try:
                        await self._register()
                    except Exception as reg_exc:
                        logger.error("Re-registration failed: %s", reg_exc)
                elif code == 4003:
                    logger.info("Agent disabled (4003) — will retry")
                    backoff = max(backoff, 10.0)
                elif self._shutting_down:
                    break
                else:
                    logger.warning("WS closed (code=%s) — retry in %.0fs", code, backoff)

            except (OSError, Exception) as exc:
                if self._shutting_down:
                    break
                logger.warning("WS error (%s) — retry in %.0fs", exc, backoff)

            if not self._shutting_down:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF_S)

    async def _run_session(self, ws) -> None:
        self._current_ws = ws
        self._status = "available"
        logger.info("WebSocket session active")
        try:
            await asyncio.gather(
                self._heartbeat_loop(ws),
                self._recv_loop(ws),
            )
        finally:
            self._current_ws = None
            self._status = "offline"

    # ── Heartbeat ──────────────────────────────────────────────────────────────

    async def _heartbeat_loop(self, ws) -> None:
        while True:
            await self._ws_send(ws, self._msg(
                "heartbeat",
                {
                    "status":       self._status,
                    "current_load": min(self._active_tasks / 3.0, 1.0),
                    "active_tasks": self._active_tasks,
                    "metrics":      self._metrics(),
                },
            ))
            await asyncio.sleep(HEARTBEAT_INTERVAL_S)

    # ── Receive loop ───────────────────────────────────────────────────────────

    async def _recv_loop(self, ws) -> None:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Non-JSON frame ignored")
                continue
            mtype = msg.get("type", "?")
            _lvl = logging.DEBUG if mtype in ("agent_registered", "agent_offline", "heartbeat_ack", "settings_push") else logging.INFO
            logger.log(_lvl, "← [%s] from=%s", mtype, msg.get("sender_id", "?"))
            await self._dispatch(ws, msg)

    async def _dispatch(self, ws, msg: dict) -> None:
        mtype   = msg.get("type", "")
        payload = msg.get("payload", {})

        if mtype == "task_request":
            asyncio.create_task(self._handle_task(ws, msg))

        elif mtype == "settings_push":
            settings = payload.get("settings", {})
            self._common_settings.update(settings)
            for k in (
                "research_model", "research_provider", "research_default_depth",
                "research_max_tokens", "research_language",
            ):
                if k in settings:
                    self._agent_settings[k] = settings[k]
            if self._engine is not None:
                self._engine.update_settings(self._common_settings, self._agent_settings)
            logger.info("Settings pushed: %d key(s) applied", len(settings))

        elif mtype in ("agent_registered", "agent_offline", "broadcast", "discovery_response"):
            logger.debug("Event [%s]: %s", mtype, payload.get("agent_id", ""))

        elif mtype == "agent_restart":
            logger.info("Restart requested by orchestrator — shutting down for restart")
            asyncio.create_task(self._graceful_shutdown())
            import sys
            asyncio.get_event_loop().call_later(1.0, lambda: sys.exit(0))

        elif mtype == "error":
            logger.error(
                "Orchestrator error [%s]: %s",
                payload.get("code"), payload.get("detail"),
            )

        else:
            logger.debug("Unhandled message type: %r", mtype)

    # ── Task handling ──────────────────────────────────────────────────────────

    async def _handle_task(self, ws, msg: dict) -> None:
        req_id     = msg.get("id")
        sender_id  = msg.get("sender_id")
        payload    = msg.get("payload", {})
        capability = payload.get("capability", "")
        input_data = payload.get("input_data", {})

        self._active_tasks += 1
        self._status = "busy"
        t0 = time.monotonic()

        try:
            output, error = await self._dispatch_capability(capability, input_data)
            duration_ms = (time.monotonic() - t0) * 1000

            if error:
                self._tasks_failed += 1
                await self._ws_send(ws, self._msg(
                    "task_response",
                    {"success": False, "error": error, "duration_ms": round(duration_ms, 1)},
                    recipient_id=sender_id,
                    correlation_id=req_id,
                ))
            else:
                self._tasks_completed += 1
                self._total_duration_ms += duration_ms
                await self._ws_send(ws, self._msg(
                    "task_response",
                    {"success": True, "output_data": output, "duration_ms": round(duration_ms, 1)},
                    recipient_id=sender_id,
                    correlation_id=req_id,
                ))

        except Exception as exc:
            duration_ms = (time.monotonic() - t0) * 1000
            self._tasks_failed += 1
            logger.exception("Unhandled error in capability %r", capability)
            await self._ws_send(ws, self._msg(
                "task_response",
                {"success": False, "error": str(exc), "duration_ms": round(duration_ms, 1)},
                recipient_id=sender_id,
                correlation_id=req_id,
            ))

        finally:
            self._active_tasks = max(0, self._active_tasks - 1)
            self._status = "draining" if self._shutting_down else (
                "busy" if self._active_tasks else "available"
            )
            await self._send_status_update(ws)

    async def _dispatch_capability(
        self, capability: str, input_data: dict
    ) -> tuple[Optional[dict], Optional[str]]:
        if capability != "research_topic":
            return None, f"Unknown capability: {capability!r}"
        if self._engine is None:
            return None, "Research engine not initialised"

        topic = str(input_data.get("topic", "")).strip()
        if not topic:
            return None, "input_data.topic is required and must be a non-empty string"

        depth           = input_data.get("depth")
        focus_areas     = input_data.get("focus_areas")
        output_sections = input_data.get("output_sections")

        result = await self._engine.research(
            topic           = topic,
            depth           = depth if isinstance(depth, str) else None,
            focus_areas     = focus_areas if isinstance(focus_areas, list) else None,
            output_sections = output_sections if isinstance(output_sections, list) else None,
        )
        return result, None

    # ── Status update ──────────────────────────────────────────────────────────

    async def _send_status_update(self, ws) -> None:
        await self._ws_send(ws, self._msg(
            "status_update",
            {
                "status":       self._status,
                "current_load": min(self._active_tasks / 3.0, 1.0),
                "active_tasks": self._active_tasks,
                "metrics":      self._metrics(),
            },
        ))

    # ── Graceful shutdown ──────────────────────────────────────────────────────

    async def _graceful_shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        logger.info("Shutdown signal received — draining …")
        self._status = "draining"

        deadline = time.monotonic() + DRAIN_TIMEOUT_S
        while self._active_tasks > 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.5)

        if self._agent_id:
            try:
                await self._http.delete(f"{self._base}/api/v1/agents/{self._agent_id}")
                logger.info("Deregistered from orchestrator.")
            except Exception as exc:
                logger.warning("Deregister failed: %s", exc)

        await self._http.aclose()
        logger.info("Shutdown complete.")

    # ── Helpers ────────────────────────────────────────────────────────────────

    async def _ws_send(self, ws, msg_str: str) -> None:
        msg   = json.loads(msg_str)
        mtype = msg.get("type", "?")
        noisy = mtype in ("heartbeat", "status_update")
        (logger.debug if noisy else logger.info)(
            "→ [%s] to=%s", mtype, msg.get("recipient_id") or "orchestrator"
        )
        try:
            await ws.send(msg_str)
        except websockets.exceptions.ConnectionClosed:
            raise
        except Exception as exc:
            logger.warning("WS send failed: %s", exc)

    def _msg(
        self,
        msg_type:       str,
        payload:        dict,
        recipient_id:   Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> str:
        return _envelope(self._agent_id, msg_type, payload, recipient_id, correlation_id)

    def _metrics(self) -> dict:
        n = self._tasks_completed + self._tasks_failed
        return {
            "tasks_completed":      self._tasks_completed,
            "tasks_failed":         self._tasks_failed,
            "avg_response_time_ms": round(self._total_duration_ms / n, 1) if n else 0.0,
            "uptime_seconds":       round(time.monotonic() - self._start_time, 1),
        }
