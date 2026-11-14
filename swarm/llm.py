"""All Claude API traffic goes through this module.

* `LLM.call`       one request -> text (streamed, so long outputs never time out)
* `LLM.call_many`  many independent requests; Message Batches API when enabled
* `LLM.agent`      a tool-using loop (implementers, literature review)

Every response is priced into state/ledger.jsonl and checked against the budget.
Set RVSWARM_MOCK_LLM=1 to exercise the orchestration offline with canned replies.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = ("claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5",
                   "claude-sonnet-5-5")


class BudgetExceeded(RuntimeError):
    pass


class Refused(RuntimeError):
    pass


class Ledger:
    def __init__(self, path: Path, pricing: dict, budget: dict):
        self.path, self.pricing, self.budget = Path(path), pricing, budget
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def add(self, model: str, usage, role: str, tag: str, cycle: int, batch: bool = False) -> float:
        g = lambda k: int(getattr(usage, k, 0) or 0)
        p = self.pricing.get(model, {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0})
        usd = (g("input_tokens") * p["input"] + g("output_tokens") * p["output"]
               + g("cache_read_input_tokens") * p["cache_read"]
               + g("cache_creation_input_tokens") * p["cache_write"]) / 1e6
        if batch:
            usd *= 0.5
        row = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "cycle": cycle, "role": role, "tag": tag,
               "model": model, "in": g("input_tokens"), "out": g("output_tokens"),
               "cache_read": g("cache_read_input_tokens"),
               "cache_write": g("cache_creation_input_tokens"), "usd": round(usd, 5), "batch": batch}
        with self._lock, open(self.path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        return usd

    def spent(self, cycle: int | None = None) -> dict:
        rows = self._rows()
        month = time.strftime("%Y-%m")
        return {"total": sum(r["usd"] for r in rows),
                "month": sum(r["usd"] for r in rows if r["t"].startswith(month)),
                "cycle": sum(r["usd"] for r in rows if r["cycle"] == cycle)}

    def check(self, cycle: int) -> None:
        s = self.spent(cycle)
        for key, lim in (("cycle", "usd_per_cycle"), ("month", "usd_per_month"), ("total", "usd_total")):
            if s[key] >= self.budget[lim]:
                raise BudgetExceeded(f"{lim} reached: spent ${s[key]:.2f} of ${self.budget[lim]}")


def _text(msg) -> str:
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


class LLM:
    def __init__(self, cfg: dict, state_dir: Path):
        self.cfg = cfg
        self.state = Path(state_dir)
        self.ledger = Ledger(self.state / "ledger.jsonl", cfg["pricing"], cfg["budget"])
        self.cycle = 0
        self.mock = None
        if os.environ.get("RVSWARM_MOCK_LLM"):
            from . import mock
            self.mock = mock
            self.client = None
        else:
            self.client = anthropic.Anthropic(max_retries=cfg["api"]["max_retries"], timeout=1800.0)

    # ------------------------------------------------------------------ request building
    def _role(self, role: str) -> tuple[str, dict]:
        r = self.cfg["roles"][role]
        return self.cfg["models"][r["model"]], r

    def _params(self, role: str, system: str, user, schema: dict | None, tools=None) -> dict:
        model, r = self._role(role)
        oc: dict = {"effort": r["effort"]}
        if schema is not None:
            oc["format"] = {"type": "json_schema", "schema": schema}
        p = {"model": model, "max_tokens": r["max_tokens"], "output_config": oc,
             "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
             "messages": user if isinstance(user, list) else [{"role": "user", "content": user}]}
        if tools:
            p["tools"] = tools
        return p

    def _live(self, params: dict, role: str, tag: str):
        """One streamed request with typed error handling; returns the final message."""
        self.ledger.check(self.cycle)
        kw = dict(params)
        if self.cfg["api"]["refusal_fallbacks"] and kw["model"] in FALLBACK_MODELS:
            kw["betas"] = [FALLBACK_BETA]
            kw["fallbacks"] = "default"
        for attempt in range(8):
            try:
                with self.client.beta.messages.stream(**kw) as stream:
                    msg = stream.get_final_message()
                break
            except anthropic.RateLimitError as e:
                wait = float(e.response.headers.get("retry-after", 0) or 0) or min(600, 30 * 2 ** attempt)
                print(f"  rate limited ({tag}); sleeping {wait:.0f}s", flush=True)
                time.sleep(wait)
            except anthropic.APIStatusError as e:
                if e.status_code >= 500 or e.status_code == 409:
                    time.sleep(min(600, 20 * 2 ** attempt))
                else:
                    raise
            except (anthropic.APIConnectionError, anthropic.APITimeoutError):
                time.sleep(min(600, 20 * 2 ** attempt))
        else:
            raise RuntimeError(f"request {tag} failed after repeated retries")
        self.ledger.add(msg.model, msg.usage, role, tag, self.cycle)
        if msg.stop_reason == "refusal":
            cat = getattr(getattr(msg, "stop_details", None), "category", None)
            raise Refused(f"{tag}: request declined (category={cat})")
        return msg

    # ------------------------------------------------------------------ single calls
    def call(self, role: str, system: str, user: str, schema: dict | None = None, tag: str = ""):
        """Returns text, or the parsed object when `schema` is given."""
        if self.mock:
            return self.mock.respond(role, user, schema, tag)
        params = self._params(role, system, user, schema)
        msg = self._live(params, role, tag)
        if msg.stop_reason == "max_tokens" and params["max_tokens"] < 64000:
            params["max_tokens"] = 64000
            msg = self._live(params, role, tag + "#retry")
        return self._finish(_text(msg), schema, tag)

    @staticmethod
    def _finish(text: str, schema, tag: str):
        if schema is None:
            return text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                return json.loads(m.group(0))
            raise ValueError(f"{tag}: model did not return JSON")

    def call_many(self, role: str, system: str, items: dict[str, str], schema: dict | None = None,
                  phase_tag: str = "") -> dict:
        """items: {key: user_prompt}. Returns {key: text-or-object}; failed keys are omitted."""
        if not items:
            return {}
        if self.mock:
            return {k: self.mock.respond(role, u, schema, k) for k, u in items.items()}
        if self.cfg["api"]["use_batches"] and len(items) > 1:
            return self._batch(role, system, items, schema, phase_tag)
        out = {}

        def one(kv):
            k, u = kv
            try:
                return k, self.call(role, system, u, schema, tag=f"{phase_tag}:{k}")
            except (Refused, ValueError) as e:
                print(f"  {phase_tag}:{k} skipped: {e}", flush=True)
                return k, None

        with ThreadPoolExecutor(self.cfg["api"]["max_concurrency"]) as ex:
            for k, v in ex.map(one, items.items()):
                if v is not None:
                    out[k] = v
        return out

    def _batch(self, role, system, items, schema, phase_tag) -> dict:
        """Submit once, remember the batch id on disk, poll until done (resumable)."""
        self.ledger.check(self.cycle)
        bdir = self.state / "batches"
        bdir.mkdir(parents=True, exist_ok=True)
        rec_file = bdir / (re.sub(r"[^A-Za-z0-9_.-]", "_", phase_tag) + ".json")
        keys = list(items)
        ids = {f"r{i:04d}": k for i, k in enumerate(keys)}
        if rec_file.exists():
            rec = json.loads(rec_file.read_text())
        else:
            reqs = [{"custom_id": cid, "params": self._params(role, system, items[k], schema)}
                    for cid, k in ids.items()]
            b = self.client.messages.batches.create(requests=reqs)
            rec = {"batch_id": b.id, "ids": ids, "created": time.time()}
            rec_file.write_text(json.dumps(rec))
            print(f"  batch {b.id} submitted for {phase_tag} ({len(reqs)} requests)", flush=True)
        ids = rec["ids"]
        deadline = rec["created"] + self.cfg["api"]["batch_max_wait_hours"] * 3600
        while True:
            b = self.client.messages.batches.retrieve(rec["batch_id"])
            if b.processing_status == "ended":
                break
            if time.time() > deadline:
                raise RuntimeError(f"batch {rec['batch_id']} did not finish in time; re-run to resume")
            time.sleep(self.cfg["api"]["batch_poll_seconds"])
        out, redo = {}, []
        for res in self.client.messages.batches.results(rec["batch_id"]):
            k = ids.get(res.custom_id)
            if k is None or k not in items:
                continue
            if res.result.type != "succeeded":
                redo.append(k)
                continue
            msg = res.result.message
            self.ledger.add(msg.model, msg.usage, role, f"{phase_tag}:{k}", self.cycle, batch=True)
            if msg.stop_reason in ("refusal", "max_tokens"):
                redo.append(k)
                continue
            try:
                out[k] = self._finish(_text(msg), schema, k)
            except ValueError:
                redo.append(k)
        for k in redo:                                   # live retry (with fallbacks, larger limit)
            try:
                out[k] = self.call(role, system, items[k], schema, tag=f"{phase_tag}:{k}#live")
            except (Refused, ValueError) as e:
                print(f"  {phase_tag}:{k} skipped: {e}", flush=True)
        rec_file.rename(rec_file.with_suffix(".done.json"))
        return out

    # ------------------------------------------------------------------ tool-using agent
    def agent(self, role: str, system: str, user: str, toolset, max_turns: int, tag: str = "",
              server_tools: list | None = None) -> str:
        """Run a tool loop until the model stops calling tools. Returns its final text."""
        if self.mock:
            return self.mock.agent(role, user, toolset, tag)
        tools = list(toolset.definitions()) + list(server_tools or [])
        messages = [{"role": "user", "content": user}]
        final = ""
        for turn in range(max_turns):
            params = self._params(role, system, messages, None, tools)
            params["cache_control"] = {"type": "ephemeral"}       # cache the growing transcript
            msg = self._live(params, role, f"{tag}#t{turn}")
            messages.append({"role": "assistant", "content": msg.content})
            final = _text(msg) or final
            calls = [b for b in msg.content if b.type == "tool_use"]
            if msg.stop_reason == "pause_turn":
                continue
            if calls:
                results = []
                for c in calls:
                    if msg.stop_reason == "max_tokens":
                        body, err = ("This tool call was cut off by the output limit and was NOT executed. "
                                     "Write large files in several smaller calls (write_file, then "
                                     "append_file)."), True
                    else:
                        body, err = toolset.execute(c.name, c.input)
                    results.append({"type": "tool_result", "tool_use_id": c.id, "content": body[:60000],
                                    "is_error": err})
                if turn == max_turns - 3:
                    results.append({"type": "text", "text": "You have 2 turns left. Finish now: make sure "
                                    "every required file is written, then stop calling tools."})
                messages.append({"role": "user", "content": results})
                continue
            if msg.stop_reason == "max_tokens":
                messages.append({"role": "user", "content": "Your message was cut off. Continue from where "
                                                             "you stopped."})
                continue
            break
        return final
