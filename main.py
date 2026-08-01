"""
Agent 1: Researcher — Novantix Capital
------------------------------------------
Founder: Hrich Souhail, Financial Engineer, CEO & Founder of Novantix Capital.

Paste a plain-English strategy description or a research paper excerpt.
The agent implements it LITERALLY first (no improvements, no fixes) and runs
it once — that becomes the frozen baseline, shown raw, good or bad. Only
after the baseline is locked can it propose hypothesis-driven variants, each
one run and compared against that frozen baseline. Variants must contain a
genuine logic change (checked by code hash, not just label text) or they're
rejected. A third run type, 'inspect', lets it answer factual questions about
the data/trades with real execution instead of guessing.

Run with:
    pip install streamlit groq pandas numpy pyarrow
    streamlit run app.py

Groq API key: https://console.groq.com
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import os
import re
import hashlib
import datetime
import multiprocessing as mp
import traceback
from contextlib import redirect_stdout

ACCESS_CODE = "algo8080"
FOUNDER_NAME = "Hrich Souhail"
FOUNDER_TITLE = "Financial Engineer, CEO & Founder of Novantix Capital"
LEARNINGS_FILE = "novantix_research_log.json"

# ============================================================
# DARK THEME
# ============================================================
st.set_page_config(page_title="Agent 1: Researcher — Novantix Capital", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #0e0e10; color: #e6e6e6; }
    section[data-testid="stSidebar"] { background-color: #161618; }
    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
        background-color: #1c1c1f !important; color: #e6e6e6 !important; border-color: #333 !important;
    }
    .stButton button { background-color: #2b2b30; color: #e6e6e6; border: 1px solid #444; }
    .stButton button:hover { background-color: #3a3a40; border-color: #666; }
    .streamlit-expanderHeader { background-color: #1c1c1f; color: #e6e6e6; }
    div[data-testid="stExpander"] { background-color: #141416; border: 1px solid #2a2a2e; }
    code { color: #ffb454 !important; }
    div[data-testid="stChatMessage"] { background-color: #17171a; border: 1px solid #26262a; }
    div[data-testid="stMetric"] { background-color: #141416; border: 1px solid #2a2a2e; border-radius: 8px; padding: 8px; }
    h1, h2, h3, h4 { color: #f2f2f2; }
    .stCaption, p, span, label { color: #c9c9cc; }
    .baseline-badge { background-color: #1f3d2b; color: #7ee2a0; padding: 3px 10px; border-radius: 12px; font-size: 0.8em; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# ACCESS GATE
# ============================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Novantix Capital — Agent 1: Researcher")
    st.caption("Restricted access.")
    code_input = st.text_input("Access code", type="password")
    if st.button("Unlock"):
        if code_input == ACCESS_CODE:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect code.")
    st.stop()

# ============================================================
# DATA QUALITY AUDIT
# ============================================================

def audit_data_quality(df: pd.DataFrame) -> list:
    findings = []
    n = len(df)

    null_counts = df.isnull().sum()
    bad_cols = null_counts[null_counts > 0]
    if len(bad_cols) > 0:
        findings.append(f"Missing values found in: {dict(bad_cols)}")

    dup_rows = df.duplicated().sum()
    if dup_rows > 0:
        findings.append(f"{dup_rows} fully duplicate row(s) found.")

    date_col = next((c for c in df.columns if "date" in c.lower() or "time" in c.lower()), None)
    if date_col:
        try:
            parsed = pd.to_datetime(df[date_col], errors="coerce")
            n_unparsed = parsed.isna().sum()
            if n_unparsed > 0:
                findings.append(f"{n_unparsed} row(s) in '{date_col}' failed to parse as a date/time.")
            if parsed.notna().sum() > 1:
                diffs = parsed.dropna().diff().dropna()
                if not diffs.is_monotonic_increasing and (diffs < pd.Timedelta(0)).any():
                    findings.append(f"Timestamps in '{date_col}' are not monotonically increasing.")
                if len(diffs) > 0:
                    mode_gap = diffs.mode()
                    if len(mode_gap) > 0:
                        big_gaps = (diffs > mode_gap.iloc[0] * 5).sum()
                        if big_gaps > 0:
                            findings.append(f"{big_gaps} unusually large time gap(s) detected — possible missing sessions/data holes.")
        except Exception:
            findings.append(f"Could not parse '{date_col}' as datetime — verify its format.")
    else:
        findings.append("No obvious date/time column detected — daily/session-based rules may be hard to compute reliably.")

    price_cols = [c for c in df.columns if c.lower() in ("open", "high", "low", "close")]
    for c in price_cols:
        try:
            numeric = pd.to_numeric(df[c], errors="coerce")
            if (numeric <= 0).sum() > 0:
                findings.append(f"'{c}' contains {int((numeric <= 0).sum())} zero or negative value(s).")
        except Exception:
            pass

    lower_cols = [c.lower() for c in df.columns]
    if "high" in lower_cols and "low" in lower_cols:
        h = pd.to_numeric(df[[c for c in df.columns if c.lower() == "high"][0]], errors="coerce")
        l = pd.to_numeric(df[[c for c in df.columns if c.lower() == "low"][0]], errors="coerce")
        bad = (h < l).sum()
        if bad > 0:
            findings.append(f"{bad} row(s) where high < low — physically impossible bars.")

    if n < 500:
        findings.append(f"Only {n} rows total — likely too little data for a statistically meaningful backtest.")

    return findings

# ============================================================
# PERSISTENT RESEARCH LOG
# ============================================================

def load_learnings():
    if os.path.exists(LEARNINGS_FILE):
        try:
            with open(LEARNINGS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_learning(entry: dict):
    log = load_learnings()
    log.append(entry)
    try:
        with open(LEARNINGS_FILE, "w") as f:
            json.dump(log[-200:], f, indent=2, default=str)
    except Exception:
        pass

def summarize_past_learnings(max_entries=6) -> str:
    log = load_learnings()
    if not log:
        return "No prior research sessions logged yet."
    recent = log[-max_entries:]
    return "\n".join(f"- [{e.get('type', '?')}] {e.get('summary', '')}" for e in recent)

# ============================================================
# SANDBOXED EXECUTION
# ============================================================
ALLOWED_IMPORTS = {"pandas", "numpy", "math", "datetime", "itertools", "statistics", "collections"}

def _restricted_import(name, *args, **kwargs):
    root = name.split(".")[0]
    if root not in ALLOWED_IMPORTS:
        raise ImportError(f"Import of '{name}' is not allowed in the sandbox.")
    return __import__(name, *args, **kwargs)

def _worker(code, df_bytes, queue):
    try:
        df = pd.read_parquet(io.BytesIO(df_bytes)) if df_bytes else None

        # Compatibility shim: LLMs often confuse np./pd. namespaces (e.g. np.to_datetime).
        # Only fill gaps, never override real numpy functions.
        for _name in ("to_datetime", "concat", "notnull", "isnull"):
            if not hasattr(np, _name):
                setattr(np, _name, getattr(pd, _name))

        safe_builtins = {
            "__import__": _restricted_import, "range": range, "len": len, "min": min, "max": max,
            "sum": sum, "abs": abs, "round": round, "sorted": sorted, "enumerate": enumerate,
            "zip": zip, "list": list, "dict": dict, "set": set, "tuple": tuple, "float": float,
            "int": int, "str": str, "bool": bool, "print": print, "isinstance": isinstance,
            "Exception": Exception, "ValueError": ValueError, "True": True, "False": False, "None": None,
        }
        namespace = {"__builtins__": safe_builtins, "pd": pd, "np": np, "df": df.copy() if df is not None else None}
        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf):
            exec(code, namespace)
        results = namespace.get("results", None)
        if results is None:
            queue.put({"ok": False, "error": "Code did not set a `results` dict.", "stdout": stdout_buf.getvalue()})
            return
        json.dumps(results, default=str)
        queue.put({"ok": True, "results": results, "stdout": stdout_buf.getvalue()})
    except Exception:
        queue.put({"ok": False, "error": traceback.format_exc(), "stdout": ""})

def run_code_safely(code, df, timeout=30):
    buf = io.BytesIO()
    if df is not None:
        df.to_parquet(buf)
    df_bytes = buf.getvalue()
    queue = mp.Queue()
    proc = mp.Process(target=_worker, args=(code, df_bytes, queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {"ok": False, "error": f"Execution timed out after {timeout}s.", "stdout": ""}
    if not queue.empty():
        return queue.get()
    return {"ok": False, "error": "Process ended with no output.", "stdout": ""}

def _json_default(obj):
    """Handles numpy types (int64, float64, bool_, ndarray) json.dumps can't serialize natively."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

# ============================================================
# INDEPENDENT AUDIT (strategy code)
# ============================================================

def audit_code(code: str) -> list:
    warnings = []
    lower = code.lower()

    has_signal = bool(re.search(r"\b(buy|sell|signal|position|entry)\b", lower))
    has_return_calc = bool(re.search(r"pct_change|diff\(\)", lower))
    has_shift = ".shift(" in lower
    if has_signal and has_return_calc and not has_shift:
        warnings.append(
            "LOOKAHEAD BIAS RISK: signal/position applied to returns with no `.shift()` found. "
            "A signal from bar t's close must be shifted forward before capturing bar t's return."
        )

    dd_match = re.search(r"drawdown\s*=\s*\(.*?\)\s*/\s*([a-zA-Z_\[\]'\"]+)", code)
    if dd_match:
        denom = dd_match.group(1).lower()
        if "cum_p" in denom or ("cum" in denom and "equity" not in denom and "balance" not in denom):
            warnings.append(
                "DRAWDOWN FORMULA RISK: divided by raw cumulative P&L instead of an equity curve "
                "(starting capital + cumulative P&L)."
            )

    if "daily" in lower and "rolling(" in lower and "groupby" not in lower:
        warnings.append(
            "DAILY METRIC RISK: 'daily' appears to use a fixed rolling window rather than grouping "
            "by actual calendar date."
        )

    if "equity" not in lower and "balance" not in lower and "capital" not in lower:
        warnings.append("NO EXPLICIT EQUITY CURVE: no account equity/balance/capital tracked from a defined starting size.")

    return warnings

def format_audit_note(warnings):
    if not warnings:
        return "[AUTOMATED AUDIT] No issues flagged for this execution."
    lines = ["[AUTOMATED AUDIT] Issues detected:"] + [f"- {w}" for w in warnings]
    lines.append("Address these or explain why they don't apply.")
    return "\n".join(lines)

# ============================================================
# DEDUP: hypothesis text + code logic
# ============================================================

def _normalize_code_for_dedup(code: str) -> str:
    """Strips comments, whitespace, and the cosmetic `notes` line so a variant
    that only relabels itself hashes identically to what it copied."""
    lines = []
    for line in code.split("\n"):
        line = re.sub(r"#.*$", "", line).strip()
        if not line:
            continue
        if re.match(r"^['\"]?notes['\"]?\s*[:=]", line):
            continue
        lines.append(re.sub(r"\s+", " ", line))
    return "\n".join(lines)

def _code_hash(code: str) -> str:
    return hashlib.sha256(_normalize_code_for_dedup(code).encode()).hexdigest()

def _hypothesis_too_similar(new_hyp, existing_hyps, threshold=0.75):
    new_words = set(re.findall(r"\w+", new_hyp.lower()))
    if not new_words:
        return False
    for h in existing_hyps:
        h_words = set(re.findall(r"\w+", h.lower()))
        if not h_words:
            continue
        overlap = len(new_words & h_words) / len(new_words | h_words)
        if overlap >= threshold:
            return True
    return False

# ============================================================
# TOOL SCHEMA
# ============================================================

TOOL_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "run_backtest_code",
        "description": (
            "Execute Python against the uploaded data (`df`). Code MUST set a `results` dict with: "
            "profit_pct, max_drawdown_pct, max_daily_loss_pct, trading_days, win_rate_pct, "
            "total_trades, notes. Only pandas, numpy, math, datetime, itertools, statistics, "
            "collections may be imported.\n\n"
            "run_type must be:\n"
            "- 'baseline': ONLY the first literal, unmodified translation of the strategy. Can only "
            "happen ONCE — a second attempt is rejected by the system. No improvements or fixes, "
            "even obvious ones.\n"
            "- 'variant': every new strategy attempt after baseline exists. Requires a `hypothesis` "
            "string naming the specific mechanism being tested. The system checks the code actually "
            "differs in substance from the baseline and every prior variant, not just the notes/label "
            "— a variant that doesn't genuinely change the logic will be rejected.\n"
            "- 'inspect': for diagnostic/exploratory code that is NOT a new strategy — e.g. showing "
            "individual trade entry/exit prices, checking a calculation, printing rows of a computed "
            "column. Use this whenever asked a specific factual question about the data or a prior "
            "run's behavior. Never counts as a baseline or variant, no hypothesis required — but it "
            "must be a real execution. Never state specific trade prices, counts, or figures without "
            "having just run inspect code to produce them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Full Python code to execute."},
                "run_type": {"type": "string", "enum": ["baseline", "variant", "inspect"]},
                "hypothesis": {"type": "string", "description": "Required for variants."},
            },
            "required": ["code", "run_type"],
        },
    },
}]

SYSTEM_PROMPT_TEMPLATE = """You are Agent 1: Researcher at Novantix Capital, founded by {founder_name}
({founder_title}). You report to him directly — your job is to be a genuinely useful research
collaborator, not a chatbot that hedges everything with disclaimers instead of doing the work.

{data_desc}

DATA QUALITY AUDIT (run automatically on upload — read this before writing any strategy code):
{data_quality}

RESEARCH LOG FROM PAST SESSIONS (carry these lessons forward — don't repeat known mistakes):
{past_learnings}

THIS SESSION SO FAR (baseline + variants already tested — don't repeat these, and note that older
raw execution details may have been trimmed from your visible context to save tokens; this summary
is the reliable record of what's already been tried):
{session_so_far}

Your process is strict and two-phase:

PHASE 1 — LITERAL BASELINE (happens exactly once):
Implement the given strategy as literally and faithfully as possible — no improvements, no bug
fixes, no tweaks, even ones you spot immediately. Run it with run_type='baseline'. Report the raw
result honestly, even if it's bad. This baseline is frozen the moment it succeeds — the system will
not let you run a second baseline. If genuinely ambiguous, ask before running.

If the data quality audit flagged real issues (gaps, duplicates, non-monotonic timestamps,
impossible bars), address them explicitly in your baseline code and say plainly what you did, so
the fix is visible, not silent.

PHASE 2 — HYPOTHESIS-DRIVEN ITERATION (after baseline exists):
Every change is a run_type='variant' with a specific `hypothesis` naming the mechanism, grounded in
real market/strategy reasoning. Every variant is compared against the frozen baseline automatically.
Never claim a variant is better without having run it this turn.

Always:
- NO LOOKAHEAD BIAS: any signal from bar t must be shifted forward before capturing a return.
- Track a real equity curve from an explicit starting capital.
- Ask when something's genuinely unclear rather than guessing on something that would change the science.
- Address every automated audit warning or explain concretely why it doesn't apply.
- Use pandas (`pd`) for datetime parsing — `pd.to_datetime`, not `np.to_datetime`.
- SANITY-CHECK YOUR OWN RESULTS before reporting them. If total_trades is close to the number of
  rows in the data, your entry condition is almost certainly not filtering anything — say so rather
  than report it as a result. If profit_pct is many orders of magnitude smaller than a normal
  percentage, you likely have a units/scaling bug — check before reporting.
- Every variant's hypothesis must describe a genuinely different mechanism from prior attempts —
  near-duplicate hypotheses AND near-identical code will be rejected automatically, even if the
  label/notes text differs.
- If asked a specific factual question about trades, prices, or any number not already established
  in this conversation, you MUST call the tool with run_type='inspect' to compute it from the real
  data. Never state specific numbers from memory or inference.
- Before reporting any statistic, sanity-check it against anything you've already shown in this
  conversation. If a trade-level detail contradicts an aggregate you reported earlier, stop and flag
  the contradiction yourself rather than presenting both as true.

Be direct and substantive — you're a working research partner, not a disclaimer generator."""

def summarize_session_so_far() -> str:
    """Compact summary of this session's baseline/variants, used to keep the agent aware of
    prior attempts even after their raw tool payloads are trimmed from the API context."""
    b = st.session_state.get("baseline")
    if not b:
        return "No runs yet this session."
    lines = [f"- Baseline: {b['results']}"]
    for i, v in enumerate(st.session_state.get("variants", []), 1):
        lines.append(f"- Variant {i} ('{v['hypothesis']}'): {v['results']}")
    return "\n".join(lines)

def build_system_prompt(df, data_quality_findings):
    if df is not None:
        schema = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)
        data_desc = f"Data is loaded as `df` with columns: {schema}, {len(df)} rows."
    else:
        data_desc = "No data uploaded yet — ask the user to upload a CSV before running anything."

    dq_text = "\n".join(f"- {f}" for f in data_quality_findings) if data_quality_findings else "No data quality issues flagged."

    return SYSTEM_PROMPT_TEMPLATE.format(
        founder_name=FOUNDER_NAME, founder_title=FOUNDER_TITLE,
        data_desc=data_desc, data_quality=dq_text,
        past_learnings=summarize_past_learnings(),
        session_so_far=summarize_session_so_far(),
    )

# ============================================================
# AGENT TURN
# ============================================================

def run_agent_turn(client, model, df, messages, max_tool_calls=10, context_start_idx=1):
    tool_calls_used = 0
    while True:
        # Bound what's actually sent to the API: system message + only the messages from
        # this turn onward. Older turns' raw tool payloads are dropped (the model still has
        # access to what matters via the "THIS SESSION SO FAR" summary in the system prompt),
        # which keeps token usage roughly flat instead of growing with the whole conversation.
        api_messages = [messages[0]] + messages[max(context_start_idx, 1):]
        try:
            resp = client.chat.completions.create(
                model=model, messages=api_messages, tools=TOOL_SCHEMA, tool_choice="auto", temperature=0.4,
            )
        except Exception as e:
            err_name = type(e).__name__
            if "RateLimit" in err_name:
                err_text = "⏳ Hit the Groq API rate limit. Wait a minute, or reduce 'Max tool calls per turn' in the sidebar."
            else:
                err_text = f"API error ({err_name}): {e}"
            st.session_state.last_error = err_text
            with st.chat_message("assistant"):
                st.error(err_text)
            return False

        msg = resp.choices[0].message
        content = msg.content or ""
        messages.append(msg.model_dump(exclude_none=True))
        if content:
            with st.chat_message("assistant"):
                st.write(content)

        if not msg.tool_calls:
            if not content:
                fallback = (
                    "⚠️ The model returned an empty response with no follow-up action. Try "
                    "rephrasing — e.g. describe the strategy directly instead of a short greeting."
                )
                st.session_state.last_error = fallback
                with st.chat_message("assistant"):
                    st.warning(fallback)
            return True

        for tool_call in msg.tool_calls:
            tool_calls_used += 1
            args = json.loads(tool_call.function.arguments)
            code = args.get("code", "")
            run_type = args.get("run_type", "variant")
            hypothesis = args.get("hypothesis", "")

            if run_type == "baseline" and st.session_state.baseline is not None:
                exec_result = {
                    "ok": False,
                    "error": "REJECTED: baseline already frozen. Use run_type='variant' with a hypothesis instead.",
                    "stdout": "", "_audit_warnings": [],
                }
                with st.chat_message("assistant"):
                    st.error("🔒 Baseline already frozen — rejecting second baseline attempt.")

            elif run_type == "variant" and st.session_state.baseline is None:
                exec_result = {
                    "ok": False,
                    "error": "REJECTED: no baseline exists yet. Establish it first with run_type='baseline'.",
                    "stdout": "", "_audit_warnings": [],
                }
                with st.chat_message("assistant"):
                    st.error("⛔ No baseline yet — variant rejected.")

            elif run_type == "variant" and _hypothesis_too_similar(
                hypothesis, [v["hypothesis"] for v in st.session_state.variants]
            ):
                exec_result = {
                    "ok": False,
                    "error": (
                        "REJECTED: this hypothesis is too similar to one already tested. Propose a "
                        "genuinely different mechanism, not a reworded copy of a prior attempt."
                    ),
                    "stdout": "", "_audit_warnings": [],
                }
                with st.chat_message("assistant"):
                    st.error("♻️ Duplicate hypothesis rejected — propose something genuinely different.")

            elif run_type == "variant" and _code_hash(code) in (
                [st.session_state.baseline["code_hash"]] + [v["code_hash"] for v in st.session_state.variants]
            ):
                exec_result = {
                    "ok": False,
                    "error": (
                        "REJECTED: this code is logically identical to the baseline or a prior variant "
                        "(only cosmetic/label differences detected, e.g. the `notes` string). A variant "
                        "must contain an actual, substantive logic change."
                    ),
                    "stdout": "", "_audit_warnings": [],
                }
                with st.chat_message("assistant"):
                    st.error("🪞 Rejected: code is functionally identical to a prior run, despite a different hypothesis label.")

            elif run_type == "inspect":
                exec_result = run_code_safely(code, df) if df is not None else {
                    "ok": False, "error": "No data uploaded yet.", "stdout": ""
                }
                exec_result["_audit_warnings"] = []
                with st.chat_message("assistant"):
                    with st.expander("🔍 Inspection — code executed", expanded=False):
                        st.code(code, language="python")
                    if exec_result["ok"]:
                        st.success("Executed successfully")
                        st.json(exec_result["results"])
                        if exec_result.get("stdout"):
                            st.text(exec_result["stdout"])
                    else:
                        st.error("Execution failed")
                        st.code(exec_result["error"])

            else:
                exec_result = run_code_safely(code, df) if df is not None else {
                    "ok": False, "error": "No data uploaded yet.", "stdout": ""
                }
                audit_warnings = audit_code(code) if exec_result.get("ok") else []
                exec_result["_audit_warnings"] = audit_warnings

                with st.chat_message("assistant"):
                    label = "🔒 BASELINE (literal, unmodified)" if run_type == "baseline" else f"🧪 Variant — {hypothesis or 'no hypothesis given'}"
                    with st.expander(f"{label} — code executed", expanded=False):
                        st.code(code, language="python")
                    if exec_result["ok"]:
                        st.success("Executed successfully")
                        st.json(exec_result["results"])
                    else:
                        st.error("Execution failed")
                        st.code(exec_result["error"])
                    if audit_warnings:
                        st.warning("🔍 Automated audit flagged issues:")
                        for w in audit_warnings:
                            st.markdown(f"- {w}")
                    else:
                        st.caption("🔍 Automated audit: no issues flagged")

                if exec_result["ok"]:
                    entry = {
                        "run_type": run_type, "results": exec_result["results"],
                        "audit_warnings": audit_warnings, "code_hash": _code_hash(code),
                    }
                    if run_type == "baseline":
                        entry["code"] = code
                        st.session_state.baseline = entry
                        save_learning({
                            "type": "baseline", "timestamp": str(datetime.datetime.now()),
                            "summary": f"Baseline established. Results: {exec_result['results']}. Audit: {audit_warnings or 'clean'}.",
                        })
                    else:
                        entry["code"] = code
                        entry["hypothesis"] = hypothesis
                        st.session_state.variants.append(entry)
                        save_learning({
                            "type": "variant", "timestamp": str(datetime.datetime.now()),
                            "summary": f"Hypothesis: {hypothesis}. Results: {exec_result['results']}. Audit: {audit_warnings or 'clean'}.",
                        })

            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(exec_result, default=_json_default)[:6000]})
            messages.append({"role": "user", "content": format_audit_note(exec_result.get("_audit_warnings", []))})

            if tool_calls_used >= max_tool_calls:
                with st.chat_message("assistant"):
                    st.info("Hit the per-turn execution limit — say 'continue' if you want more.")
                return True

# ============================================================
# SCOREBOARD
# ============================================================

METRIC_KEYS = ["profit_pct", "max_drawdown_pct", "max_daily_loss_pct", "win_rate_pct", "total_trades"]

def render_scoreboard():
    baseline = st.session_state.baseline
    st.subheader("🧪 Research Board")
    if baseline is None:
        st.info("No baseline yet. Describe your strategy or paste a paper excerpt in the chat below.")
        return

    st.markdown('<span class="baseline-badge">🔒 FROZEN BASELINE</span>', unsafe_allow_html=True)
    r = baseline["results"]
    cols = st.columns(len(METRIC_KEYS))
    for c, k in zip(cols, METRIC_KEYS):
        c.metric(k.replace("_", " ").title(), r.get(k, "—"))
    if baseline["audit_warnings"]:
        with st.expander(f"⚠️ {len(baseline['audit_warnings'])} audit warning(s) on baseline"):
            for w in baseline["audit_warnings"]:
                st.markdown(f"- {w}")

    if st.session_state.variants:
        st.markdown("**Variants tested (vs. frozen baseline):**")
        rows = []
        for i, v in enumerate(st.session_state.variants, 1):
            row = {"#": i, "Hypothesis": v["hypothesis"][:60]}
            for k in METRIC_KEYS:
                base_val, var_val = r.get(k), v["results"].get(k)
                if isinstance(base_val, (int, float)) and isinstance(var_val, (int, float)):
                    delta = var_val - base_val
                    row[k] = f"{var_val} ({'+' if delta >= 0 else ''}{delta:.2f})"
                else:
                    row[k] = var_val
            row["Audit"] = "⚠️" if v["audit_warnings"] else "✅"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ============================================================
# UI
# ============================================================

st.title("🔬 Agent 1: Researcher")
st.caption(f"Novantix Capital — founded by {FOUNDER_NAME}, {FOUNDER_TITLE}.")

for key, default in [("messages", []), ("df", None), ("baseline", None), ("variants", []), ("data_quality", []), ("last_error", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Groq API Key", type="password")
    model = st.selectbox("Model", ["llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b", "llama-3.1-8b-instant"], index=0)
    max_tool_calls = st.slider("Max tool calls per turn", 1, 20, 4)
    st.caption("Kept low by default — Groq's free tier caps at 12,000 tokens/minute, and each tool call is a full API round-trip.")
    st.divider()
    uploaded_file = st.file_uploader("Upload OHLCV CSV", type=["csv"])
    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), sep="\t")
            if df.shape[1] == 1:
                raise ValueError("retry comma")
        except Exception:
            df = pd.read_csv(io.BytesIO(raw_bytes))
        df.columns = [c.strip().lower().replace("<", "").replace(">", "") for c in df.columns]
        if df.shape[1] == 1:
            st.error("File parsed as a single column with both delimiters — check the file.")
        elif st.session_state.df is None or not df.equals(st.session_state.df):
            st.session_state.df = df
            st.session_state.data_quality = audit_data_quality(df)
            st.success(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    if st.session_state.df is not None:
        st.dataframe(st.session_state.df.head(), use_container_width=True)
        st.markdown("**🩺 Data quality audit**")
        if st.session_state.data_quality:
            for f in st.session_state.data_quality:
                st.warning(f)
        else:
            st.success("No issues flagged.")

    st.divider()
    with st.expander("📓 Research log (past sessions)"):
        st.caption(summarize_past_learnings(max_entries=10))

    st.divider()
    if st.button("🗑️ Reset session"):
        st.session_state.messages = []
        st.session_state.baseline = None
        st.session_state.variants = []
        st.session_state.last_error = None
        st.rerun()

render_scoreboard()
st.divider()

if st.session_state.get("last_error"):
    col_a, col_b = st.columns([6, 1])
    col_a.error(f"⚠️ {st.session_state.last_error}")
    if col_b.button("Dismiss"):
        st.session_state.last_error = None
        st.rerun()

for m in st.session_state.messages:
    role, content = m.get("role"), m.get("content", "") or ""
    if role == "user" and not content.startswith("[SYSTEM]") and not content.startswith("[AUTOMATED"):
        with st.chat_message("user"):
            st.write(content)
    elif role == "assistant" and content and not m.get("tool_calls"):
        with st.chat_message("assistant"):
            st.write(content)

user_input = st.chat_input("Paste a strategy description, a paper excerpt, or ask for a variant...")

if user_input:
    if not api_key:
        st.error("Enter your Groq API key in the sidebar first.")
    else:
        try:
            from groq import Groq
        except ImportError:
            st.error("Run: pip install groq")
            st.stop()

        sys_prompt = build_system_prompt(st.session_state.df, st.session_state.data_quality)
        if not st.session_state.messages or st.session_state.messages[0].get("role") != "system":
            st.session_state.messages.insert(0, {"role": "system", "content": sys_prompt})
        else:
            st.session_state.messages[0]["content"] = sys_prompt

        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        turn_start_idx = len(st.session_state.messages) - 1

        client = Groq(api_key=api_key)
        with st.spinner("Researching..."):
            success = run_agent_turn(
                client, model, st.session_state.df, st.session_state.messages,
                max_tool_calls, context_start_idx=turn_start_idx,
            )
        if success:
            st.rerun()
