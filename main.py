"""
Agent 1: Researcher
----------------------
Paste a plain-English strategy description or a research paper excerpt.
The agent implements it LITERALLY first (no improvements, no fixes) and runs
it once — that becomes the frozen baseline, shown raw, good or bad. Only
after the baseline is locked can it propose hypothesis-driven variants, each
one run and compared against that frozen baseline. The baseline cannot be
silently overwritten — the app enforces this structurally, not just by prompt.

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
import re
import multiprocessing as mp
import traceback
from contextlib import redirect_stdout

# ============================================================
# DARK THEME
# ============================================================
st.set_page_config(page_title="Agent 1: Researcher", layout="wide")
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
# For a native dark theme instead: add .streamlit/config.toml with [theme] base = "dark"

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

# ============================================================
# INDEPENDENT AUDIT
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
                "DRAWDOWN FORMULA RISK: divided by raw cumulative P&L (can approach zero early on) "
                "instead of an equity curve (starting capital + cumulative P&L)."
            )

    if "daily" in lower and "rolling(" in lower and "groupby" not in lower:
        warnings.append(
            "DAILY METRIC RISK: 'daily' appears to use a fixed rolling window rather than grouping "
            "by actual calendar date — verify the window truly equals one trading day."
        )

    if "equity" not in lower and "balance" not in lower and "capital" not in lower:
        warnings.append(
            "NO EXPLICIT EQUITY CURVE: no account equity/balance/capital tracked from a defined "
            "starting size."
        )

    return warnings

def format_audit_note(warnings):
    if not warnings:
        return "[AUTOMATED AUDIT] No issues flagged for this execution."
    lines = ["[AUTOMATED AUDIT] Issues detected:"] + [f"- {w}" for w in warnings]
    lines.append("Address these or explain why they don't apply.")
    return "\n".join(lines)

# ============================================================
# TOOL SCHEMA — run_type is the structural enforcement mechanism
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
            "You MUST set run_type correctly:\n"
            "- 'baseline': ONLY for the very first literal, unmodified translation of the strategy "
            "as described. This can only ever be done ONCE — a second baseline attempt will be "
            "rejected by the system. Do not add improvements or fix issues in the baseline run, "
            "even ones you notice — implement it exactly as described, flaws included, and report "
            "what actually happens.\n"
            "- 'variant': for every run after the baseline exists. Requires a `hypothesis` string "
            "explaining the specific mechanism you're testing and why you expect it to help."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Full Python code to execute."},
                "run_type": {"type": "string", "enum": ["baseline", "variant"]},
                "hypothesis": {
                    "type": "string",
                    "description": "Required for variants: the specific mechanism being tested and why.",
                },
            },
            "required": ["code", "run_type"],
        },
    },
}]

SYSTEM_PROMPT_TEMPLATE = """You are Agent 1: Researcher — a quant research collaborator whose defining
discipline is NEVER hiding or skipping the honest starting point of an idea.

{data_desc}

Your process is strict and two-phase:

PHASE 1 — LITERAL BASELINE (happens exactly once):
When the user gives you a strategy description or a research paper excerpt, your FIRST job is to
implement it as literally and faithfully as possible — no improvements, no bug fixes, no "obviously
better" tweaks, even ones you spot immediately. Run it with run_type='baseline'. Report the raw
result honestly, even if it's bad, even if it loses money, even if it barely trades. This baseline
is frozen the moment it succeeds — the system will not let you run a second baseline, by design.
If the description is ambiguous in a way that would change the literal implementation, ask the user
before running — don't silently pick an interpretation for the baseline.

PHASE 2 — HYPOTHESIS-DRIVEN ITERATION (after baseline exists):
Only now can you propose changes. Every change must be run with run_type='variant' and a specific
`hypothesis` naming the mechanism (e.g. "adding a volatility filter should reduce whipsaw losses in
choppy regimes" — not "trying to improve performance"). Every variant is automatically compared
against the frozen baseline so the user sees the real delta, not just a final number. Never claim a
variant is better without having actually run it this turn.

Rules that always apply:
- NO LOOKAHEAD BIAS: any signal from bar t must be shifted forward before capturing a return.
- Track a real equity curve from an explicit starting capital.
- If something is genuinely unclear (data columns, an ambiguous rule in a paper, what "improvement"
  the user actually wants), ask — don't guess on something that would change the science.
- Every automated audit warning must be addressed or explicitly explained, not ignored.

Be a real collaborator: explain your reasoning before and after each run, not just tool calls."""

def build_system_prompt(df):
    if df is not None:
        schema = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)
        data_desc = f"Data is loaded as `df` with columns: {schema}, {len(df)} rows."
    else:
        data_desc = "No data uploaded yet — ask the user to upload a CSV before running anything."
    return SYSTEM_PROMPT_TEMPLATE.format(data_desc=data_desc)

# ============================================================
# AGENT TURN
# ============================================================

def run_agent_turn(client, model, df, messages, max_tool_calls=10):
    tool_calls_used = 0
    while True:
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=TOOL_SCHEMA, tool_choice="auto", temperature=0.4,
        )
        msg = resp.choices[0].message
        content = msg.content or ""
        messages.append(msg.model_dump(exclude_none=True))
        if content:
            with st.chat_message("assistant"):
                st.write(content)

        if not msg.tool_calls:
            return

        for tool_call in msg.tool_calls:
            tool_calls_used += 1
            args = json.loads(tool_call.function.arguments)
            code = args.get("code", "")
            run_type = args.get("run_type", "variant")
            hypothesis = args.get("hypothesis", "")

            # STRUCTURAL GATE: baseline can only be set once, ever.
            if run_type == "baseline" and st.session_state.baseline is not None:
                exec_result = {
                    "ok": False,
                    "error": (
                        "REJECTED: a baseline already exists and is frozen. You cannot run a second "
                        "baseline. If you want to test a change, call this again with "
                        "run_type='variant' and a hypothesis."
                    ),
                    "stdout": "",
                    "_audit_warnings": [],
                }
                with st.chat_message("assistant"):
                    st.error("🔒 Baseline already frozen — rejecting second baseline attempt.")
            elif run_type == "variant" and st.session_state.baseline is None:
                exec_result = {
                    "ok": False,
                    "error": "REJECTED: no baseline exists yet. Establish the literal baseline first with run_type='baseline'.",
                    "stdout": "",
                    "_audit_warnings": [],
                }
                with st.chat_message("assistant"):
                    st.error("⛔ No baseline yet — variant rejected.")
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
                    if run_type == "baseline":
                        st.session_state.baseline = {
                            "code": code, "results": exec_result["results"],
                            "audit_warnings": audit_warnings,
                        }
                    else:
                        st.session_state.variants.append({
                            "code": code, "results": exec_result["results"],
                            "audit_warnings": audit_warnings, "hypothesis": hypothesis,
                        })

            messages.append({
                "role": "tool", "tool_call_id": tool_call.id,
                "content": json.dumps(exec_result)[:6000],
            })
            messages.append({"role": "user", "content": format_audit_note(exec_result.get("_audit_warnings", []))})

            if tool_calls_used >= max_tool_calls:
                with st.chat_message("assistant"):
                    st.info("Hit the per-turn execution limit — say 'continue' if you want more.")
                return

# ============================================================
# COMPARISON PANEL
# ============================================================

METRIC_KEYS = ["profit_pct", "max_drawdown_pct", "max_daily_loss_pct", "win_rate_pct", "total_trades"]

def render_scoreboard():
    baseline = st.session_state.baseline
    st.subheader("🧪 Research Board")
    if baseline is None:
        st.info("No baseline yet. Describe your strategy or paste a paper excerpt in the chat below to establish one.")
        return

    st.markdown('<span class="baseline-badge">🔒 FROZEN BASELINE</span>', unsafe_allow_html=True)
    r = baseline["results"]
    cols = st.columns(len(METRIC_KEYS))
    for c, k in zip(cols, METRIC_KEYS):
        c.metric(k.replace("_", " ").title(), r.get(k, "—"))
    if baseline["audit_warnings"]:
        with st.expander(f"⚠️ {len(baseline['audit_warnings'])} audit warning(s) on baseline", expanded=False):
            for w in baseline["audit_warnings"]:
                st.markdown(f"- {w}")

    if st.session_state.variants:
        st.markdown("**Variants tested (vs. frozen baseline):**")
        rows = []
        for i, v in enumerate(st.session_state.variants, 1):
            row = {"#": i, "Hypothesis": v["hypothesis"][:60]}
            for k in METRIC_KEYS:
                base_val = r.get(k)
                var_val = v["results"].get(k)
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
st.caption("Literal baseline first, frozen and honest. Every improvement idea is a labeled hypothesis, tested and compared — never hidden.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "baseline" not in st.session_state:
    st.session_state.baseline = None
if "variants" not in st.session_state:
    st.session_state.variants = []

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Groq API Key", type="password")
    model = st.selectbox(
        "Model", ["llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b", "llama-3.1-8b-instant"], index=0,
    )
    max_tool_calls = st.slider("Max tool calls per turn", 1, 20, 10)
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
            st.success(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    if st.session_state.df is not None:
        st.dataframe(st.session_state.df.head(), use_container_width=True)

    st.divider()
    if st.button("🗑️ Reset everything"):
        st.session_state.messages = []
        st.session_state.baseline = None
        st.session_state.variants = []
        st.rerun()

render_scoreboard()
st.divider()

for m in st.session_state.messages:
    role = m.get("role")
    content = m.get("content", "") or ""
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

        if not st.session_state.messages or st.session_state.messages[0].get("role") != "system":
            st.session_state.messages.insert(0, {"role": "system", "content": build_system_prompt(st.session_state.df)})
        else:
            st.session_state.messages[0]["content"] = build_system_prompt(st.session_state.df)

        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        client = Groq(api_key=api_key)
        with st.spinner("Researching..."):
            run_agent_turn(client, model, st.session_state.df, st.session_state.messages, max_tool_calls)
        st.rerun()
