# =============================================================================
# app/agent/graph.py — LangGraph Agent Graph
# =============================================================================
# This file defines the actual agent as a directed graph.
# Each node is a Python async function that receives AgentState
# and returns a dict of updates to merge back into state.
#
# EXECUTION FLOW:
#   context_builder (pre-step, outside graph)
#       ↓
#   router_node  ──────────────────────────────────────────────────────────►
#       ↓                                                                    |
#   tool_dispatcher (runs tools in parallel based on router flags)          |
#       ↓                                                                    |
#   response_node                                                            |
#       ↓                                                                    |
#   agent_state_updater (async, doesn't block response)                    |
#       ↓                                                                    ▼
#   END                                                        (all traced in Langfuse)
# =============================================================================

import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from app.agent.state import AgentState
from app.agent.tools.sentiment import run_sentiment_tool
from app.agent.tools.event_extractor import run_event_extractor_tool
from app.agent.tools.memory_retriever import run_memory_retriever_tool
from app.agent.tools.memory_writer import run_memory_writer_tool
from app.agent.prompts import (
    ROUTER_PROMPT, SENTIMENT_PROMPT, EVENT_EXTRACTOR_PROMPT,
    AGENT_STATE_EVALUATOR_PROMPT, build_system_prompt, PERSONA_PROMPTS
)
from app.core.config import settings

logger = logging.getLogger(__name__)

# =============================================================================
# LLM SETUP — provider selected by LLM_PROVIDER in .env
# =============================================================================
# Change LLM_PROVIDER to switch providers with zero other code changes:
#   LLM_PROVIDER=anthropic  → Claude haiku (tools) + Claude sonnet (response)
#   LLM_PROVIDER=openai     → GPT-4o-mini (tools) + GPT-4o (response)
#   LLM_PROVIDER=ollama     → local Ollama for both (free, slow on CPU)
# =============================================================================

def _build_llms():
    from langchain_ollama import ChatOllama
    fast = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0.1,
        format="json",
    )
    resp = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0.75,
    )
    logger.info(f"[LLM] Provider: Ollama ({settings.ollama_model})")

    return fast, resp

llm_fast, llm_response = _build_llms()


# =============================================================================
# HELPER: safe JSON parse
# =============================================================================
def _coerce_booleans(obj: dict) -> dict:
    """
    Recursively coerce string booleans to real booleans throughout a dict.
    LLMs sometimes return "true"/"false" as strings even in JSON mode.
    This fixes it everywhere in one pass rather than at every call site.
    """
    result = {}
    for key, val in obj.items():
        if isinstance(val, str) and val.lower() in ("true", "false"):
            result[key] = val.lower() == "true"
        elif isinstance(val, dict):
            result[key] = _coerce_booleans(val)
        elif isinstance(val, list):
            result[key] = [
                _coerce_booleans(i) if isinstance(i, dict) else i
                for i in val
            ]
        else:
            result[key] = val
    return result


def safe_json_parse(text: str, fallback: dict) -> dict:
    """
    Parse LLM JSON output safely and completely.

    Handles all known LLM JSON quirks:
    1. Markdown code fences (```json ... ```)
    2. Leading/trailing garbage text around the JSON object
    3. Partial responses (fragment of JSON)
    4. String booleans ("true"/"false" instead of true/false)
    5. Empty or whitespace-only responses

    Falls back gracefully to `fallback` on any parse failure.
    Every LLM call in this system goes through this function.
    """
    if not text or not text.strip():
        logger.warning("JSON parse: empty response, using fallback")
        return fallback

    try:
        cleaned = text.strip()

        # Fix 1: Strip markdown code fences
        # LLMs often wrap JSON in ```json ... ``` even when told not to
        if "```" in cleaned:
            parts = cleaned.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    cleaned = part
                    break

        # Fix 2: Extract JSON object boundaries
        # Handles cases where LLM adds preamble like "Here is the JSON: {...}"
        # or postamble like "{...} Hope this helps!"
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start == -1 or end <= start:
            logger.warning(f"JSON parse: no object found in: {text[:200]}")
            return fallback
        cleaned = cleaned[start:end]

        # Fix 3: Parse the JSON
        parsed = json.loads(cleaned)

        # Fix 4: Coerce string booleans throughout the entire response
        # "true" -> True, "false" -> False at every level of nesting
        return _coerce_booleans(parsed)

    except (json.JSONDecodeError, ValueError, IndexError) as e:
        logger.warning(f"JSON parse failed: {e} | Raw: {text[:300]}")
        return fallback


# =============================================================================
# NODE 1: ROUTER
# =============================================================================
# =============================================================================
# PYTHON RULE ENGINE — all tool flags computed here, not by the LLM
# =============================================================================
# WHY NOT LET THE LLM DECIDE TOOL FLAGS?
#   We tried. Gemini Flash consistently ignores run_event_extractor even with
#   explicit prompting. The problem is fundamental — multi-task JSON prompts
#   cause models to anchor on the most prominent signal and drop secondary ones.
#
#   Solution: LLM decides ONLY intent and urgency (requires language understanding).
#   Everything else is computed by deterministic Python rules (fast, 100% reliable).
#   This is the correct architectural boundary.
# =============================================================================

import re as _re

# Emotional keywords — if any appear, run sentiment analysis
_EMOTION_PATTERN = _re.compile(
    r"("
    r"stress(ed|ful)?|anxious|anxiety|nervous|worried|worry|"
    r"scared|fear(ful)?|panic|overwhelm(ed)?|"
    r"sad|depress(ed|ing)?|down|low|unhappy|miserable|"
    r"angry|frustrated|annoyed|upset|furious|mad|"
    r"happy|excited|thrilled|great|amazing|wonderful|fantastic|"
    r"tired|exhausted|drained|burned?\s*out|"
    r"lonely|alone|lost|confused|stuck|"
    r"proud|confident|motivated|inspired|"
    r"awful|terrible|horrible|rough|tough|hard|difficult"
    r")",
    _re.IGNORECASE
)

# Memory keywords — references to past conversations or ongoing situations
_MEMORY_PATTERN = _re.compile(
    r"("
    r"remember|mentioned|told you|last time|before|previously|"
    r"we talked|you said|you suggested|earlier|the other day|"
    r"still|update|follow.?up|"
    r"how did|how.s it going|what happened|any news|any update|did you know|"
    r"my (sister|brother|mom|dad|friend|boss|colleague|partner|girlfriend|boyfriend)"
    r")",
    _re.IGNORECASE
)


def _compute_tool_flags(message: str, intent: str) -> dict:
    """
    Compute tool flags from Python rules.

    WHY PYTHON FOR SENTIMENT AND MEMORY BUT LLM FOR EVENTS?

    run_sentiment:
        Computed by Python keyword check. Emotional words are well-defined
        and a regex catches them reliably. No ambiguity.

    run_event_extractor:
        ALWAYS TRUE — the event extractor is a dedicated focused LLM call
        that returns has_event:false when nothing is found. This is cheap
        and correct. We never want to miss an event because the router
        didn't detect it.
        WHY NOT REGEX? Events are open-ended — "annual day", "farewell",
        "convocation", "puja", "family function", "orientation" — no regex
        can cover all of them. Only an LLM with a focused prompt can.
        The event extractor tool already handles this correctly.

    run_memory_retriever:
        Computed by Python keyword check. References to past conversations
        have clear linguistic markers ("you mentioned", "last time" etc).
        Will activate Qdrant vector search in Step 6.
    """
    has_emotion_keyword = bool(_EMOTION_PATTERN.search(message))
    has_memory_keyword  = bool(_MEMORY_PATTERN.search(message))

    # Sentiment: run if emotional keywords present OR intent signals emotion
    run_sentiment = has_emotion_keyword or intent in ("emotional", "venting", "mixed")

    # Event: ALWAYS run — let the dedicated event extractor LLM decide
    # It returns has_event:false cleanly when nothing is there
    # This is the only correct approach for open-ended event detection
    run_event = True

    # Memory: run if memory keywords present (Qdrant activates in Step 6)
    run_memory = has_memory_keyword

    return {
        "run_sentiment":        run_sentiment,
        "run_event_extractor":  run_event,
        "run_memory_retriever": run_memory,
        "_flags_debug": {
            "emotion_keyword": has_emotion_keyword,
            "memory_keyword":  has_memory_keyword,
            "event_extractor": "always runs — dedicated LLM call",
        }
    }


async def router_node(state: AgentState) -> dict:
    """
    Two responsibilities:
    1. LLM call — classify intent and urgency (needs language understanding)
    2. Python rules — compute all tool flags (deterministic, no LLM needed)
    """
    logger.info(f"[Router] Processing message for user {state['user_id']}")

    recent = state.get("recent_turns", [])[-3:]
    recent_context = " | ".join([
        f"user: {t.get('user_message', '')[:40]} | agent: {t.get('agent_reply', '')[:40]}"
        for t in recent
        if t.get("user_message")
    ]) if recent else "No prior context"

    prompt = ROUTER_PROMPT.format(
        message=state["message"],
        recent_context=recent_context,
    )

    try:
        response = await llm_fast.ainvoke([HumanMessage(content=prompt)])
        result = safe_json_parse(response.content, {
            "intent": "casual",
            "urgency": "low",
        })

        # Validate LLM output — only intent and urgency now
        valid_intents = {"casual","emotional","event","question","venting","playful","mixed"}
        valid_urgency  = {"low","medium","high"}

        intent = result.get("intent", "casual")
        if intent not in valid_intents:
            logger.warning(f"[Router] Invalid intent '{intent}', defaulting to casual")
            intent = "casual"

        urgency = result.get("urgency", "low")
        if urgency not in valid_urgency:
            urgency = "low"

        # Compute all tool flags via Python rules — no LLM involvement
        flags = _compute_tool_flags(state["message"], intent)
        debug = flags.pop("_flags_debug")

        logger.info(
            f"[Router] intent={intent} urgency={urgency} | "
            f"sentiment={flags['run_sentiment']} "
            f"event={flags['run_event_extractor']} "
            f"memory={flags['run_memory_retriever']}"
        )
        logger.debug(f"[Router] Flag reasons: {debug}")

        return {
            "intent":  intent,
            "urgency": urgency,
            **flags,
        }

    except Exception as e:
        logger.error(f"[Router] Error: {e}")
        # Even on LLM failure, still compute tool flags from rules
        flags = _compute_tool_flags(state["message"], "casual")
        flags.pop("_flags_debug", None)
        return {
            "intent":  "casual",
            "urgency": "low",
            "error":   f"Router LLM failed: {str(e)}",
            **flags,
        }


# =============================================================================
# NODE 2: TOOL DISPATCHER
# =============================================================================
async def tool_dispatcher(state: AgentState) -> dict:
    """
    Runs the appropriate tools IN PARALLEL based on router flags.
    asyncio.gather means all tools run simultaneously — no waiting.

    WHY PARALLEL?
      Sentiment + event extraction are independent.
      Running them sequentially doubles latency for no reason.
      asyncio.gather fires them all at once and waits for all to finish.
    """
    tasks = []
    task_names = []

    if state.get("run_sentiment"):
        tasks.append(_run_sentiment(state))
        task_names.append("sentiment")

    if state.get("run_event_extractor"):
        tasks.append(_run_event_extractor(state))
        task_names.append("event")

    if not tasks:
        logger.info(
            f"[Tools] No tools needed | "
            f"sentiment={state.get('run_sentiment')} "
            f"event={state.get('run_event_extractor')} "
            f"memory={state.get('run_memory_retriever')}"
        )
        return {}

    logger.info(f"[Tools] Running in parallel: {task_names}")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    updates = {}
    for name, result in zip(task_names, results):
        if isinstance(result, Exception):
            logger.error(f"[Tools] {name} failed: {result}")
        else:
            updates.update(result)

    return updates


async def _run_sentiment(state: AgentState) -> dict:
    """Delegates to the standalone sentiment tool module."""
    return await run_sentiment_tool(state)


async def _run_event_extractor(state: AgentState) -> dict:
    """Delegates to the standalone event extractor tool module."""
    return await run_event_extractor_tool(state)


# =============================================================================
# NODE 3: RESPONSE NODE
# =============================================================================
async def response_node(state: AgentState) -> dict:
    """
    Generates the agent's reply using the full quality LLM.

    This node has access to EVERYTHING assembled so far:
    - The rich persona system prompt (from context_builder)
    - Agent emotional state (how the agent is feeling)
    - User mood (from sentiment tool)
    - Extracted events (from event extractor)
    - Recent conversation history
    - Message behavioral signals (caps, punctuation)
    """
    logger.info(f"[Response] Generating reply | intent: {state.get('intent')}")
    logger.info(f"[Response] Current message: {state.get('message', '')[:100]}")
    logger.info(f"[Response] Recent turns count: {len(state.get('recent_turns', []))}")
    logger.debug(f"[Response] System prompt preview: {state.get('persona_prompt', '')[:300]}")

    # Build the complete system prompt
    # At this point persona_prompt already contains persona + agent state + mood context
    # We enrich it further with tool results
    system_prompt = state.get("persona_prompt", "")

    # Inject event context if an event was found
    event = state.get("extracted_event", {})
    if event and event.get("has_event"):
        system_prompt += f"""

[EVENT DETECTED]
The user mentioned: {event.get('title')} on {event.get('date')}
A reminder has been scheduled. You can naturally acknowledge you've noted it
without making a big deal of it — "I've got that noted" is enough.
"""

    # Inject sentiment context
    sentiment = state.get("sentiment", {})
    if sentiment:
        system_prompt += f"""

[CURRENT EMOTIONAL READING]
Valence: {sentiment.get('valence', 0)} | Energy: {sentiment.get('energy', 'medium')}
Label: {sentiment.get('label', 'neutral')} | Tone: {sentiment.get('tone', 'neutral')}
Sarcasm detected: {sentiment.get('sarcasm_detected', False)}
{"CAPS and intensity signals suggest elevated emotional state — match that register carefully." if state.get('caps_ratio', 0) > 0.5 else ""}
"""

    # Inject urgency flag
    if state.get("urgency") == "high":
        system_prompt += """

[URGENCY: HIGH]
This person may be in distress. Prioritize presence over advice.
Keep response focused, warm, and grounding.
"""

    # Build conversation history for the LLM
    # WHY THIS KEY MAPPING?
    #   MongoDB turns are stored as:
    #     { "user_message": "...", "agent_reply": "..." }
    #   LangChain messages need HumanMessage / AIMessage objects.
    #   We map user_message → HumanMessage, agent_reply → AIMessage.
    #   Skip turns with empty content — they add noise not signal.
    messages = [SystemMessage(content=system_prompt)]

    for turn in state.get("recent_turns", [])[-6:]:
        user_msg = turn.get("user_message", "").strip()
        agent_msg = turn.get("agent_reply", "").strip()

        if user_msg:
            messages.append(HumanMessage(content=user_msg))
        if agent_msg:
            messages.append(AIMessage(content=agent_msg))

    # Add the current message last
    messages.append(HumanMessage(content=state["message"]))

    logger.debug(
        f"[Response] Context: {len(messages)-1} history messages + current message"
    )

    try:
        response = await llm_response.ainvoke(messages)
        reply = response.content.strip()

        logger.info(f"[Response] Generated {len(reply)} chars")

        return {
            "reply": reply,
            "reply_ts": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"[Response] Error: {e}")
        # Graceful fallback — never leave user with an error message
        fallback = (
            "I'm here, I just had a small hiccup. What were you saying?"
            if state.get("intent") != "emotional"
            else "I'm still with you — something glitched on my end. Tell me more."
        )
        return {
            "reply": fallback,
            "reply_ts": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
        }


# =============================================================================
# NODE 4: AGENT STATE UPDATER
# =============================================================================
async def agent_state_updater(state: AgentState) -> dict:
    """
    Evaluates how the user's message affected the agent's emotional state.
    Updates MongoDB with the new state.

    This runs AFTER the response is generated — it affects the NEXT turn,
    not the current one. The agent has already replied warmly even if hurt.
    The emotional impact registers for next time.
    """
    logger.info("[AgentState] Evaluating impact on agent state")

    sentiment = state.get("sentiment", {})
    user_tone = sentiment.get("tone", "neutral") if sentiment else "neutral"

    prompt = AGENT_STATE_EVALUATOR_PROMPT.format(
        current_mood=state.get("agent_mood", "neutral"),
        current_trust=state.get("agent_trust", 0.3),
        current_openness=state.get("agent_openness", 0.4),
        message=state["message"],
        caps_ratio=state.get("caps_ratio", 0.0),
        punctuation_intensity=state.get("punctuation_intensity", 0),
        intent=state.get("intent", "casual"),
        user_tone=user_tone,
    )

    try:
        response = await llm_fast.ainvoke([HumanMessage(content=prompt)])
        delta = safe_json_parse(response.content, {
            "mood_change": "no_change",
            "trust_delta": 0.0,
            "openness_delta": 0.0,
            "energy_change": "no_change",
            "impact_description": "neutral interaction",
            "should_acknowledge": False,
        })

        # Apply deltas with bounds clamping
        new_trust = max(0.0, min(1.0,
            state.get("agent_trust", 0.3) + delta.get("trust_delta", 0.0)
        ))
        new_openness = max(0.0, min(1.0,
            state.get("agent_openness", 0.4) + delta.get("openness_delta", 0.0)
        ))
        new_mood = delta.get("mood_change", "no_change")
        if new_mood == "no_change":
            new_mood = state.get("agent_mood", "neutral")

        # Persist to MongoDB
        from app.db.mongo import get_mood_logs_collection
        mood_col = get_mood_logs_collection()

        await mood_col.update_one(
            {"user_id": state["user_id"], "doc_type": "agent_state"},
            {"$set": {
                "mood":          new_mood,
                "trust_level":   new_trust,
                "openness":      new_openness,
                "energy":        delta.get("energy_change", state.get("agent_energy", "medium")),
                "last_impact":   delta.get("impact_description", ""),
                "updated_at":    datetime.now(timezone.utc).isoformat(),
            }},
            upsert=True,
        )

        logger.info(f"[AgentState] mood: {new_mood} | trust: {new_trust:.2f} | openness: {new_openness:.2f}")

        return {
            "state_delta": delta,
            "agent_mood":     new_mood,
            "agent_trust":    new_trust,
            "agent_openness": new_openness,
        }

    except Exception as e:
        logger.error(f"[AgentState] Error: {e}")
        return {"state_delta": {}}


# =============================================================================
# CONDITIONAL EDGE: should we run tools?
# =============================================================================
def should_run_tools(state: AgentState) -> str:
    """
    After routing, decide next node.
    If any tool needs to run → go to tool_dispatcher.
    Otherwise → go straight to response.
    """
    needs_tools = (
        state.get("run_sentiment") or
        state.get("run_event_extractor") or
        state.get("run_memory_retriever")
    )
    return "tool_dispatcher" if needs_tools else "response_node"


# =============================================================================
# GRAPH ASSEMBLY
# =============================================================================
def build_agent_graph():
    """
    Assembles and compiles the LangGraph agent.

    WHY compile()?
      compile() validates the graph — checks for unreachable nodes,
      missing edges, type mismatches. Catches errors at startup not runtime.
      It also enables features like streaming and checkpointing later.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router_node",        router_node)
    graph.add_node("tool_dispatcher",    tool_dispatcher)
    graph.add_node("response_node",      response_node)
    graph.add_node("agent_state_updater", agent_state_updater)

    # Entry point
    graph.set_entry_point("router_node")

    # Conditional edge: after routing, tools or direct to response
    graph.add_conditional_edges(
        "router_node",
        should_run_tools,
        {
            "tool_dispatcher": "tool_dispatcher",
            "response_node":   "response_node",
        }
    )

    # After tools → always go to response
    graph.add_edge("tool_dispatcher", "response_node")

    # After response → update agent state
    graph.add_edge("response_node", "agent_state_updater")

    # Agent state update → end
    graph.add_edge("agent_state_updater", END)

    return graph.compile()


# Module-level compiled graph — built once, reused for every request
agent_graph = build_agent_graph()