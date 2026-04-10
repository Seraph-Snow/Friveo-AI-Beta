# =============================================================================
# app/agent/state.py — LangGraph Shared State
# =============================================================================
# WHY A TYPED STATE OBJECT?
#   Every node in the graph receives this state and returns updates to it.
#   TypedDict makes every field explicit — no mystery keys, no silent bugs.
#   LangGraph merges node outputs back into state automatically.
#   If a node only updates `intent`, it only returns {"intent": "emotional"}.
#   Everything else is preserved from before.
#
# THINK OF IT AS: the "working memory" of one conversation turn.
#   It starts almost empty, gets filled as each node runs, and by the
#   end contains everything that happened during this turn.
# =============================================================================

from typing import TypedDict, Optional
from datetime import datetime


class AgentState(TypedDict):

    # -------------------------------------------------------------------------
    # INPUT — set before the graph starts
    # -------------------------------------------------------------------------
    user_id:        str
    session_id:     str
    message:        str          # the raw user message
    message_ts:     str          # ISO timestamp of when message was received

    # -------------------------------------------------------------------------
    # USER CONTEXT — loaded by context_builder before graph runs
    # These come from Postgres + MongoDB + Redis — NOT from the LLM
    # -------------------------------------------------------------------------
    # Static profile from Postgres
    user_profile: dict           # { display_name, timezone, personality_code }
    persona_prompt: str          # the assembled system prompt prefix for this user
                                 # e.g. full ENFJ behavioral description

    # Dynamic context from MongoDB
    recent_turns: list           # last 10 { role, content, timestamp } dicts
    session_summary: str         # rolling summary of current session so far
    mood_history: list           # last 3 { valence, energy, label, timestamp }

    # Relevant memories from Qdrant vector search
    memories: list               # [ { text, relevance_score, session_id, date } ]

    # -------------------------------------------------------------------------
    # AGENT EMOTIONAL STATE — loaded from MongoDB, updated each turn
    # This is what makes the agent feel like a real friend, not a reset bot
    # -------------------------------------------------------------------------
    agent_mood:         str      # "happy"|"neutral"|"playful"|"hurt"|"concerned"|"withdrawn"
    agent_trust:        float    # 0.0 → 1.0 — builds with kindness, drops with cruelty
    agent_openness:     float    # 0.0 → 1.0 — how freely agent shares opinions/humour
    agent_energy:       str      # "high"|"medium"|"low" — mirrors interaction quality
    agent_last_impact:  str      # human-readable: "user was warm and grateful"
    agent_state_prompt: str      # injected into system prompt — see prompts.py

    # -------------------------------------------------------------------------
    # ROUTER OUTPUT — set by router_node
    # -------------------------------------------------------------------------
    intent:               str    # "casual"|"emotional"|"event"|"question"|"venting"|"playful"
    run_sentiment:        bool   # should sentiment tool run this turn?
    run_event_extractor:  bool   # should event extractor run?
    run_memory_retriever: bool   # should we do a Qdrant search?
    urgency:              str    # "low"|"medium"|"high" — affects response priority

    # -------------------------------------------------------------------------
    # MESSAGE SIGNALS — computed before LLM call (pure Python)
    # These give the LLM raw behavioral signals it can't infer itself
    # -------------------------------------------------------------------------
    caps_ratio:             float  # 0.0-1.0 — high = user may be yelling/excited
    punctuation_intensity:  int    # count of !, ?, ... — emotional intensity signal
    message_length_signal:  str    # "very_short"|"short"|"medium"|"long"|"very_long"
                                   # very short = terse/dismissive or just brief
                                   # very long = user needs to vent, don't cut them off

    # -------------------------------------------------------------------------
    # TOOL OUTPUTS — set by tools, read by response_node
    # -------------------------------------------------------------------------
    # Sentiment tool output
    sentiment: dict          # { valence: float, energy: str, label: str,
                             #   sarcasm_detected: bool, tone: str }

    # Event extractor output
    extracted_event: dict    # { has_event: bool, title: str, date: str,
                             #   event_type: str, confidence: float }
                             # OR {} if no event found

    # Memory retriever output (already in `memories` above, updated by tool)

    # -------------------------------------------------------------------------
    # AGENT STATE EVALUATOR OUTPUT — set after response is generated
    # How did the user's message affect the agent?
    # -------------------------------------------------------------------------
    state_delta: dict        # { mood_change: str, trust_delta: float,
                             #   openness_delta: float, impact_description: str }

    # -------------------------------------------------------------------------
    # RESPONSE — set by response_node
    # -------------------------------------------------------------------------
    reply:             str           # the final agent response text
    reply_ts:          str           # timestamp when reply was generated
    langfuse_trace_id: str           # link back to Langfuse for debugging

    # -------------------------------------------------------------------------
    # ERROR HANDLING
    # -------------------------------------------------------------------------
    error:             Optional[str] # if any node fails, error is set here
                                     # response_node has a fallback for this