# =============================================================================
# app/agent/analytics_engine.py — Wellbeing Analytics Engine
# =============================================================================
# WHAT THIS DOES:
#   Reads raw mood_logs + sessions + events and derives meaningful insights:
#   - Mood timeline (valence over time)
#   - Trigger analysis (what correlates with low mood)
#   - Recovery patterns (how long from low → baseline)
#   - What helps (humour/venting/talking → mood lift correlation)
#   - Trust growth (relationship deepening over time)
#   - Inferred goals (after 30 days or 50 snapshots)
#
# CALLED BY:
#   Weekly Celery task (Sunday 10pm) → stores results in analytics collection
#   GET /analytics/me → reads pre-computed results
#   Agent context_builder → injects weekly_insight into system prompt
#
# THRESHOLD FOR GOAL INFERENCE:
#   Minimum: 30 days of chat history OR 50 sentiment snapshots
#   Below threshold: return partial data, no goals inferred
# =============================================================================

import logging
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Optional

from app.db.mongo import (
    get_mood_logs_collection,
    get_sessions_collection,
    get_events_collection,
    get_analytics_collection,
    get_goals_collection,
)

logger = logging.getLogger(__name__)

GOAL_THRESHOLD_DAYS      = 30
GOAL_THRESHOLD_SNAPSHOTS = 50


# =============================================================================
# DATA FETCHING HELPERS
# =============================================================================

async def _get_sentiment_snapshots(user_id: str, days: int = 90) -> list[dict]:
    """Get sentiment snapshots for the last N days, newest first."""
    mood_col  = get_mood_logs_collection()
    since     = datetime.now(timezone.utc) - timedelta(days=days)
    cursor    = mood_col.find(
        {
            "user_id":   user_id,
            "doc_type":  "sentiment_snapshot",
            "timestamp": {"$gte": since.isoformat()},
        },
        sort=[("timestamp", 1)],
    )
    return await cursor.to_list(length=10000)


async def _get_agent_states(user_id: str, days: int = 90) -> list[dict]:
    """Get agent state snapshots for the last N days."""
    mood_col = get_mood_logs_collection()
    since    = datetime.now(timezone.utc) - timedelta(days=days)
    cursor   = mood_col.find(
        {
            "user_id":    user_id,
            "doc_type":   "agent_state",
            "updated_at": {"$gte": since.isoformat()},
        },
        sort=[("updated_at", 1)],
    )
    return await cursor.to_list(length=10000)


async def _get_sessions_with_turns(user_id: str, days: int = 90) -> list[dict]:
    """Get sessions with their turns for the last N days."""
    sessions_col = get_sessions_collection()
    since        = datetime.now(timezone.utc) - timedelta(days=days)
    cursor       = sessions_col.find(
        {
            "user_id":    user_id,
            "created_at": {"$gte": since.isoformat()},
            "turns.0":    {"$exists": True},
        },
        sort=[("created_at", 1)],
    )
    return await cursor.to_list(length=5000)


# =============================================================================
# MOOD TIMELINE
# =============================================================================

def compute_mood_timeline(snapshots: list[dict]) -> list[dict]:
    """
    Daily average valence from sentiment snapshots.
    Returns list of { date, avg_valence, snapshot_count, dominant_label }
    """
    daily = defaultdict(list)
    for s in snapshots:
        ts    = s.get("timestamp", "")[:10]  # YYYY-MM-DD
        val   = s.get("valence", 0.0)
        label = s.get("label", "neutral")
        daily[ts].append((val, label))

    timeline = []
    for date in sorted(daily.keys()):
        entries       = daily[date]
        values        = [e[0] for e in entries]
        labels        = [e[1] for e in entries]
        avg_valence   = sum(values) / len(values)
        # Find most common label
        label_counts  = defaultdict(int)
        for l in labels:
            label_counts[l] += 1
        dominant = max(label_counts, key=label_counts.get)

        timeline.append({
            "date":           date,
            "avg_valence":    round(avg_valence, 3),
            "snapshot_count": len(entries),
            "dominant_label": dominant,
        })

    return timeline


def compute_rolling_average(timeline: list[dict], window: int = 7) -> list[dict]:
    """Add 7-day rolling average to mood timeline."""
    for i, day in enumerate(timeline):
        window_vals = [
            timeline[j]["avg_valence"]
            for j in range(max(0, i - window + 1), i + 1)
        ]
        day["rolling_avg"] = round(sum(window_vals) / len(window_vals), 3)
    return timeline


# =============================================================================
# TRIGGER ANALYSIS
# =============================================================================

def compute_trigger_analysis(snapshots: list[dict], sessions: list[dict]) -> dict:
    """
    Identify what topics/events correlate with low mood.

    Method:
    - Find turns where valence < -0.5 (significantly low)
    - Look at the intent and any event in those turns
    - Count which intents/event_types appear most in low-mood turns
    """
    low_mood_intents     = defaultdict(int)
    low_mood_event_types = defaultdict(int)
    total_low            = 0

    # Build a map of message_id → sentiment for fast lookup
    sentiment_by_session = defaultdict(list)
    for s in snapshots:
        sentiment_by_session[s.get("session_id", "")].append(s)

    for session in sessions:
        for turn in session.get("turns", []):
            sentiment = turn.get("sentiment", {})
            valence   = sentiment.get("valence", 0.0) if sentiment else 0.0

            if valence < -0.5:
                total_low += 1
                intent = turn.get("intent", "unknown")
                low_mood_intents[intent] += 1

                event = turn.get("extracted_event", {})
                if event and event.get("has_event"):
                    etype = event.get("event_type", "other")
                    low_mood_event_types[etype] += 1

    return {
        "total_low_mood_turns": total_low,
        "top_intents": dict(sorted(low_mood_intents.items(), key=lambda x: -x[1])[:5]),
        "top_event_types": dict(sorted(low_mood_event_types.items(), key=lambda x: -x[1])[:5]),
    }


# =============================================================================
# RECOVERY PATTERNS
# =============================================================================

def compute_recovery_patterns(timeline: list[dict]) -> dict:
    """
    Measure how long it takes to recover from low mood to baseline.

    Method:
    - Find days where avg_valence < -0.4 (low)
    - Count consecutive days until valence >= -0.1 (baseline)
    - Average and trend these recovery times
    """
    recovery_times = []
    in_low         = False
    low_start_idx  = 0

    for i, day in enumerate(timeline):
        val = day["avg_valence"]
        if not in_low and val < -0.4:
            in_low        = True
            low_start_idx = i
        elif in_low and val >= -0.1:
            recovery_days = i - low_start_idx
            recovery_times.append(recovery_days)
            in_low        = False

    if not recovery_times:
        return {
            "average_recovery_days": None,
            "recovery_count":        0,
            "trend":                 "insufficient_data",
            "improving":             None,
        }

    avg = sum(recovery_times) / len(recovery_times)

    # Check if recovery is improving (getting shorter) over time
    improving = None
    if len(recovery_times) >= 3:
        first_half = recovery_times[:len(recovery_times)//2]
        second_half = recovery_times[len(recovery_times)//2:]
        avg_first  = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        improving  = avg_second < avg_first

    return {
        "average_recovery_days": round(avg, 1),
        "recovery_count":        len(recovery_times),
        "all_recovery_times":    recovery_times,
        "trend":                 "improving" if improving else "stable" if improving is None else "worsening",
        "improving":             improving,
    }


# =============================================================================
# WHAT HELPS — RESPONSE STYLE CORRELATION
# =============================================================================

def compute_what_helps(sessions: list[dict]) -> dict:
    """
    Correlate agent response style with subsequent mood change.

    For each consecutive turn pair (N, N+1):
    - Detect style of agent reply at turn N (humour, empathetic, advice, question)
    - Measure valence change: turn N+1 valence - turn N valence
    - Average mood lift per style

    Simple heuristics for style detection:
    - Humour: contains "😄 😂 haha lol funny joke 😊" or "!" + short reply
    - Empathetic: contains "understand feel sorry hard must be"
    - Advice: contains "try should could maybe consider"
    - Question: ends with "?"
    """
    style_lifts = defaultdict(list)

    for session in sessions:
        turns = session.get("turns", [])
        for i in range(len(turns) - 1):
            curr = turns[i]
            nxt  = turns[i+1]

            curr_val  = (curr.get("sentiment") or {}).get("valence")
            next_val  = (nxt.get("sentiment") or {}).get("valence")

            if curr_val is None or next_val is None:
                continue

            delta = next_val - curr_val
            reply = (curr.get("agent_reply") or "").lower()

            # Detect style
            if any(w in reply for w in ["haha", "lol", "😄", "😂", "😊", "funny", "joke"]):
                style_lifts["humour"].append(delta)
            if any(w in reply for w in ["understand", "feel", "must be", "sounds hard", "that's tough"]):
                style_lifts["empathetic"].append(delta)
            if any(w in reply for w in ["try", "should", "could", "maybe", "consider", "what if"]):
                style_lifts["advice"].append(delta)
            if reply.strip().endswith("?"):
                style_lifts["question"].append(delta)

    result = {}
    for style, lifts in style_lifts.items():
        if lifts:
            result[style] = {
                "avg_mood_lift":   round(sum(lifts) / len(lifts), 3),
                "sample_count":    len(lifts),
                "helps":           sum(lifts) / len(lifts) > 0.05,
            }

    return result


# =============================================================================
# TRUST GROWTH
# =============================================================================

def compute_trust_growth(agent_states: list[dict]) -> dict:
    """
    Track how trust_level has evolved over time.
    Returns timeline and growth rate.
    """
    if not agent_states:
        return {"timeline": [], "current": None, "growth_rate": None}

    timeline = []
    for state in agent_states:
        ts    = (state.get("updated_at") or state.get("timestamp", ""))[:10]
        trust = state.get("trust_level", 0.3)
        timeline.append({"date": ts, "trust": round(float(trust), 3)})

    if len(timeline) >= 2:
        growth = timeline[-1]["trust"] - timeline[0]["trust"]
    else:
        growth = None

    return {
        "timeline":    timeline,
        "current":     timeline[-1]["trust"] if timeline else None,
        "growth_rate": round(growth, 3) if growth is not None else None,
        "growing":     growth > 0.05 if growth is not None else None,
    }


# =============================================================================
# GOAL INFERENCE
# =============================================================================

GOAL_DEFINITIONS = {
    "calmer_responses": {
        "label":       "Become calmer under pressure",
        "description": "Reduce emotional intensity when facing stressors",
        "metric":      "recovery_days_shortening",
        "signal":      lambda analytics: (
            analytics.get("recovery", {}).get("improving") == True
        ),
        "progress_fn": lambda analytics: (
            min(100, max(0, int(
                (1 - min(analytics.get("recovery", {}).get("average_recovery_days", 5) / 5, 1)) * 100
            )))
        ),
    },
    "positive_baseline": {
        "label":       "Raise your emotional baseline",
        "description": "More days feeling neutral or positive",
        "metric":      "rolling_avg_valence_trend",
        "signal":      lambda analytics: (
            len(analytics.get("timeline", [])) >= 14 and
            sum(d["avg_valence"] for d in analytics["timeline"][-7:]) / 7 >
            sum(d["avg_valence"] for d in analytics["timeline"][:7]) / 7
        ),
        "progress_fn": lambda analytics: (
            min(100, max(0, int(
                (min(analytics.get("timeline", [{}])[-1].get("rolling_avg", 0) + 1, 2) / 2) * 100
            )))
        ),
    },
    "open_up_more": {
        "label":       "Open up more easily",
        "description": "Share feelings more freely and deeply",
        "metric":      "trust_level_growth",
        "signal":      lambda analytics: (
            analytics.get("trust", {}).get("growing") == True
        ),
        "progress_fn": lambda analytics: (
            min(100, max(0, int(
                analytics.get("trust", {}).get("current", 0.3) * 100
            )))
        ),
    },
}


async def infer_goals(
    user_id: str,
    analytics: dict,
    snapshots: list[dict],
    first_session_date: Optional[str],
) -> dict:
    """
    Infer goals from analytics data if threshold is met.
    Returns { eligible, goals, days_until_eligible }
    """
    # Check threshold
    snapshot_count = len(snapshots)
    days_of_history = 0

    if first_session_date:
        try:
            first_dt    = datetime.fromisoformat(first_session_date[:10])
            days_of_history = (datetime.now(timezone.utc).date() - first_dt.date()).days
        except Exception:
            days_of_history = 0

    eligible = (
        days_of_history >= GOAL_THRESHOLD_DAYS or
        snapshot_count >= GOAL_THRESHOLD_SNAPSHOTS
    )

    if not eligible:
        days_remaining = max(
            GOAL_THRESHOLD_DAYS - days_of_history,
            0
        )
        snapshots_remaining = max(GOAL_THRESHOLD_SNAPSHOTS - snapshot_count, 0)
        return {
            "eligible":             False,
            "days_until_eligible":  days_remaining,
            "snapshots_remaining":  snapshots_remaining,
            "goals":                [],
            "message": (
                f"Keep chatting — insights unlock in about {days_remaining} days "
                f"or after {snapshots_remaining} more conversations."
            ),
        }

    # Infer applicable goals
    inferred = []
    for goal_id, goal_def in GOAL_DEFINITIONS.items():
        try:
            signal_present = goal_def["signal"](analytics)
            if signal_present:
                progress = goal_def["progress_fn"](analytics)
                inferred.append({
                    "goal_id":     goal_id,
                    "label":       goal_def["label"],
                    "description": goal_def["description"],
                    "progress":    progress,
                    "metric":      goal_def["metric"],
                })
        except Exception as e:
            logger.debug(f"[Analytics] Goal {goal_id} signal check failed: {e}")

    return {
        "eligible": True,
        "goals":    inferred,
        "message":  None,
    }


# =============================================================================
# MAIN ANALYTICS COMPUTATION
# =============================================================================

async def compute_user_analytics(user_id: str, days: int = 90) -> dict:
    """
    Compute the full analytics suite for a user.
    Called by the weekly Celery task and cached in MongoDB.

    Returns the complete analytics dict.
    """
    logger.info(f"[Analytics] Computing for user {user_id[:8]}...")

    # Fetch raw data
    snapshots    = await _get_sentiment_snapshots(user_id, days)
    agent_states = await _get_agent_states(user_id, days)
    sessions     = await _get_sessions_with_turns(user_id, days)

    if not snapshots:
        logger.info(f"[Analytics] No data yet for user {user_id[:8]}")
        return {
            "user_id":         user_id,
            "computed_at":     datetime.now(timezone.utc).isoformat(),
            "has_data":        False,
            "snapshot_count":  0,
            "message":         "Start chatting to unlock your wellbeing insights.",
        }

    # Compute all metrics
    timeline   = compute_mood_timeline(snapshots)
    timeline   = compute_rolling_average(timeline)
    triggers   = compute_trigger_analysis(snapshots, sessions)
    recovery   = compute_recovery_patterns(timeline)
    what_helps = compute_what_helps(sessions)
    trust      = compute_trust_growth(agent_states)

    analytics = {
        "timeline":   timeline,
        "triggers":   triggers,
        "recovery":   recovery,
        "what_helps": what_helps,
        "trust":      trust,
    }

    # Get first session date for goal threshold check
    first_date = sessions[0].get("created_at") if sessions else None

    # Infer goals
    goals_result = await infer_goals(user_id, analytics, snapshots, first_date)

    result = {
        "user_id":        user_id,
        "computed_at":    datetime.now(timezone.utc).isoformat(),
        "has_data":       True,
        "snapshot_count": len(snapshots),
        "session_count":  len(sessions),
        "days_tracked":   len(set(s["date"] for s in timeline)),
        "timeline":       timeline,
        "triggers":       triggers,
        "recovery":       recovery,
        "what_helps":     what_helps,
        "trust":          trust,
        "goals":          goals_result,
    }

    logger.info(
        f"[Analytics] Done: {len(snapshots)} snapshots, "
        f"{len(sessions)} sessions, "
        f"{len(timeline)} days tracked"
    )
    return result


async def save_analytics(user_id: str, week_start: str, analytics: dict) -> None:
    """Save computed analytics to MongoDB (upsert by user_id + week_start)."""
    col = get_analytics_collection()
    await col.update_one(
        {"user_id": user_id, "week_start": week_start},
        {"$set": {**analytics, "week_start": week_start}},
        upsert=True,
    )
    logger.info(f"[Analytics] Saved for user {user_id[:8]} week {week_start}")


async def get_latest_analytics(user_id: str) -> Optional[dict]:
    """Get the most recently computed analytics for a user."""
    col    = get_analytics_collection()
    result = await col.find_one(
        {"user_id": user_id},
        sort=[("week_start", -1)],
    )
    return result


async def build_analytics_context_for_prompt(user_id: str) -> str:
    """
    Build a concise analytics summary for injection into system prompt.
    Called by context_builder — gives agent awareness of user's patterns.
    """
    analytics = await get_latest_analytics(user_id)
    if not analytics or not analytics.get("has_data"):
        return ""

    lines = []

    # Recovery pattern
    recovery = analytics.get("recovery", {})
    if recovery.get("average_recovery_days") is not None:
        trend = recovery.get("trend", "stable")
        days  = recovery["average_recovery_days"]
        lines.append(
            f"Recovery pattern: typically bounces back from low mood in ~{days} days "
            f"({trend})"
        )

    # What helps most
    what_helps = analytics.get("what_helps", {})
    helpful = [s for s, d in what_helps.items() if d.get("helps")]
    if helpful:
        lines.append(f"What tends to help this person: {', '.join(helpful)}")

    # Active goals
    goals = analytics.get("goals", {})
    if goals.get("eligible") and goals.get("goals"):
        goal_labels = [g["label"] for g in goals["goals"]]
        lines.append(f"User's growth goals: {'; '.join(goal_labels)}")

    # Trust level
    trust = analytics.get("trust", {})
    if trust.get("current") is not None:
        t = trust["current"]
        trust_desc = "deep trust" if t > 0.8 else "good trust" if t > 0.6 else "building trust" if t > 0.4 else "early relationship"
        lines.append(f"Relationship: {trust_desc} (level {t:.2f})")

    return "\n".join(lines) if lines else ""