# =============================================================================
# app/api/personality.py — Personality Assessment Endpoints
# =============================================================================
# ENDPOINTS:
#   GET  /personality/questions        → BFI-10 quiz questions
#   GET  /personality/agent-roles      → all agent roles with descriptions
#   POST /personality/assess/quiz      → submit quiz answers → assign type
#   POST /personality/assess/converse  → submit conversation → LLM infers type
#   GET  /personality/me               → current user's personality + agent
#   POST /personality/reassess         → reset and retake (resets onboarding)
# =============================================================================

import json
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional

from app.db.postgres import get_db
from app.models.sql_models import User, PersonalityType, CompatibilityMap
from app.api.auth import get_current_user
from app.agent.personality_engine import (
    BFI_10_QUESTIONS, RESPONSE_SCALE, AGENT_ROLES,
    OceanScores, score_bfi10, ocean_to_mbti,
    DimensionConfidence, TARGETED_QUESTIONS,
)
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/personality", tags=["Personality"])


# =============================================================================
# REQUEST SCHEMAS
# =============================================================================

class QuizAnswers(BaseModel):
    """BFI-10 quiz answers — one integer (1-5) per question."""
    answers: dict[int, int]
    # e.g. { "1": 3, "2": 4, "3": 2, "4": 5, "5": 1,
    #         "6": 4, "7": 2, "8": 5, "9": 3, "10": 4 }


class AgentRoleSelection(BaseModel):
    """Agent role chosen during onboarding Step 1."""
    agent_role: str  # one of: friend, therapist, mentor, sibling, role_model


class QuizAssessmentRequest(BaseModel):
    """Combined request — agent role + quiz answers in one body."""
    agent_role: str
    answers: dict[int, int]
    # e.g. {
    #   "agent_role": "friend",
    #   "answers": { "1": 3, "2": 4, ..., "10": 4 }
    # }


class ConversationalAssessment(BaseModel):
    """
    Result from conversational assessment.
    The LLM infers OCEAN scores from the conversation turns.
    """
    conversation: list[dict]  # [ { role: user/assistant, content: str } ]
    # Optional: if LLM already assessed, pass scores directly
    ocean_scores: Optional[dict] = None


# =============================================================================
# HELPER: assign personality and agent type from MBTI code
# =============================================================================

async def _assign_personality(
    user: User,
    mbti_code: str,
    ocean: OceanScores,
    agent_role: str,
    assessment_method: str,
    db: AsyncSession,
) -> dict:
    """
    Given an MBTI code and OCEAN scores:
    1. Look up the PersonalityType in Postgres
    2. Find the best matching agent type from compatibility_map
    3. Update the user record
    4. Return the assignment details
    """

    # Find personality type
    pt_result = await db.execute(
        select(PersonalityType).where(PersonalityType.code == mbti_code)
    )
    personality_type = pt_result.scalar_one_or_none()

    if not personality_type:
        # Fallback to INFP if code not found (shouldn't happen)
        logger.warning(f"[Personality] Code {mbti_code} not found, defaulting to INFP")
        mbti_code = "INFP"
        pt_result = await db.execute(
            select(PersonalityType).where(PersonalityType.code == "INFP")
        )
        personality_type = pt_result.scalar_one_or_none()

    # Find best agent match from compatibility_map
    compat_result = await db.execute(
        select(CompatibilityMap, PersonalityType)
        .join(PersonalityType, CompatibilityMap.agent_type_id == PersonalityType.id)
        .where(CompatibilityMap.user_type_id == personality_type.id)
        .order_by(CompatibilityMap.compatibility_score.desc())
        .limit(1)
    )
    compat_row = compat_result.first()

    if not compat_row:
        # No compatibility entry — use ENFJ as universal fallback
        agent_result = await db.execute(
            select(PersonalityType).where(PersonalityType.code == "ENFJ")
        )
        agent_type = agent_result.scalar_one_or_none()
        compatibility_score = 0.85
        compatibility_reason = "Default warm companion match"
    else:
        compat_map, agent_type = compat_row
        compatibility_score = float(compat_map.compatibility_score)
        compatibility_reason = compat_map.reason or ""

    # Update user record
    user.personality_type_id   = personality_type.id
    user.agent_persona_type_id = agent_type.id if agent_type else None
    user.is_onboarded          = True
    user.agent_role            = agent_role
    user.ocean_openness          = ocean.openness
    user.ocean_conscientiousness = ocean.conscientiousness
    user.ocean_extraversion      = ocean.extraversion
    user.ocean_agreeableness     = ocean.agreeableness
    user.ocean_neuroticism       = ocean.neuroticism
    user.assessment_method     = assessment_method

    db.add(user)
    await db.flush()

    return {
        "personality_type": {
            "code":        personality_type.code,
            "name":        personality_type.name,
            "description": personality_type.description,
        },
        "agent_type": {
            "code":        agent_type.code if agent_type else "ENFJ",
            "name":        agent_type.name if agent_type else "The Protagonist",
        },
        "agent_role":           agent_role,
        "compatibility_score":  compatibility_score,
        "compatibility_reason": compatibility_reason,
        "ocean_scores":         ocean.to_dict(),
        "assessment_method":    assessment_method,
    }


# =============================================================================
# GET /personality/questions
# =============================================================================

@router.get("/questions")
async def get_quiz_questions():
    """
    Return the BFI-10 quiz questions.
    No auth required — public endpoint for the onboarding screen.
    """
    return {
        "questions": BFI_10_QUESTIONS,
        "scale":     RESPONSE_SCALE,
        "instructions": (
            "For each statement, indicate how much you agree or disagree "
            "on a scale of 1 (Disagree strongly) to 5 (Agree strongly). "
            "Answer based on how you generally are, not how you wish you were."
        ),
    }


# =============================================================================
# GET /personality/agent-roles
# =============================================================================

@router.get("/agent-roles")
async def get_agent_roles():
    """
    Return all agent roles with descriptions for the card selection UI.
    No auth required — public endpoint.
    """
    roles = []
    for role_id, data in AGENT_ROLES.items():
        roles.append({
            "id":          role_id,
            "name":        data["name"],
            "emoji":       data["emoji"],
            "tagline":     data["tagline"],
            "description": data["description"],
        })
    return {"roles": roles}


# =============================================================================
# POST /personality/assess/quiz
# =============================================================================

@router.post("/assess/quiz")
async def assess_via_quiz(
    request:      QuizAssessmentRequest,
    current_user: User = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    """
    Submit BFI-10 quiz answers to complete personality assessment.

    Send as a single JSON body:
    {
        "agent_role": "friend",
        "answers": { "1": 3, "2": 4, "3": 2, "4": 5, "5": 1,
                     "6": 4, "7": 2, "8": 5, "9": 3, "10": 4 }
    }

    Returns the assigned personality type, agent match, and OCEAN scores.
    Sets is_onboarded: true on the user.
    """
    # Validate agent role
    if request.agent_role not in AGENT_ROLES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid agent role. Choose from: {list(AGENT_ROLES.keys())}"
        )

    # Validate and score answers
    try:
        # Convert string keys to int (JSON always sends string keys)
        int_answers = {int(k): v for k, v in request.answers.items()}
        ocean = score_bfi10(int_answers)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Map OCEAN → MBTI code
    mbti_code = ocean_to_mbti(ocean)
    logger.info(
        f"[Personality] Quiz scored: {ocean.to_dict()} → {mbti_code} "
        f"for user {current_user.id}"
    )

    # Assign and save
    result = await _assign_personality(
        user=current_user,
        mbti_code=mbti_code,
        ocean=ocean,
        agent_role=request.agent_role,
        assessment_method="quiz",
        db=db,
    )

    # Invalidate Redis profile cache so next chat loads fresh persona
    from app.db.redis_client import cache_delete
    await cache_delete(f"user:{current_user.id}:profile")

    return {
        "message": "Assessment complete! Your companion has been matched.",
        **result,
    }


# =============================================================================
# POST /personality/assess/converse
# =============================================================================

@router.post("/assess/converse")
async def assess_via_conversation(
    request:      ConversationalAssessment,
    agent_role:   str,
    current_user: User = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    """
    Submit a conversational assessment.

    If ocean_scores are provided (pre-computed by frontend after LLM inference),
    use them directly. Otherwise, run the LLM inference here.
    """
    if agent_role not in AGENT_ROLES:
        raise HTTPException(status_code=400, detail="Invalid agent role")

    # Use provided scores or infer from conversation
    if request.ocean_scores:
        ocean = OceanScores(**request.ocean_scores)
    else:
        # Run LLM inference on the conversation
        ocean = await _infer_ocean_from_conversation(request.conversation)

    mbti_code = ocean_to_mbti(ocean)
    logger.info(
        f"[Personality] Conversational scored: {ocean.to_dict()} → {mbti_code}"
    )

    result = await _assign_personality(
        user=current_user,
        mbti_code=mbti_code,
        ocean=ocean,
        agent_role=agent_role,
        assessment_method="conversational",
        db=db,
    )

    from app.db.redis_client import cache_delete
    await cache_delete(f"user:{current_user.id}:profile")

    return {
        "message": "Assessment complete! Your companion has been matched.",
        **result,
    }


async def _infer_ocean_from_conversation(conversation: list[dict]) -> OceanScores:
    """
    Use LLM to infer OCEAN scores from a conversation.
    Returns OceanScores with values 0-100.
    """
    from langchain_core.messages import HumanMessage

    conv_text = "\n".join([
        f"{t.get('role', 'user').upper()}: {t.get('content', '')}"
        for t in conversation
    ])

    prompt = f"""You are a personality psychologist analyzing a conversation to assess
Big Five (OCEAN) personality dimensions.

Conversation:
{conv_text}

Based on this conversation, estimate scores for each Big Five dimension on a 0-100 scale.
Consider:
- Openness: curiosity, creativity, openness to new experiences
- Conscientiousness: organization, discipline, reliability
- Extraversion: sociability, assertiveness, positive emotionality
- Agreeableness: cooperativeness, trust, empathy
- Neuroticism: emotional instability, anxiety, moodiness

Also provide a confidence score (0-1) for each dimension based on how much evidence
the conversation provides.

Return ONLY valid JSON:
{{
  "openness": <0-100>,
  "conscientiousness": <0-100>,
  "extraversion": <0-100>,
  "agreeableness": <0-100>,
  "neuroticism": <0-100>,
  "confidence": {{
    "openness": <0-1>,
    "conscientiousness": <0-1>,
    "extraversion": <0-1>,
    "agreeableness": <0-1>,
    "neuroticism": <0-1>
  }}
}}"""

    from langchain_ollama import ChatOllama
    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0.1,
        format="json",
    )

    response = await llm.ainvoke([HumanMessage(content=prompt)])

    # Parse response
    raw = response.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    data = json.loads(raw.strip())

    return OceanScores(
        openness=          float(data.get("openness", 50)),
        conscientiousness= float(data.get("conscientiousness", 50)),
        extraversion=      float(data.get("extraversion", 50)),
        agreeableness=     float(data.get("agreeableness", 50)),
        neuroticism=       float(data.get("neuroticism", 50)),
    )


# =============================================================================
# POST /personality/assess/converse/start
# =============================================================================

@router.post("/assess/converse/start")
async def start_conversational_assessment(
    current_user: User = Depends(get_current_user),
):
    """
    Start a conversational personality assessment.
    Returns the first question to ask the user.
    The assessment_id tracks progress across turns.
    """
    import uuid
    from app.db.redis_client import cache_set

    assessment_id = f"assessment_{uuid.uuid4().hex[:12]}"

    # Initial state stored in Redis (TTL: 30 minutes)
    state = {
        "assessment_id":  assessment_id,
        "user_id":        str(current_user.id),
        "turns":          [],
        "turn_count":     0,
        "complete":       False,
        "ocean_scores":   None,
        "confidence":     {
            "openness": 0.0, "conscientiousness": 0.0,
            "extraversion": 0.0, "agreeableness": 0.0, "neuroticism": 0.0
        }
    }

    await cache_set(
        f"assessment:{assessment_id}",
        state,
        ttl_seconds=1800
    )

    # Opening question — warm and conversational, not clinical
    opening = (
        "Before we get started properly, I'd love to get a sense of who you are. "
        "Tell me about a recent situation where you had to make a tough call — "
        "could be anything, big or small. What was going through your mind?"
    )

    return {
        "assessment_id": assessment_id,
        "message":       opening,
        "turn":          1,
        "complete":      False,
    }


# =============================================================================
# POST /personality/assess/converse/turn
# =============================================================================

class ConverseTurnRequest(BaseModel):
    assessment_id: str
    user_response: str
    agent_role:    Optional[str] = None  # only needed on final turn


@router.post("/assess/converse/turn")
async def conversational_assessment_turn(
    request:      ConverseTurnRequest,
    current_user: User = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    """
    Submit one turn of the conversational assessment.

    The agent asks follow-up questions until all 5 OCEAN dimensions
    have sufficient confidence (>= 0.7). After that, uses targeted
    fallback questions for any remaining unclear dimensions.

    Returns either:
    - Next question if assessment incomplete
    - Final result if assessment complete (all dimensions confident)
    """
    from app.db.redis_client import cache_get, cache_set, cache_delete
    from langchain_core.messages import HumanMessage

    # Load assessment state from Redis
    state = await cache_get(f"assessment:{request.assessment_id}")
    if not state:
        raise HTTPException(status_code=404, detail="Assessment session expired or not found")

    if state["user_id"] != str(current_user.id):
        raise HTTPException(status_code=403, detail="Not your assessment")

    if state["complete"]:
        raise HTTPException(status_code=400, detail="Assessment already complete")

    # Add this turn to history
    state["turns"].append({
        "role":    "user",
        "content": request.user_response
    })
    state["turn_count"] += 1

    # ── Run LLM to assess dimensions and generate next question ──────────────
    conv_text = "\n".join([
        f"{t['role'].upper()}: {t['content']}"
        for t in state["turns"]
    ])

    # Truncate conversation to last 6 turns to stay within token limits
    recent_turns = state["turns"][-6:] if len(state["turns"]) > 6 else state["turns"]
    conv_text = "\n".join([
        f"{t['role'].upper()}: {t['content'][:200]}"
        for t in recent_turns
    ])

    assessment_prompt = f"""Analyze this conversation for Big Five personality traits.
Conversation: {conv_text}
Turn {state["turn_count"]} of max 8.

Return ONLY JSON (no explanation):
{{"confidence":{{"openness":0.0,"conscientiousness":0.0,"extraversion":0.0,"agreeableness":0.0,"neuroticism":0.0}},"current_scores":{{"openness":50,"conscientiousness":50,"extraversion":50,"agreeableness":50,"neuroticism":50}},"next_question":"one short follow-up question","assessment_complete":false}}

Set assessment_complete true if all confidence >= 0.7 or turn >= 8.
Keep next_question under 30 words and conversational."""

    from langchain_ollama import ChatOllama
    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0.3,
        format="json",
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=assessment_prompt)])
        raw = response.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        llm_result = json.loads(raw.strip())
    except Exception as e:
        error_str = str(e).lower()
        if "quota" in error_str or "rate" in error_str or "429" in error_str:
            raise HTTPException(
                status_code=429,
                detail="Rate limit reached. Please wait 60 seconds and try again."
            )
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")

    confidence  = llm_result.get("confidence", {})
    scores      = llm_result.get("current_scores", {})
    is_complete = llm_result.get("assessment_complete", False)
    next_q      = llm_result.get("next_question", "")

    # Force complete after 8 turns regardless
    if state["turn_count"] >= 8:
        is_complete = True

    # Update state
    state["confidence"]   = confidence
    state["ocean_scores"] = scores

    if is_complete:
        state["complete"] = True

        # Need agent_role to finalize — either from this request or ask for it
        agent_role = request.agent_role
        if not agent_role:
            # Save state and ask for role
            await cache_set(
                f"assessment:{request.assessment_id}",
                state,
                ttl_seconds=1800
            )
            return {
                "assessment_id":   request.assessment_id,
                "complete":        False,
                "needs_role":      True,
                "message": (
                    "I think I have a good sense of you now. "
                    "One last thing — what kind of companion are you looking for? "
                    "A friend, mentor, therapist, sibling, or role model?"
                ),
                "turn": state["turn_count"] + 1,
            }

        # Finalize assessment
        ocean = OceanScores(
            openness=          float(scores.get("openness", 50)),
            conscientiousness= float(scores.get("conscientiousness", 50)),
            extraversion=      float(scores.get("extraversion", 50)),
            agreeableness=     float(scores.get("agreeableness", 50)),
            neuroticism=       float(scores.get("neuroticism", 50)),
        )
        mbti_code = ocean_to_mbti(ocean)

        result = await _assign_personality(
            user=current_user,
            mbti_code=mbti_code,
            ocean=ocean,
            agent_role=agent_role,
            assessment_method="conversational",
            db=db,
        )

        # Clean up Redis
        await cache_delete(f"assessment:{request.assessment_id}")

        # Invalidate profile cache
        from app.db.redis_client import cache_delete as cd
        await cd(f"user:{current_user.id}:profile")

        return {
            "assessment_id": request.assessment_id,
            "complete":      True,
            "message":       "Assessment complete! Your companion has been matched.",
            **result,
        }

    else:
        # Add agent question to history and save state
        state["turns"].append({
            "role":    "assistant",
            "content": next_q
        })
        await cache_set(
            f"assessment:{request.assessment_id}",
            state,
            ttl_seconds=1800
        )

        return {
            "assessment_id": request.assessment_id,
            "complete":      False,
            "message":       next_q,
            "turn":          state["turn_count"] + 1,
            "confidence":    confidence,
        }


# =============================================================================
# GET /personality/me
# =============================================================================

@router.get("/me")
async def get_my_personality(
    current_user: User = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    """Get the current user's personality type, agent match, and role."""
    if not current_user.is_onboarded:
        return {
            "is_onboarded":    False,
            "message":         "Assessment not yet completed",
            "personality_type": None,
            "agent_type":       None,
            "agent_role":       None,
        }

    personality_type = None
    agent_type = None

    if current_user.personality_type_id:
        pt = await db.execute(
            select(PersonalityType).where(
                PersonalityType.id == current_user.personality_type_id
            )
        )
        personality_type = pt.scalar_one_or_none()

    if current_user.agent_persona_type_id:
        at = await db.execute(
            select(PersonalityType).where(
                PersonalityType.id == current_user.agent_persona_type_id
            )
        )
        agent_type = at.scalar_one_or_none()

    return {
        "is_onboarded": True,
        "personality_type": {
            "code":        personality_type.code if personality_type else None,
            "name":        personality_type.name if personality_type else None,
            "description": personality_type.description if personality_type else None,
        },
        "agent_type": {
            "code": agent_type.code if agent_type else None,
            "name": agent_type.name if agent_type else None,
        },
        "agent_role":        getattr(current_user, "agent_role", "friend"),
        "ocean_scores": {
            "openness":          getattr(current_user, "ocean_openness", None),
            "conscientiousness": getattr(current_user, "ocean_conscientiousness", None),
            "extraversion":      getattr(current_user, "ocean_extraversion", None),
            "agreeableness":     getattr(current_user, "ocean_agreeableness", None),
            "neuroticism":       getattr(current_user, "ocean_neuroticism", None),
        },
        "assessment_method": getattr(current_user, "assessment_method", None),
    }


# =============================================================================
# POST /personality/reassess
# =============================================================================

@router.post("/reassess")
async def reassess(
    current_user: User = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
):
    """
    Reset onboarding so user can retake the assessment.
    The new persona takes effect on the next session.
    Partial trust reset — relationship history remains.
    """
    current_user.is_onboarded          = False
    current_user.personality_type_id   = None
    current_user.agent_persona_type_id = None
    current_user.agent_role            = None

    db.add(current_user)
    await db.flush()

    # Partial trust reset in MongoDB
    from app.db.mongo import get_mood_logs_collection
    mood_col = get_mood_logs_collection()
    await mood_col.update_one(
        {"user_id": str(current_user.id), "doc_type": "agent_state"},
        {"$set": {
            "trust_level": 0.4,   # partial reset, not full — history remains
            "openness":    0.5,
            "mood":        "neutral",
            "last_impact": "User reset their personality assessment",
        }}
    )

    # Invalidate Redis cache
    from app.db.redis_client import cache_delete
    await cache_delete(f"user:{current_user.id}:profile")

    return {
        "message": "Assessment reset. Complete onboarding again to get a new companion match.",
        "is_onboarded": False,
    }