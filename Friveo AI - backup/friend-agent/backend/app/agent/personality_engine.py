# =============================================================================
# app/agent/personality_engine.py — Personality Assessment Logic
# =============================================================================
# THREE RESPONSIBILITIES:
#   1. Score BFI-10 quiz answers → OCEAN dimensions (0-100 each)
#   2. Map OCEAN scores → MBTI-style personality code (16 types)
#   3. Look up best agent match from compatibility_map in Postgres
#
# WHY HERE AND NOT IN THE API LAYER?
#   This logic is pure calculation — no HTTP, no DB writes.
#   Keeping it here makes it testable independently and reusable
#   by both the quiz endpoint and the conversational endpoint.
# =============================================================================

from dataclasses import dataclass
from typing import Optional


# =============================================================================
# BFI-10 QUESTIONS
# =============================================================================
# The actual validated BFI-10 questions used in published research.
# Each question maps to an OCEAN dimension.
# Reversed items (R) are scored as 6 - response before calculating.
# Source: Rammstedt & John (2007), Journal of Research in Personality
# =============================================================================

BFI_10_QUESTIONS = [
    {
        "id":        1,
        "text":      "I see myself as someone who is reserved",
        "dimension": "extraversion",
        "reversed":  True,
        "example":   "e.g. quiet at parties, prefers small groups",
    },
    {
        "id":        2,
        "text":      "I see myself as someone who is generally trusting",
        "dimension": "agreeableness",
        "reversed":  False,
        "example":   "e.g. assumes good intentions in others",
    },
    {
        "id":        3,
        "text":      "I see myself as someone who tends to be lazy",
        "dimension": "conscientiousness",
        "reversed":  True,
        "example":   "e.g. puts things off, not very organized",
    },
    {
        "id":        4,
        "text":      "I see myself as someone who is relaxed and handles stress well",
        "dimension": "neuroticism",
        "reversed":  True,
        "example":   "e.g. calm under pressure, not easily upset",
    },
    {
        "id":        5,
        "text":      "I see myself as someone who has few artistic interests",
        "dimension": "openness",
        "reversed":  True,
        "example":   "e.g. not drawn to art, music, or creative work",
    },
    {
        "id":        6,
        "text":      "I see myself as someone who is outgoing and sociable",
        "dimension": "extraversion",
        "reversed":  False,
        "example":   "e.g. loves meeting new people, energized by crowds",
    },
    {
        "id":        7,
        "text":      "I see myself as someone who tends to find fault with others",
        "dimension": "agreeableness",
        "reversed":  True,
        "example":   "e.g. critical, skeptical of others' motives",
    },
    {
        "id":        8,
        "text":      "I see myself as someone who does a thorough job",
        "dimension": "conscientiousness",
        "reversed":  False,
        "example":   "e.g. careful, detailed, follows through",
    },
    {
        "id":        9,
        "text":      "I see myself as someone who gets nervous easily",
        "dimension": "neuroticism",
        "reversed":  False,
        "example":   "e.g. anxious, worries a lot, mood changes easily",
    },
    {
        "id":        10,
        "text":      "I see myself as someone who has an active imagination",
        "dimension": "openness",
        "reversed":  False,
        "example":   "e.g. creative, daydreams, enjoys abstract ideas",
    },
]

RESPONSE_SCALE = {
    1: "Disagree strongly",
    2: "Disagree a little",
    3: "Neither agree nor disagree",
    4: "Agree a little",
    5: "Agree strongly",
}


# =============================================================================
# OCEAN SCORING
# =============================================================================

@dataclass
class OceanScores:
    openness:          float  # 0-100
    conscientiousness: float  # 0-100
    extraversion:      float  # 0-100
    agreeableness:     float  # 0-100
    neuroticism:       float  # 0-100

    def to_dict(self) -> dict:
        return {
            "openness":          round(self.openness, 1),
            "conscientiousness": round(self.conscientiousness, 1),
            "extraversion":      round(self.extraversion, 1),
            "agreeableness":     round(self.agreeableness, 1),
            "neuroticism":       round(self.neuroticism, 1),
        }


def score_bfi10(answers: dict[int, int]) -> OceanScores:
    """
    Score BFI-10 answers into OCEAN dimensions.

    Args:
        answers: { question_id: response (1-5) }
                 e.g. { 1: 3, 2: 4, 3: 2, 4: 5, 5: 1, 6: 4, 7: 2, 8: 5, 9: 3, 10: 4 }

    Returns:
        OceanScores with each dimension 0-100

    Scoring method:
        Each dimension has 2 questions.
        Reversed items: score = 6 - answer
        Raw score per dimension: sum of 2 items (range 2-10)
        Normalized to 0-100: (raw - 2) / 8 * 100
    """
    dimension_scores = {
        "openness":          [],
        "conscientiousness": [],
        "extraversion":      [],
        "agreeableness":     [],
        "neuroticism":       [],
    }

    for question in BFI_10_QUESTIONS:
        qid  = question["id"]
        dim  = question["dimension"]
        rev  = question["reversed"]

        if qid not in answers:
            raise ValueError(f"Missing answer for question {qid}")

        raw = answers[qid]
        if raw not in range(1, 6):
            raise ValueError(f"Answer for Q{qid} must be 1-5, got {raw}")

        # Apply reversal
        score = (6 - raw) if rev else raw
        dimension_scores[dim].append(score)

    # Normalize each dimension to 0-100
    def normalize(scores: list[int]) -> float:
        raw = sum(scores)          # range 2-10
        return (raw - 2) / 8 * 100

    return OceanScores(
        openness=          normalize(dimension_scores["openness"]),
        conscientiousness= normalize(dimension_scores["conscientiousness"]),
        extraversion=      normalize(dimension_scores["extraversion"]),
        agreeableness=     normalize(dimension_scores["agreeableness"]),
        neuroticism=       normalize(dimension_scores["neuroticism"]),
    )


# =============================================================================
# OCEAN → MBTI MAPPING
# =============================================================================
# This crosswalk is based on published meta-analyses comparing the two models.
# Primary reference: McCrae & Costa (1989), Furnham (1996)
#
# The mapping:
#   E/I ← Extraversion (threshold: 50)
#   N/S ← Openness     (threshold: 50)
#   T/F ← Agreeableness + Neuroticism (combined signal)
#   J/P ← Conscientiousness (threshold: 50)
#
# T/F mapping rationale:
#   High Agreeableness → empathetic, values harmony → F
#   High Neuroticism   → emotionally reactive, values feelings → F tendency
#   Low both           → objective, detached → T
#   The 50-point threshold on A is the primary signal; N is secondary.
# =============================================================================

def ocean_to_mbti(scores: OceanScores) -> str:
    """
    Map OCEAN scores to a 4-letter MBTI code.

    Returns one of the 16 MBTI codes: INFP, ENTJ, etc.
    """
    # E vs I
    ei = "E" if scores.extraversion >= 50 else "I"

    # N vs S
    ns = "N" if scores.openness >= 50 else "S"

    # T vs F — Agreeableness is primary signal, Neuroticism adjusts
    # High A → F
    # Low A + High N → still leans F (emotional reactivity)
    # Low A + Low N  → T
    if scores.agreeableness >= 50:
        tf = "F"
    elif scores.neuroticism >= 60:
        tf = "F"
    else:
        tf = "T"

    # J vs P
    jp = "J" if scores.conscientiousness >= 50 else "P"

    return f"{ei}{ns}{tf}{jp}"


# =============================================================================
# AGENT ROLE DEFINITIONS
# =============================================================================
# Each role has:
#   name:        display name
#   description: shown to user during card selection
#   system_prompt_layer: injected at the TOP of every system prompt
#                        before the personality layer
# =============================================================================

AGENT_ROLES = {
    "friend": {
        "name":        "Friend",
        "emoji":       "👋",
        "tagline":     "Someone who just gets you",
        "description": (
            "A genuine peer who listens without judgment, celebrates your wins, "
            "and sits with you through the hard stuff. No agenda, no advice unless "
            "you ask. Just present."
        ),
        "system_prompt_layer": """
ROLE: You are this person's close friend.
- Peer relationship — you're equals, not mentor/student
- Be yourself — have opinions, share reactions, push back when you disagree
- Casual tone — match their energy, use humour when it fits
- Celebrate their wins genuinely, not performatively
- When they're struggling: presence first, advice only if asked
- You can tease, be sarcastic (lightly), be honest even when it's uncomfortable
- Don't be a yes-machine — real friends tell the truth
""".strip(),
    },

    "therapist": {
        "name":        "Therapist",
        "emoji":       "🧠",
        "tagline":     "A space to think out loud",
        "description": (
            "A reflective, non-judgmental presence that helps you understand yourself "
            "better. Not here to fix you — here to help you find your own answers."
        ),
        "system_prompt_layer": """
ROLE: You are this person's supportive companion with a therapeutic approach.
- Reflect before advising — ask clarifying questions, don't rush to solutions
- Notice patterns: "This is the third time you've mentioned X — what's that about?"
- Gently surface cognitive distortions without labeling them clinically
- Never tell them what to do — help them arrive there themselves
- Validate emotions before exploring them: "That makes sense given..."
- Use Socratic questions: "What would you tell a friend in this situation?"
- Long silences (very short responses) are okay — don't fill every space
- You are NOT a licensed therapist — if crisis signals appear, encourage professional help
""".strip(),
    },

    "mentor": {
        "name":        "Mentor",
        "emoji":       "🎯",
        "tagline":     "Someone who believes in what you can become",
        "description": (
            "An experienced guide who helps you think bigger, make better decisions, "
            "and hold yourself accountable. Direct, honest, invested in your growth."
        ),
        "system_prompt_layer": """
ROLE: You are this person's mentor.
- Long-term orientation — connect daily situations to bigger goals and growth
- Direct and honest — give your actual opinion when asked, not hedged non-answers
- Hold them accountable: "Last week you said you'd do X — what happened?"
- Help them think in frameworks: "There are really two things going on here..."
- Celebrate growth, not just outcomes: "The fact that you noticed that is progress"
- Challenge limiting beliefs directly but kindly
- Share relevant experience/perspective in first person when genuinely useful
- Don't coddle — high standards + genuine care = good mentoring
""".strip(),
    },

    "sibling": {
        "name":        "Sibling",
        "emoji":       "🤝",
        "tagline":     "Honest in the way only family can be",
        "description": (
            "The sibling who tells you the truth even when it's uncomfortable, "
            "teases you about your nonsense, but would do anything for you."
        ),
        "system_prompt_layer": """
ROLE: You are this person's older/younger sibling figure.
- Unconditional positive regard underneath, but honest on the surface
- Call out their BS — lovingly, but directly: "That's a terrible idea and you know it"
- Light teasing is natural — don't take everything seriously
- Protective instinct — get genuinely concerned when something seems off
- Family-level honesty: "I'm saying this because I care, not to be harsh"
- Shared history matters — reference past conversations naturally
- Can be blunt in a way a friend or therapist can't
- The warmth is always there, just not always on the surface
""".strip(),
    },

    "role_model": {
        "name":        "Role Model",
        "emoji":       "⭐",
        "tagline":     "The person you're trying to become",
        "description": (
            "An inspiring presence that consistently reflects the qualities "
            "you aspire to. Calm under pressure, clear in thinking, principled."
        ),
        "system_prompt_layer": """
ROLE: You are this person's role model and source of inspiration.
- Embody the qualities they aspire to: calm, clear, principled, resilient
- Inspire through questions more than statements: "What would the best version of you do here?"
- Connect their daily choices to their values and long-term identity
- Model equanimity — don't overreact, don't catastrophize
- Gentle but consistent challenge: "Is that the standard you want to hold yourself to?"
- Share perspective that elevates their thinking
- Don't preach — one well-placed insight > five motivational statements
- You believe in their potential more than they currently believe in themselves
""".strip(),
    },
}


def get_role_system_prompt(agent_role: str) -> str:
    """Get the system prompt layer for a given agent role."""
    role_data = AGENT_ROLES.get(agent_role, AGENT_ROLES["friend"])
    return role_data["system_prompt_layer"]


# =============================================================================
# CONVERSATIONAL ASSESSMENT SUFFICIENCY CHECK
# =============================================================================

@dataclass
class DimensionConfidence:
    openness:          float = 0.0
    conscientiousness: float = 0.0
    extraversion:      float = 0.0
    agreeableness:     float = 0.0
    neuroticism:       float = 0.0

    def all_sufficient(self, threshold: float = 0.7) -> bool:
        return all([
            self.openness          >= threshold,
            self.conscientiousness >= threshold,
            self.extraversion      >= threshold,
            self.agreeableness     >= threshold,
            self.neuroticism       >= threshold,
        ])

    def weakest_dimension(self) -> Optional[str]:
        """Returns the dimension with lowest confidence, if below threshold."""
        scores = {
            "openness":          self.openness,
            "conscientiousness": self.conscientiousness,
            "extraversion":      self.extraversion,
            "agreeableness":     self.agreeableness,
            "neuroticism":       self.neuroticism,
        }
        weakest = min(scores, key=scores.get)
        return weakest if scores[weakest] < 0.7 else None


# Targeted fallback questions for each dimension
# Used when conversational assessment hasn't resolved a dimension
TARGETED_QUESTIONS = {
    "extraversion": (
        "One more thing — when you have free time with no obligations, "
        "do you find yourself wanting to be around people or needing some time alone?"
    ),
    "openness": (
        "Quick one — do you tend to stick to what you know works, "
        "or are you drawn to trying new things even when the outcome is uncertain?"
    ),
    "conscientiousness": (
        "When you have something important coming up, do you plan it out carefully "
        "in advance, or do you tend to figure it out as you go?"
    ),
    "agreeableness": (
        "If a friend asks for your honest opinion on something they've worked hard on "
        "but you think has real problems — what do you usually do?"
    ),
    "neuroticism": (
        "Last one — when something stressful happens that you can't control, "
        "how long does it usually take you to settle back to your baseline?"
    ),
}