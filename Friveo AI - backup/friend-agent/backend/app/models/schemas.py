# =============================================================================
# app/models/schemas.py — Pydantic Request/Response Schemas
# =============================================================================
# WHY SEPARATE SCHEMAS FROM ORM MODELS?
#   ORM models (sql_models.py) represent DATABASE tables.
#   Pydantic schemas represent what the API ACCEPTS and RETURNS.
#   These are deliberately different because:
#
#   1. You never want to return hashed_password to the client
#      UserResponse has no password field — the ORM model does.
#
#   2. Input validation is different from DB shape
#      RegisterRequest requires password confirmation.
#      The DB just stores the hash.
#
#   3. Nested/computed fields
#      The API response might include personality_type_name (a JOIN result)
#      while the DB model just has personality_type_id (a foreign key).
#
# PYDANTIC v2 SYNTAX:
#   Field(min_length=...) replaces v1's Field(..., min_length=...)
#   model_config replaces class Config.
#   These are important — v1 syntax silently fails in v2.
# =============================================================================

from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional
from datetime import datetime
import uuid


# =============================================================================
# AUTH SCHEMAS
# =============================================================================

class RegisterRequest(BaseModel):
    """What the client sends to /auth/register."""
    email: EmailStr                              # validates email format
    password: str = Field(min_length=8)          # minimum 8 chars
    display_name: Optional[str] = Field(default=None, max_length=100)

    # WHY A VALIDATOR?
    #   We could check password strength on the frontend, but backend
    #   validation is the security boundary. Frontend validation is UX,
    #   backend validation is security.
    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one number")
        return v


class LoginRequest(BaseModel):
    """What the client sends to /auth/login."""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """
    What /auth/login returns on success.

    WHY access_token + token_type?
      This follows the OAuth2 spec that FastAPI's security utilities expect.
      token_type is always "bearer" — it tells the client how to send the
      token: Authorization: Bearer <token>
    """
    access_token: str
    token_type: str = "bearer"
    expires_in: int       # seconds until expiry — useful for frontend refresh logic
    user: "UserResponse"  # include user info so frontend doesn't need a second call


class RefreshRequest(BaseModel):
    """For token refresh — not implemented yet, placeholder."""
    refresh_token: str


# =============================================================================
# USER SCHEMAS
# =============================================================================

class UserResponse(BaseModel):
    """
    Safe user data returned to clients.
    NOTE: hashed_password is deliberately ABSENT.
    """
    id: uuid.UUID
    email: str
    display_name: Optional[str]
    is_onboarded: bool
    timezone: str
    personality_type_code: Optional[str] = None    # e.g. "INFP"
    personality_type_name: Optional[str] = None    # e.g. "The Mediator"
    agent_persona_code: Optional[str] = None       # e.g. "ENFJ"
    created_at: datetime

    # model_config with from_attributes=True lets Pydantic read from
    # SQLAlchemy ORM objects directly (not just dicts).
    # Without this, UserResponse(**user.__dict__) would fail.
    model_config = {"from_attributes": True}


class UpdateProfileRequest(BaseModel):
    """For PATCH /users/me — all fields optional."""
    display_name: Optional[str] = Field(default=None, max_length=100)
    timezone: Optional[str] = None
    journal_time: Optional[str] = None   # "HH:MM" format


# =============================================================================
# PERSONALITY SCHEMAS
# =============================================================================

class PersonalityTypeResponse(BaseModel):
    """Returned when listing personality types for onboarding."""
    id: int
    code: str
    name: str
    description: Optional[str]

    model_config = {"from_attributes": True}


class PersonalityAssessmentRequest(BaseModel):
    """
    The frontend sends this after the user completes the personality quiz.
    assessed_type: the MBTI code determined from quiz answers ("INFP" etc.)
    """
    assessed_type: str = Field(min_length=4, max_length=4)


class PersonalityAssessmentResponse(BaseModel):
    """What we return after processing the assessment."""
    user_type: PersonalityTypeResponse
    agent_type: PersonalityTypeResponse
    compatibility_score: float
    compatibility_reason: str
    message: str    # a friendly explanation of the match


# =============================================================================
# CHAT SCHEMAS (used later in Step 3, defined here for completeness)
# =============================================================================

class ChatMessageRequest(BaseModel):
    """A single message from the user."""
    content: str = Field(min_length=1, max_length=4000)
    session_id: Optional[str] = None    # None = start new session


class ChatMessageResponse(BaseModel):
    """The agent's reply."""
    content: str
    session_id: str
    message_id: str
    timestamp: datetime


# =============================================================================
# GENERIC RESPONSE SCHEMAS
# =============================================================================

class SuccessResponse(BaseModel):
    """Generic success wrapper."""
    success: bool = True
    message: str


class ErrorResponse(BaseModel):
    """Generic error wrapper — matches FastAPI's HTTPException format."""
    detail: str