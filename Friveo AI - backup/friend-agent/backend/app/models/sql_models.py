# =============================================================================
# app/models/sql_models.py â€” SQLAlchemy ORM Models
# =============================================================================
# WHAT IS AN ORM MODEL?
#   ORM = Object Relational Mapper. Instead of writing raw SQL like:
#     SELECT * FROM users WHERE email = 'test@example.com'
#   You write Python like:
#     await db.execute(select(User).where(User.email == 'test@example.com'))
#   SQLAlchemy translates your Python into SQL and maps results back to objects.
#
# WHY ORM OVER RAW SQL?
#   1. Type safety â€” your IDE knows User.email is a string
#   2. No SQL injection by default â€” parameters are always escaped
#   3. Database-agnostic â€” swap Postgres for SQLite in tests trivially
#   4. Relationships â€” define them once, SQLAlchemy handles the JOIN
#   The tradeoff: complex queries are sometimes cleaner in raw SQL.
#   SQLAlchemy lets you drop to raw SQL when needed â€” best of both worlds.
#
# THESE MODELS MUST MIRROR init.sql EXACTLY.
#   The init.sql creates the actual tables in Postgres.
#   These Python classes tell SQLAlchemy what those tables look like.
#   If they diverge, queries will fail with column-not-found errors.
# =============================================================================

from datetime import time as PyTime
from sqlalchemy import (
    Column, String, Integer, Boolean, Text, DateTime,
    ForeignKey, Numeric, Time, func
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import uuid

from app.db.postgres import Base


class PersonalityType(Base):
    """
    The 16 MBTI personality types.
    Seeded in init.sql â€” read-only at runtime, never written by the app.

    WHY STORE AGENT PERSONA AS JSONB?
      The persona is a flexible dict (tone, style, avoid, strengths).
      JSONB stores it as binary JSON in Postgres â€” queryable and indexable.
      If we used a Text column we couldn't query inside the JSON.
      Example: find all types where persona->>'tone' = 'warm and gentle'
    """
    __tablename__ = "personality_types"

    id          = Column(Integer, primary_key=True)
    code        = Column(String(4), nullable=False, unique=True)   # "INFP"
    name        = Column(String(50), nullable=False)               # "The Mediator"
    description = Column(Text)
    agent_persona = Column(JSONB, nullable=False, default=dict)

    # Relationships â€” makes it easy to navigate from Python objects
    # "Give me all compatibility entries where this type is the user type"
    as_user_type  = relationship("CompatibilityMap", foreign_keys="CompatibilityMap.user_type_id", back_populates="user_type")
    as_agent_type = relationship("CompatibilityMap", foreign_keys="CompatibilityMap.agent_type_id", back_populates="agent_type")

    def __repr__(self):
        return f"<PersonalityType {self.code}: {self.name}>"


class CompatibilityMap(Base):
    """
    Maps user personality type â†’ best matching agent personality type.

    WHY A SEPARATE TABLE NOT A COLUMN ON PersonalityType?
      One user type can have MULTIPLE compatible agent types (with scores).
      We want to pick the top match, but also show alternatives.
      A separate table with a score column supports both use cases cleanly.
    """
    __tablename__ = "compatibility_map"

    id                  = Column(Integer, primary_key=True)
    user_type_id        = Column(Integer, ForeignKey("personality_types.id"))
    agent_type_id       = Column(Integer, ForeignKey("personality_types.id"))
    compatibility_score = Column(Numeric(3, 2), nullable=False)
    reason              = Column(Text)

    # Relationships â€” navigate to the PersonalityType objects directly
    user_type  = relationship("PersonalityType", foreign_keys=[user_type_id],  back_populates="as_user_type")
    agent_type = relationship("PersonalityType", foreign_keys=[agent_type_id], back_populates="as_agent_type")

    def __repr__(self):
        return f"<CompatibilityMap {self.user_type_id} â†’ {self.agent_type_id}: {self.compatibility_score}>"


class User(Base):
    """
    Application users â€” auth + static profile data only.
    Dynamic data (sessions, mood) lives in MongoDB.

    WHY UUID NOT INTEGER PRIMARY KEY?
      Integer IDs are predictable â€” an attacker can enumerate users by
      trying /users/1, /users/2 etc. UUIDs are random and non-enumerable.
      Also works better in distributed systems if you ever shard the DB.

    WHY STORE ONLY HASHED_PASSWORD?
      We NEVER store the plain text password. bcrypt hashes it one-way.
      Even if someone dumps the entire users table, they get useless hashes.
      Verification happens by hashing the attempt and comparing hashes.
    """
    __tablename__ = "users"

    id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email        = Column(String(255), nullable=False, unique=True)
    hashed_password = Column(String(255), nullable=False)
    display_name = Column(String(100))

    # Personality assessment (null until onboarding complete)
    personality_type_id   = Column(Integer, ForeignKey("personality_types.id"), nullable=True)
    agent_persona_type_id = Column(Integer, ForeignKey("personality_types.id"), nullable=True)
    is_onboarded          = Column(Boolean, nullable=False, default=False)

    # Preferences
    timezone     = Column(String(50), nullable=False, default="UTC")
    journal_time = Column(Time, nullable=False, default=PyTime(21, 0, 0))

    # Timestamps â€” server-side defaults, never set by application code
    created_at   = Column(DateTime(timezone=True), server_default=func.now())
    updated_at   = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_seen_at = Column(DateTime(timezone=True), nullable=True)

    # Agent role + assessment method â€” set during onboarding
    agent_role            = Column(String(20), nullable=True)
    assessment_method     = Column(String(20), nullable=True)

    # OCEAN Big Five scores â€” set during personality assessment
    ocean_openness          = Column(Numeric(5,2), nullable=True)
    ocean_conscientiousness = Column(Numeric(5,2), nullable=True)
    ocean_extraversion      = Column(Numeric(5,2), nullable=True)
    ocean_agreeableness     = Column(Numeric(5,2), nullable=True)
    ocean_neuroticism       = Column(Numeric(5,2), nullable=True)

    # Relationships
    personality_type  = relationship("PersonalityType", foreign_keys=[personality_type_id])
    agent_persona_type = relationship("PersonalityType", foreign_keys=[agent_persona_type_id])

    def __repr__(self):
        return f"<User {self.email}>"