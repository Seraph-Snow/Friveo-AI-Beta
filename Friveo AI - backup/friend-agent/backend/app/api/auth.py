# =============================================================================
# app/api/auth.py — Authentication Endpoints
# =============================================================================
# ENDPOINTS:
#   POST /auth/register  — create account, return JWT
#   POST /auth/login     — verify credentials, return JWT
#   GET  /auth/me        — return current user (requires valid JWT)
#
# FASTAPI DEPENDENCY INJECTION:
#   FastAPI's Depends() system is one of its best features.
#   Instead of manually checking auth in every endpoint, you declare:
#     current_user: User = Depends(get_current_user)
#   FastAPI calls get_current_user before your endpoint runs.
#   If auth fails, FastAPI returns 401 automatically — your code never runs.
#   This keeps endpoint logic clean and auth logic centralized.
# =============================================================================

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.postgres import get_db
from app.models.sql_models import User, PersonalityType, CompatibilityMap
from app.models.schemas import (
    RegisterRequest, LoginRequest, TokenResponse,
    UserResponse, SuccessResponse
)
from app.core.security import hash_password, verify_password, create_access_token, decode_access_token

# APIRouter groups related endpoints — mounted in main.py with a prefix
router = APIRouter(prefix="/auth", tags=["Authentication"])

# HTTPBearer extracts the token from: Authorization: Bearer <token>
bearer_scheme = HTTPBearer()


# =============================================================================
# DEPENDENCY: get_current_user
# =============================================================================
# This is the auth guard. Any endpoint that adds:
#   current_user: User = Depends(get_current_user)
# will automatically verify the JWT and get the User object injected.
#
# WHY RETURN THE FULL USER OBJECT NOT JUST THE ID?
#   Endpoints almost always need user data (personality type, timezone etc.)
#   Returning the full object means no second DB query inside the endpoint.
# =============================================================================
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Dependency that validates the JWT and returns the authenticated User.
    Raises HTTP 401 if token is missing, invalid, or expired.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Decode and verify the JWT
    user_id = decode_access_token(credentials.credentials)
    if user_id is None:
        raise credentials_exception

    # Fetch the user from Postgres
    # We use selectinload to eagerly load personality relationships
    # so we don't trigger N+1 queries when accessing user.personality_type
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()

    if user is None:
        raise credentials_exception

    return user


# =============================================================================
# REGISTER
# =============================================================================
@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new account",
)
async def register(
    request: RegisterRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Register a new user account.

    Steps:
    1. Check email isn't already taken
    2. Hash the password with bcrypt
    3. Create the user row in Postgres
    4. Generate and return a JWT token

    WHY RETURN A TOKEN ON REGISTER?
      Better UX — user doesn't have to log in immediately after registering.
      The token works exactly like a login token.
    """
    # Check if email already exists
    existing = await db.execute(
        select(User).where(User.email == request.email.lower())
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        )

    # Create the user
    user = User(
        email=request.email.lower(),          # normalize to lowercase
        hashed_password=hash_password(request.password),
        display_name=request.display_name,
    )
    db.add(user)
    await db.flush()   # flush to get the auto-generated UUID without committing
                       # flush sends SQL to DB but doesn't commit the transaction
                       # commit happens automatically in get_db() after yield

    # Generate token
    token, expires_in = create_access_token(str(user.id))

    return TokenResponse(
        access_token=token,
        expires_in=expires_in,
        user=UserResponse(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            is_onboarded=user.is_onboarded,
            timezone=user.timezone,
            created_at=user.created_at,
        )
    )


# =============================================================================
# LOGIN
# =============================================================================
@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login with email and password",
)
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Authenticate with email + password, receive a JWT token.

    WHY THE SAME ERROR FOR WRONG EMAIL AND WRONG PASSWORD?
      "Email not found" tells an attacker which emails are registered.
      "Invalid credentials" reveals nothing — attacker can't enumerate users.
      This is called "username enumeration prevention."
    """
    # Fetch user by email
    result = await db.execute(
        select(User).where(User.email == request.email.lower())
    )
    user = result.scalar_one_or_none()

    # Same error regardless of whether email exists or password is wrong
    auth_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid email or password",
    )

    if user is None:
        raise auth_error

    if not verify_password(request.password, user.hashed_password):
        raise auth_error

    # Generate token
    token, expires_in = create_access_token(str(user.id))

    # Build response — load personality names if assessment is done
    personality_code = None
    personality_name = None
    agent_code = None

    if user.personality_type_id:
        pt_result = await db.execute(
            select(PersonalityType).where(PersonalityType.id == user.personality_type_id)
        )
        pt = pt_result.scalar_one_or_none()
        if pt:
            personality_code = pt.code
            personality_name = pt.name

    if user.agent_persona_type_id:
        at_result = await db.execute(
            select(PersonalityType).where(PersonalityType.id == user.agent_persona_type_id)
        )
        at = at_result.scalar_one_or_none()
        if at:
            agent_code = at.code

    return TokenResponse(
        access_token=token,
        expires_in=expires_in,
        user=UserResponse(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            is_onboarded=user.is_onboarded,
            timezone=user.timezone,
            personality_type_code=personality_code,
            personality_type_name=personality_name,
            agent_persona_code=agent_code,
            created_at=user.created_at,
        )
    )


# =============================================================================
# GET CURRENT USER
# =============================================================================
@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get the currently authenticated user",
)
async def get_me(
    current_user: User = Depends(get_current_user),
):
    """
    Returns the currently authenticated user's profile.
    The JWT is verified by the get_current_user dependency automatically.
    """
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        display_name=current_user.display_name,
        is_onboarded=current_user.is_onboarded,
        timezone=current_user.timezone,
        created_at=current_user.created_at,
    )