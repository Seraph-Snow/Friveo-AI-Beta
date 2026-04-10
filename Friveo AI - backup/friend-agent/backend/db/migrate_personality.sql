-- =============================================================================
-- Migration: Add personality assessment columns to users table
-- =============================================================================
-- Run this manually once:
--   docker exec -it friend-postgres psql -U friendagent -d friendagent -f /docker-entrypoint-initdb.d/migrate_step5.sql
-- OR connect via TablePlus and run it directly
-- =============================================================================

-- Agent role — what role the agent plays for this user
-- Set during onboarding Step 1 (agent card selection)
ALTER TABLE users
    ADD COLUMN IF NOT EXISTS agent_role VARCHAR(20)
    CHECK (agent_role IN ('friend','therapist','mentor','sibling','role_model'));

-- Big Five (OCEAN) scores — stored for reassessment comparison
-- Set during onboarding Step 2 (quiz or conversational assessment)
ALTER TABLE users
    ADD COLUMN IF NOT EXISTS ocean_openness          DECIMAL(5,2),
    ADD COLUMN IF NOT EXISTS ocean_conscientiousness DECIMAL(5,2),
    ADD COLUMN IF NOT EXISTS ocean_extraversion      DECIMAL(5,2),
    ADD COLUMN IF NOT EXISTS ocean_agreeableness     DECIMAL(5,2),
    ADD COLUMN IF NOT EXISTS ocean_neuroticism       DECIMAL(5,2);

-- Assessment method used — useful for analytics and reassessment UX
ALTER TABLE users
    ADD COLUMN IF NOT EXISTS assessment_method VARCHAR(20)
    CHECK (assessment_method IN ('quiz', 'conversational'));

-- Verify columns were added
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'users'
  AND column_name IN (
    'agent_role', 'ocean_openness', 'ocean_conscientiousness',
    'ocean_extraversion', 'ocean_agreeableness', 'ocean_neuroticism',
    'assessment_method'
  )
ORDER BY column_name;