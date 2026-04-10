-- PostgreSQL Init Script for Friend Agent
-- Runs automatically on first container start

CREATE TABLE IF NOT EXISTS personality_types (
    id            SERIAL PRIMARY KEY,
    code          VARCHAR(4) NOT NULL UNIQUE,
    name          VARCHAR(50) NOT NULL,
    description   TEXT,
    agent_persona JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS compatibility_map (
    id                  SERIAL PRIMARY KEY,
    user_type_id        INTEGER REFERENCES personality_types(id),
    agent_type_id       INTEGER REFERENCES personality_types(id),
    compatibility_score DECIMAL(3,2) NOT NULL,
    reason              TEXT
);

CREATE TABLE IF NOT EXISTS users (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email                 VARCHAR(255) NOT NULL UNIQUE,
    hashed_password       VARCHAR(255) NOT NULL,
    display_name          VARCHAR(100),
    personality_type_id   INTEGER REFERENCES personality_types(id),
    agent_persona_type_id INTEGER REFERENCES personality_types(id),
    is_onboarded          BOOLEAN NOT NULL DEFAULT false,
    timezone              VARCHAR(50) NOT NULL DEFAULT 'UTC',
    journal_time          TIME NOT NULL DEFAULT '21:00:00',
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at          TIMESTAMPTZ
);

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

INSERT INTO personality_types (code, name, description, agent_persona) VALUES
('INFP','The Mediator','Idealistic, empathetic, deeply values authenticity.','{"tone":"warm and gentle","style":"reflective and validating","avoid":["blunt advice"],"strengths":["emotional support","deep listening"]}'),
('ENFP','The Campaigner','Enthusiastic, creative, loves connecting ideas.','{"tone":"energetic and enthusiastic","style":"brainstorming-forward","avoid":["rigid structure"],"strengths":["generating ideas","encouragement"]}'),
('INTJ','The Architect','Strategic, independent, highly values competence.','{"tone":"direct and rigorous","style":"structured and efficient","avoid":["vague answers"],"strengths":["strategic thinking","honest feedback"]}'),
('ENTP','The Debater','Quick-witted, enjoys intellectual sparring.','{"tone":"witty and playful","style":"challenge assumptions","avoid":["being overly agreeable"],"strengths":["creative problem solving","debate"]}'),
('INFJ','The Advocate','Insightful, principled, deeply empathetic.','{"tone":"thoughtful and sincere","style":"connect actions to deeper meaning","avoid":["surface-level responses"],"strengths":["insight","helping find purpose"]}'),
('ENFJ','The Protagonist','Charismatic, empathetic, natural leader.','{"tone":"warm and motivating","style":"people-focused","avoid":["being cold"],"strengths":["motivation","social coaching"]}'),
('INTP','The Thinker','Analytical, objective, loves theoretical frameworks.','{"tone":"precise and curious","style":"explore all angles","avoid":["oversimplification"],"strengths":["deep analysis","thought experiments"]}'),
('ENTJ','The Commander','Bold, strategic, decisive.','{"tone":"confident and action-oriented","style":"clear directives","avoid":["meandering"],"strengths":["decision support","planning"]}'),
('ISFP','The Adventurer','Gentle, sensitive, lives in the present.','{"tone":"gentle and non-judgmental","style":"supportive of autonomy","avoid":["pressure"],"strengths":["present-moment focus","kindness"]}'),
('ESFP','The Entertainer','Spontaneous, energetic, lives out loud.','{"tone":"fun and upbeat","style":"light-hearted","avoid":["heavy theory"],"strengths":["lifting mood","social advice"]}'),
('ISTJ','The Logistician','Reliable, practical, dutiful.','{"tone":"calm and dependable","style":"step-by-step and factual","avoid":["vague advice"],"strengths":["planning","stability"]}'),
('ESTJ','The Executive','Organized, decisive, values order.','{"tone":"matter-of-fact","style":"clear priorities","avoid":["ambiguity"],"strengths":["organization","pragmatic advice"]}'),
('ISFJ','The Defender','Caring, loyal, attentive to others.','{"tone":"caring and supportive","style":"validates feelings","avoid":["rushing to solutions"],"strengths":["remembering details","nurturing"]}'),
('ESFJ','The Consul','Warm, social, values harmony.','{"tone":"friendly and encouraging","style":"harmony-focused","avoid":["conflict-generating responses"],"strengths":["social support","encouragement"]}'),
('ISTP','The Virtuoso','Bold, practical, masters of tools.','{"tone":"calm and pragmatic","style":"hands-on and direct","avoid":["abstract theory"],"strengths":["troubleshooting","practical solutions"]}'),
('ESTP','The Entrepreneur','Smart, energetic, perceptive.','{"tone":"direct and energetic","style":"action-first","avoid":["overthinking"],"strengths":["quick decisions","momentum"]}')
ON CONFLICT (code) DO NOTHING;

INSERT INTO compatibility_map (user_type_id, agent_type_id, compatibility_score, reason)
SELECT u.id, a.id, m.score, m.reason
FROM (VALUES
    ('INFP','ENFJ',0.95,'ENFJ provides warm guidance INFPs respond well to'),
    ('INFP','INFJ',0.90,'Shared depth and values'),
    ('ENFP','INTJ',0.92,'INTJ grounds ENFPs ideas with structure'),
    ('ENFP','INFJ',0.88,'Complementary intuition, deep connection'),
    ('INTJ','ENFP',0.90,'ENFP brings warmth to INTJs world'),
    ('INTJ','ENTP',0.88,'Intellectual sparring, mutual respect'),
    ('ENTP','INTJ',0.91,'INTJ matches ENTP depth, adds decisiveness'),
    ('ENTP','INFJ',0.87,'INFJ provides emotional grounding'),
    ('INFJ','ENFP',0.93,'ENFP energizes INFJ, shared idealism'),
    ('INFJ','ENTP',0.86,'Intellectual growth, complementary strengths'),
    ('ENFJ','INFP',0.91,'INFP appreciates ENFJs genuine care'),
    ('INTP','ENTJ',0.89,'ENTJ drives INTP to act on ideas'),
    ('ENTJ','INTP',0.90,'INTP provides the analysis ENTJ executes'),
    ('ISFP','ENFJ',0.92,'ENFJ nurtures ISFPs growth gently'),
    ('ESFP','ISFJ',0.89,'ISFJ provides stability to ESFPs spontaneity'),
    ('ISTJ','ESFP',0.86,'ESFP adds joy to ISTJs reliability'),
    ('ISFJ','ESFP',0.90,'ESFP brings excitement, ISFJ provides care'),
    ('ISTP','ESTJ',0.87,'Shared pragmatism, ESTJ adds structure'),
    ('ESTP','ISFJ',0.85,'ISFJ grounds ESTPs energy')
) AS m(user_code, agent_code, score, reason)
JOIN personality_types u ON u.code = m.user_code
JOIN personality_types a ON a.code = m.agent_code
ON CONFLICT DO NOTHING;

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_compatibility_user_type ON compatibility_map(user_type_id, compatibility_score DESC);