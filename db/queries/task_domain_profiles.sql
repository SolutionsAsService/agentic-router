-- Reserve a root before acquiring its cross-replica inference lock.
-- name: InsertTaskDomainProfile :exec
INSERT INTO router.task_domain_profiles (conversation_key, root_sha256, release_sha256, evidence_sha256)
VALUES (@conversation_key::text, @root_sha256::text, @release_sha256::text, @evidence_sha256::text)
ON CONFLICT (conversation_key, root_sha256, release_sha256, evidence_sha256)
DO UPDATE SET outcome = NULL, expires_at = CURRENT_TIMESTAMP + INTERVAL '30 days'
WHERE task_domain_profiles.expires_at <= CURRENT_TIMESTAMP;

-- Completed profiles never wait behind unrelated first-turn inference transactions.
-- name: GetTaskDomainProfile :one
SELECT outcome FROM router.task_domain_profiles
WHERE conversation_key = @conversation_key::text AND root_sha256 = @root_sha256::text
AND release_sha256 = @release_sha256::text AND evidence_sha256 = @evidence_sha256::text
AND expires_at > CURRENT_TIMESTAMP AND outcome IS NOT NULL;

-- The row lock spans only the bounded first classification and content-free commit.
-- name: GetTaskDomainProfileForUpdate :one
SELECT outcome FROM router.task_domain_profiles
WHERE conversation_key = @conversation_key::text AND root_sha256 = @root_sha256::text
AND release_sha256 = @release_sha256::text AND evidence_sha256 = @evidence_sha256::text
AND expires_at > CURRENT_TIMESTAMP
FOR UPDATE;

-- Persist terminal outcomes, including optional inference failures, without prompt text.
-- name: UpdateTaskDomainProfile :exec
UPDATE router.task_domain_profiles SET outcome = @outcome::jsonb
WHERE conversation_key = @conversation_key::text AND root_sha256 = @root_sha256::text
AND release_sha256 = @release_sha256::text AND evidence_sha256 = @evidence_sha256::text;

-- A compacted conversation may recover only one unambiguous task root.
-- name: GetTaskDomainResumeProfiles :many
SELECT outcome FROM router.task_domain_profiles
WHERE conversation_key = @conversation_key::text AND release_sha256 = @release_sha256::text
AND evidence_sha256 = @evidence_sha256::text AND expires_at > CURRENT_TIMESTAMP
LIMIT 2;

-- Expiry never grants permission to classify a compaction summary as an original task.
-- name: DeleteExpiredTaskDomainProfiles :exec
DELETE FROM router.task_domain_profiles WHERE expires_at <= CURRENT_TIMESTAMP;
