BEGIN;

-- Failed classifications keep their row (so they still count toward resume ambiguity)
-- but become retryable once retry_after passes.
ALTER TABLE router.task_domain_profiles ADD COLUMN retry_after timestamptz;

COMMIT;
