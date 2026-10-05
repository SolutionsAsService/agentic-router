// Command task_domain_check verifies content-free profile persistence on a
// disposable migrated localhost database. It never loads deployment credentials.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"net/url"
	"os"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"golang.org/x/sync/errgroup"
	"weave-os/router/internal/postgres"
	"weave-os/router/internal/router/taskdomain"
)

func main() {
	if err := check(os.Getenv("ROUTER_TEST_DATABASE_URL")); err != nil {
		slog.Error("Task profile database check failed", "err", err)
		os.Exit(1)
	}
	slog.Info("Task profile database check passed: replica deduplication, cache capacity, ambiguity, version isolation, failures, rollback and expiry")
}

func identity() string {
	sum := sha256.Sum256([]byte(uuid.NewString()))
	return hex.EncodeToString(sum[:])
}

func check(dsn string) (checkErr error) {
	parsed, err := url.Parse(dsn)
	if err != nil {
		return err
	}
	if parsed.Hostname() != "localhost" && parsed.Hostname() != "127.0.0.1" && parsed.Hostname() != "::1" {
		return errors.New("ROUTER_TEST_DATABASE_URL must name a disposable localhost database")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return err
	}
	defer pool.Close()
	key := taskdomain.Key{Conversation: identity(), Root: identity(), Release: identity(), Evidence: identity()}
	defer func() {
		cleanupCtx, stop := context.WithTimeout(context.Background(), time.Second)
		defer stop()
		_, err := pool.Exec(cleanupCtx, "DELETE FROM router.task_domain_profiles WHERE conversation_key = $1", key.Conversation)
		checkErr = errors.Join(checkErr, err)
	}()
	repo, replica := postgres.NewTaskDomainRepo(pool), postgres.NewTaskDomainRepo(pool)
	var calls atomic.Int32
	profile := taskdomain.Profile{taskdomain.UI: false, taskdomain.Logic: true, taskdomain.Data: false, taskdomain.Infra: false, taskdomain.Docs: false}
	classify := func(context.Context) taskdomain.Outcome {
		calls.Add(1)
		return taskdomain.Outcome{Status: taskdomain.Ready, Profile: profile, ReleaseSHA256: key.Release, EvidenceSHA256: key.Evidence}
	}
	started, finish := make(chan struct{}), make(chan struct{})
	group, parallelCtx := errgroup.WithContext(ctx)
	group.Go(func() error {
		_, err := repo.Resolve(parallelCtx, key, false, func(ctx context.Context) taskdomain.Outcome {
			close(started)
			select {
			case <-finish:
			case <-ctx.Done():
			}
			return classify(ctx)
		})
		return err
	})
	select {
	case <-started:
	case <-ctx.Done():
		return ctx.Err()
	}
	group.Go(func() error { _, err := replica.Resolve(parallelCtx, key, false, classify); return err })
	close(finish)
	if err := group.Wait(); err != nil {
		return err
	}
	if calls.Load() != 1 {
		return fmt.Errorf("concurrent inference count %d, want 1", calls.Load())
	}
	cached, err := replica.Resolve(ctx, key, false, classify)
	if err != nil {
		return err
	}
	if !cached.Cached || !cached.Profile[taskdomain.Logic] || calls.Load() != 1 {
		return errors.New("replica did not reuse committed prediction")
	}
	resumed, err := repo.Resolve(ctx, key, true, classify)
	if err != nil || resumed.Status != taskdomain.Ready {
		return fmt.Errorf("single root resume: %v, %v", resumed.Status, err)
	}
	changedVersion := key
	changedVersion.Release = identity()
	missing, err := repo.Resolve(ctx, changedVersion, true, classify)
	if err != nil || missing.Status != taskdomain.NoTask {
		return fmt.Errorf("release isolation: %v, %v", missing.Status, err)
	}
	child := key
	child.Root = identity()
	_, err = repo.Resolve(ctx, child, false, classify)
	if err != nil {
		return err
	}
	ambiguous, err := repo.Resolve(ctx, key, true, classify)
	if err != nil || ambiguous.Status != taskdomain.NoTask {
		return fmt.Errorf("ambiguous resume: %v, %v", ambiguous.Status, err)
	}
	failed := key
	failed.Root = identity()
	failure := func(context.Context) taskdomain.Outcome {
		return taskdomain.Outcome{Status: taskdomain.TimedOut, ReleaseSHA256: key.Release, EvidenceSHA256: key.Evidence}
	}
	if _, err := repo.Resolve(ctx, failed, false, failure); err != nil {
		return err
	}
	negative, err := replica.Resolve(ctx, failed, false, classify)
	if err != nil || !negative.Cached || negative.Status != taskdomain.TimedOut {
		return fmt.Errorf("terminal failure not retained: %v, %v", negative, err)
	}
	rollback := key
	rollback.Root = identity()
	invalid := func(context.Context) taskdomain.Outcome { return taskdomain.Outcome{Status: taskdomain.Ready} }
	if _, err := repo.Resolve(ctx, rollback, false, invalid); err == nil {
		return errors.New("invalid outcome committed")
	}
	recovered, err := replica.Resolve(ctx, rollback, false, classify)
	if err != nil || recovered.Cached || recovered.Status != taskdomain.Ready {
		return fmt.Errorf("rollback recovery: %v, %v", recovered.Status, err)
	}
	if _, err := pool.Exec(ctx, "UPDATE router.task_domain_profiles SET expires_at = CURRENT_TIMESTAMP - INTERVAL '1 second' WHERE conversation_key = $1", key.Conversation); err != nil {
		return err
	}
	expired, err := replica.Resolve(ctx, key, true, classify)
	if err != nil || expired.Status != taskdomain.NoTask {
		return fmt.Errorf("expired resume: %v, %v", expired.Status, err)
	}
	renewed, err := repo.Resolve(ctx, key, false, classify)
	if err != nil || renewed.Cached || renewed.Status != taskdomain.Ready {
		return fmt.Errorf("expired root renewal: %v, %v", renewed.Status, err)
	}
	// Fill the optional inference slots; completed roots must still be readable.
	blockersStarted, releaseBlockers := make(chan struct{}, 2), make(chan struct{})
	blockers, blockedCtx := errgroup.WithContext(ctx)
	for range 2 {
		blockedKey := key
		blockedKey.Root = identity()
		blockers.Go(func() error {
			_, err := repo.Resolve(blockedCtx, blockedKey, false, func(ctx context.Context) taskdomain.Outcome {
				blockersStarted <- struct{}{}
				select {
				case <-releaseBlockers:
				case <-ctx.Done():
				}
				return classify(ctx)
			})
			return err
		})
	}
	defer close(releaseBlockers)
	for range 2 {
		select {
		case <-blockersStarted:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	stillCached, err := repo.Resolve(ctx, key, false, classify)
	if err != nil || !stillCached.Cached {
		return fmt.Errorf("cache blocked by inference capacity: %v", err)
	}
	// Release each transaction before the pool closes.
	releaseBlockers <- struct{}{}
	releaseBlockers <- struct{}{}
	return blockers.Wait()
}
