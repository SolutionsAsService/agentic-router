package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"weave-os/router/internal/observability"
	"weave-os/router/internal/policyclient"
	"weave-os/router/internal/policyregistry"
	"weave-os/router/internal/postgres"
	"weave-os/router/internal/proxy"
	"weave-os/router/internal/router/hmm/rosterdata"
	"weave-os/router/internal/router/hmm/selection"
	"weave-os/router/internal/router/taskdomain"
)

type taskDomainBinding struct {
	ReleaseFile   string            `json:"release_file"`
	ReleaseSHA256 string            `json:"release_sha256"`
	Endpoint      string            `json:"endpoint"`
	BearerEnv     string            `json:"bearer_env"`
	EvidenceFiles map[string]string `json:"evidence_files"`
}

type loadedTaskDomain struct {
	classifier taskdomain.Classifier
	release    taskdomain.Release
	evidence   map[string][]byte
}

type taskDomainRuntime struct {
	store    *postgres.TaskDomainRepo
	bindings map[string]loadedTaskDomain
}

func loadTaskDomainRuntime(pool *pgxpool.Pool) (*taskDomainRuntime, error) {
	configPath := os.Getenv("ROUTER_TASK_DOMAIN_BINDINGS_PATH")
	if configPath == "" {
		return nil, nil
	}
	if pool == nil {
		return nil, errors.New("task classification requires persistent router storage")
	}
	payload, err := readTaskDomainFile(configPath)
	if err != nil {
		return nil, err
	}
	var bindings []taskDomainBinding
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&bindings); err != nil {
		return nil, err
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) || len(bindings) == 0 || len(bindings) > 16 {
		return nil, errors.New("invalid task classifier binding inventory")
	}
	runtime := &taskDomainRuntime{store: postgres.NewTaskDomainRepo(pool), bindings: make(map[string]loadedTaskDomain)}
	for _, binding := range bindings {
		if _, exists := runtime.bindings[binding.ReleaseSHA256]; exists {
			return nil, errors.New("duplicate task classifier release")
		}
		payload, err := readTaskDomainFile(binding.ReleaseFile)
		if err != nil {
			return nil, err
		}
		release, err := taskdomain.ParseRelease(payload, binding.ReleaseSHA256)
		if err != nil {
			return nil, err
		}
		classifier, err := policyclient.NewTaskDomainClassifier(binding.Endpoint, os.Getenv(binding.BearerEnv), binding.ReleaseSHA256, nil)
		if err != nil {
			return nil, err
		}
		loaded := loadedTaskDomain{classifier: classifier, release: release, evidence: make(map[string][]byte)}
		for _, digest := range release.Evidence {
			payload, err := readTaskDomainFile(binding.EvidenceFiles[digest])
			if err != nil {
				return nil, err
			}
			if rosterdata.SHA256Hex(payload) != digest {
				return nil, errors.New("task evidence file digest mismatch")
			}
			loaded.evidence[digest] = payload
		}
		runtime.bindings[binding.ReleaseSHA256] = loaded
	}
	return runtime, nil
}

func readTaskDomainFile(path string) ([]byte, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	payload, err := io.ReadAll(io.LimitReader(file, (4<<20)+1))
	if err != nil {
		return nil, err
	}
	if len(payload) > 4<<20 {
		return nil, errors.New("task configuration exceeds size limit")
	}
	return payload, nil
}

func (r *taskDomainRuntime) bind(candidate policyregistry.Candidate) (taskdomain.Resolver, *selection.DomainEvidence, string, error) {
	ref, selected := candidate.AuxiliaryModels[taskdomain.AuxiliaryModel]
	if !selected {
		return nil, nil, "", nil
	}
	if r == nil {
		return nil, nil, "", errors.New("admitted task classifier release is not loaded")
	}
	loaded, exists := r.bindings[ref.SHA256]
	if !exists {
		return nil, nil, "", errors.New("admitted task classifier release is unavailable")
	}
	evidenceSHA := loaded.release.Evidence[candidate.Policy.SHA256]
	if evidenceSHA == "" {
		return &proxy.TaskDomainResolver{ReleaseSHA256: ref.SHA256}, nil, "", nil
	}
	evidence, err := selection.ParseDomainEvidence(loaded.evidence[evidenceSHA], candidate.Policy)
	if err != nil {
		return nil, nil, "", err
	}
	return &proxy.TaskDomainResolver{Store: r.store, Classifier: loaded.classifier, ReleaseSHA256: ref.SHA256, EvidenceSHA256: evidenceSHA}, evidence, evidenceSHA, nil
}

func (r *taskDomainRuntime) sweep(ctx context.Context) {
	ticker := time.NewTicker(time.Hour)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			sweepCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
			err := r.store.Sweep(sweepCtx)
			cancel()
			if err != nil {
				observability.FromContext(ctx).Error("Task profile sweep failed", "err", err)
			}
		}
	}
}
