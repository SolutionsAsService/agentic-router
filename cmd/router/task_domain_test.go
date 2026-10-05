package main

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"weave-os/router/internal/policyregistry"
	"weave-os/router/internal/router/hmm/rosterdata"
	"weave-os/router/internal/router/taskdomain"
)

func TestTaskBindingRequiresAdmittedAndLoadedRelease(t *testing.T) {
	var runtime *taskDomainRuntime
	resolver, evidence, digest, err := runtime.bind(policyregistry.Candidate{})
	require.NoError(t, err)
	assert.Nil(t, resolver)
	assert.Nil(t, evidence)
	assert.Empty(t, digest)
	releaseSHA := strings.Repeat("a", 64)
	candidate := policyregistry.Candidate{AuxiliaryModels: map[string]policyregistry.ObjectRef{taskdomain.AuxiliaryModel: {SHA256: releaseSHA}}, Policy: &rosterdata.Roster{SHA256: strings.Repeat("b", 64)}}
	_, _, _, err = runtime.bind(candidate)
	require.ErrorContains(t, err, "not loaded")
	runtime = &taskDomainRuntime{bindings: map[string]loadedTaskDomain{strings.Repeat("c", 64): {}}}
	_, _, _, err = runtime.bind(candidate)
	require.ErrorContains(t, err, "unavailable")
	runtime.bindings[releaseSHA] = loadedTaskDomain{release: taskdomain.Release{Evidence: map[string]string{strings.Repeat("d", 64): strings.Repeat("e", 64)}}}
	resolver, evidence, digest, err = runtime.bind(candidate)
	require.NoError(t, err)
	require.NotNil(t, resolver)
	assert.Nil(t, evidence)
	assert.Empty(t, digest)
	outcome := resolver.Resolve(context.Background(), taskdomain.Input{UserText: "Synthetic task"})
	assert.Equal(t, taskdomain.EvidenceUnavailable, outcome.Status)
	assert.Equal(t, releaseSHA, outcome.ReleaseSHA256)
	loaded := runtime.bindings[releaseSHA]
	loaded.release.Evidence[candidate.Policy.SHA256] = strings.Repeat("f", 64)
	loaded.evidence = map[string][]byte{strings.Repeat("f", 64): []byte(`{"invalid":"evidence"}`)}
	runtime.bindings[releaseSHA] = loaded
	_, _, _, err = runtime.bind(candidate)
	require.Error(t, err, "invalid admitted evidence cannot become a different release")
}
