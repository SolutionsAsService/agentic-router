package policyclient

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"weave-os/router/internal/router/taskdomain"
)

func TestTaskDomainTransportValidatesPinnedFacts(t *testing.T) {
	digest := strings.Repeat("a", 64)
	for _, tc := range []struct {
		name, output, release string
		tokens, status        int
		valid                 bool
	}{
		{"valid", "0,1,0,1,0", digest, 10, 200, true},
		{"all zero valid", "0,0,0,0,0", digest, 10, 200, true},
		{"wrong release", "0,1,0,1,0", strings.Repeat("b", 64), 10, 200, false},
		{"prose", "Answer: 0,1,0,1,0", digest, 10, 200, false},
		{"oversize", strings.Repeat("x", 5000), digest, 10, 200, false},
		{"token overflow", "0,1,0,1,0", digest, taskdomain.MaxInputTokens + 1, 200, false},
		{"missing tokens", "0,1,0,1,0", digest, 0, 200, false},
		{"redirect", "0,1,0,1,0", digest, 10, 302, false},
		{"unavailable", "0,1,0,1,0", digest, 10, 503, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, "/classify", r.URL.Path)
				assert.Equal(t, "Bearer "+strings.Repeat("s", 32), r.Header.Get("Authorization"))
				var request map[string]string
				require.NoError(t, json.NewDecoder(r.Body).Decode(&request))
				assert.Equal(t, "Review the deployment", request["user_text"])
				assert.Equal(t, taskdomain.ProjectionVersion, request["projection_version"])
				w.Header().Set("Location", "/unexpected")
				w.WriteHeader(tc.status)
				_ = json.NewEncoder(w).Encode(map[string]any{"schema_version": taskdomain.SchemaVersion, "release_sha256": tc.release, "output": tc.output, "input_tokens": tc.tokens})
			}))
			defer server.Close()
			client, err := NewTaskDomainClassifier(server.URL, strings.Repeat("s", 32), digest, server.Client())
			require.NoError(t, err)
			profile, err := client.Classify(context.Background(), "Review the deployment")
			if !tc.valid {
				require.Error(t, err)
				assert.Nil(t, profile)
				return
			}
			require.NoError(t, err)
			require.NoError(t, profile.Validate())
			assert.Equal(t, tc.output[2] == '1', profile[taskdomain.Logic])
			assert.False(t, profile[taskdomain.UI])
		})
	}
}

func TestTaskDomainTransportRejectsUnsafeConfiguration(t *testing.T) {
	for _, endpoint := range []string{"http://localhost", "https://example.test/path", "https://user:secret@example.test", "https://example.test?x=1"} {
		_, err := NewTaskDomainClassifier(endpoint, strings.Repeat("s", 32), strings.Repeat("a", 64), nil)
		require.Error(t, err)
	}
}
