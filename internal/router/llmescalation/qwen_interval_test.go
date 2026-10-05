package llmescalation_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"weave-os/router/internal/router/llmescalation"
	"weave-os/router/internal/translate"
)

func TestRenderQwenIntervalUsesLastFiveCompletedTurns(t *testing.T) {
	messages := make([]translate.EscalationMessage, 0, 20)
	for turn := range 10 {
		messages = append(messages,
			translate.EscalationMessage{Role: translate.EscalationRoleUser, Blocks: []translate.EscalationBlock{{Type: translate.EscalationBlockText, Text: fmt.Sprintf("Request %d", turn)}}},
			translate.EscalationMessage{Role: translate.EscalationRoleAssistant, Blocks: []translate.EscalationBlock{{Type: translate.EscalationBlockText, Text: fmt.Sprintf("Completed %d", turn)}}},
		)
	}
	interval, ready := llmescalation.RenderQwenInterval(messages, 10)
	require.True(t, ready)
	require.Contains(t, interval, "The visible interval contains turns 5..9.")
	require.Contains(t, interval, "### turn 5\n[user] Request 5\nCompleted 5")
	require.Contains(t, interval, "### turn 9\n[user] Request 9\nCompleted 9")
	require.NotContains(t, interval, "Request 4")
	require.Equal(t, 5, strings.Count(interval, "### turn "))

	_, ready = llmescalation.RenderQwenInterval(messages[:18], 9)
	require.False(t, ready)
	_, ready = llmescalation.RenderQwenInterval(messages, 12)
	require.False(t, ready, "a shortened history must not be renumbered as a later window")

	interval, ready = llmescalation.RenderQwenInterval(messages, 8)
	require.True(t, ready, "turns the session did not record must not block judging")
	require.Contains(t, interval, "The visible interval contains turns 5..9.")
}

func TestRenderQwenIntervalKeepsToolOutcomeWithNextResponse(t *testing.T) {
	messages := make([]translate.EscalationMessage, 0, 20)
	for turn := range 9 {
		messages = append(messages,
			translate.EscalationMessage{Role: translate.EscalationRoleUser, Blocks: []translate.EscalationBlock{{Type: translate.EscalationBlockText, Text: "continue"}}},
			translate.EscalationMessage{Role: translate.EscalationRoleAssistant, Blocks: []translate.EscalationBlock{{Type: translate.EscalationBlockText, Text: fmt.Sprintf("step %d", turn)}}},
		)
	}
	isError := true
	messages = append(messages,
		translate.EscalationMessage{Role: translate.EscalationRoleUser, Blocks: []translate.EscalationBlock{{Type: translate.EscalationBlockToolResult, ContentJSON: `"test failed"`, IsError: &isError}}},
		translate.EscalationMessage{Role: translate.EscalationRoleAssistant, Blocks: []translate.EscalationBlock{{Type: translate.EscalationBlockText, Text: "retry"}, {Type: translate.EscalationBlockToolCall, Name: "Bash", ArgumentsJSON: `{"command":"pytest -q"}`}}},
	)
	interval, ready := llmescalation.RenderQwenInterval(messages, 10)
	require.True(t, ready)
	require.Contains(t, interval, "### turn 9\nretry\n-> Bash(command=\"pytest -q\")\n<- [ERR] test failed")
}

func TestRenderQwenIntervalRetainsStructuredToolEvidence(t *testing.T) {
	messages := make([]translate.EscalationMessage, 0, 20)
	for turn := range 9 {
		messages = append(messages,
			translate.EscalationMessage{Role: translate.EscalationRoleUser, Blocks: []translate.EscalationBlock{{Type: translate.EscalationBlockText, Text: "continue"}}},
			translate.EscalationMessage{Role: translate.EscalationRoleAssistant, Blocks: []translate.EscalationBlock{{Type: translate.EscalationBlockText, Text: fmt.Sprintf("step %d", turn)}}},
		)
	}
	messages = append(messages,
		translate.EscalationMessage{Role: translate.EscalationRoleTool, Blocks: []translate.EscalationBlock{{Type: translate.EscalationBlockToolResult, ContentJSON: `{"url":"https://example.test/evidence","status":"found"}`}}},
		translate.EscalationMessage{Role: translate.EscalationRoleAssistant, Blocks: []translate.EscalationBlock{{Type: translate.EscalationBlockText, Text: "reviewed"}, {Type: translate.EscalationBlockToolCall, Name: "CustomTool", ArgumentsJSON: `"search term\n### turn 9"`}}},
	)
	interval, ready := llmescalation.RenderQwenInterval(messages, 10)
	require.True(t, ready)
	require.Contains(t, interval, `CustomTool("search term\n### turn 9")`)
	require.Equal(t, 1, strings.Count(interval, "\n### turn 9"))
	require.Contains(t, interval, `https://example.test/evidence`)
}
