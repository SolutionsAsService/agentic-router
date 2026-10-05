package selection

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"time"

	"weave-os/router/internal/router/hmm/rosterdata"
	"weave-os/router/internal/router/taskdomain"
)

// Domain is one independent work-category bit, not a complexity label.
type Domain = taskdomain.Domain

const (
	DomainUI    = taskdomain.UI
	DomainLogic = taskdomain.Logic
	DomainData  = taskdomain.Data
	DomainInfra = taskdomain.Infra
	DomainDocs  = taskdomain.Docs
)

const (
	sparseSchemaVersion = "domain_wmi_evidence_v2"
	sparseRecipeVersion = "domain_wmi_terminal_sparse_v1"
	terminalBenchmark   = "terminalbench_v2_1"
)

// DomainProfile is absent when the classifier cannot establish a human-turn profile.
// A present profile with no active bits is a distinct, valid prediction.
type DomainProfile = taskdomain.Profile

// DomainArmEvidence retains exact-effort benchmark identity from the pinned snapshot.
type DomainArmEvidence struct {
	GlobalWII              float64  `json:"global_wii"`
	WPI                    float64  `json:"wpi"`
	TerminalQuality        *float64 `json:"terminal_quality"`
	TerminalCarriedForward bool     `json:"terminal_carried_forward"`
	TerminalClipped        bool     `json:"terminal_clipped"`
}

type domainRecipe struct {
	Influence float64            `json:"influence"`
	Weights   map[string]float64 `json:"weights"`
}

// DomainEvidence is a version-bound, candidate-complete task scoring policy.
type DomainEvidence struct {
	SchemaVersion          string                       `json:"schema_version"`
	RecipeVersion          string                       `json:"recipe_version"`
	SourceSnapshotSHA256   string                       `json:"source_snapshot_sha256"`
	SourceIngestDate       string                       `json:"source_ingest_date"`
	RosterSHA256           string                       `json:"roster_sha256"`
	WIIScoreVersion        string                       `json:"wii_score_version"`
	WIINormalizationSHA256 string                       `json:"wii_normalization_sha256"`
	WPIScoreVersion        string                       `json:"wpi_score_version"`
	WPINormalizationSHA256 string                       `json:"wpi_normalization_sha256"`
	Recipes                map[Domain]domainRecipe      `json:"recipes"`
	Arms                   map[string]DomainArmEvidence `json:"arms"`
}

// ParseDomainEvidence validates exact-arm coverage and roster/version binding
// before the quality correction can be computed. It never falls back to base effort.
func ParseDomainEvidence(payload []byte, roster *rosterdata.Roster) (*DomainEvidence, error) {
	var evidence DomainEvidence
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&evidence); err != nil {
		return nil, fmt.Errorf("parse sparse domain evidence: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return nil, errors.New("sparse domain evidence has trailing content")
	}
	if err := validateDomainEvidence(&evidence, roster); err != nil {
		return nil, err
	}
	return &evidence, nil
}

func validateDomainEvidence(evidence *DomainEvidence, roster *rosterdata.Roster) error {
	if roster == nil || roster.SHA256 == "" || evidence.RosterSHA256 != roster.SHA256 {
		return errors.New("sparse domain evidence roster binding mismatch")
	}
	if !isSHA256(evidence.SourceSnapshotSHA256) || !isSHA256(evidence.RosterSHA256) ||
		!isSHA256(evidence.WIINormalizationSHA256) || !isSHA256(evidence.WPINormalizationSHA256) {
		return errors.New("sparse domain evidence has invalid provenance")
	}
	if _, err := time.Parse(time.DateOnly, evidence.SourceIngestDate); err != nil {
		return fmt.Errorf("sparse domain evidence has invalid ingest date: %w", err)
	}
	if evidence.SchemaVersion != sparseSchemaVersion || evidence.RecipeVersion != sparseRecipeVersion ||
		evidence.WIIScoreVersion != roster.Ranking.WIIScoreVersion ||
		evidence.WIINormalizationSHA256 != roster.Ranking.WIINormalizationSHA256 ||
		evidence.WPIScoreVersion != roster.Ranking.WPIScoreVersion ||
		evidence.WPINormalizationSHA256 != roster.Ranking.WPINormalizationSHA256 {
		return errors.New("sparse domain evidence version binding mismatch")
	}
	weights := map[Domain]float64{DomainUI: 0, DomainLogic: 0.15, DomainData: 0, DomainInfra: 0.25, DomainDocs: 0}
	if len(evidence.Recipes) != len(weights) || len(evidence.Arms) != len(roster.AllArms()) {
		return errors.New("sparse domain evidence has incomplete recipes or candidate coverage")
	}
	for domain, weight := range weights {
		recipe, exists := evidence.Recipes[domain]
		if !exists || recipe.Influence != weight || len(recipe.Weights) != boolToInt(weight > 0) || (weight > 0 && recipe.Weights[terminalBenchmark] != 1) {
			return fmt.Errorf("sparse domain recipe drift for %q", domain)
		}
	}
	for _, arm := range roster.AllArms() {
		cell, exists := evidence.Arms[arm]
		if !exists || cell.TerminalQuality == nil || !boundedIndex(cell.GlobalWII) || !boundedIndex(cell.WPI) || !boundedIndex(*cell.TerminalQuality) {
			return fmt.Errorf("sparse domain evidence missing valid exact arm %q", arm)
		}
		for _, cluster := range roster.Clusters {
			if indices, found := cluster.ArmIndices[arm]; found && (math.Abs(indices.WII-cell.GlobalWII) > 1e-5 || math.Abs(indices.WPI-cell.WPI) > 1e-5) {
				return fmt.Errorf("sparse domain indices mismatch for %q", arm)
			}
		}
	}
	for _, cluster := range roster.Clusters {
		for arm := range cluster.ArmScores {
			if cell, exists := evidence.Arms[arm]; !exists || cell.TerminalQuality == nil {
				return fmt.Errorf("sparse domain evidence missing scored arm %q", arm)
			}
		}
		for arm := range cluster.ArmIndices {
			if cell, exists := evidence.Arms[arm]; !exists || cell.TerminalQuality == nil {
				return fmt.Errorf("sparse domain evidence missing indexed arm %q", arm)
			}
		}
	}
	return nil
}

func boundedIndex(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value >= 0 && value <= 100
}

func isSHA256(digest string) bool {
	if len(digest) != 64 {
		return false
	}
	_, err := hex.DecodeString(digest)
	return err == nil
}

func boolToInt(value bool) int {
	if value {
		return 1
	}
	return 0
}

// TerminalInfluence averages supported corrections across all positive bits.
// An unavailable profile and a valid all-zero profile both score at baseline.
func TerminalInfluence(profile DomainProfile) float64 {
	if len(profile) != 5 {
		return 0
	}
	influence, active := 0.0, 0
	for _, domain := range []Domain{DomainUI, DomainLogic, DomainData, DomainInfra, DomainDocs} {
		if _, known := profile[domain]; !known {
			return 0
		}
		if profile[domain] {
			active++
			switch domain {
			case DomainLogic:
				influence += 0.15
			case DomainInfra:
				influence += 0.25
			}
		}
	}
	if active == 0 {
		return 0
	}
	return influence / float64(active)
}
