package model

import (
	"math"
	"strconv"
	"strings"
)

// Coercion helpers for upstream JSON whose numeric fields arrive inconsistently
// as either JSON numbers or strings (MLflow does both across versions/fields).
// A json.RawMessage is decoded leniently: quotes are stripped, then parsed.

func asInt64(raw []byte) int64 {
	s := strings.Trim(string(raw), `"`)
	if s == "" || s == "null" {
		return 0
	}
	if n, err := strconv.ParseInt(s, 10, 64); err == nil {
		return n
	}
	// Some values arrive as floats (e.g. "1.699e12"); parse then truncate.
	if f, err := strconv.ParseFloat(s, 64); err == nil {
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return 0
		}
		return int64(f)
	}
	return 0
}

func asFiniteFloat64(raw []byte) (float64, bool) {
	s := strings.Trim(string(raw), `"`)
	if s == "" || s == "null" {
		return 0, false
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil || math.IsNaN(f) || math.IsInf(f, 0) {
		return 0, false
	}
	return f, true
}

// parseDurationSeconds parses a Flyte protobuf-JSON duration string ("123.4s")
// into whole seconds. Returns 0 for an empty/unparseable value.
func parseDurationSeconds(s string) int64 {
	s = strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(s), "s"))
	if s == "" {
		return 0
	}
	if f, err := strconv.ParseFloat(s, 64); err == nil {
		return int64(f)
	}
	return 0
}
