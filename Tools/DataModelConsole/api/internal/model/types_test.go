package model

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestGeoPoseMarshalsTimestampWithoutPrecisionLoss(t *testing.T) {
	body, err := json.Marshal(GeoPose{TimestampNS: 1700000000000000001})
	if err != nil {
		t.Fatalf("marshal GeoPose: %v", err)
	}
	if !strings.Contains(string(body), `"timestamp_ns":"1700000000000000001"`) {
		t.Fatalf("timestamp was not encoded as a decimal string: %s", body)
	}
}
