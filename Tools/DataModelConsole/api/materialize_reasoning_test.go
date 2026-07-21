package main

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/config"
)

func TestRunReasoningMaterializerRejectsInvalidCoordinatesBeforeAWS(
	t *testing.T,
) {
	tests := []struct {
		name string
		args []string
		want string
	}{
		{
			name: "missing coordinate",
			args: []string{"--dataset", "kitscenes"},
			want: "--dataset, --version, and --manifest-sha256 are required",
		},
		{
			name: "invalid version",
			args: []string{
				"--dataset", "kitscenes",
				"--version", "../../latest",
				"--manifest-sha256", strings.Repeat("a", 64),
			},
			want: "invalid dataset version",
		},
		{
			name: "invalid manifest digest",
			args: []string{
				"--dataset", "kitscenes",
				"--version", "v2.1",
				"--manifest-sha256", "latest",
			},
			want: "invalid publication manifest SHA-256",
		},
		{
			name: "positional argument",
			args: []string{
				"--dataset", "kitscenes",
				"--version", "v2.1",
				"--manifest-sha256", strings.Repeat("a", 64),
				"unexpected",
			},
			want: "--dataset, --version, and --manifest-sha256 are required",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := runReasoningMaterializer(
				context.Background(),
				&config.Config{},
				test.args,
				&bytes.Buffer{},
			)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want substring %q", err, test.want)
			}
		})
	}
}
