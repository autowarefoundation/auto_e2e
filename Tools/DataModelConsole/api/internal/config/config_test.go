package config

import "testing"

func TestLoadODDOperationGates(t *testing.T) {
	t.Setenv("ODD_OPERATIONS_ENABLED", "true")
	t.Setenv("ODD_OPERATIONS_ALLOW_FULL", "false")
	t.Setenv("ODD_OPERATIONS_REQUIRED_ROLE", "odd-operator-test")
	t.Setenv("ODD_LAUNCH_PLAN_NAME", "odd-dataset-labeler")
	t.Setenv("ODD_LABELER_INPUTS_JSON", `{"immutable":"inputs"}`)

	config := Load()

	if !config.ODDOperationsEnabled ||
		config.ODDOperationsAllowFull ||
		config.ODDOperationsRequiredRole != "odd-operator-test" ||
		config.ODDLaunchPlanName != "odd-dataset-labeler" ||
		config.ODDLabelerInputsJSON != `{"immutable":"inputs"}` {
		t.Fatalf("ODD operation config = %+v", config)
	}
}

func TestLoadODDOperationDefaultsAreDisabled(t *testing.T) {
	t.Setenv("ODD_OPERATIONS_ENABLED", "")
	t.Setenv("ODD_OPERATIONS_ALLOW_FULL", "")
	t.Setenv("ODD_OPERATIONS_REQUIRED_ROLE", "")
	t.Setenv("ODD_LAUNCH_PLAN_NAME", "")
	t.Setenv("ODD_LABELER_INPUTS_JSON", "")

	config := Load()

	if config.ODDOperationsEnabled ||
		config.ODDOperationsAllowFull ||
		config.ODDOperationsRequiredRole != "console-odd-operator" ||
		config.ODDLaunchPlanName != "odd-dataset-labeler" ||
		config.ODDLabelerInputsJSON != "" {
		t.Fatalf("default ODD operation config = %+v", config)
	}
}
