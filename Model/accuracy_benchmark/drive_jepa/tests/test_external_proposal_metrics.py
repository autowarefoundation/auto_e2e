import torch

from accuracy_benchmark.drive_jepa.inspect_external_proposals import candidate_totals


def test_oracle_ade_and_fde_choose_independently_and_samples_have_equal_weight():
    errors = torch.zeros(2, 2, 10)
    errors[0, 0] = 1.0
    errors[0, 0, -1] = 10.0
    errors[0, 1] = 3.0
    errors[1, 0] = 2.0
    errors[1, 1] = 4.0
    totals = candidate_totals(errors)
    # Sample0 ADE candidate0=1.9 wins; endpoint candidate1=3 wins.
    assert abs(totals["oracle_min"][1][0] - 3.9) < 1e-6
    assert totals["oracle_min"][1][2] == 5.0
    assert totals["oracle_min"][1][1] == 2
    assert abs(totals["candidate_mean"][1][0] - 5.45) < 1e-6
    assert totals["candidate_first"][1][2] == 12.0
