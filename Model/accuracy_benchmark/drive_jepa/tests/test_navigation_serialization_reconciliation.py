import json
from pathlib import Path
import runpy
import pytest

reconcile = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "regenerate_navigation.py")
)["reconcile_quality_scalars"]


def fixture():
    data = {
        "navigation_map": {"geometry": [[1.25, 2.5]]},
        "navigation_route": {
            "route_id": "route",
            "map_version": "map",
            "valid": True,
            "confidence": 0.8,
            "quality": {"median_lateral_distance_m": 0.7935814989867734},
            "lane_sequence": [{"centerline_enu_m": [[0, 0], [10, 2]]}],
        },
    }
    metadata = {
        "route_id": "route",
        "map_version": "map",
        "route_valid": True,
        "route_confidence": 0.8,
        "route_quality_median_lateral_distance_m": 0.7935814989867733,
    }
    return data, metadata


def test_only_published_roundoff_scalar_changes():
    data, metadata = fixture()
    payload, changes = reconcile(json.dumps(data).encode(), metadata)
    result = json.loads(payload)
    assert result["navigation_map"] == data["navigation_map"]
    assert (
        result["navigation_route"]["lane_sequence"]
        == data["navigation_route"]["lane_sequence"]
    )
    assert list(changes) == ["route_quality_median_lateral_distance_m"]


@pytest.mark.parametrize(
    "field,value",
    [("route_id", "different"), ("route_quality_median_lateral_distance_m", 0.9)],
)
def test_material_or_identity_changes_rejected(field, value):
    data, metadata = fixture()
    metadata[field] = value
    with pytest.raises(ValueError):
        reconcile(json.dumps(data).encode(), metadata)


def test_invalid_raster_identity_does_not_replace_scene_route_identity_or_confidence():
    data, metadata = fixture()
    data["navigation_route"]["valid"] = False
    metadata.update(route_valid=False, route_id="", route_confidence=0.0)
    payload, changes = reconcile(json.dumps(data).encode(), metadata)
    route = json.loads(payload)["navigation_route"]
    assert route["route_id"] == "route"
    assert route["confidence"] == 0.8
    assert not route["valid"]
    assert list(changes) == ["route_quality_median_lateral_distance_m"]
