from pathlib import Path


def test_data_prep_image_installs_and_imports_nuplan_map_backend() -> None:
    dockerfile = (
        Path(__file__).resolve().parents[2]
        / "Platform"
        / "docker"
        / "data-prep"
        / "Dockerfile"
    ).read_text(encoding="utf-8")

    assert '"flytekit==1.16.24"' in dockerfile
    assert '"pyogrio==0.12.1"' in dockerfile
    assert '"rasterio==1.4.3"' in dockerfile
    assert '"pytest==9.0.3"' in dockerfile
    assert 'pip install --no-cache-dir --no-deps "aioboto3==15.5.0"' in dockerfile
    assert (
        'python -c "from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB"'
        in dockerfile
    )
    assert (
        'python -c "from nuplan.planning.scenario_builder.nuplan_db.'
        'nuplan_scenario_builder import NuPlanScenarioBuilder"'
        in dockerfile
    )
