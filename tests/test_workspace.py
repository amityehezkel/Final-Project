from pathlib import Path

from mvp_pipeline.workspace import PROJECT_ROOT, check_workspace


def test_current_workspace_is_complete_and_portable() -> None:
    report = check_workspace(PROJECT_ROOT)
    assert report["crop_modes_ready"]
    assert report["whole_image_ready"]
    assert report["benchmark_ready_and_portable"]
    assert report["complete_workspace_ready"]
    assert "guided_workflow_ready" in report
    assert "interactive_annotation" in report
    assert report["benchmark"]["rows"] == 43
    assert report["benchmark"]["absolute_path_count"] == 0
    assert Path(report["detector_config"]["path"]).exists()
