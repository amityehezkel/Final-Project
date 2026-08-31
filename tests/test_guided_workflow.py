from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from mvp_pipeline.cli import build_parser
from mvp_pipeline.config import DetectorConfig
from mvp_pipeline.guided import (
    complete_crop_masks,
    finish_source_crops,
    load_guided_state,
    prepare_guided_crop_folder,
    prepare_guided_whole_folder,
    run_guided_detection,
    skip_source,
    validate_manual_masks,
)
from mvp_pipeline import guided
from mvp_pipeline.io import read_grayscale, write_grayscale


def _fiber_image(shape: tuple[int, int] = (100, 120)) -> np.ndarray:
    image = np.full(shape, 95, dtype=np.uint8)
    image[20:80, 25:95] = 35
    image[34:66, 42:78] = 130
    image[43:55, 82:90] = 240
    return image


def _valid_masks(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    center_y, center_x = shape[0] // 2, shape[1] // 2
    axon = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= 12**2
    outer = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= 24**2
    return axon, outer


def test_whole_folder_accepts_many_crops_keeps_input_and_resumes_without_duplicates(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "whole_inputs"
    inputs.mkdir()
    source = inputs / "whole.tif"
    write_grayscale(source, np.tile(np.arange(160, dtype=np.uint8), (120, 1)))
    output = tmp_path / "guided_output"

    state_path = prepare_guided_whole_folder(
        inputs, output, scale_nm_per_px=4.0, archive_mode="move"
    )
    source_id = load_guided_state(state_path)["sources"][0]["id"]
    rectangles = [
        [0, 0, 20, 20],
        [20, 0, 40, 20],
        [40, 0, 60, 20],
        [60, 0, 80, 20],
        [80, 0, 100, 20],
        [100, 0, 120, 20],
    ]
    crop_ids = finish_source_crops(state_path, source_id, rectangles)

    assert len(crop_ids) == 6
    assert source.exists()
    assert not (output / "processed_whole_images").exists()
    assert not (output / "skipped_whole_images").exists()
    assert len(list((output / "crops" / "images").glob("*.png"))) == 6
    status = pd.read_csv(output / "results" / "input_status.csv")
    assert status.loc[0, "status"] == "processed"
    assert int(status.loc[0, "crop_count"]) == 6

    # The original remains in the queue, but the saved identity prevents duplicates.
    resumed = prepare_guided_whole_folder(
        inputs, output, scale_nm_per_px=4.0
    )
    state = load_guided_state(resumed)
    assert state["sources"][0]["status"] == "crops_complete"
    assert len(state["crops"]) == 6


def test_guided_crop_folder_references_inputs_without_copying_and_supports_scale_csv(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "crops"
    inputs.mkdir()
    write_grayscale(inputs / "fiber_a.tif", _fiber_image())
    write_grayscale(inputs / "fiber_b.png", _fiber_image())
    scales = tmp_path / "scales.csv"
    pd.DataFrame(
        [
            {"filename": "fiber_a.tif", "scale_nm_per_px": 3.5},
            {"filename": "fiber_b.png", "scale_nm_per_px": 4.5},
        ]
    ).to_csv(scales, index=False)

    state_path = prepare_guided_crop_folder(
        inputs, tmp_path / "output", scales_csv=scales
    )
    state = load_guided_state(state_path)

    assert len(state["crops"]) == 2
    assert {row["scale_nm_per_px"] for row in state["crops"]} == {3.5, 4.5}
    for crop in state["crops"]:
        original = Path(crop["image_path"])
        assert original.parent == inputs
        assert original.exists()
        assert read_grayscale(original).shape == (100, 120)
    assert not list((tmp_path / "output" / "crops" / "images").iterdir())


def test_mask_validation_and_final_detection_outputs(tmp_path: Path) -> None:
    inputs = tmp_path / "crops"
    inputs.mkdir()
    write_grayscale(inputs / "fiber.tif", _fiber_image())
    output = tmp_path / "output"
    state_path = prepare_guided_crop_folder(
        inputs, output, scale_nm_per_px=5.0
    )
    state = load_guided_state(state_path)
    crop = state["crops"][0]
    shape = read_grayscale(crop["image_path"]).shape
    axon, outer = _valid_masks(shape)

    invalid_axon = axon.copy()
    invalid_axon[0, 0] = True
    errors, _ = validate_manual_masks(invalid_axon, outer)
    assert any("inside" in error for error in errors)
    with pytest.raises(ValueError, match="inside"):
        complete_crop_masks(state_path, crop["id"], invalid_axon, outer)

    assert complete_crop_masks(state_path, crop["id"], axon, outer) == []
    results = run_guided_detection(
        state_path,
        DetectorConfig(
            min_area_um2=0.00001,
            intensity_threshold_offset=0.05,
            gaussian_sigma_um=0.0,
            morphology_radius_um=0.0,
        ),
    )

    assert results["n_processed"] == 1
    assert (output / "input_manifest.csv").exists()
    assert (output / "results" / "metrics.csv").exists()
    assert (output / "results" / "masks" / "fiber_vacuole.png").exists()
    assert (output / "results" / "overlays" / "fiber_overlay.png").exists()
    status = pd.read_csv(output / "results" / "input_status.csv")
    assert status.loc[0, "input_filename"] == "fiber.tif"
    assert status.loc[0, "status"] == "processed"
    assert results["n_inputs_processed"] == 1
    assert results["n_inputs_skipped"] == 0
    assert load_guided_state(state_path)["status"] == "complete"


def test_skipped_whole_image_stays_in_input_and_is_reported(tmp_path: Path) -> None:
    inputs = tmp_path / "whole_inputs"
    inputs.mkdir()
    source = inputs / "skip_me.tif"
    write_grayscale(source, _fiber_image())
    output = tmp_path / "guided_output"
    state_path = prepare_guided_whole_folder(
        inputs, output, scale_nm_per_px=5.0
    )
    source_id = load_guided_state(state_path)["sources"][0]["id"]

    skip_source(state_path, source_id, reason="not_suitable")
    results = run_guided_detection(state_path, DetectorConfig())

    assert source.exists()
    assert results["n_inputs_skipped"] == 1
    status = pd.read_csv(output / "results" / "input_status.csv")
    assert status.loc[0, "status"] == "skipped"
    assert status.loc[0, "skip_reason"] == "not_suitable"


def test_cli_exposes_both_guided_workflows() -> None:
    parser = build_parser()
    whole = parser.parse_args(
        [
            "guided-whole-folder",
            "--input",
            "incoming",
            "--nm-per-pixel",
            "4.93",
            "--output",
            "guided",
        ]
    )
    crops = parser.parse_args(
        [
            "guided-crop-folder",
            "--images",
            "crops",
            "--scales-csv",
            "scales.csv",
            "--output",
            "guided",
        ]
    )

    assert whole.mode == "guided-whole-folder"
    assert whole.archive_mode == "leave"
    assert crops.mode == "guided-crop-folder"
    assert crops.scales_csv == "scales.csv"


def test_gui_bootstrap_creates_qapplication_before_returning_widgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    fake_napari = ModuleType("napari")
    fake_napari_qt = ModuleType("napari.qt")

    def get_qapp() -> object:
        calls.append("get_qapp")
        return object()

    fake_napari_qt.get_qapp = get_qapp  # type: ignore[attr-defined]
    fake_qtpy = ModuleType("qtpy")
    fake_widgets = ModuleType("qtpy.QtWidgets")
    for name in (
        "QLabel",
        "QMessageBox",
        "QPushButton",
        "QVBoxLayout",
        "QWidget",
    ):
        setattr(fake_widgets, name, type(name, (), {}))

    monkeypatch.setitem(sys.modules, "napari", fake_napari)
    monkeypatch.setitem(sys.modules, "napari.qt", fake_napari_qt)
    monkeypatch.setitem(sys.modules, "qtpy", fake_qtpy)
    monkeypatch.setitem(sys.modules, "qtpy.QtWidgets", fake_widgets)

    returned = guided._require_gui()

    assert calls == ["get_qapp"]
    assert returned[0] is fake_napari


def test_gui_windows_retain_layer_data_when_napari_removes_closed_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Signal:
        def connect(self, callback) -> None:
            self.callback = callback

    class Widget:
        pass

    class Label:
        def __init__(self, text: str) -> None:
            self.text = text

        def setWordWrap(self, value: bool) -> None:
            pass

    class Button:
        def __init__(self, text: str) -> None:
            self.text = text
            self.clicked = Signal()

    class Layout:
        def __init__(self, widget: Widget) -> None:
            pass

        def addWidget(self, widget: object) -> None:
            pass

        def addStretch(self) -> None:
            pass

    class MessageBox:
        Yes = 1

    class Layer:
        def __init__(self, data: object) -> None:
            self.data = data

    class Window:
        def add_dock_widget(self, *args, **kwargs) -> None:
            pass

    class Viewer:
        def __init__(self, *args, **kwargs) -> None:
            self.layers: dict[str, Layer] = {}
            self.window = Window()
            fake_napari.current = self

        def add_image(self, data, name: str, **kwargs) -> Layer:
            layer = Layer(data)
            self.layers[name] = layer
            return layer

        def add_shapes(self, data, name: str, **kwargs) -> Layer:
            layer = Layer(data)
            self.layers[name] = layer
            return layer

        def add_labels(self, data, name: str, **kwargs) -> Layer:
            layer = Layer(data)
            self.layers[name] = layer
            return layer

        def close(self) -> None:
            self.layers.clear()

    class FakeNapari:
        current = None

        @classmethod
        def run(cls) -> None:
            assert cls.current is not None
            # Napari 0.9 clears the layer list while closing its window.
            cls.current.layers.clear()

    fake_napari = FakeNapari
    fake_napari.Viewer = Viewer
    monkeypatch.setattr(
        guided,
        "_require_gui",
        lambda: (fake_napari, Label, MessageBox, Button, Layout, Widget),
    )
    image_path = tmp_path / "image.png"
    write_grayscale(image_path, _fiber_image())
    source = {
        "id": "source",
        "current_path": str(image_path),
        "draft_rectangles": [[10, 12, 40, 45]],
    }
    crop = {
        "id": "crop",
        "image_path": str(image_path),
        "axon_mask_path": str(tmp_path / "axon.png"),
        "outer_fiber_mask_path": str(tmp_path / "outer.png"),
    }

    crop_action, rectangles = guided._crop_selection_window(source, 1, 1)
    mask_action, axon, outer = guided._mask_annotation_window(crop, 1, 1)

    assert crop_action == "pause"
    assert rectangles == [[10.0, 12.0, 40.0, 45.0]]
    assert mask_action == "pause"
    assert axon.shape == outer.shape == (100, 120)
