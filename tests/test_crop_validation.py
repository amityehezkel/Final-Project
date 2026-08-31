import numpy as np

from mvp_pipeline.instances import FiberInstance, ExtractionResult
from mvp_pipeline.crop_validation import select_central_fiber, target_scale_for_crop


def _fiber(number: int, bbox: tuple[int, int, int, int]) -> FiberInstance:
    x0, y0, x1, y1 = bbox
    shape = (y1 - y0, x1 - x0)
    mask = np.ones(shape, dtype=bool)
    return FiberInstance(
        number=number,
        bbox=bbox,
        axon=mask,
        outer_fiber=mask,
        extraction_flags=(),
        source_cluster_axon_count=1,
        myelin_coverage=1.0,
        axon_area_um2=0.1,
        axon_solidity=1.0,
    )


def test_crop_scale_selection() -> None:
    assert target_scale_for_crop(1.0908) == 2.36
    assert target_scale_for_crop(5.523) == 4.93


def test_selects_fiber_nearest_crop_center() -> None:
    corner = _fiber(1, (0, 0, 10, 10))
    center = _fiber(2, (45, 45, 55, 55))
    extraction = ExtractionResult((corner, center), 2, {})
    assert select_central_fiber(extraction, (100, 100)) is center
    assert select_central_fiber(ExtractionResult((), 0, {}), (100, 100)) is None
