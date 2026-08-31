from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class DetectorConfig:
    """Frozen detector settings selected on the development split."""

    detector: str = "intensity"
    min_area_um2: float = 0.003
    clahe_clip_limit: float = 0.01
    intensity_threshold_offset: float = 0.0
    intensity_low_threshold_offset: float | None = None
    gaussian_sigma_um: float = 0.02
    morphology_radius_um: float = 0.01
    boundary_refinement: bool = False
    refinement_max_distance_um: float = 0.03
    refinement_growth_offset: float = 0.0
    refinement_max_area_ratio: float = 3.0
    thin_seed_rescue: bool = False
    rescue_morphology_radius_um: float = 0.005
    rescue_min_area_um2: float = 0.0015
    rescue_min_thickness_um: float = 0.045
    rescue_max_radial_position: float = 0.55
    rescue_max_eccentricity: float = 0.95
    rescue_min_solidity: float = 0.85
    exclude_scale_bar: bool = False
    scale_bar_right_fraction: float = 0.25
    scale_bar_bottom_fraction: float = 0.20
    tuned_on_split: str | None = None
    development_median_dice: float | None = None

    def __post_init__(self) -> None:
        if self.detector not in {"intensity", "geometry"}:
            raise ValueError("detector must be 'intensity' or 'geometry'")
        if self.min_area_um2 < 0:
            raise ValueError("min_area_um2 must be non-negative")
        if self.clahe_clip_limit <= 0:
            raise ValueError("clahe_clip_limit must be positive")
        if not 0 <= self.intensity_threshold_offset <= 1:
            raise ValueError("intensity_threshold_offset must be between 0 and 1")
        if self.intensity_low_threshold_offset is not None:
            if not 0 <= self.intensity_low_threshold_offset <= 1:
                raise ValueError(
                    "intensity_low_threshold_offset must be between 0 and 1"
                )
            if self.intensity_low_threshold_offset > self.intensity_threshold_offset:
                raise ValueError(
                    "intensity_low_threshold_offset cannot exceed the high threshold offset"
                )
        if self.gaussian_sigma_um < 0 or self.morphology_radius_um < 0:
            raise ValueError("physical radii must be non-negative")
        if self.refinement_max_distance_um < 0:
            raise ValueError("refinement_max_distance_um must be non-negative")
        if not -1 <= self.refinement_growth_offset <= 1:
            raise ValueError("refinement_growth_offset must be between -1 and 1")
        if self.refinement_max_area_ratio < 1:
            raise ValueError("refinement_max_area_ratio must be at least 1")
        if self.rescue_morphology_radius_um < 0:
            raise ValueError("rescue_morphology_radius_um must be non-negative")
        if self.rescue_min_area_um2 < 0 or self.rescue_min_thickness_um < 0:
            raise ValueError("rescue physical thresholds must be non-negative")
        for name, value in (
            ("rescue_max_radial_position", self.rescue_max_radial_position),
            ("rescue_max_eccentricity", self.rescue_max_eccentricity),
            ("rescue_min_solidity", self.rescue_min_solidity),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        for value in (self.scale_bar_right_fraction, self.scale_bar_bottom_fraction):
            if not 0 <= value <= 1:
                raise ValueError("scale-bar fractions must be between 0 and 1")

    @classmethod
    def from_json(cls, path: str | Path) -> "DetectorConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls(**json.load(stream))

    def to_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as stream:
            json.dump(asdict(self), stream, indent=2, sort_keys=True)
