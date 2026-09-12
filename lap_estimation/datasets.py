from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    raw_root: Path
    artifact_root: Path
    csv_format: str
    native_sample_rate_hz: float
    clock_provenance: str
    enabled: bool
    analysis_channels: tuple[str, ...]
    trajectory_channels: tuple[str, ...]
    rpm_column: str
    motion_policy: str
    semantic_mapping: dict[str, str]
    semantic_mapping_version: str


DATASET_DEFINITIONS = {
    "drive_day_2026": {
        "raw_path": "Drive Day 7_18",
        "csv_format": "flat_timestamped_csv",
        "native_sample_rate_hz": 10.0,
        "clock_provenance": "recorded_time_column",
        "enabled": True,
        "analysis_channels": (
            "Motor RPM [Rpm]",
            "MC Duty Cycle [%]",
            "AC Current",
            "DC Current [A]",
            "Pack_Current",
        ),
        "trajectory_channels": (
            "ACC Long [g]",
            "ACC Lat [g]",
            "ACC Vert [g]",
            "Pitch Rate [deg/sec]",
            "Roll Rate [deg/sec]",
            "Yaw Rate [deg/sec]",
            "Steering Wheel Angle",
            "Wheel Speed FL",
            "Wheel Speed FR",
            "Wheel Speed RL",
            "Wheel Speed RR",
            "Oil Pressure",
        ),
        "rpm_column": "Motor RPM [Rpm]",
        "motion_policy": "motor_rpm_threshold",
        "semantic_mapping": {
            "Motor RPM [Rpm]": "motor_rpm",
            "MC Duty Cycle [%]": "motor_controller_duty_pct",
            "AC Current": "motor_ac_current",
            "DC Current [A]": "motor_dc_current_a",
            "Pack_Current": "pack_current",
        },
        "semantic_mapping_version": "1.0",
    },
    "aim_2023": {
        "raw_path": "2023",
        "csv_format": "aim_csv_missing_sample_time",
        "native_sample_rate_hz": 20.0,
        "clock_provenance": "inferred_uniform_from_metadata",
        "enabled": True,
        "analysis_channels": ("InlineAcc", "LateralAcc", "RollRate", "PitchRate", "YawRate"),
        "trajectory_channels": (
            "InlineAcc",
            "LateralAcc",
            "VerticalAcc",
            "RollRate",
            "PitchRate",
            "YawRate",
            "ECU VehSpeed",
            "ECU WheelSpdFL",
            "ECU WheelSpdFR",
            "ECU WheelSpdRL",
            "ECU WheelSpdRR",
            "Oil Pressure",
        ),
        "rpm_column": "ECU RPM",
        "motion_policy": "training_low_dynamics_quantile",
        "semantic_mapping": {
            "InlineAcc": "longitudinal_accel_g",
            "LateralAcc": "lateral_accel_g",
            "VerticalAcc": "vertical_accel_g",
            "RollRate": "roll_rate_deg_s",
            "PitchRate": "pitch_rate_deg_s",
            "YawRate": "yaw_rate_deg_s",
            "ECU RPM": "engine_rpm",
            "ECU ThrottlePos": "throttle_pct",
            "Oil Pressure": "oil_pressure_bar",
            "Rear Brake": "rear_brake_pressure_bar",
        },
        "semantic_mapping_version": "1.0",
    },
}


def resolve_dataset(key, project_root):
    if key not in DATASET_DEFINITIONS:
        raise KeyError(f"Unknown dataset: {key}")
    definition = DATASET_DEFINITIONS[key]
    project_root = Path(project_root)
    return DatasetSpec(
        key=key,
        raw_root=project_root / definition["raw_path"],
        artifact_root=project_root / "artifacts" / key,
        csv_format=definition["csv_format"],
        native_sample_rate_hz=definition["native_sample_rate_hz"],
        clock_provenance=definition["clock_provenance"],
        enabled=definition["enabled"],
        analysis_channels=tuple(definition["analysis_channels"]),
        trajectory_channels=tuple(definition["trajectory_channels"]),
        rpm_column=definition["rpm_column"],
        motion_policy=definition["motion_policy"],
        semantic_mapping=dict(definition["semantic_mapping"]),
        semantic_mapping_version=definition["semantic_mapping_version"],
    )
