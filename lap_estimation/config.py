SAMPLE_RATE_HZ = 10.0
RANDOM_SEED = 42
STATIONARY_RPM_THRESHOLD = 100.0

CORE_SIGNAL_COLUMNS = [
    "Motor RPM [Rpm]",
    "MC Duty Cycle [%]",
    "AC Current",
    "DC Current [A]",
    "Pack_Current",
]

SUPPLEMENTARY_SIGNAL_COLUMNS = [
    "MC Volts [V]",
    "Actual_FOC_id",
    "Actual_FOC_iq",
    "Actual_Brake [%]",
]

TRAJECTORY_SIGNAL_COLUMNS = [
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
    "Wheel Angle FL",
    "Wheel Angle FR",
    "Wheel Angle RL",
    "Wheel Angle RR",
    "Oil Pressure",
]

THERMAL_COLUMNS = [
    "MC Temp [°C]",
    "Motor Temp [°C]",
    "RadIn_Temp [°C]",
    "RadOut_Tem [°C]",
    "Int Temp [°C]",
]
