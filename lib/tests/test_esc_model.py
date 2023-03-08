import numpy as np
from hypothesis import given, example, settings, strategies

import lib.esc_model as esc_model


@given(
    maximum_input_power=strategies.floats(min_value=0.0, max_value=1.0),
    throttle=strategies.floats(min_value=0.0, max_value=1.0),
)
def test_esc_input(maximum_input_power: np.float64, throttle: np.float64):
    esc = esc_model.ESC(
        efficiency=np.float64(1.0), maximum_input_power=maximum_input_power
    )

    power = maximum_input_power * throttle

    assert np.isclose(esc.solve_input(throttle), np.min([maximum_input_power, power]))


@given(
    power=strategies.floats(min_value=0.0, max_value=1.0),
)
def test_esc_output(power: np.float64):
    esc = esc_model.ESC(efficiency=np.float64(1.0), maximum_input_power=np.float64(1.0))

    assert np.isclose(esc.solve_output(power), power)
