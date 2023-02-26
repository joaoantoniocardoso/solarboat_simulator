import numpy as np

from lib.esc_model import ESC


def test_esc_input():
    maximum_input_power = np.float64(1.0)
    esc = ESC(efficiency=np.float64(1.0), maximum_input_power=maximum_input_power)

    for throttle in np.linspace(0, 1, 100):
        assert esc.solve_input(throttle) == throttle

    assert esc.solve_input(maximum_input_power * 10) == maximum_input_power


def test_esc_output():
    esc = ESC(efficiency=np.float64(1.0), maximum_input_power=np.float64(1.0))

    for power in np.linspace(0, 1, 100):
        assert esc.solve_output(power) == power
