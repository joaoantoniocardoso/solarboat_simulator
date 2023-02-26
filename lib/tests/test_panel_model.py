from lib.panel_model import Panel

import numpy as np


def test_solve_output():
    efficiency = np.float64(1.0)
    area = np.float64(1.0)
    maximum_output_power = np.float64(100.0)

    panel = Panel(efficiency, area, maximum_output_power)

    irradiation = np.float64(0.0)
    assert panel.solve_output(irradiation) == irradiation

    irradiation = np.float64(1.0)
    assert panel.solve_output(irradiation) == irradiation

    irradiation = maximum_output_power * 10
    assert panel.solve_output(irradiation) == maximum_output_power
