from lib.panel_model import Panel


def test_solve_output():
    efficiency = 1.0
    area = 1.0
    maximum_output_power = 100.0

    panel = Panel(efficiency, area, maximum_output_power)

    irradiation = 0.0
    assert panel.solve_output(irradiation) == irradiation

    irradiation = 1.0
    assert panel.solve_output(irradiation) == irradiation

    irradiation = maximum_output_power * 10
    assert panel.solve_output(irradiation) == maximum_output_power
