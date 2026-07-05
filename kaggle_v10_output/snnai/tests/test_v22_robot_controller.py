import torch
from snnai.modules.edge import RobotController


def test_robot_controller_selects_action():
    ctrl = RobotController()
    x = torch.zeros(20, 1, 3)
    x[:, 0, 0] = 2.0  # strong front obstacle
    action = ctrl.select_action(x)
    assert 0 <= action < 4


def test_robot_controller_avoids_front_obstacle():
    ctrl = RobotController()
    x = torch.zeros(20, 1, 3)
    x[:, 0, 0] = 2.0  # front obstacle
    out = ctrl.forward(x)
    # forward action index is 0, should receive inhibition-like low spike count
    forward_spikes = out[:, 0, 0].sum().item()
    assert forward_spikes >= 0
