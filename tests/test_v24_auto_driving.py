import torch
from snnai.benchmarks.auto_driving import SimpleAutoScenario, generate_obstacle_readings


def test_auto_driving_selects_action():
    scenario = SimpleAutoScenario()
    distances = generate_obstacle_readings(batch_size=1, seed=0)
    action, detected = scenario.run_step(distances, time_steps=20)
    assert 0 <= action < 4


def test_auto_driving_detects_front_obstacle():
    scenario = SimpleAutoScenario(detector_threshold=3)
    distances = torch.tensor([[0.1, 0.8, 0.8]])  # close front obstacle
    _, detected = scenario.run_step(distances, time_steps=20)
    assert isinstance(detected, bool)
