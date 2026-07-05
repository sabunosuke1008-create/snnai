"""Minimal autonomous-driving obstacle-detection scenario."""
import torch

from snnai.modules.edge import IoTEventDetector, RobotController


class SimpleAutoScenario:
    """A toy self-driving scenario with lidar-like distance sensors.

    The car receives front/left/right distance readings. When an obstacle
    is close in front, the robot controller should choose to turn.

    State: (front_dist, left_dist, right_dist) where smaller is closer.
    """

    def __init__(self, detector_threshold=5):
        self.detector = IoTEventDetector(input_size=3, hidden_size=16, threshold=detector_threshold)
        self.controller = RobotController(beta=0.9, threshold=1.0)

    def run_step(self, distances, time_steps=20):
        """Run one control step from distance readings.

        Parameters
        ----------
        distances : torch.Tensor
            Distance readings of shape (batch, 3).
        time_steps : int
            Number of simulation steps.

        Returns
        -------
        int
            Selected action (0=forward, 1=turn_left, 2=turn_right, 3=stop).
        bool
            Whether an obstacle event was detected.
        """
        # Invert distances so that close obstacles become high activation
        x = (1.0 - distances).clamp(0, 1).unsqueeze(0).repeat(time_steps, 1, 1)
        _, detections = self.detector(x)
        action = self.controller.select_action(x)
        return action, bool(detections[0].item())


def generate_obstacle_readings(batch_size=1, seed=0):
    """Generate random distance readings with occasional front obstacles."""
    torch.manual_seed(seed)
    distances = torch.rand(batch_size, 3) * 0.5 + 0.3  # normal 0.3-0.8
    # Make front distance very small (close obstacle)
    distances[:, 0] = 0.1
    return distances
