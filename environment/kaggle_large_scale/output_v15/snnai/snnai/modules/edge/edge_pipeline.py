"""Lightweight edge pipeline integrating filter, detector, and controller."""
import torch
import torch.nn as nn

from .sensor_filter import SensorFilter
from .iot_event_detector import IoTEventDetector
from .robot_controller import RobotController


class EdgePipeline(nn.Module):
    """End-to-end edge processing pipeline.

    Sensor data is filtered, anomalies are detected, and actions are selected.

    Parameters
    ----------
    input_size : int
        Number of sensor channels.
    filter_threshold : float, optional
        Threshold for sensor filter (default 1.0).
    event_threshold : int, optional
        Threshold for event detector (default 5).
    """

    def __init__(self, input_size=4, filter_threshold=1.0, event_threshold=5):
        super().__init__()
        self.input_size = input_size
        self.filter = SensorFilter(input_size=input_size, beta=0.9, threshold=filter_threshold)
        self.detector = IoTEventDetector(input_size=input_size, hidden_size=16, threshold=event_threshold)
        self.controller = RobotController(beta=0.9, threshold=1.0)

    def forward(self, x):
        """Run edge pipeline.

        Parameters
        ----------
        x : torch.Tensor
            Raw sensor input of shape (time, batch, input_size).

        Returns
        -------
        filtered : torch.Tensor
            Filtered spikes (time, batch, input_size).
        events : torch.Tensor
            Event detection flags (batch,).
        action : int
            Selected action.
        """
        if x.shape[2] != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {x.shape[2]}")
        filtered = self.filter(x)
        _, events = self.detector(filtered)
        # Pad/truncate to 3 channels for controller if needed
        if self.input_size >= 3:
            controller_input = filtered[:, :, :3]
        else:
            pad = torch.zeros(filtered.shape[0], filtered.shape[1], 3 - self.input_size)
            controller_input = torch.cat([filtered, pad], dim=2)
        action = self.controller.select_action(controller_input)
        return filtered, events, action
