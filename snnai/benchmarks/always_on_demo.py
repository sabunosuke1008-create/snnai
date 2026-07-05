"""Always-on event-driven monitoring demo using SNNAI edge modules."""
import torch

from snnai.modules.edge import IoTEventDetector


class AlwaysOnMonitor:
    """Simulate 24/7 monitoring with event-driven activation.

    The monitor keeps an SNN detector in a low-power eval mode. It only
    reports events when significant activity is detected.
    """

    def __init__(self, input_size=4, hidden_size=16, threshold=5):
        self.detector = IoTEventDetector(input_size=input_size, hidden_size=hidden_size, threshold=threshold)
        self.event_log = []

    def monitor(self, stream, window_size=20):
        """Monitor a sensor stream and log events.

        Parameters
        ----------
        stream : torch.Tensor
            Sensor stream of shape (total_time, batch, input_size).
        window_size : int
            Number of time steps per detection window.

        Returns
        -------
        list
            Indices of windows where an event was detected.
        """
        self.event_log = []
        self.detector.eval()
        total_time = stream.size(0)
        with torch.no_grad():
            for start in range(0, total_time - window_size + 1, window_size):
                window = stream[start:start + window_size]
                _, detections = self.detector(window)
                if detections.any():
                    self.event_log.append(start)
        return self.event_log

    def energy_estimate(self, stream, window_size=20, joules_per_spike=1e-9):
        """Estimate energy for the monitoring period."""
        from snnai.benchmarks.energy_estimation import estimate_energy
        self.detector.eval()
        total_spikes = 0.0
        with torch.no_grad():
            for start in range(0, stream.size(0) - window_size + 1, window_size):
                window = stream[start:start + window_size]
                spikes, _ = self.detector(window)
                total_spikes += float(spikes.sum().item())
        return estimate_energy(total_spikes, joules_per_spike)
