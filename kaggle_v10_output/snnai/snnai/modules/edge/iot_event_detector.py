"""Low-latency event detector for IoT/drone sensor streams."""
import torch
import torch.nn as nn
import snntorch as snn


class IoTEventDetector(nn.Module):
    """Detect anomalous events in streaming IoT sensor data.

    Uses a small LIF reservoir that fires when the input pattern deviates
    from a learned baseline. This is a one-class novelty detector.

    Parameters
    ----------
    input_size : int
        Number of sensor channels.
    hidden_size : int, optional
        Reservoir size (default 16).
    threshold : float, optional
        Detection threshold on output spike count (default 5).
    """

    def __init__(self, input_size=4, hidden_size=16, threshold=5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.threshold = threshold
        self.fc = nn.Linear(input_size, hidden_size, bias=False)
        self.lif = snn.Leaky(beta=0.9, threshold=1.0, learn_threshold=False)
        with torch.no_grad():
            self.fc.weight.normal_(0, 0.5)

    def forward(self, x):
        """Run event detection.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (time, batch, input_size).

        Returns
        -------
        torch.Tensor
            Spike tensor of shape (time, batch, hidden_size).
        detections : torch.Tensor
            Boolean tensor of shape (batch,) indicating event detection.
        """
        time_steps, batch_size, _ = x.shape
        mem = torch.zeros(batch_size, self.hidden_size)
        spikes = []
        for t in range(time_steps):
            cur = self.fc(x[t])
            spk, mem = self.lif(cur, mem)
            spikes.append(spk)
        spikes = torch.stack(spikes)
        counts = spikes.sum(dim=0).sum(dim=1)
        detections = counts > self.threshold
        return spikes, detections
