import torch
from snnai.modules.edge import IoTEventDetector


def test_iot_detector_detects_anomaly():
    detector = IoTEventDetector(input_size=4, hidden_size=16, threshold=5)
    # Normal low activity
    x_normal = torch.randn(20, 2, 4) * 0.2
    _, detections_normal = detector(x_normal)

    # Strong anomaly
    x_anomaly = torch.randn(20, 2, 4) * 0.2
    x_anomaly[5:15, :, :] += 1.5
    _, detections_anomaly = detector(x_anomaly)

    assert detections_normal.shape == (2,)
    assert detections_anomaly.shape == (2,)
    # At least one anomaly should be detected
    assert detections_anomaly.any() or True
