import torch
from snnai.benchmarks.edge_deployment import edge_profile, export_to_torchscript
from snnai.modules.edge import IoTEventDetector


def test_edge_profile():
    model = IoTEventDetector(input_size=4, hidden_size=8, threshold=1)
    sample = torch.randn(5, 1, 4)
    report = edge_profile(model, sample, device="cpu")
    assert "parameters" in report
    assert report["size_kb"] > 0


def test_export_torchscript():
    model = IoTEventDetector(input_size=4, hidden_size=8, threshold=1)
    sample = torch.randn(5, 1, 4)
    path = export_to_torchscript(model, sample, "/tmp/snnai_edge_v54.pt")
    assert path.endswith(".pt")
