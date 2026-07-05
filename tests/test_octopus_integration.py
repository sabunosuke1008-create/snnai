import torch
import torch.nn.functional as F
from snnai.modules.octopus import OctopusNetwork, SubModule


def test_submodule_independence():
    m = SubModule(input_size=2, hidden_size=4)
    x = torch.zeros(20, 1, 2)
    x[:, 0, 0] = 2.0
    spikes = m(x)
    assert spikes.shape == (20, 1, 4)
    assert spikes.sum().item() > 0


def test_distributed_parallel_output_shape():
    net = OctopusNetwork(n_modules=2, input_size=2, hidden_size=4, n_classes=3, max_workers=2)
    x1 = torch.zeros(20, 2, 2)
    x2 = torch.zeros(20, 2, 2)
    logits = net([x1, x2])
    assert logits.shape == (2, 3)


def test_multimodal_classification_trend():
    torch.manual_seed(0)
    net = OctopusNetwork(n_modules=2, input_size=2, hidden_size=4, n_classes=3, max_workers=1)
    # Hand-craft classifier weights for interpretability.
    with torch.no_grad():
        net.classifier.fc.weight.zero_()
        # Class 2 (warning): both channels strong -> high total activity
        net.classifier.fc.weight[2, :] = 0.5
        # Class 1 (attention): only first channel strong
        net.classifier.fc.weight[1, :4] = 0.5
        net.classifier.fc.weight[1, 4:] = -0.5
        # Class 0 (safe): low total activity
        net.classifier.fc.weight[0, :] = -0.5

    def make_input(ch0_strong, ch1_strong):
        x = torch.zeros(20, 1, 2)
        if ch0_strong:
            x[:, 0, 0] = 2.0
        if ch1_strong:
            x[:, 0, 1] = 2.0
        return x

    logits_safe = net([make_input(False, False), make_input(False, False)])
    logits_warn = net([make_input(True, True), make_input(True, True)])
    logits_att = net([make_input(True, False), make_input(False, False)])

    cls_safe = int(torch.argmax(logits_safe, dim=1)[0])
    cls_warn = int(torch.argmax(logits_warn, dim=1)[0])
    cls_att = int(torch.argmax(logits_att, dim=1)[0])

    assert cls_safe == 0
    assert cls_warn == 2
    assert cls_att == 1
