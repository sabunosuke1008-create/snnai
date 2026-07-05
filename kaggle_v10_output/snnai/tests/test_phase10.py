import torch

from snnai.integration.trainable_pipeline import (
    TrainablePipeline,
    generate_integration_data,
    train_pipeline,
)


def test_trainable_pipeline_forward():
    model = TrainablePipeline(time_steps=10)
    x = torch.randn(10, 2, 2)
    logits = model(x)
    assert logits.shape == (2, 4)


def test_end_to_end_training_improves():
    model, acc = train_pipeline(n_samples=400, epochs=10, lr=0.02, seed=1)
    print(f"Phase10 test accuracy: {acc:.3f}")
    assert acc > 0.25, f"Accuracy {acc:.3f} should exceed random chance"
