import torch
from snnai.modules.personalization import UserAdapter


def test_user_adapter():
    adapter = UserAdapter(feature_size=4, num_users=2, rank=2)
    features = torch.randn(5, 1, 4)
    out = adapter(features, user_id=0)
    assert out.shape == features.shape
