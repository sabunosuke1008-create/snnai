"""Integration pipeline that connects biological modules via bridges."""
import torch

from snnai.modules.c_elegans import ReflexModule
from snnai.modules.honeybee import SNNAgent

from .encoding_bridge import EncodingBridge


class SNNAIPipeline:
    """Minimal heterogeneous SNN pipeline.

    Connects a reflex module to a honeybee navigation agent. The reflex
    module monitors sensory input and can override the agent's selected
    action when an immediate threat is detected. The modules communicate
    through an encoding bridge that converts the reflex output into a
    temporal context signal for the agent.

    This is a proof-of-concept integration for Phase6.
    """

    def __init__(self, size=3, time_steps=10):
        self.time_steps = time_steps
        self.reflex = ReflexModule(beta=0.9, threshold=1.0)
        self.agent = SNNAgent(
            size=size,
            n_cells=25,
            hidden_size=20,
            n_steps=time_steps,
        )
        self.bridge = EncodingBridge(in_dim=2, out_dim=2, time_steps=time_steps)

    def forward(self, reflex_input, agent_pos):
        """Run the integrated pipeline for one decision step.

        Parameters
        ----------
        reflex_input : torch.Tensor
            Reflex sensory input of shape (time, batch, 2).
        agent_pos : torch.Tensor
            Current agent position of shape (batch, 2).

        Returns
        -------
        int
            Selected action (possibly overridden by reflex).
        """
        batch_size = reflex_input.shape[1]

        # Reflex output: which side is active
        reflex_spikes, _ = self.reflex(reflex_input)
        reflex_rates = self.bridge.spikes_to_rates(reflex_spikes)

        # Agent proposes an action from position
        action, *_ = self.agent.select_action(agent_pos)

        # Reflex override: strong left stimulus -> move right, strong right -> move left
        left_strength = reflex_rates[:, 0].mean().item()
        right_strength = reflex_rates[:, 1].mean().item()
        if right_strength > 0.1 and right_strength > left_strength:
            action = 3  # move right
        elif left_strength > 0.1 and left_strength > right_strength:
            action = 2  # move left

        return action
