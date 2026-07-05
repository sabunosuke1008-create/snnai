"""Always-on AI v1.0 release demo."""
import torch

from snnai.benchmarks.always_on_demo import AlwaysOnMonitor
from snnai.modules.llm_bridge.pipeline import LLMCollaborationPipeline


class AlwaysOnAIV1:
    """Combined always-on monitoring and lightweight LLM pipeline."""

    def __init__(self, vocab, sensor_input_size=4):
        self.monitor = AlwaysOnMonitor(input_size=sensor_input_size, hidden_size=16, threshold=3)
        self.pipeline = LLMCollaborationPipeline(vocab, feature_size=4, hidden_size=8)

    def run_cycle(self, sensor_stream, user_query=None):
        """Run one monitoring + optional response cycle."""
        events = self.monitor.monitor(sensor_stream, window_size=20)
        alert = len(events) > 0
        response = None
        if alert and user_query:
            response, _ = self.pipeline.run(user_query, max_chars=15)
        return {"alert": alert, "events": events, "response": response}
