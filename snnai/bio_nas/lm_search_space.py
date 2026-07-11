"""LM architecture search space for Bio-NAS Phase 7.0.

Extends the biological module search space with LM-specific layer types:
- feedforward: standard feed-forward SNN block
- recurrent: sequence-axis recurrent coupling (GRU/LIF)
- attention: spiking self-attention (SSA) block
- hippocampus_gate: hippocampal memory gating

Biological modules from the original search space can also be included
as regularization / auxiliary pathways.
"""
import copy
import random


LM_LAYER_TYPES = ["feedforward", "recurrent", "attention", "hippocampus_gate"]
BIO_MODULES = ["c_elegans", "honeybee", "crow", "octopus"]
ALL_MODULE_TYPES = LM_LAYER_TYPES + BIO_MODULES


class LmLayer:
    """One node in the LM architecture graph."""

    def __init__(self, name, layer_type, hidden_dim=None):
        if layer_type not in ALL_MODULE_TYPES:
            raise ValueError(f"Unknown layer type: {layer_type}")
        self.name = name
        self.layer_type = layer_type
        self.hidden_dim = hidden_dim


class LmArchitecture:
    """Directed acyclic graph of LM layers.

    Nodes are layers. The first node is ``input`` (embedding) and the last
    node is ``output`` (language modeling head). Edges describe the forward
    data flow. Each non-input/output node can have multiple predecessors;
    predecessor outputs are summed before being passed to the layer.
    """

    def __init__(self):
        self.nodes = ["input", "output"]
        self.edges = []
        self.layers = {}

    def add_layer(self, name, layer_type, hidden_dim=None):
        """Add a layer node."""
        if name in self.nodes:
            raise ValueError(f"Node {name} already exists")
        self.nodes.insert(-1, name)
        self.layers[name] = LmLayer(name, layer_type, hidden_dim=hidden_dim)

    def add_edge(self, src, dst):
        """Add a directed edge."""
        if src not in self.nodes or dst not in self.nodes:
            raise ValueError("Edge nodes must exist")
        if src == dst:
            raise ValueError("Self-loops are not allowed")
        self.edges.append((src, dst))

    def is_valid(self):
        """Check structural constraints.

        Constraints:
        1. The graph must be a DAG.
        2. ``output`` must be reachable from ``input``.
        3. All non-input/output nodes have known layer types.
        4. At least one LM-specific layer type is used.
        """
        adj = {n: [] for n in self.nodes}
        indeg = {n: 0 for n in self.nodes}
        for src, dst in self.edges:
            adj[src].append(dst)
            indeg[dst] += 1

        # Every non-input node must have at least one predecessor
        for n in self.nodes:
            if n != "input" and indeg[n] == 0:
                return False

        # DAG detection via topological sort
        queue = [n for n in self.nodes if indeg[n] == 0]
        order = []
        local_indeg = dict(indeg)
        while queue:
            node = queue.pop(0)
            order.append(node)
            for nxt in adj[node]:
                local_indeg[nxt] -= 1
                if local_indeg[nxt] == 0:
                    queue.append(nxt)
        if len(order) != len(self.nodes):
            return False

        if "output" not in order:
            return False

        for n in self.nodes:
            if n in ("input", "output"):
                continue
            if n not in self.layers:
                return False
            if self.layers[n].layer_type not in ALL_MODULE_TYPES:
                return False

        used_lm_types = {
            self.layers[n].layer_type
            for n in self.layers
            if self.layers[n].layer_type in LM_LAYER_TYPES
        }
        if len(used_lm_types) == 0:
            return False

        return True

    def topological_order(self):
        """Return topological order from input to output."""
        adj = {n: [] for n in self.nodes}
        indeg = {n: 0 for n in self.nodes}
        for src, dst in self.edges:
            adj[src].append(dst)
            indeg[dst] += 1
        queue = [n for n in self.nodes if indeg[n] == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for nxt in adj[node]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)
        if len(order) != len(self.nodes):
            raise ValueError("Graph contains a cycle")
        return order

    def predecessors(self, node):
        """Return list of predecessor nodes."""
        return [src for src, dst in self.edges if dst == node]

    def mutate(self, rng=None):
        """Return a mutated copy."""
        if rng is None:
            rng = random
        child = copy.deepcopy(self)

        # Remove a random non-essential edge
        if rng.random() < 0.25 and len(child.edges) > 1:
            removable = [e for e in child.edges if e[1] != "output" or e[0] != "input"]
            if removable:
                child.edges.remove(rng.choice(removable))

        # Add a random valid edge
        if rng.random() < 0.35:
            candidates = [
                (src, dst)
                for src in child.nodes
                for dst in child.nodes
                if src != dst
                and src != "output"
                and dst != "input"
                and (src, dst) not in child.edges
            ]
            if candidates:
                child.edges.append(rng.choice(candidates))

        # Mutate a layer type
        if rng.random() < 0.3:
            layer_nodes = [n for n in child.nodes if n not in ("input", "output")]
            if layer_nodes:
                target = rng.choice(layer_nodes)
                current = child.layers[target].layer_type
                new_type = rng.choice([m for m in ALL_MODULE_TYPES if m != current])
                child.layers[target].layer_type = new_type

        # Add a new layer (skip-connection-like expansion)
        if rng.random() < 0.15:
            new_name = f"L{len(child.nodes)}"
            if new_name not in child.nodes:
                new_type = rng.choice(LM_LAYER_TYPES)
                child.add_layer(new_name, new_type)
                # Connect from a random non-output node to new layer
                src = rng.choice([n for n in child.nodes if n != "output" and n != new_name])
                child.add_edge(src, new_name)
                child.add_edge(new_name, "output")

        return child

    def count_lm_layer_types(self):
        """Return set of LM layer types used."""
        return {
            self.layers[n].layer_type
            for n in self.layers
            if self.layers[n].layer_type in LM_LAYER_TYPES
        }


def lm_serial_architecture(hidden_dim=None):
    """Input -> feedforward -> recurrent -> attention -> output."""
    arch = LmArchitecture()
    arch.add_layer("l1", "feedforward", hidden_dim=hidden_dim)
    arch.add_layer("l2", "recurrent", hidden_dim=hidden_dim)
    arch.add_layer("l3", "attention", hidden_dim=hidden_dim)
    arch.add_edge("input", "l1")
    arch.add_edge("l1", "l2")
    arch.add_edge("l2", "l3")
    arch.add_edge("l3", "output")
    return arch


def lm_diverse_architecture(hidden_dim=None):
    """Input -> feedforward -> recurrent -> attention -> hippocampus_gate -> output."""
    arch = LmArchitecture()
    arch.add_layer("l1", "feedforward", hidden_dim=hidden_dim)
    arch.add_layer("l2", "recurrent", hidden_dim=hidden_dim)
    arch.add_layer("l3", "attention", hidden_dim=hidden_dim)
    arch.add_layer("l4", "hippocampus_gate", hidden_dim=hidden_dim)
    arch.add_edge("input", "l1")
    arch.add_edge("l1", "l2")
    arch.add_edge("l2", "l3")
    arch.add_edge("l3", "l4")
    arch.add_edge("l4", "output")
    return arch
