"""Search space and biological constraints for Bio-NAS."""
import copy
import random


MODULES = ["c_elegans", "honeybee", "crow", "octopus"]


class Architecture:
    """A directed acyclic graph of biological modules.

    Nodes are modules. Edges are feed-forward connections. The first
    node is always ``input`` and the last node is always ``output``.
    """

    def __init__(self):
        self.nodes = ["input", "output"]
        self.edges = []
        self.module_types = {}

    def add_module(self, name, module_type):
        """Add a module node and remember its biological type."""
        if name in self.nodes:
            raise ValueError(f"Node {name} already exists")
        self.nodes.insert(-1, name)
        self.module_types[name] = module_type

    def add_edge(self, src, dst):
        """Add a directed edge from src to dst."""
        if src not in self.nodes or dst not in self.nodes:
            raise ValueError("Edge nodes must exist")
        if src == dst:
            raise ValueError("Self-loops are not allowed")
        self.edges.append((src, dst))

    def is_valid(self):
        """Check biological and structural constraints.

        Constraints:
        1. ``c_elegans`` must be directly reachable from ``input`` within one step.
        2. The graph must be a DAG.
        3. ``output`` must be reachable from ``input``.
        4. All module nodes have known types.
        """
        # Constraint 1: c_elegans adjacent to input
        ce_nodes = [n for n, t in self.module_types.items() if t == "c_elegans"]
        for ce in ce_nodes:
            if not any(src == "input" and dst == ce for src, dst in self.edges):
                return False

        # Constraint 2: DAG detection
        adj = {n: [] for n in self.nodes}
        for src, dst in self.edges:
            adj[src].append(dst)
        visited = set()
        rec_stack = set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            for nxt in adj[node]:
                if nxt not in visited:
                    if dfs(nxt):
                        return True
                elif nxt in rec_stack:
                    return True
            rec_stack.remove(node)
            return False

        if dfs("input"):
            return False
        # Ensure output reachable
        if "output" not in visited:
            return False

        # Constraint 4: known module types
        for n in self.nodes:
            if n in ("input", "output"):
                continue
            if n not in self.module_types:
                return False
            if self.module_types[n] not in MODULES:
                return False

        return True

    def topological_order(self):
        """Return a topological order from input to output."""
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

    def mutate(self):
        """Return a mutated copy of this architecture."""
        child = copy.deepcopy(self)
        if random.random() < 0.3 and len(child.edges) > 1:
            # Remove a random non-essential edge
            removable = [e for e in child.edges if e[1] != "output" or e[0] != "input"]
            if removable:
                child.edges.remove(random.choice(removable))
        if random.random() < 0.4:
            # Add a random valid edge
            candidates = [
                (src, dst)
                for src in child.nodes
                for dst in child.nodes
                if src != dst and src != "output" and dst != "input"
                and (src, dst) not in child.edges
            ]
            if candidates:
                child.edges.append(random.choice(candidates))
        if random.random() < 0.3:
            # Mutate a module type
            module_nodes = [n for n in child.nodes if n not in ("input", "output")]
            if module_nodes:
                target = random.choice(module_nodes)
                current = child.module_types[target]
                new_type = random.choice([m for m in MODULES if m != current])
                child.module_types[target] = new_type
        return child


def serial_architecture():
    """Input -> c_elegans -> honeybee -> crow -> octopus -> output."""
    arch = Architecture()
    for name, mod in [("m1", "c_elegans"), ("m2", "honeybee"), ("m3", "crow"), ("m4", "octopus")]:
        arch.add_module(name, mod)
    arch.add_edge("input", "m1")
    arch.add_edge("m1", "m2")
    arch.add_edge("m2", "m3")
    arch.add_edge("m3", "m4")
    arch.add_edge("m4", "output")
    return arch


def hub_architecture():
    """All modules connect to a central hub node."""
    arch = Architecture()
    arch.add_module("hub", "crow")
    for name, mod in [("m1", "c_elegans"), ("m2", "honeybee"), ("m3", "octopus")]:
        arch.add_module(name, mod)
        arch.add_edge("input", name)
        arch.add_edge(name, "hub")
    arch.add_edge("input", "m1")
    arch.add_edge("hub", "output")
    return arch


def parallel_architecture():
    """c_elegans feeds multiple parallel modules whose outputs merge."""
    arch = Architecture()
    arch.add_module("reflex", "c_elegans")
    arch.add_module("path", "honeybee")
    arch.add_module("memory", "crow")
    arch.add_module("dist", "octopus")
    arch.add_edge("input", "reflex")
    arch.add_edge("reflex", "path")
    arch.add_edge("reflex", "memory")
    arch.add_edge("reflex", "dist")
    arch.add_edge("path", "output")
    arch.add_edge("memory", "output")
    arch.add_edge("dist", "output")
    return arch
