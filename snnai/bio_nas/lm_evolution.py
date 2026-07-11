"""Multi-objective evolutionary search for LM architectures (Bio-NAS Phase 7.0).

Uses a scalarized weighted objective by default. The search returns the best
architecture along with a Pareto front approximation.
"""
import random

from .lm_evaluator import evaluate_lm_architecture
from .lm_search_space import lm_serial_architecture


class LmEvolutionSearch:
    """Evolutionary search over LM layer architectures."""

    def __init__(
        self,
        population_size=6,
        n_generations=3,
        top_k=3,
        mutation_rate=1.0,
        seed=0,
        weights=None,
        eval_kwargs=None,
    ):
        """Initialize search.

        Parameters
        ----------
        population_size : int
        n_generations : int
        top_k : int
        mutation_rate : float
        seed : int
        weights : dict or None
            Weights for scalarized objective. Keys: val_ppl, latency_sec,
            energy_proxy_joules, bleu1, biological_penalty.
        eval_kwargs : dict or None
            Passed to evaluate_lm_architecture.
        """
        self.population_size = population_size
        self.n_generations = n_generations
        self.top_k = top_k
        self.mutation_rate = mutation_rate
        self.seed = seed
        self.rng = random.Random(seed)
        self.weights = weights or {
            "val_ppl": -0.4,
            "latency_sec": -2.0,
            "energy_proxy_joules": -200.0,
            "bleu1": 2.0,
            "biological_penalty": -1.0,
        }
        self.eval_kwargs = eval_kwargs or {}
        self._eval_cache = {}

    def _initialize(self):
        """Start from serial architecture and create mutants."""
        base = lm_serial_architecture()
        pop = [base]
        while len(pop) < self.population_size:
            mutant = base.mutate(rng=self.rng)
            if mutant.is_valid():
                pop.append(mutant)
        return pop

    def _scalar_score(self, metrics):
        """Convert metrics dict to scalar score (higher is better)."""
        score = 0.0
        for key, weight in self.weights.items():
            score += weight * metrics[key]
        # Bonus for using more diverse LM layer types
        score += 0.05 * metrics.get("layer_type_count", 0)
        return score

    def _evaluate(self, arch, gen):
        """Evaluate an architecture with caching."""
        key = (tuple(arch.nodes), tuple(arch.edges), tuple(sorted(arch.layers.items())))
        if key in self._eval_cache:
            return self._eval_cache[key]
        metrics = evaluate_lm_architecture(
            arch,
            seed=self.seed + gen,
            **self.eval_kwargs,
        )
        self._eval_cache[key] = metrics
        return metrics

    def search(self):
        """Run evolution and return best arch, metrics, history, pareto_front.

        Returns
        -------
        best_arch, best_metrics, history, pareto_front
        """
        population = self._initialize()
        history = []
        pareto_front = []

        for gen in range(self.n_generations):
            scored = []
            metrics_list = []
            for arch in population:
                metrics = self._evaluate(arch, gen)
                score = self._scalar_score(metrics)
                scored.append((score, arch, metrics))
                metrics_list.append(metrics)

            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[: self.top_k]
            best_score, best_arch, best_metrics = top[0]
            history.append((gen, best_score, best_metrics))

            # Update Pareto front approximation
            for _, arch, metrics in scored:
                self._update_pareto(pareto_front, arch, metrics)

            # Elitism + mutate
            new_pop = [s[1] for s in top]
            while len(new_pop) < self.population_size:
                parent = self.rng.choice([s[1] for s in top])
                child = parent.mutate(rng=self.rng)
                if self.rng.random() < self.mutation_rate and child.is_valid():
                    new_pop.append(child)
                else:
                    new_pop.append(parent)
            population = new_pop

        return best_arch, best_metrics, history, pareto_front

    def _update_pareto(self, front, arch, metrics):
        """Add (arch, metrics) to Pareto front if non-dominated.

        Objectives to minimize: val_ppl, latency_sec, energy_proxy_joules,
                                biological_penalty.
        Objectives to maximize: bleu1.
        """
        obj = {
            "val_ppl": metrics["val_ppl"],
            "latency_sec": metrics["latency_sec"],
            "energy_proxy_joules": metrics["energy_proxy_joules"],
            "biological_penalty": metrics["biological_penalty"],
            "bleu1": -metrics["bleu1"],  # negate for minimization
        }
        dominated = []
        for i, (_, existing_metrics) in enumerate(front):
            existing = {
                "val_ppl": existing_metrics["val_ppl"],
                "latency_sec": existing_metrics["latency_sec"],
                "energy_proxy_joules": existing_metrics["energy_proxy_joules"],
                "biological_penalty": existing_metrics["biological_penalty"],
                "bleu1": -existing_metrics["bleu1"],
            }
            # Check if existing dominates new
            if all(existing[k] <= obj[k] for k in obj) and any(
                existing[k] < obj[k] for k in obj
            ):
                return
            # Check if new dominates existing
            if all(obj[k] <= existing[k] for k in obj) and any(
                obj[k] < existing[k] for k in obj
            ):
                dominated.append(i)

        # Remove dominated entries
        for idx in reversed(dominated):
            front.pop(idx)
        front.append((arch, metrics))


def search_lm_architecture(**kwargs):
    """Convenience wrapper around LmEvolutionSearch."""
    search = LmEvolutionSearch(**kwargs)
    return search.search()
