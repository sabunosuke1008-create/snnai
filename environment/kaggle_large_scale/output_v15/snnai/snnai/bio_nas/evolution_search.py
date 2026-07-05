"""Evolutionary architecture search for Bio-NAS."""
import random

from .evaluator import evaluate_architecture
from .search_space import serial_architecture


class EvolutionSearch:
    """Simple evolutionary search over SNN module architectures."""

    def __init__(
        self,
        population_size=8,
        n_generations=5,
        top_k=4,
        mutation_rate=1.0,
        seed=0,
    ):
        self.population_size = population_size
        self.n_generations = n_generations
        self.top_k = top_k
        self.mutation_rate = mutation_rate
        self.seed = seed
        self.rng = random.Random(seed)

    def _initialize(self):
        """Start from serial architecture and create mutants."""
        base = serial_architecture()
        pop = [base]
        while len(pop) < self.population_size:
            mutant = base.mutate()
            if mutant.is_valid():
                pop.append(mutant)
        return pop

    def search(self):
        """Run evolution and return the best architecture + score.

        Returns
        -------
        best_arch, best_score, history
        """
        population = self._initialize()
        history = []
        for gen in range(self.n_generations):
            scores = []
            for arch in population:
                score = evaluate_architecture(arch, seed=self.seed + gen)
                scores.append(score)
            ranked = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
            top = [arch for _, arch in ranked[: self.top_k]]
            best_score, best_arch = ranked[0]
            history.append((gen, best_score))

            # Elitism + mutate
            new_pop = top[:]
            while len(new_pop) < self.population_size:
                parent = self.rng.choice(top)
                child = parent.mutate()
                if self.rng.random() < self.mutation_rate and child.is_valid():
                    new_pop.append(child)
                else:
                    new_pop.append(parent)
            population = new_pop
        return best_arch, best_score, history
