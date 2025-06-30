
import random
import numpy as np
from .core import calculate_makespan, load_instance

class GeneticAlgorithmFSSP:
    """
    Resolve o Flow Shop Scheduling Problem (FSSP) usando um Algoritmo Genético.

    Atributos:
        processing_times (np.ndarray): Matriz com os tempos de processamento.
        population_size (int): Número de indivíduos na população.
        n_generations (int): Número de gerações que o AG executará.
        mutation_rate (float): Probabilidade de um indivíduo sofrer mutação.
        elitism_rate (float): Porcentagem dos melhores indivíduos a serem mantidos.
        crossover_method (str): Método de cruzamento ('ox' ou 'cx').
        parent_selection_method (str): Método de seleção de pais ('tournament' ou 'rank').
    """
    def __init__(self, processing_times, population_size, n_generations, mutation_rate,
                 elitism_rate, crossover_method, parent_selection_method):
        self.processing_times = processing_times
        self.n_jobs = processing_times.shape[0]
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.crossover_method = crossover_method
        self.parent_selection_method = parent_selection_method

        self.population = self._initialize_population()
        self.best_sequence_so_far = None
        self.best_makespan_so_far = float('inf')

    def _initialize_population(self):
        """Cria a população inicial com permutações aleatórias."""
        population = []
        base_sequence = list(range(self.n_jobs))
        for _ in range(self.population_size):
            shuffled_sequence = random.sample(base_sequence, len(base_sequence))
            population.append(shuffled_sequence)
        return population

    def _evaluate_fitness(self, population):
        """Avalia o fitness (makespan) de cada indivíduo na população."""
        return [calculate_makespan(self.processing_times, seq) for seq in population]

    def _selection(self, population, fitnesses):
        """Seleciona os pais para a próxima geração."""
        if self.parent_selection_method == 'tournament':
            return self._tournament_selection(population, fitnesses)
        elif self.parent_selection_method == 'rank':
            return self._rank_selection(population, fitnesses)
        else:
            raise ValueError(f"Método de seleção desconhecido: {self.parent_selection_method}")

    def _tournament_selection(self, population, fitnesses, k=3):
        """Seleção por torneio."""
        selected_parents = []
        for _ in range(len(population)):
            participants_indices = random.sample(range(len(population)), k)
            best_participant_index = min(participants_indices, key=lambda i: fitnesses[i])
            selected_parents.append(population[best_participant_index])
        return selected_parents

    def _rank_selection(self, population, fitnesses):
        """Seleção por ranking."""
        sorted_population = [p for _, p in sorted(zip(fitnesses, population), key=lambda pair: pair[0])]
        ranks = np.arange(1, len(population) + 1)[::-1]  # Inverte para que o melhor tenha o maior rank
        rank_probabilities = ranks / np.sum(ranks)

        selected_indices = np.random.choice(len(population), size=len(population), p=rank_probabilities)
        return [sorted_population[i] for i in selected_indices]

    def _crossover(self, parent1, parent2):
        """Realiza o cruzamento entre dois pais para gerar um filho."""
        if self.crossover_method == 'ox':
            return self._order_crossover(parent1, parent2)
        elif self.crossover_method == 'cx':
            return self._cycle_crossover(parent1, parent2)
        else:
            raise ValueError(f"Método de cruzamento desconhecido: {self.crossover_method}")

    def _order_crossover(self, parent1, parent2):
        """Order Crossover (OX)."""
        size = len(parent1)
        child = [None] * size
        start, end = sorted(random.sample(range(size), 2))

        child[start:end+1] = parent1[start:end+1]

        p2_genes = [item for item in parent2 if item not in child]

        child_idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = p2_genes.pop(0)
        return child

    def _cycle_crossover(self, parent1, parent2):
        """Cycle Crossover (CX)."""
        size = len(parent1)
        child = [None] * size
        p1_map = {val: i for i, val in enumerate(parent1)}

        cycles = []
        visited = [False] * size
        for i in range(size):
            if not visited[i]:
                cycle = []
                curr_idx = i
                while not visited[curr_idx]:
                    cycle.append(curr_idx)
                    visited[curr_idx] = True
                    curr_idx = p1_map[parent2[curr_idx]]
                cycles.append(cycle)

        for i, cycle in enumerate(cycles):
            if i % 2 == 0:  # Alterna os ciclos entre os pais
                for idx in cycle:
                    child[idx] = parent1[idx]
            else:
                for idx in cycle:
                    child[idx] = parent2[idx]
        return child

    def _mutation(self, individual):
        """Mutação por troca (Swap Mutation)."""
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def _elitism(self, population, fitnesses):
        """Preserva os melhores indivíduos."""
        elite_size = int(self.population_size * self.elitism_rate)
        if elite_size == 0:
            return []
        sorted_population = [p for _, p in sorted(zip(fitnesses, population), key=lambda pair: pair[0])]
        return sorted_population[:elite_size]

    def run(self, track_convergence=False, verbose=True):
        """Executa o loop principal do Algoritmo Genético."""
        convergence_history = []

        for generation in range(self.n_generations):
            fitnesses = self._evaluate_fitness(self.population)
            
            current_best_makespan = min(fitnesses)
            if current_best_makespan < self.best_makespan_so_far:
                self.best_makespan_so_far = current_best_makespan
                best_idx = fitnesses.index(current_best_makespan)
                self.best_sequence_so_far = self.population[best_idx][:]

            if track_convergence:
                convergence_history.append(self.best_makespan_so_far)

            elites = self._elitism(self.population, fitnesses)
            parents = self._selection(self.population, fitnesses)
            next_population = elites[:]
            
            while len(next_population) < self.population_size:
                p1, p2 = random.sample(parents, 2)
                child = self._crossover(p1, p2)
                child = self._mutation(child)
                next_population.append(child)
            
            self.population = next_population
            
            if verbose and (generation + 1) % 10 == 0:
                print(f"Geração {generation + 1}/{self.n_generations} | Melhor Makespan: {self.best_makespan_so_far}")

        if track_convergence:
            return self.best_sequence_so_far, self.best_makespan_so_far, convergence_history
        else:
            return self.best_sequence_so_far, self.best_makespan_so_far

# Bloco de teste
if __name__ == '__main__':
    # Carrega a instância de exemplo
    instance_filepath = '../instances/example.txt'
    processing_times = load_instance(instance_filepath)

    print("Instância FSSP carregada.")
    print(f"Número de tarefas: {processing_times.shape[0]}")
    print(f"Número de máquinas: {processing_times.shape[1]}")
    print("-" * 30)

    # Parâmetros do Algoritmo Genético
    params = {
        'population_size': 50,
        'n_generations': 100,
        'mutation_rate': 0.1,
        'elitism_rate': 0.1,
        'crossover_method': 'ox',  # 'ox' ou 'cx'
        'parent_selection_method': 'tournament'  # 'tournament' ou 'rank'
    }

    print("Iniciando o Algoritmo Genético...")
    print(f"Parâmetros: {params}")
    print("-" * 30)

    # Cria e executa o AG
    ga = GeneticAlgorithmFSSP(processing_times, **params)
    best_sequence, best_makespan = ga.run()

    print("-" * 30)
    print("Execução do AG concluída.")
    print(f"Melhor sequência encontrada: {best_sequence}")
    print(f"Melhor Makespan (C_max): {best_makespan}")
