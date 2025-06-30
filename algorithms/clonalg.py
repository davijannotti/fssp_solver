
import random
import numpy as np
from .core import calculate_makespan, load_instance

class ClonalgForFSSP:
    """
    Resolve o Flow Shop Scheduling Problem (FSSP) usando o algoritmo CLONALG.

    Atributos:
        processing_times (np.ndarray): Matriz com os tempos de processamento.
        population_size (int): Número total de anticorpos na população.
        n_generations (int): Número de gerações que o algoritmo executará.
        selection_size (int): (n) Número de melhores indivíduos a serem selecionados.
        clone_factor (float): (β) Fator para determinar o número de clones.
        num_replace (int): (d) Número de piores indivíduos a serem substituídos.
    """
    def __init__(self, processing_times, population_size, n_generations, 
                 selection_size, clone_factor, num_replace):
        self.processing_times = processing_times
        self.n_jobs = processing_times.shape[0]
        self.population_size = population_size
        self.n_generations = n_generations
        self.selection_size = selection_size
        self.clone_factor = clone_factor
        self.num_replace = num_replace

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

    def _evaluate_affinity(self, population):
        """Calcula o makespan e a afinidade para uma população."""
        makespans = np.array([calculate_makespan(self.processing_times, seq) for seq in population])
        # Afinidade é o inverso do makespan (maior é melhor)
        affinities = 1 / makespans
        return makespans, affinities

    def _select_and_clone(self, population, affinities):
        """Seleciona os n melhores indivíduos e os clona proporcionalmente à afinidade."""
        # Ordena a população pela afinidade (maior primeiro)
        sorted_indices = np.argsort(affinities)[::-1]
        
        selected_population = [population[i] for i in sorted_indices[:self.selection_size]]
        selected_affinities = affinities[sorted_indices[:self.selection_size]]

        clones = []
        # Normaliza as afinidades dos selecionados para o cálculo de clones
        affinity_sum = np.sum(selected_affinities)
        if affinity_sum == 0:
            normalized_affinities = np.ones(len(selected_affinities)) / len(selected_affinities)
        else:
            normalized_affinities = selected_affinities / affinity_sum

        for i in range(self.selection_size):
            num_clones = int(round(self.clone_factor * self.population_size * normalized_affinities[i]))
            for _ in range(num_clones):
                clones.append({
                    'sequence': selected_population[i][:],
                    'original_affinity': selected_affinities[i]
                })
        return clones

    def _hypermutation(self, clones):
        """Aplica mutação aos clones com taxa inversamente proporcional à afinidade."""
        mutated_clones = []
        
        # Encontra a afinidade máxima para normalização
        max_affinity = max(c['original_affinity'] for c in clones) if clones else 0

        for clone in clones:
            # A taxa de mutação (alpha) é inversamente proporcional à afinidade
            # Quanto maior a afinidade, menor a taxa de mutação
            normalized_affinity = clone['original_affinity'] / max_affinity if max_affinity > 0 else 0
            mutation_rate = np.exp(-normalized_affinity) # Exponencial decrescente
            
            mutated_sequence = clone['sequence']
            
            # O número de trocas é proporcional à taxa de mutação
            num_swaps = int(np.ceil(mutation_rate * self.n_jobs / 2))

            for _ in range(num_swaps):
                if len(mutated_sequence) > 1:
                    idx1, idx2 = random.sample(range(self.n_jobs), 2)
                    mutated_sequence[idx1], mutated_sequence[idx2] = mutated_sequence[idx2], mutated_sequence[idx1]
            
            mutated_clones.append(mutated_sequence)
            
        return mutated_clones

    def _population_replacement(self, population, affinities):
        """Substitui os `d` piores indivíduos por novos aleatórios."""
        if self.num_replace == 0:
            return population

        sorted_indices = np.argsort(affinities) # Pior afinidade primeiro
        
        new_population = population[:]
        base_sequence = list(range(self.n_jobs))

        for i in range(self.num_replace):
            worst_idx = sorted_indices[i]
            new_population[worst_idx] = random.sample(base_sequence, len(base_sequence))
            
        return new_population

    def run(self, verbose=True):
        """Executa o loop principal do algoritmo CLONALG."""
        for generation in range(self.n_generations):
            # 1. Avaliar a afinidade da população atual
            makespans, affinities = self._evaluate_affinity(self.population)

            # Atualiza a melhor solução global encontrada
            current_best_idx = np.argmin(makespans)
            if makespans[current_best_idx] < self.best_makespan_so_far:
                self.best_makespan_so_far = makespans[current_best_idx]
                self.best_sequence_so_far = self.population[current_best_idx][:]

            # 2. Selecionar os n melhores e clonar
            clones = self._select_and_clone(self.population, affinities)
            
            # 3. Hipermutação dos clones
            mutated_clones = self._hypermutation(clones)

            # 4. Avaliar os clones mutados e selecionar os melhores para reintroduzir
            if mutated_clones:
                mutated_makespans, _ = self._evaluate_affinity(mutated_clones)
                best_mutated_idx = np.argmin(mutated_makespans)
                
                # O melhor clone maturado compete para entrar na população
                # Substitui o pior da população original se for melhor
                worst_original_idx = np.argmin(affinities)
                if mutated_makespans[best_mutated_idx] < makespans[worst_original_idx]:
                    self.population[worst_original_idx] = mutated_clones[best_mutated_idx]

            # 5. Substituir os `d` piores indivíduos
            self.population = self._population_replacement(self.population, affinities)

            if verbose and (generation + 1) % 10 == 0:
                print(f"Geração {generation + 1}/{self.n_generations} | Melhor Makespan: {self.best_makespan_so_far}")

        return self.best_sequence_so_far, self.best_makespan_so_far

# Bloco de teste
if __name__ == '__main__':
    # Carrega a instância de exemplo
    instance_filepath = '../instances/example.txt'
    processing_times = load_instance(instance_filepath)
    
    print("Instância FSSP carregada para o CLONALG.")
    print(f"Tarefas: {processing_times.shape[0]}, Máquinas: {processing_times.shape[1]}")
    print("-" * 40)

    # Parâmetros do CLONALG
    params = {
        'population_size': 50,
        'n_generations': 100,
        'selection_size': 10,   # n: número de anticorpos a serem selecionados
        'clone_factor': 0.5,    # β: fator de clonagem
        'num_replace': 5        # d: número de piores a serem substituídos
    }

    print("Iniciando o CLONALG...")
    print(f"Parâmetros: {params}")
    print("-" * 40)

    # Cria e executa o algoritmo
    clonalg = ClonalgForFSSP(processing_times, **params)
    best_sequence, best_makespan = clonalg.run()

    print("-" * 40)
    print("Execução do CLONALG concluída.")
    print(f"Melhor sequência encontrada: {best_sequence}")
    print(f"Melhor Makespan (C_max): {best_makespan}")
