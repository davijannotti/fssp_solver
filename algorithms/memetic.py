
import random
import numpy as np
from .core import load_instance, calculate_makespan
from .genetic import GeneticAlgorithmFSSP
from .clonalg import ClonalgForFSSP

class MemeticFSSP(GeneticAlgorithmFSSP):
    """
    Resolve o FSSP usando um Algoritmo Memético, que combina um Algoritmo
    Genético com o CLONALG como mecanismo de busca local.
    """
    def __init__(self, processing_times, ga_params, clonalg_params, 
                 local_search_rate, clonalg_generations):
        
        # Inicializa a parte do Algoritmo Genético
        super().__init__(processing_times=processing_times, **ga_params)
        
        # Parâmetros específicos do Algoritmo Memético
        self.clonalg_params = clonalg_params
        self.local_search_rate = local_search_rate
        self.clonalg_generations = clonalg_generations

    def _local_improvement(self, individual):
        """
        Aplica uma busca local usando CLONALG para refinar um indivíduo.
        """
        # Define os parâmetros para a busca local do CLONALG
        local_clonalg_params = self.clonalg_params.copy()
        local_clonalg_params['n_generations'] = self.clonalg_generations
        
        # A população inicial do CLONALG será baseada no indivíduo
        # Criamos uma pequena população de clones para dar diversidade inicial
        initial_pop_size = local_clonalg_params.get('population_size', 10)
        initial_population = [individual[:] for _ in range(initial_pop_size)]

        # Instancia e executa o CLONALG
        local_search_solver = ClonalgForFSSP(self.processing_times, **local_clonalg_params)
        local_search_solver.population = initial_population # Define a população inicial
        
        # Executa o CLONALG de forma silenciosa
        best_sequence, _ = local_search_solver.run(verbose=False)
        
        return best_sequence

    def run(self, verbose=True, track_convergence=False):
        """Executa o loop principal do Algoritmo Memético."""
        convergence_history = []

        for generation in range(self.n_generations):
            fitnesses = self._evaluate_fitness(self.population)
            
            # Atualiza a melhor solução global
            current_best_makespan = min(fitnesses)
            if current_best_makespan < self.best_makespan_so_far:
                self.best_makespan_so_far = current_best_makespan
                best_idx = fitnesses.index(current_best_makespan)
                self.best_sequence_so_far = self.population[best_idx][:]

            if track_convergence:
                convergence_history.append(self.best_makespan_so_far)

            # Elitismo
            elites = self._elitism(self.population, fitnesses)
            
            # Seleção e Crossover/Mutação (como no AG)
            parents = self._selection(self.population, fitnesses)
            
            children = []
            while len(children) < self.population_size - len(elites):
                p1, p2 = random.sample(parents, 2)
                child = self._crossover(p1, p2)
                child = self._mutation(child)
                children.append(child)

            # Etapa de Melhoria Local (Busca Memética)
            improved_children = []
            for child in children:
                if random.random() < self.local_search_rate:
                    improved_child = self._local_improvement(child)
                    improved_children.append(improved_child)
                else:
                    improved_children.append(child)
            
            next_population = elites + improved_children
            self.population = next_population
            
            if verbose and (generation + 1) % 5 == 0:
                print(f"Geração {generation + 1}/{self.n_generations} | Melhor Makespan: {self.best_makespan_so_far}")

        if track_convergence:
            return self.best_sequence_so_far, self.best_makespan_so_far, convergence_history
        else:
            return self.best_sequence_so_far, self.best_makespan_so_far

# Bloco de teste
if __name__ == '__main__':
    # Carrega a instância
    instance_filepath = '../instances/example.txt'
    processing_times = load_instance(instance_filepath)
    
    print("Instância FSSP carregada para o Algoritmo Memético.")
    print("-" * 50)

    # Parâmetros para o Algoritmo Genético (a parte global)
    ga_params = {
        'population_size': 50,
        'n_generations': 50, # Menos gerações globais, pois a busca local refina
        'mutation_rate': 0.2,
        'elitism_rate': 0.1,
        'crossover_method': 'ox',
        'parent_selection_method': 'tournament'
    }

    # Parâmetros para o CLONALG (a parte de busca local)
    clonalg_params = {
        'population_size': 10, # População pequena para a busca local
        'selection_size': 3,
        'clone_factor': 0.5,
        'num_replace': 1
    }

    # Parâmetros do Algoritmo Memético
    memetic_params = {
        'local_search_rate': 0.3, # 30% de chance de aplicar busca local
        'clonalg_generations': 5  # Gerações para o refinamento local
    }

    print("Iniciando o Algoritmo Memético...")
    
    # Cria e executa o algoritmo
    memetic_solver = MemeticFSSP(
        processing_times=processing_times, 
        ga_params=ga_params, 
        clonalg_params=clonalg_params, 
        **memetic_params
    )
    best_sequence, best_makespan = memetic_solver.run()

    print("-" * 50)
    print("Execução do Algoritmo Memético concluída.")
    print(f"Melhor sequência encontrada: {best_sequence}")
    print(f"Melhor Makespan (C_max): {best_makespan}")
