import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithms.core import load_instance
from algorithms.genetic import GeneticAlgorithmFSSP

def main():
    """
    Script principal para resolver uma instância do FSSP com o Algoritmo Genético.
    """
    # 1. Carregar a instância do FSSP
    instance_filepath = 'fssp_solver/instances/fssp_instance_04.txt'
    print(f"Carregando instância de: {instance_filepath}")
    processing_times = load_instance(instance_filepath)
    print(f"Instância carregada: {processing_times.shape[0]} tarefas, {processing_times.shape[1]} máquinas.")
    print("-" * 40)

    # 2. Definir os parâmetros do Algoritmo Genético
    ga_params = {
        'population_size': 100,
        'n_generations': 200,
        'mutation_rate': 0.15,
        'elitism_rate': 0.1,
        'crossover_method': 'ox',  # Opções: 'ox', 'cx'
        'parent_selection_method': 'tournament'  # Opções: 'tournament', 'rank'
    }
    print("Parâmetros do Algoritmo Genético:")
    for key, value in ga_params.items():
        print(f"  - {key.replace('_', ' ').title()}: {value}")
    print("-" * 40)

    # 3. Instanciar e executar o algoritmo
    print("Iniciando a execução do Algoritmo Genético...")
    start_time = time.time()

    ga_solver = GeneticAlgorithmFSSP(processing_times, **ga_params)
    best_sequence, best_makespan = ga_solver.run()

    end_time = time.time()
    total_time = end_time - start_time
    print("Execução concluída.")
    print("-" * 40)

    # 4. Imprimir os resultados
    print("Resultados:")
    print(f"  - Melhor sequência encontrada: {best_sequence}")
    print(f"  - Melhor Makespan (C_max): {best_makespan}")
    print(f"  - Tempo total de execução: {total_time:.4f} segundos")

if __name__ == "__main__":
    main()
