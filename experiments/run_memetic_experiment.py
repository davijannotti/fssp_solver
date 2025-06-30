import time
import itertools
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.core import load_instance
from algorithms.memetic import MemeticFSSP

# --- Funções Auxiliares para Geração de Relatórios (adaptadas para o Memético) ---

def generate_latex_table(results_df, filename="memetic_results.tex"):
    """Gera uma tabela LaTeX com os resultados do experimento memético."""
    df_sorted = results_df.sort_values(by='mean_makespan').reset_index(drop=True)

    latex_str = "\\begin{table}[h!]\n"
    latex_str += "\\centering\n"
    latex_str += "\\caption{Resultados do Experimento Fatorial do Algoritmo Memético}\n"
    latex_str += "\\label{tab:memetic_results}\n"
    latex_str += "\\resizebox{\\textwidth}{!}{\n"
    latex_str += "\\begin{tabular}{ccccccc} \\hline \n"
    latex_str += "\\textbf{Rank} & \\textbf{Mutação} & \\textbf{Crossover} & \\textbf{Clone Factor} & \\textbf{LS Rate} & \\textbf{LS Gens} & \\textbf{Makespan (Média \\pm DP)} \\\\ \\hline \n"

    for i, row in df_sorted.head(15).iterrows(): # Mostra as 15 melhores
        makespan_str = f"${row['mean_makespan']:.2f} \\pm {row['std_makespan']:.2f}$"
        latex_str += f"{i+1} & {row['mutation']:.2f} & {row['crossover']} & {row['clone_factor']:.2f} & {row['ls_rate']:.2f} & {row['ls_gens']} & {makespan_str} \\\\ \n"

    latex_str += "\\hline \n"
    latex_str += "\\end{tabular}} \n"
    latex_str += "\\end{table}"

    with open(filename, 'w') as f:
        f.write(latex_str)
    print(f"Tabela de resultados salva em: {filename}")

def generate_convergence_plot(results_df, filename="memetic_convergence.png"):
    """Gera um gráfico de convergência para as melhores configurações do AM."""
    top_10_configs = results_df.sort_values(by='mean_makespan').head(10)

    plt.figure(figsize=(14, 9))

    for i, row in top_10_configs.iterrows():
        mean_convergence = np.mean(row['convergence_histories'], axis=0)
        label = f"C:{row['crossover']}, M:{row['mutation']:.2f}, CF:{row['clone_factor']:.2f}, LSR:{row['ls_rate']:.2f}, LSG:{row['ls_gens']}"
        plt.plot(mean_convergence, label=label, alpha=0.8)

    plt.title('Convergência Média das 10 Melhores Configurações do Algoritmo Memético')
    plt.xlabel('Geração')
    plt.ylabel('Melhor Makespan Médio')
    plt.legend(loc='upper right', fontsize='medium')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(filename)
    print(f"Gráfico de convergência salvo em: {filename}")

# --- Função de Execução para Multiprocessamento ---

def run_single_memetic_config(config):
    """Executa o Algoritmo Memético para uma única configuração, N vezes."""
    config_id, ga_params, clonalg_params, memetic_params, n_runs, processing_times = config

    print(f"Testando Configuração #{config_id}")

    run_makespans = []
    run_histories = []

    for _ in range(n_runs):
        solver = MemeticFSSP(processing_times, ga_params, clonalg_params, **memetic_params)
        _, makespan, history = solver.run(verbose=False, track_convergence=True)
        run_makespans.append(makespan)
        run_histories.append(history)

    return {
        'config_id': config_id,
        'ga_params': ga_params,
        'clonalg_params': clonalg_params,
        'memetic_params': memetic_params,
        'mean_makespan': np.mean(run_makespans),
        'std_makespan': np.std(run_makespans),
        'convergence_histories': run_histories
    }

# --- Script Principal ---

def main():
    instance_filepath = '../instances/fssp_instance_05.txt'
    processing_times = load_instance(instance_filepath)

    # Parâmetros fixos (ajustados para um experimento mais rápido)
    N_RUNS = 5
    POPULATION_SIZE = 100
    N_GENERATIONS = 100

    # Espaço de parâmetros a ser explorado
    param_space = {
        'mutation_rate': [0.15, 0.3],
        'crossover_method': ['ox', 'cx'],
        'clone_factor': [0.3, 0.6, 0.9],
        'local_search_rate': [0.2, 0.4],
        'clonalg_generations': [5, 10]
    }

    keys, values = zip(*param_space.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    tasks = []
    for i, params in enumerate(param_combinations):
        ga_params = {
            'population_size': POPULATION_SIZE,
            'n_generations': N_GENERATIONS,
            'mutation_rate': params['mutation_rate'],
            'elitism_rate': 0.1, # Fixo para simplificar
            'crossover_method': params['crossover_method'],
            'parent_selection_method': 'tournament' # Fixo
        }
        clonalg_params = {
            'population_size': 10,
            'selection_size': 4,
            'clone_factor': params['clone_factor'],
            'num_replace': 2
        }
        memetic_params = {
            'local_search_rate': params['local_search_rate'],
            'clonalg_generations': params['clonalg_generations']
        }
        tasks.append((i, ga_params, clonalg_params, memetic_params, N_RUNS, processing_times))

    print(f"Iniciando experimento memético com {len(tasks)} configurações.")
    print(f"Cada configuração será executada {N_RUNS} vezes.")
    print(f"Total de execuções do AM: {len(tasks) * N_RUNS}")
    print("-" * 50)

    start_time = time.time()
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(run_single_memetic_config, tasks)
    end_time = time.time()

    print(f"\nExperimento concluído em {end_time - start_time:.2f} segundos.")
    print("-" * 50)

    # Processa e salva os resultados
    results_df = pd.DataFrame([
        {
        'mutation': r['ga_params']['mutation_rate'],
        'crossover': r['ga_params']['crossover_method'],
        'clone_factor': r['clonalg_params']['clone_factor'],
        'ls_rate': r['memetic_params']['local_search_rate'],
        'ls_gens': r['memetic_params']['clonalg_generations'],
        'mean_makespan': r['mean_makespan'],
        'std_makespan': r['std_makespan'],
        'convergence_histories': r['convergence_histories']
    } for r in results])

    generate_latex_table(results_df, filename="../results/memetic_results.tex")
    generate_convergence_plot(results_df, filename="../results/memetic_convergence.png")

if __name__ == "__main__":
    main()
