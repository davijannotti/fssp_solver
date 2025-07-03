

import time
import itertools
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.core import load_instance
from algorithms.genetic import GeneticAlgorithmFSSP

# --- Funções Auxiliares para Geração de Relatórios ---

def generate_latex_table(results_df, filename="ga_results.tex"):
    """Gera uma tabela LaTeX com os resultados do experimento."""
    # Ordena os resultados pela média do makespan
    df_sorted = results_df.sort_values(by='mean_makespan').reset_index(drop=True)

    latex_str = "\\begin{table}[h!]\n"
    latex_str += "\centering\n"
    latex_str += "\caption{Resultados do Experimento Fatorial do Algoritmo Genético}\n"
    latex_str += "\label{tab:ga_results}\n"
    latex_str += "\\begin{tabular}{cccccc} \\hline \n"
    latex_str += "\\textbf{Rank} & \\textbf{Mutação} & \\textbf{Elitismo} & \\textbf{Cruzamento} & \\textbf{Seleção} & \\textbf{Makespan (Média \\pm DP)} \\\\ \\hline \n"

    for i, row in df_sorted.iterrows():
        makespan_str = f"${row['mean_makespan']:.2f} \\pm {row['std_makespan']:.2f}$"
        latex_str += f"{i+1} & {row['mutation']:.2f} & {row['elitism']:.2f} & {row['crossover']} & {row['selection']} & {makespan_str} \\\\ \n"

    latex_str += "\\hline \n"
    latex_str += "\end{tabular} \n"
    latex_str += "\end{table}"

    with open(filename, 'w') as f:
        f.write(latex_str)
    print(f"Tabela de resultados salva em: {filename}")

def generate_convergence_plot(results_df, filename="ga_convergence.png"):
    """Gera um gráfico de convergência para as melhores configurações."""
    # Ordena e pega as 10 melhores configurações
    top_10_configs = results_df.sort_values(by='mean_makespan').head(10)

    plt.figure(figsize=(12, 8))

    for i, row in top_10_configs.iterrows():
        # Calcula a média da convergência ao longo das execuções
        mean_convergence = np.mean(row['convergence_histories'], axis=0)
        label = f"C: {row['crossover']}, S: {row['selection']}, M: {row['mutation']:.2f}, E: {row['elitism']:.2f}"
        plt.plot(mean_convergence, label=label)

    plt.title('Convergência Média das 10 Melhores Configurações do AG')
    plt.xlabel('Geração')
    plt.ylabel('Melhor Makespan Médio')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(filename)
    print(f"Gráfico de convergência salvo em: {filename}")

# --- Função de Execução para Multiprocessamento ---

def run_single_config(config):
    """
    Executa o AG para uma única configuração, N vezes.
    Esta função é projetada para ser usada com multiprocessing.Pool.
    """
    # Desempacota a configuração
    config_id, params, n_runs, processing_times = config

    print(f"Testando Configuração #{config_id}: {params}")

    run_makespans = []
    run_histories = []

    for _ in range(n_runs):
        ga = GeneticAlgorithmFSSP(processing_times=processing_times, **params)
        _, makespan, history = ga.run(track_convergence=True, verbose=False)
        run_makespans.append(makespan)
        run_histories.append(history)

    # Calcula as métricas
    mean_makespan = np.mean(run_makespans)
    std_makespan = np.std(run_makespans)

    return {
        'config_id': config_id,
        'params': params,
        'mean_makespan': mean_makespan,
        'std_makespan': std_makespan,
        'convergence_histories': run_histories
    }

# --- Script Principal ---

def main():
    # Carrega a instância
    instance_filepath = '../instances/fssp_instance_05.txt'
    processing_times = load_instance(instance_filepath)

    # Parâmetros fixos
    N_RUNS = 10  # Número de execuções por configuração para robustez estatística
    POPULATION_SIZE = 50
    N_GENERATIONS = 100

    # Espaço de parâmetros a ser explorado (fatorial)
    param_space = {
        'mutation_rate': [0.05, 0.15, 0.30],
        'elitism_rate': [0.05, 0.1, 0.2],
        'crossover_method': ['ox', 'cx'],
        'parent_selection_method': ['tournament', 'rank']
    }

    # Gera todas as combinações de parâmetros
    keys, values = zip(*param_space.items())
    experiment_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Adiciona os parâmetros fixos
    for config in experiment_configs:
        config['population_size'] = POPULATION_SIZE
        config['n_generations'] = N_GENERATIONS

    # Prepara os argumentos para o pool de multiprocessamento
    tasks = [
        (i, params, N_RUNS, processing_times)
        for i, params in enumerate(experiment_configs)
    ]

    print(f"Iniciando experimento fatorial com {len(tasks)} configurações.")
    print(f"Cada configuração será executada {N_RUNS} vezes.")
    print(f"Total de execuções do AG: {len(tasks) * N_RUNS}")
    print("-" * 50)

    start_time = time.time()

    # Executa o experimento em paralelo
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(run_single_config, tasks)

    end_time = time.time()
    print(f"\nExperimento concluído em {end_time - start_time:.2f} segundos.")
    print("-" * 50)

    # Processa os resultados
    import pandas as pd
    results_df = pd.DataFrame([{
        'mutation': r['params']['mutation_rate'],
        'elitism': r['params']['elitism_rate'],
        'crossover': r['params']['crossover_method'],
        'selection': r['params']['parent_selection_method'],
        'mean_makespan': r['mean_makespan'],
        'std_makespan': r['std_makespan'],
        'convergence_histories': r['convergence_histories']
    } for r in results])

    # Gera os artefatos de saída
    generate_latex_table(results_df, filename="./ga_results.tex")
    generate_convergence_plot(results_df, filename="./ga_convergence.png")

if __name__ == "__main__":
    main()
