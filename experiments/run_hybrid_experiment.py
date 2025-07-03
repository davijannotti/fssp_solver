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
from algorithms.hybrid_aco_bat import HybridAcoBatFSSP

# --- Funções Auxiliares para Geração de Relatórios ---

def generate_latex_table(results_df, filename="hybrid_results.tex"):
    """Gera uma tabela LaTeX com os resultados do experimento híbrido."""
    df_sorted = results_df.sort_values(by='mean_makespan').reset_index(drop=True)

    latex_str = "\\begin{table}[h!]\n"
    latex_str += "\centering\n"
    latex_str += "\caption{Resultados do Experimento Fatorial do Algoritmo Híbrido ACO-BA}\n"
    latex_str += "\label{tab:hybrid_results}\n"
    latex_str += "\\begin{tabular}{cccccc} \\ \hline \n"
    latex_str += "\\textbf{Rank} & \\textbf{Alpha (ACO)} & \\textbf{Rho (ACO)} & \\textbf{Loudness (BA)} & \\textbf{Pulse Rate (BA)} & \\textbf{Makespan (Média \\pm DP)} \\ \\ \\ \hline \n"

    for i, row in df_sorted.head(15).iterrows():
        makespan_str = f"${row['mean_makespan']:.2f} \\pm {row['std_makespan']:.2f}$"
        latex_str += f"{i+1} & {row['alpha']:.2f} & {row['rho']:.2f} & {row['loudness']:.2f} & {row['pulse_rate']:.2f} & {makespan_str} \\ \n"

    latex_str += "\\hline \n"
    latex_str += "\end{tabular} \n"
    latex_str += "\end{table}"

    with open(filename, 'w') as f:
        f.write(latex_str)
    print(f"Tabela de resultados salva em: {filename}")

def generate_convergence_plot(results_df, filename="hybrid_convergence.png"):
    """Gera um gráfico de convergência para as melhores configurações do algoritmo híbrido."""
    top_10_configs = results_df.sort_values(by='mean_makespan').head(10)

    plt.figure(figsize=(14, 9))

    for i, row in top_10_configs.iterrows():
        mean_convergence = np.mean(row['convergence_histories'], axis=0)
        label = f"A:{row['alpha']:.2f}, R:{row['rho']:.2f}, L:{row['loudness']:.2f}, PR:{row['pulse_rate']:.2f}"
        plt.plot(mean_convergence, label=label, alpha=0.8)

    plt.title('Convergência Média das 10 Melhores Configurações do Algoritmo Híbrido ACO-BA')
    plt.xlabel('Geração (Fase Bat Algorithm)')
    plt.ylabel('Melhor Makespan Médio')
    plt.legend(loc='upper right', fontsize='medium')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(filename)
    print(f"Gráfico de convergência salvo em: {filename}")

# --- Função de Execução para Multiprocessamento ---

def run_single_hybrid_config(config):
    """Executa o algoritmo híbrido para uma única configuração, N vezes."""
    config_id, aco_params, bat_params, hybrid_params, n_runs, processing_times = config

    print(f"Testando Configuração #{config_id}")

    run_makespans = []
    run_histories = []

    for _ in range(n_runs):
        solver = HybridAcoBatFSSP(processing_times, aco_params, bat_params, **hybrid_params)
        _, makespan, history = solver.run(verbose=False, track_convergence=True)
        run_makespans.append(makespan)
        run_histories.append(history)

    return {
        'config_id': config_id,
        'aco_params': aco_params,
        'bat_params': bat_params,
        'mean_makespan': np.mean(run_makespans),
        'std_makespan': np.std(run_makespans),
        'convergence_histories': run_histories
    }

# --- Script Principal ---

def main():
    instance_filepath = '../instances/fssp_instance_05.txt'
    processing_times = load_instance(instance_filepath)

    # Parâmetros fixos
    N_RUNS = 5
    ACO_GENERATIONS = 100
    BAT_GENERATIONS = 100
    NUM_ELITES_FOR_BA = 10

    # Espaço de parâmetros a ser explorado
    param_space = {
        'alpha': [0.5, 1.0, 2.0],  # Parâmetro do ACO
        'evaporation_rate': [0.3, 0.5, 0.7], # Parâmetro do ACO (rho)
        'loudness_initial': [0.8, 0.95], # Parâmetro do BA
        'pulse_rate_initial': [0.4, 0.6]  # Parâmetro do BA (r)
    }

    keys, values = zip(*param_space.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    tasks = []
    for i, params in enumerate(param_combinations):
        aco_params = {
            'n_ants': 30,
            'n_generations': ACO_GENERATIONS,
            'alpha': params['alpha'],
            'beta': 2.0, # Fixo
            'evaporation_rate': params['evaporation_rate'],
            'q0': 0.9 # Fixo
        }
        bat_params = {
            'n_bats': 20,
            'n_generations': BAT_GENERATIONS,
            'loudness_initial': params['loudness_initial'],
            'pulse_rate_initial': params['pulse_rate_initial'],
            'f_min': 0.0, # Fixo
            'f_max': 2.0  # Fixo
        }
        hybrid_params = {
            'num_elites_for_ba': NUM_ELITES_FOR_BA
        }
        tasks.append((i, aco_params, bat_params, hybrid_params, N_RUNS, processing_times))

    print(f"Iniciando experimento híbrido com {len(tasks)} configurações.")
    print(f"Cada configuração será executada {N_RUNS} vezes.")
    print(f"Total de execuções: {len(tasks) * N_RUNS}")
    print("-" * 50)

    start_time = time.time()
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(run_single_hybrid_config, tasks)
    end_time = time.time()

    print(f"\nExperimento concluído em {end_time - start_time:.2f} segundos.")
    print("-" * 50)

    # Processa e salva os resultados
    results_df = pd.DataFrame([{
        'alpha': r['aco_params']['alpha'],
        'rho': r['aco_params']['evaporation_rate'],
        'loudness': r['bat_params']['loudness_initial'],
        'pulse_rate': r['bat_params']['pulse_rate_initial'],
        'mean_makespan': r['mean_makespan'],
        'std_makespan': r['std_makespan'],
        'convergence_histories': r['convergence_histories']
    } for r in results])

    generate_latex_table(results_df, filename="../results/hybrid_results.tex")
    generate_convergence_plot(results_df, filename="../results/hybrid_convergence.png")

if __name__ == "__main__":
    main()
