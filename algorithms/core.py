import numpy as np

def load_instance(filepath):
    """
    Carrega uma instância do FSSP de um arquivo.

    O formato do arquivo é:
    - A primeira linha contém o número de tarefas (N) e o número de máquinas (M).
    - As N linhas seguintes contêm os M tempos de processamento para cada tarefa.

    Args:
        filepath (str): O caminho para o arquivo da instância.

    Returns:
        numpy.ndarray: Uma matriz com os tempos de processamento (tarefas x máquinas).
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Lê N e M da primeira linha
    n_jobs, n_machines = map(int, lines[0].strip().split())

    # Lê os tempos de processamento das linhas seguintes
    processing_times = []
    for i in range(n_jobs):
        parts = list(map(int, lines[i + 1].strip().split()))
        processing_times.append(parts)

    return np.array(processing_times)

def calculate_makespan(processing_times, sequence):
    """
    Calcula o makespan (C_max) para uma dada sequência de tarefas.

    Args:
        processing_times (numpy.ndarray): A matriz de tempos de processamento.
        sequence (list[int]): A sequência (permutação) de tarefas.

    Returns:
        int: O makespan (C_max).
    """
    n_jobs, n_machines = processing_times.shape

    # A matriz C armazena os tempos de conclusão
    # C[i][j] é o tempo de conclusão da j-ésima tarefa da sequência na máquina i
    c = np.zeros((n_machines, n_jobs))

    # Calcula os tempos de conclusão
    for j in range(n_jobs):  # Para cada tarefa na sequência
        for i in range(n_machines):  # Para cada máquina
            job_index = sequence[j]

            if i == 0 and j == 0:
                # Primeira tarefa na primeira máquina
                c[i][j] = processing_times[job_index][i]
            elif i == 0:
                # Tarefas seguintes na primeira máquina
                c[i][j] = c[i][j-1] + processing_times[job_index][i]
            elif j == 0:
                # Primeira tarefa nas máquinas seguintes
                c[i][j] = c[i-1][j] + processing_times[job_index][i]
            else:
                # Todas as outras
                c[i][j] = max(c[i-1][j], c[i][j-1]) + processing_times[job_index][i]

    # O makespan é o tempo de conclusão da última tarefa na última máquina
    makespan = c[n_machines-1][n_jobs-1]
    return int(makespan)

# Bloco de teste
if __name__ == '__main__':
    # Exemplo de tempos de processamento (3 tarefas, 3 máquinas)
    # Este exemplo foi criado para que a sequência [0, 1, 2] resulte em um makespan de 31.
    test_processing_times = np.array([
        [8, 1, 5],  # Tarefa 0
        [3, 7, 2],  # Tarefa 1
        [4, 6, 7]   # Tarefa 2
    ])

    # Sequência de teste
    test_sequence = [0, 1, 2]

    # Calcular o makespan esperado
    expected_makespan = 31
    calculated_makespan = calculate_makespan(test_processing_times, test_sequence);

    print(f"Sequência de teste: {test_sequence}")
    print(f"Tempos de processamento:\n{test_processing_times}")
    print(f"Makespan calculado: {calculated_makespan}")
    print(f"Makespan esperado: {expected_makespan}")

    # Verificar se o resultado está correto
    assert calculated_makespan == expected_makespan, "O makespan calculado está incorreto!"

    print("\nTeste do `calculate_makespan` passou com sucesso!")
