pub use std::cmp::max;
pub use std::fs::File;
pub use std::io::{BufRead, BufReader};
pub use std::path::Path;

/// Representa uma instância do Problema de Escalonamento Flow Shop (FSSP).
#[derive(Debug, Clone)]
pub struct FSSPInstance {
    pub n_jobs: usize,                   // Número de tarefas.
    pub n_machines: usize,               // Número de máquinas.
    pub processing_times: Vec<Vec<u32>>, // Tempos de processamento [tarefa][máquina].
}

/// Carrega uma instância FSSP de um arquivo.
/// O arquivo deve conter N e M na primeira linha, seguidos pelos tempos de processamento.
pub fn load_instance(filepath: &str) -> Result<FSSPInstance, std::io::Error> {
    let file = File::open(Path::new(filepath))?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Lê N (número de tarefas) e M (número de máquinas) da primeira linha.
    let first_line = lines.next().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Arquivo vazio ou mal formatado",
        )
    })??;
    let parts: Vec<usize> = first_line
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    if parts.len() < 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "A primeira linha deve conter N e M",
        ));
    }
    let (n_jobs, n_machines) = (parts[0], parts[1]);

    // Lê os tempos de processamento das N linhas seguintes.
    let mut processing_times = Vec::with_capacity(n_jobs);
    for line in lines.take(n_jobs) {
        let row: Vec<u32> = line?
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        // Valida se o número de tempos por linha corresponde a M.
        if row.len() != n_machines {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Número de tempos em uma linha não corresponde ao número de máquinas.",
            ));
        }
        processing_times.push(row);
    }

    // Valida se o número de linhas de tempo lidas corresponde a N.
    if processing_times.len() != n_jobs {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Número de linhas de tarefas não corresponde ao especificado.",
        ));
    }

    Ok(FSSPInstance {
        n_jobs,
        n_machines,
        processing_times,
    })
}

impl FSSPInstance {
    /// Calcula o **Makespan** (tempo total de conclusão) para uma dada sequência de tarefas.
    /// O Makespan é o tempo em que a última tarefa é finalizada na última máquina.
    pub fn calculate_makespan(&self, sequence: &[usize]) -> u32 {
        // Matriz 'c' armazena os tempos de conclusão: c[máquina][tarefa_na_sequência].
        let mut c = vec![vec![0; self.n_jobs]; self.n_machines];

        // Preenche a matriz de tempos de conclusão.
        for j in 0..self.n_jobs {
            // Itera sobre as tarefas na sequência.
            for i in 0..self.n_machines {
                // Itera sobre as máquinas.
                let job_index = sequence[j]; // ID da tarefa original.
                let p_time = self.processing_times[job_index][i]; // Tempo de processamento.

                if i == 0 && j == 0 {
                    c[i][j] = p_time; // Primeira tarefa na primeira máquina.
                } else if i == 0 {
                    c[i][j] = c[i][j - 1] + p_time; // Primeira máquina, tarefas seguintes.
                } else if j == 0 {
                    c[i][j] = c[i - 1][j] + p_time; // Primeira tarefa, máquinas seguintes.
                } else {
                    // Outros casos: tempo de conclusão é o máximo entre:
                    // - Término da mesma tarefa na máquina anterior.
                    // - Término da tarefa anterior na mesma máquina.
                    c[i][j] = max(c[i - 1][j], c[i][j - 1]) + p_time;
                }
            }
        }

        // O Makespan final é o tempo de conclusão da última tarefa na última máquina.
        c[self.n_machines - 1][self.n_jobs - 1]
    }
}
