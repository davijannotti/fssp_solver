use std::fs::File;
use std::io::{BufRead, BufReader, ErrorKind};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct FSSPInstance {
    pub n_jobs: usize,
    pub n_machines: usize,
    pub processing_times: Vec<Vec<u32>>,
}

pub fn load_instance(filepath: &str) -> Result<FSSPInstance, std::io::Error> {
    let file = File::open(Path::new(filepath))?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Lê a primeira linha para obter N e M
    let first_line = lines.next().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Arquivo de instância está vazio ou mal formatado",
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

    // Lê as N linhas de tempos de processamento
    let mut processing_times = Vec::with_capacity(n_jobs);
    for line in lines.take(n_jobs) {
        let row: Vec<u32> = line?
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        // Validação crucial: verifica se o número de colunas bate com n_machines
        if row.len() != n_machines {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "O número de tempos de processamento em uma linha não corresponde ao número de máquinas."));
        }
        processing_times.push(row);
    }

    // Validação final para garantir que o número de tarefas lido bate com n_jobs
    if processing_times.len() != n_jobs {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "O número de linhas de tarefas não corresponde ao número de tarefas especificado.",
        ));
    }

    Ok(FSSPInstance {
        n_jobs,
        n_machines,
        processing_times,
    })
}

pub fn calculate_makespan(instance: &FSSPInstance, sequence: &[usize]) -> u32 {
    let n_jobs = instance.n_jobs;
    let n_machines = instance.n_machines;
    let mut completion_times = vec![vec![0u32; n_machines]; n_jobs];

    // Primeira tarefa na sequência
    let first_job_id = sequence[0];
    completion_times[0][0] = instance.processing_times[first_job_id][0];
    for m_idx in 1..n_machines {
        completion_times[0][m_idx] =
            completion_times[0][m_idx - 1] + instance.processing_times[first_job_id][m_idx];
    }

    // Demais tarefas na sequência
    for j_idx in 1..n_jobs {
        // j_idx é o índice na *sequência*
        let current_job_id = sequence[j_idx];

        // Primeira máquina
        completion_times[j_idx][0] =
            completion_times[j_idx - 1][0] + instance.processing_times[current_job_id][0];

        // Demais máquinas
        for m_idx in 1..n_machines {
            completion_times[j_idx][m_idx] = u32::max(
                completion_times[j_idx - 1][m_idx], // Conclusão da tarefa anterior na mesma máquina
                completion_times[j_idx][m_idx - 1], // Conclusão desta tarefa na máquina anterior
            ) + instance.processing_times[current_job_id][m_idx];
        }
    }

    completion_times[n_jobs - 1][n_machines - 1]
}
