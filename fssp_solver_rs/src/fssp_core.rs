pub use std::cmp::max;
pub use std::fs::File;
pub use std::io::{BufRead, BufReader};
pub use std::path::Path;

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
impl FSSPInstance {
    pub fn calculate_makespan(&self, sequence: &[usize]) -> u32 {
        // A matriz C armazena os tempos de conclusão.
        // c[i][j] é o tempo de conclusão da j-ésima tarefa da sequência na máquina i.
        let mut c = vec![vec![0; self.n_jobs]; self.n_machines];

        // Calcula os tempos de conclusão
        for j in 0..self.n_jobs {
            // Para cada tarefa na sequência
            for i in 0..self.n_machines {
                // Para cada máquina
                let job_index = sequence[j];
                let p_time = self.processing_times[job_index][i];

                if i == 0 && j == 0 {
                    // Primeira tarefa na primeira máquina
                    c[i][j] = p_time;
                } else if i == 0 {
                    // Tarefas seguintes na primeira máquina
                    c[i][j] = c[i][j - 1] + p_time;
                } else if j == 0 {
                    // Primeira tarefa nas máquinas seguintes
                    c[i][j] = c[i - 1][j] + p_time;
                } else {
                    // Todas as outras
                    c[i][j] = max(c[i - 1][j], c[i][j - 1]) + p_time;
                }
            }
        }

        // O makespan é o tempo de conclusão da última tarefa na última máquina.
        c[self.n_machines - 1][self.n_jobs - 1]
    }
}
