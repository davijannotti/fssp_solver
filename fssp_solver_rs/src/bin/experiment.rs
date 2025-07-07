use fssp_solver_rs::fssp_core::load_instance;
use fssp_solver_rs::solver::MemeticAlgorithm;
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

fn main() -> io::Result<()> {
    // 1. Carrega a instância mais desafiadora (`fssp_instance_05.txt`).
    let instance_path = "./src/instances/fssp_instance_05.txt";
    let instance = load_instance(instance_path).expect("Failed to load FSSP instance");

    // 2. Define uma gama de parâmetros a serem testados para o Algoritmo Memético
    let population_sizes = vec![50, 100];
    let generations = vec![100, 200];
    let mutation_rates = vec![0.01, 0.05, 0.1];
    let local_search_rates = vec![0.1, 0.2, 0.3];
    let num_runs = 5; // Para robustez estatística

    // Itera sobre todas as combinações de parâmetros para gerar a lista de tarefas
    let mut all_combinations: Vec<(usize, usize, f64, f64)> = Vec::new();
    for &pop_size in &population_sizes {
        for &gens in &generations {
            for &mut_rate in &mutation_rates {
                for &ls_rate in &local_search_rates {
                    all_combinations.push((pop_size, gens, mut_rate, ls_rate));
                }
            }
        }
    }

    // Process combinations em paralelo
    let results: Vec<String> = all_combinations
        .par_iter()
        .map(|&(pop_size, gens, mut_rate, ls_rate)| {
            let mut makespans = Vec::with_capacity(num_runs);
            for _ in 0..num_runs {
                let current_instance = instance.clone(); // Clona a instância para cada execução
                let mut solver =
                    MemeticAlgorithm::new(current_instance, pop_size, gens, mut_rate, ls_rate);
                solver.run();
                makespans.push(solver.best_makespan);
            }

            let sum: u32 = makespans.iter().sum();
            let mean = sum as f64 / num_runs as f64;

            let variance = makespans
                .iter()
                .map(|&m| {
                    let diff = m as f64 - mean;
                    diff * diff
                })
                .sum::<f64>()
                / num_runs as f64;
            let std_dev = variance.sqrt();

            format!(
                "{},{},{},{},{:.2},{:.2}",
                pop_size, gens, mut_rate, ls_rate, mean, std_dev
            )
        })
        .collect();

    // 5. Salva os resultados em um arquivo `results.csv`
    let output_path = Path::new("results.csv");
    let mut file = File::create(&output_path)?;

    writeln!(file, "population_size,generations,mutation_rate,local_search_rate,mean_makespan,std_dev_makespan")?;
    for line in results {
        writeln!(file, "{}", line)?;
    }

    println!("Experiment results saved to results.csv");

    Ok(())
}
