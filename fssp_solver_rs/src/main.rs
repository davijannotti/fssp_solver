mod fssp_core;
mod solver; // Mude de 'genetic_algorithm' para 'solver'

use fssp_core::load_instance;
use solver::MemeticAlgorithm;
use std::time::Instant; // Mude o nome da classe se necessário

fn main() {
    let filepath = "./src/instances/fssp_instance_05.txt";
    println!("Resolvendo instância: {}", filepath);

    let instance = load_instance(filepath).expect("Falha ao carregar instância.");
    println!(
        "Instância carregada: {} tarefas, {} máquinas.",
        instance.n_jobs, instance.n_machines
    );

    // Parâmetros do Algoritmo Memético
    let population_size = 30;
    let generations = 100;
    let mutation_rate = 0.3;
    let local_search_rate = 0.3;

    let start_time = Instant::now();

    let mut solver = MemeticAlgorithm::new(
        instance,
        population_size,
        generations,
        mutation_rate,
        local_search_rate,
    );

    solver.run();

    let execution_time = start_time.elapsed();

    println!("\n--- Resultados Finais ---");
    println!("Melhor Makespan: {}", solver.best_makespan);

    let sequence_str: Vec<String> = solver
        .best_sequence
        .iter()
        .map(|&x| (x + 1).to_string())
        .collect();
    println!("Melhor Sequencia: {}", sequence_str.join(" "));

    println!(
        "Tempo de Execucao (segundos): {:.4}",
        execution_time.as_secs_f64()
    );
}
