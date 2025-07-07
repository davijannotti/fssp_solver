use clap::Parser;
use fssp_solver_rs::fssp_core::load_instance;
use fssp_solver_rs::solver::MemeticAlgorithm;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// O caminho para o arquivo da instância FSSP.
    #[arg(required = true)]
    instance_path: PathBuf,

    /// Número máximo de gerações que o algoritmo irá executar.
    #[arg(long, default_value_t = 100)]
    max_generations: usize,

    /// Duração máxima da execução em segundos. O algoritmo encerrará se exceder este tempo.
    #[arg(long)]
    max_duration: Option<u64>,

    /// Diretório para salvar o arquivo de resultado.
    #[arg(long, default_value = ".")]
    output_dir: PathBuf,

    // --- Parâmetros do Algoritmo ---
    /// Tamanho da população.
    #[arg(long, default_value_t = 100)]
    population_size: usize,

    /// Taxa de mutação (probabilidade de um indivíduo sofrer mutação).
    #[arg(long, default_value_t = 0.3)]
    mutation_rate: f64,

    /// Taxa de busca local (probabilidade de um indivíduo passar por busca local).
    #[arg(long, default_value_t = 0.6)]
    local_search_rate: f64,
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();

    let instance =
        load_instance(cli.instance_path.to_str().unwrap()).expect("Falha ao carregar instância.");

    let start_time = Instant::now();
    let max_duration = cli.max_duration.map(Duration::from_secs);

    let mut solver = MemeticAlgorithm::new(
        instance,
        cli.population_size,
        cli.max_generations,
        cli.mutation_rate,
        cli.local_search_rate,
    );

    // Executa o solver com os limites de tempo e geração.
    solver.run(start_time, max_duration);

    let execution_time = start_time.elapsed();

    // --- Exibição dos resultados no console ---
    println!("\n--- Resultados Finais ---");
    println!("Melhor Makespan: {}", solver.best_makespan);

    let sequence_str_display: Vec<String> = solver
        .best_sequence
        .iter()
        .map(|&x| (x + 1).to_string()) // +1 para visualização (base 1)
        .collect();
    println!("Melhor Sequencia: {}", sequence_str_display.join(" "));
    println!(
        "Tempo de Execucao (segundos): {:.4}",
        execution_time.as_secs_f64()
    );

    // --- Geração do arquivo de resultado ---
    write_results_to_file(&cli, &solver, execution_time.as_secs_f64())?;

    Ok(())
}

fn write_results_to_file(
    cli: &Cli,
    solver: &MemeticAlgorithm,
    exec_time: f64,
) -> std::io::Result<()> {
    // Extrai o nome do arquivo da instância, ex: "fssp_instance_05"
    let instance_stem = Path::new(&cli.instance_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("resultado_desconhecido");

    // Monta o nome do arquivo de saída
    let output_filename = format!("resultado_{}.txt", instance_stem);
    let output_path = cli.output_dir.join(output_filename);

    println!("\nSalvando resultados em: {}", output_path.display());

    // Cria e abre o arquivo para escrita
    let mut file = File::create(&output_path)?;

    // Formata a sequência para o arquivo (base 0, como nos dados)
    let sequence_str_file: Vec<String> = solver
        .best_sequence
        .iter()
        .map(|&x| x.to_string())
        .collect();

    // Escreve os resultados no arquivo
    writeln!(file, "Melhor Makespan: {}", solver.best_makespan)?;
    writeln!(file, "Melhor Sequencia: {}", sequence_str_file.join(" "))?;
    writeln!(file, "Tempo de Execucao (segundos): {:.4}", exec_time)?;

    println!("Resultados salvos com sucesso.");
    Ok(())
}
