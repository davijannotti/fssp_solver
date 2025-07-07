use crate::fssp_core::FSSPInstance;
use rand::seq::SliceRandom;
use rand::Rng;
use std::time::Instant;

/// Estrutura que representa o Algoritmo Memético para resolver o Problema de Escalonamento Flow Shop.
pub struct MemeticAlgorithm {
    instance: FSSPInstance,        // Instância do problema FSSP.
    population_size: usize,        // Tamanho da população.
    generations: usize,            // Número máximo de gerações.
    mutation_rate: f64,            // Taxa de mutação.
    local_search_rate: f64,        // Taxa de aplicação da busca local.
    population: Vec<Vec<usize>>,   // População atual de sequências de tarefas.
    fitness: Vec<u32>,             // Makespan (aptidão) de cada indivíduo na população.
    pub best_sequence: Vec<usize>, // A melhor sequência de tarefas encontrada.
    pub best_makespan: u32,        // O makespan da melhor sequência encontrada.
}

impl MemeticAlgorithm {
    /// Cria uma nova instância do `MemeticAlgorithm`.
    pub fn new(
        instance: FSSPInstance,
        population_size: usize,
        generations: usize,
        mutation_rate: f64,
        local_search_rate: f64,
    ) -> Self {
        MemeticAlgorithm {
            instance,
            population_size,
            generations,
            mutation_rate,
            local_search_rate,
            population: Vec::new(),
            fitness: Vec::new(),
            best_sequence: Vec::new(),
            best_makespan: u32::MAX,
        }
    }

    /// Executa o Algoritmo Memético.
    pub fn run(&mut self, start_time: Instant, max_duration: Option<std::time::Duration>) {
        self._initialize_population(); // Inicializa a população.

        for gen in 0..self.generations {
            // Verifica se o tempo de execução excedeu o limite.
            if let Some(duration) = max_duration {
                if start_time.elapsed() > duration {
                    println!(
                        "\nLimite de tempo de {:.1?}s atingido. Encerrando...",
                        duration.as_secs_f32()
                    );
                    break;
                }
            }

            self._evaluate_fitness(); // Avalia a aptidão dos indivíduos.

            // Encontra o melhor indivíduo na geração atual.
            let (current_best_idx, current_best_fitness) = self
                .fitness
                .iter()
                .enumerate()
                .min_by_key(|&(_, f)| f)
                .unwrap();

            // Atualiza a melhor solução global encontrada.
            if *current_best_fitness < self.best_makespan {
                self.best_makespan = *current_best_fitness;
                self.best_sequence = self.population[current_best_idx].clone();
            }

            // Imprime o progresso a cada 20 gerações.
            if (gen + 1) % 20 == 0 {
                println!(
                    "Geração {}: Melhor Makespan = {}",
                    gen + 1,
                    self.best_makespan
                );
            }

            let parents_indices = self._selection_tournament(); // Seleção dos pais.
            let mut next_population = self._crossover(&parents_indices); // Cruzamento.
            self._mutation(&mut next_population); // Mutação.
            self._apply_local_search(&mut next_population); // Aplica busca local (memético).
            self._elitism(&mut next_population); // Aplica elitismo.

            self.population = next_population; // Atualiza a população.
        }
    }

    /// Inicializa a população com soluções gulosas e aleatórias.
    fn _initialize_population(&mut self) {
        self.population.clear();

        // Calcula o tempo total de processamento para cada tarefa.
        let mut job_metrics: Vec<(usize, u32)> = (0..self.instance.n_jobs)
            .map(|job_idx| {
                let total_time: u32 = self.instance.processing_times[job_idx].iter().sum();
                (job_idx, total_time)
            })
            .collect();

        // Adiciona a primeira solução gulosa (tempos ascendentes).
        if self.population_size > 0 {
            job_metrics.sort_by_key(|&(_, total_time)| total_time);
            let greedy_solution_asc: Vec<usize> =
                job_metrics.iter().map(|&(job_idx, _)| job_idx).collect();
            self.population.push(greedy_solution_asc);
        }

        // Adiciona a segunda solução gulosa (tempos descendentes).
        if self.population_size > 1 {
            let greedy_solution_desc: Vec<usize> = job_metrics
                .iter()
                .rev()
                .map(|&(job_idx, _)| job_idx)
                .collect();
            self.population.push(greedy_solution_desc);
        }

        // Preenche o restante da população com soluções aleatórias.
        let mut rng = rand::thread_rng();
        let num_random_to_generate = self.population_size.saturating_sub(self.population.len());

        for _ in 0..num_random_to_generate {
            let mut random_solution: Vec<usize> = (0..self.instance.n_jobs).collect();
            random_solution.shuffle(&mut rng);
            self.population.push(random_solution);
        }
    }

    /// Avalia o makespan (aptidão) de cada indivíduo na população.
    fn _evaluate_fitness(&mut self) {
        self.fitness = self
            .population
            .iter()
            .map(|seq| self.instance.calculate_makespan(seq))
            .collect();
    }

    /// Realiza a seleção por torneio para escolher os pais.
    fn _selection_tournament(&self) -> Vec<usize> {
        let mut parents = Vec::with_capacity(self.population_size);
        let mut rng = rand::thread_rng();
        let candidates: Vec<usize> = (0..self.population_size).collect();

        for _ in 0..self.population_size {
            // Seleciona 3 candidatos aleatórios para o torneio.
            let selected_indices = candidates
                .choose_multiple(&mut rng, 3)
                .cloned()
                .collect::<Vec<_>>();
            // O vencedor é o indivíduo com o menor makespan.
            let winner_index = *selected_indices
                .iter()
                .min_by_key(|&&idx| self.fitness[idx])
                .unwrap();
            parents.push(winner_index);
        }
        parents
    }

    /// Realiza o cruzamento (PMX) entre pares de pais para gerar filhos.
    fn _crossover(&self, parents: &[usize]) -> Vec<Vec<usize>> {
        let mut children = Vec::with_capacity(self.population_size);
        let mut rng = rand::thread_rng();

        for i in (0..self.population_size).step_by(2) {
            let p1_idx = parents[i];
            let p2_idx = if i + 1 < self.population_size {
                parents[i + 1]
            } else {
                parents[0]
            };

            let p1 = &self.population[p1_idx];
            let p2 = &self.population[p2_idx];

            let mut c1 = vec![usize::MAX; self.instance.n_jobs];
            let mut c2 = vec![usize::MAX; self.instance.n_jobs];

            // Define os pontos de corte para o cruzamento.
            let (start, end) = {
                let mut v = [
                    rng.gen_range(0..self.instance.n_jobs),
                    rng.gen_range(0..self.instance.n_jobs),
                ];
                v.sort_unstable();
                (v[0], v[1])
            };

            // Copia o segmento central dos pais para os filhos.
            c1[start..=end].copy_from_slice(&p1[start..=end]);
            c2[start..=end].copy_from_slice(&p2[start..=end]);

            // Preenche os restantes dos filhos com genes dos outros pais.
            let p2_rem: Vec<usize> = p2
                .iter()
                .copied()
                .filter(|&gene| !c1.contains(&gene))
                .collect();
            let p1_rem: Vec<usize> = p1
                .iter()
                .copied()
                .filter(|&gene| !c2.contains(&gene))
                .collect();

            let mut p2_iter = p2_rem.iter();
            let mut p1_iter = p1_rem.iter();

            for i_gene in 0..self.instance.n_jobs {
                if c1[i_gene] == usize::MAX {
                    c1[i_gene] = *p2_iter.next().unwrap();
                }
                if c2[i_gene] == usize::MAX {
                    c2[i_gene] = *p1_iter.next().unwrap();
                }
            }

            children.push(c1);
            if children.len() < self.population_size {
                children.push(c2);
            }
        }
        children
    }

    /// Aplica mutação por troca em indivíduos selecionados.
    fn _mutation(&self, population: &mut Vec<Vec<usize>>) {
        let mut rng = rand::thread_rng();
        for individual in population.iter_mut() {
            if rng.gen::<f64>() < self.mutation_rate {
                let i = rng.gen_range(0..self.instance.n_jobs);
                let j = rng.gen_range(0..self.instance.n_jobs);
                individual.swap(i, j); // Troca dois elementos aleatórios na sequência.
            }
        }
    }

    /// Aplica busca local (swap 2-opt) em indivíduos selecionados.
    fn _apply_local_search(&self, population: &mut Vec<Vec<usize>>) {
        let mut rng = rand::thread_rng();
        for individual in population.iter_mut() {
            if rng.gen::<f64>() < self.local_search_rate {
                self._local_search_swap(individual);
            }
        }
    }

    /// Realiza uma busca local 2-opt para otimizar uma sequência.
    fn _local_search_swap(&self, sequence: &mut Vec<usize>) {
        let mut current_makespan = self.instance.calculate_makespan(sequence);
        let mut improved = true;

        while improved {
            improved = false;
            for i in 0..self.instance.n_jobs {
                for j in (i + 1)..self.instance.n_jobs {
                    sequence.swap(i, j); // Tenta uma troca.
                    let new_makespan = self.instance.calculate_makespan(sequence);
                    if new_makespan < current_makespan {
                        current_makespan = new_makespan;
                        improved = true;
                    } else {
                        sequence.swap(i, j); // Desfaz a troca se não houver melhoria.
                    }
                }
            }
        }
    }

    /// Implementa o elitismo, preservando o melhor indivíduo da geração atual.
    fn _elitism(&mut self, next_population: &mut Vec<Vec<usize>>) {
        // Encontra o melhor indivíduo da população atual.
        let best_current_idx = self
            .fitness
            .iter()
            .enumerate()
            .min_by_key(|&(_, &f)| f)
            .unwrap()
            .0;

        let elite_individual = self.population[best_current_idx].clone();

        // Encontra o pior indivíduo na próxima população.
        let mut worst_idx = 0;
        let mut max_makespan = u32::MIN;

        for (idx, seq) in next_population.iter().enumerate() {
            let makespan = self.instance.calculate_makespan(seq);
            if makespan > max_makespan {
                max_makespan = makespan;
                worst_idx = idx;
            }
        }
        // Substitui o pior indivíduo pelo elite.
        next_population[worst_idx] = elite_individual;
    }
}
