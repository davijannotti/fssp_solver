use crate::fssp_core::FSSPInstance;
use rand::seq::SliceRandom;
use rand::Rng;

pub struct MemeticAlgorithm {
    instance: FSSPInstance,
    population_size: usize,
    generations: usize,
    mutation_rate: f64,
    local_search_rate: f64,
    population: Vec<Vec<usize>>,
    fitness: Vec<u32>,
    pub best_sequence: Vec<usize>,
    pub best_makespan: u32,
}

impl MemeticAlgorithm {
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

    pub fn run(&mut self) {
        self._initialize_population();

        for gen in 0..self.generations {
            self._evaluate_fitness();

            let (current_best_idx, current_best_fitness) = self
                .fitness
                .iter()
                .enumerate()
                .min_by_key(|&(_, f)| f)
                .map(|(idx, &f)| (idx, f))
                .unwrap();

            if current_best_fitness < self.best_makespan {
                self.best_makespan = current_best_fitness;
                self.best_sequence = self.population[current_best_idx].clone();
            }

            if (gen + 1) % 20 == 0 {
                println!(
                    "Geração {}: Melhor Makespan = {}",
                    gen + 1,
                    self.best_makespan
                );
            }

            let parents_indices = self._selection_tournament();
            let mut next_population = self._crossover(&parents_indices);
            self._mutation(&mut next_population);
            self._apply_local_search(&mut next_population);
            self._elitism(&mut next_population);

            self.population = next_population;
        }
    }

    fn _initialize_population(&mut self) {
        // Garante que a população esteja vazia antes de começar.
        self.population.clear();

        // 1. Calcula a soma dos tempos de processamento para cada tarefa (job).
        let mut job_metrics: Vec<(usize, u32)> = (0..self.instance.n_jobs)
            .map(|job_idx| {
                let total_time: u32 = self.instance.processing_times[job_idx].iter().sum();
                (job_idx, total_time)
            })
            .collect();

        // 2. Cria a primeira solução gulosa (ordenada do menor para o maior tempo total).
        // Esta verificação garante que não haja pânico se o tamanho da população for 0.
        if self.population_size > 0 {
            // Ordena as tarefas pela soma de seus tempos.
            job_metrics.sort_by_key(|&(_, total_time)| total_time);

            let greedy_solution_asc: Vec<usize> =
                job_metrics.iter().map(|&(job_idx, _)| job_idx).collect();
            self.population.push(greedy_solution_asc);
        }

        // 3. Cria a segunda solução gulosa (ordenada do maior para o menor tempo total).
        // Esta verificação garante que não haja pânico se o tamanho da população for < 2.
        if self.population_size > 1 {
            let greedy_solution_desc: Vec<usize> = job_metrics
                .iter()
                .rev() // Apenas inverte a lista já ordenada. É mais eficiente.
                .map(|&(job_idx, _)| job_idx)
                .collect();
            self.population.push(greedy_solution_desc);
        }

        // 4. Preenche o resto da população com soluções aleatórias.
        let mut rng = rand::thread_rng();
        // Calcula quantos indivíduos aleatórios ainda precisam ser gerados.
        // `saturating_sub` evita pânico se self.population_size for < 2.
        let num_random_to_generate = self.population_size.saturating_sub(self.population.len());

        for _ in 0..num_random_to_generate {
            let mut random_solution: Vec<usize> = (0..self.instance.n_jobs).collect();
            random_solution.shuffle(&mut rng);
            self.population.push(random_solution);
        }
    }

    fn _evaluate_fitness(&mut self) {
        self.fitness = self
            .population
            .iter()
            .map(|seq| (&self.instance).calculate_makespan(seq))
            .collect();
    }

    fn _selection_tournament(&self) -> Vec<usize> {
        let mut parents = Vec::with_capacity(self.population_size);
        let mut rng = rand::thread_rng();
        for _ in 0..self.population_size {
            let candidates: Vec<usize> = (0..self.population_size).collect();
            let selected_indices = candidates
                .choose_multiple(&mut rng, 3)
                .cloned()
                .collect::<Vec<_>>();
            let winner_index = *selected_indices
                .iter()
                .min_by_key(|&&idx| self.fitness[idx])
                .unwrap();
            parents.push(winner_index);
        }
        parents
    }

    fn _crossover(&self, parents: &[usize]) -> Vec<Vec<usize>> {
        let mut children = Vec::with_capacity(self.population_size);
        for i in (0..self.population_size).step_by(2) {
            let p1_idx = parents[i];
            let p2_idx = if i + 1 < self.population_size {
                parents[i + 1]
            } else {
                parents[0]
            };

            let p1 = &self.population[p1_idx];
            let p2 = &self.population[p2_idx];

            let mut rng = rand::thread_rng();
            let mut c1 = vec![usize::MAX; self.instance.n_jobs];
            let mut c2 = vec![usize::MAX; self.instance.n_jobs];

            let (start, end) = {
                let mut v = [
                    rng.gen_range(0..self.instance.n_jobs),
                    rng.gen_range(0..self.instance.n_jobs),
                ];
                v.sort();
                (v[0], v[1])
            };

            c1[start..=end].copy_from_slice(&p1[start..=end]);
            c2[start..=end].copy_from_slice(&p2[start..=end]);

            let p2_rem: Vec<_> = p2.iter().filter(|&gene| !c1.contains(gene)).collect();
            let p1_rem: Vec<_> = p1.iter().filter(|&gene| !c2.contains(gene)).collect();

            let mut p2_iter = p2_rem.iter();
            let mut p1_iter = p1_rem.iter();

            for i in 0..self.instance.n_jobs {
                if c1[i] == usize::MAX {
                    c1[i] = **p2_iter.next().unwrap();
                }
                if c2[i] == usize::MAX {
                    c2[i] = **p1_iter.next().unwrap();
                }
            }

            children.push(c1);
            if children.len() < self.population_size {
                children.push(c2);
            }
        }
        children
    }

    fn _mutation(&self, population: &mut Vec<Vec<usize>>) {
        let mut rng = rand::thread_rng();
        for individual in population.iter_mut() {
            if rng.gen::<f64>() < self.mutation_rate {
                let (i, j) = (
                    rng.gen_range(0..self.instance.n_jobs),
                    rng.gen_range(0..self.instance.n_jobs),
                );
                individual.swap(i, j);
            }
        }
    }

    fn _apply_local_search(&self, population: &mut Vec<Vec<usize>>) {
        let mut rng = rand::thread_rng();
        for individual in population.iter_mut() {
            if rng.gen::<f64>() < self.local_search_rate {
                self._local_search_swap(individual);
            }
        }
    }

    fn _local_search_swap(&self, sequence: &mut Vec<usize>) {
        let mut current_makespan = (&self.instance).calculate_makespan(sequence);
        let mut improved = true;
        while improved {
            improved = false;
            for i in 0..self.instance.n_jobs {
                for j in (i + 1)..self.instance.n_jobs {
                    sequence.swap(i, j);
                    let new_makespan = (&self.instance).calculate_makespan(sequence);
                    if new_makespan < current_makespan {
                        current_makespan = new_makespan;
                        improved = true;
                    } else {
                        sequence.swap(i, j); // Desfaz a troca
                    }
                }
            }
        }
    }

    fn _elitism(&mut self, next_population: &mut Vec<Vec<usize>>) {
        // Encontra o índice do melhor indivíduo na população atual.
        let best_current_idx = self
            .fitness
            .iter()
            .position(|&f| f == *self.fitness.iter().min().unwrap())
            .unwrap();

        let elite_individual = self.population[best_current_idx].clone();

        let fitness_next_pop: Vec<u32> = next_population
            .iter()
            .map(|seq| (&self.instance).calculate_makespan(seq))
            .collect();

        let worst_idx = fitness_next_pop
            .iter()
            .position(|&f| f == *fitness_next_pop.iter().max().unwrap())
            .unwrap();

        next_population[worst_idx] = elite_individual;
    }
}
