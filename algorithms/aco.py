import random
import numpy as np
from .core import calculate_makespan

class AntColonyOptimizer:
    def __init__(self, processing_times, n_ants, n_generations, alpha, beta, evaporation_rate, q0):
        self.processing_times = processing_times
        self.n_jobs = processing_times.shape[0]
        self.n_machines = processing_times.shape[1]
        self.n_ants = n_ants
        self.n_generations = n_generations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q0 = q0

        # Initialize pheromone trails
        self.pheromone = np.ones((self.n_jobs, self.n_jobs))

        # Heuristic information (using a simple heuristic for now)
        self.heuristic = self._initialize_heuristic()

        self.best_sequence = None
        self.best_makespan = float('inf')
        self.convergence_history = []

    def _initialize_heuristic(self):
        """
        Initializes the heuristic information matrix.
        A simple heuristic: 1 / (average processing time of a job).
        """
        avg_processing_time = np.mean(self.processing_times, axis=1)
        heuristic = np.zeros((self.n_jobs, self.n_jobs))
        for i in range(self.n_jobs):
            for j in range(self.n_jobs):
                if i != j:
                    heuristic[i, j] = 1.0 / avg_processing_time[j]
        return heuristic

    def run(self, track_convergence=False, verbose=False):
        elite_solutions = []

        for generation in range(self.n_generations):
            all_ant_solutions = []

            for _ in range(self.n_ants):
                sequence = self._construct_solution()
                makespan = calculate_makespan(self.processing_times, sequence)
                all_ant_solutions.append((sequence, makespan))

                if makespan < self.best_makespan:
                    self.best_makespan = makespan
                    self.best_sequence = sequence

            # Update elite solutions
            # Sort all ant solutions by makespan and add the top ones to elites
            all_ant_solutions.sort(key=lambda x: x[1])
            for sol in all_ant_solutions:
                if sol not in elite_solutions:
                    elite_solutions.append(sol)

            # Keep the list of elites sorted and trimmed
            elite_solutions.sort(key=lambda x: x[1])
            elite_solutions = elite_solutions[:self.n_ants] # Keep a number of elites equal to n_ants

            self._update_pheromone([sol[0] for sol in all_ant_solutions], [sol[1] for sol in all_ant_solutions])

            if track_convergence:
                self.convergence_history.append(self.best_makespan)

            if verbose and (generation + 1) % 10 == 0:
                print(f"Geração {generation + 1}/{self.n_generations} | Melhor Makespan: {self.best_makespan}")

        return elite_solutions

    def _construct_solution(self):
        sequence = []
        remaining_jobs = list(range(self.n_jobs))

        # Start with a random job
        current_job = random.choice(remaining_jobs)
        sequence.append(current_job)
        remaining_jobs.remove(current_job)

        while remaining_jobs:
            next_job = self._select_next_job(current_job, remaining_jobs)
            sequence.append(next_job)
            remaining_jobs.remove(next_job)
            current_job = next_job

        return sequence

    def _select_next_job(self, current_job, remaining_jobs):
        probabilities = []

        # Calculate probabilities for all remaining jobs
        for job in remaining_jobs:
            pheromone_val = self.pheromone[current_job, job] ** self.alpha
            heuristic_val = self.heuristic[current_job, job] ** self.beta
            probabilities.append(pheromone_val * heuristic_val)

        # Normalize probabilities
        sum_probs = sum(probabilities)
        if sum_probs == 0: # Avoid division by zero
            probabilities = [1.0 / len(remaining_jobs)] * len(remaining_jobs)
        else:
            probabilities = [p / sum_probs for p in probabilities]

        # ACO selection logic (with exploitation/exploration)
        if random.random() < self.q0:
            # Exploitation: choose the best next job
            max_prob_index = np.argmax(probabilities)
            next_job = remaining_jobs[max_prob_index]
        else:
            # Exploration: choose based on probability distribution
            next_job = random.choices(remaining_jobs, weights=probabilities, k=1)[0]

        return next_job

    def _update_pheromone(self, ant_sequences, ant_makespans):
        # Evaporation
        self.pheromone *= (1 - self.evaporation_rate)

        # Reinforcement
        for sequence, makespan in zip(ant_sequences, ant_makespans):
            # Add pheromone based on the quality of the solution
            pheromone_deposit = 1.0 / makespan
            for i in range(self.n_jobs - 1):
                job1 = sequence[i]
                job2 = sequence[i+1]
                self.pheromone[job1, job2] += pheromone_deposit
