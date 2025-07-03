
import random
import numpy as np
from .core import calculate_makespan

class BatAlgorithmFSSP:
    """
    Solves the Flow Shop Scheduling Problem (FSSP) using the Bat Algorithm.

    This implementation adapts the standard Bat Algorithm, which operates on continuous
    spaces, to the permutation-based nature of the FSSP. It uses a ranking-based
    encoding where the "position" of a bat is a vector of continuous values
    representing the priority of each job. The job sequence is obtained by sorting
    these priorities.

    Attributes:
        processing_times (np.ndarray): Matrix of processing times (jobs x machines).
        n_bats (int): The number of bats in the population.
        n_generations (int): The number of iterations to run the algorithm.
        loudness_initial (float): Initial loudness value (A).
        pulse_rate_initial (float): Initial pulse emission rate (r).
        f_min (float): Minimum frequency for the bats' search.
        f_max (float): Maximum frequency for the bats' search.
    """
    def __init__(self, processing_times, n_bats, n_generations,
                 loudness_initial, pulse_rate_initial, f_min, f_max):
        self.processing_times = processing_times
        self.n_jobs = processing_times.shape[0]
        self.n_bats = n_bats
        self.n_generations = n_generations
        self.loudness_initial = loudness_initial
        self.pulse_rate_initial = pulse_rate_initial
        self.f_min = f_min
        self.f_max = f_max

        # Initialize bat population
        self.positions = np.random.rand(self.n_bats, self.n_jobs)
        self.velocities = np.zeros((self.n_bats, self.n_jobs))
        self.frequencies = np.zeros(self.n_bats)
        self.loudness = np.full(self.n_bats, self.loudness_initial)
        self.pulse_rates = np.full(self.n_bats, self.pulse_rate_initial)

        # Best solution found so far
        self.best_sequence = None
        self.best_makespan = float('inf')
        self.best_position = None

    def _decode_position(self, position):
        """
        Decodes a continuous position vector into a job sequence permutation.
        The sequence is determined by the order of the indices of the sorted
        priority values in the position vector.
        """
        return np.argsort(position).tolist()

    def run(self, track_convergence=False, verbose=True):
        """
        Executes the main loop of the Bat Algorithm.
        """
        convergence_history = []

        # Evaluate initial population
        for i in range(self.n_bats):
            sequence = self._decode_position(self.positions[i])
            makespan = calculate_makespan(self.processing_times, sequence)
            if makespan < self.best_makespan:
                self.best_makespan = makespan
                self.best_sequence = sequence
                self.best_position = self.positions[i]

        for t in range(self.n_generations):
            for i in range(self.n_bats):
                # Update frequency, velocity, and position
                self.frequencies[i] = self.f_min + (self.f_max - self.f_min) * np.random.rand()
                self.velocities[i] += (self.positions[i] - self.best_position) * self.frequencies[i]
                new_position = self.positions[i] + self.velocities[i]

                # Local search phase
                if np.random.rand() > self.pulse_rates[i]:
                    # Generate a new solution around the best solution
                    new_position = self.best_position + 0.01 * np.random.randn(self.n_jobs)

                # Decode and evaluate the new position
                new_sequence = self._decode_position(new_position)
                new_makespan = calculate_makespan(self.processing_times, new_sequence)

                # Acceptance of the new solution
                if new_makespan < self.best_makespan and np.random.rand() < self.loudness[i]:
                    self.positions[i] = new_position
                    self.loudness[i] *= 0.9  # Alpha (loudness reduction factor)
                    self.pulse_rates[i] = self.pulse_rate_initial * (1 - np.exp(-0.1 * t)) # Gamma (pulse rate increase factor)

                    # Update the global best solution
                    self.best_makespan = new_makespan
                    self.best_sequence = new_sequence
                    self.best_position = new_position

            if track_convergence:
                convergence_history.append(self.best_makespan)

            if verbose and (t + 1) % 10 == 0:
                print(f"Generation {t + 1}/{self.n_generations} | Best Makespan: {self.best_makespan}")

        if track_convergence:
            return self.best_sequence, self.best_makespan, convergence_history
        else:
            return self.best_sequence, self.best_makespan

if __name__ == '__main__':
    from .core import load_instance

    # Load an instance
    instance_filepath = '../instances/example.txt'
    processing_times = load_instance(instance_filepath)

    print("FSSP Instance Loaded.")
    print(f"Jobs: {processing_times.shape[0]}, Machines: {processing_times.shape[1]}")
    print("-" * 30)

    # Bat Algorithm parameters
    params = {
        'n_bats': 20,
        'n_generations': 100,
        'loudness_initial': 0.95,
        'pulse_rate_initial': 0.5,
        'f_min': 0.0,
        'f_max': 2.0
    }

    print("Starting Bat Algorithm...")
    print(f"Parameters: {params}")
    print("-" * 30)

    # Create and run the Bat Algorithm
    ba = BatAlgorithmFSSP(processing_times, **params)
    best_sequence, best_makespan = ba.run()

    print("-" * 30)
    print("Bat Algorithm execution finished.")
    print(f"Best sequence found: {best_sequence}")
    print(f"Best Makespan (C_max): {best_makespan}")
