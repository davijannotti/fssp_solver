

import numpy as np
from .aco import AntColonyOptimizer
from .bat_algorithm import BatAlgorithmFSSP
from .core import load_instance

class HybridAcoBatFSSP:
    """
    A hybrid algorithm that combines Ant Colony Optimization (ACO) and the Bat Algorithm (BA)
    to solve the Flow Shop Scheduling Problem (FSSP).

    The algorithm first runs ACO to find a set of promising solutions (elites) and then
    uses these elites to seed the initial population of the Bat Algorithm, which refines
    the search.
    """
    def __init__(self, processing_times, aco_params, bat_params, num_elites_for_ba):
        self.processing_times = processing_times
        self.aco_params = aco_params
        self.bat_params = bat_params
        self.num_elites_for_ba = num_elites_for_ba

    def _encode_sequence_to_position(self, sequence):
        """
        Encodes a job sequence into a continuous priority vector for the Bat Algorithm.
        The job that appears earlier in the sequence gets a higher priority value.
        """
        n_jobs = len(sequence)
        position = np.zeros(n_jobs)
        priorities = np.linspace(1.0, 0.1, n_jobs) # Higher priority for earlier jobs
        for i, job_index in enumerate(sequence):
            position[job_index] = priorities[i]
        return position

    def run(self, verbose=False, track_convergence=False):
        """
        Executes the hybrid ACO-BA algorithm.
        """
        # Phase 1: Run Ant Colony Optimization
        if verbose:
            print("--- Phase 1: Running Ant Colony Optimization ---")
        aco = AntColonyOptimizer(self.processing_times, **self.aco_params)
        aco_elite_solutions = aco.run()

        # Select the top unique elite sequences from ACO
        elite_sequences = []
        seen_sequences = set()
        for seq, _ in sorted(aco_elite_solutions, key=lambda x: x[1]):
            if tuple(seq) not in seen_sequences:
                elite_sequences.append(seq)
                seen_sequences.add(tuple(seq))
            if len(elite_sequences) >= self.num_elites_for_ba:
                break

        if verbose:
            print(f"ACO found {len(elite_sequences)} unique elite solutions to seed the Bat Algorithm.")
            print("ACO Best Makespan:", aco.best_makespan)
            print("-" * 50)

        # Phase 2: Run Bat Algorithm
        if verbose:
            print("--- Phase 2: Running Bat Algorithm ---")
        ba = BatAlgorithmFSSP(self.processing_times, **self.bat_params)

        # Initialize Bat Algorithm population with ACO elites
        for i in range(len(elite_sequences)):
            if i < ba.n_bats:
                ba.positions[i] = self._encode_sequence_to_position(elite_sequences[i])

        # Run the Bat Algorithm to refine the solutions
        results = ba.run(verbose=verbose, track_convergence=track_convergence)

        if track_convergence:
            best_sequence, best_makespan, history = results
            if verbose:
                print("-" * 50)
                print("Hybrid ACO-BA execution finished.")
            return best_sequence, best_makespan, history
        else:
            best_sequence, best_makespan = results
            if verbose:
                print("-" * 50)
                print("Hybrid ACO-BA execution finished.")
            return best_sequence, best_makespan

if __name__ == '__main__':
    # Load an instance
    instance_filepath = '../instances/example.txt'
    processing_times = load_instance(instance_filepath)

    # Parameters for the hybrid algorithm
    aco_parameters = {
        'n_ants': 10,
        'n_generations': 20, # Generations for ACO phase
        'alpha': 1.0,
        'beta': 2.0,
        'evaporation_rate': 0.5,
        'q0': 0.9
    }

    bat_parameters = {
        'n_bats': 20,
        'n_generations': 80, # Generations for BA phase
        'loudness_initial': 0.95,
        'pulse_rate_initial': 0.5,
        'f_min': 0.0,
        'f_max': 2.0
    }

    hybrid_params = {
        'num_elites_for_ba': 5 # Number of ACO elites to seed BA
    }

    print("Starting Hybrid ACO-BA FSSP Solver...")
    print("-" * 50)

    # Create and run the hybrid solver
    hybrid_solver = HybridAcoBatFSSP(
        processing_times,
        aco_params=aco_parameters,
        bat_params=bat_parameters,
        **hybrid_params
    )

    best_sequence, best_makespan = hybrid_solver.run()

    print(f"\nFinal Best Sequence: {best_sequence}")
    print(f"Final Best Makespan (C_max): {best_makespan}")
