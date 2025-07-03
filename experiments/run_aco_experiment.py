
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.core import load_instance
from algorithms.aco import AntColonyOptimizer

def main():
    """
    Script to run an ACO experiment for the FSSP.
    """
    # 1. Load the FSSP instance
    instance_filepath = './instances/fssp_instance_05.txt'
    print(f"Loading instance from: {instance_filepath}")
    processing_times = load_instance(instance_filepath)
    print(f"Instance loaded: {processing_times.shape[0]} jobs, {processing_times.shape[1]} machines.")
    print("-" * 40)

    # 2. Define ACO parameters
    aco_params = {
        'n_ants': 20,
        'n_generations': 100,
        'alpha': 1.0,
        'beta': 2.0,
        'evaporation_rate': 0.1,
        'q0': 0.9, # Exploitation factor
    }
    print("ACO Parameters:")
    for key, value in aco_params.items():
        print(f"  - {key.replace('_', ' ').title()}: {value}")
    print("-" * 40)

    # 3. Instantiate and run the algorithm
    print("Starting ACO execution...")
    start_time = time.time()

    aco_solver = AntColonyOptimizer(processing_times, **aco_params)
    best_sequence, best_makespan = aco_solver.run(verbose=True)

    end_time = time.time()
    total_time = end_time - start_time
    print("Execution finished.")
    print("-" * 40)

    # 4. Print the results
    print("Results:")
    print(f"  - Best sequence found: {best_sequence}")
    print(f"  - Best Makespan (C_max): {best_makespan}")
    print(f"  - Total execution time: {total_time:.4f} seconds")

if __name__ == "__main__":
    main()
