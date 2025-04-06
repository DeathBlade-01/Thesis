#!/usr/bin/env python3

import random
import time
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

class HEFT:
    """Heterogeneous Earliest Finish Time (HEFT) algorithm implementation"""
    
    def __init__(self, dag, execution_times, communication_times, num_tasks, computing_units):
        """Initialize the HEFT algorithm with the given parameters"""
        self.dag = dag  # Workflow DAG - adjacency list format
        self.execution_times = execution_times  # Matrix of task execution times on diff units
        self.communication_times = communication_times  # Matrix of communication times between tasks
        self.num_tasks = num_tasks  # Number of tasks
        self.computing_units = computing_units  # List of computing units
        self.num_units = len(computing_units)
        
        # For storing results
        self.task_schedule = []  # List of tasks in scheduled order
        self.unit_schedule = []  # List of units corresponding to tasks
        self.start_times = [0] * num_tasks  # Start time for each task
        self.finish_times = [0] * num_tasks  # Finish time for each task
    
    def get_successors(self, task):
        """Get successors of a task from the DAG"""
        return self.dag.get(task, [])
    
    def get_predecessors(self, task):
        """Get predecessors of a task from the DAG"""
        predecessors = []
        for potential_pred in range(self.num_tasks):
            if task in self.get_successors(potential_pred):
                predecessors.append(potential_pred)
        return predecessors
    
    def calculate_upward_rank(self):
        """
        Calculate upward rank for all tasks
        Upward rank is defined recursively as:
        rank_u(i) = w_i + max_j∈succ(i) {c_i,j + rank_u(j)}
        where w_i is the average execution time of task i
        and c_i,j is the average communication time between tasks i and j
        """
        ranks = [0] * self.num_tasks
        
        # Calculate average execution time for each task
        avg_execution = [sum(self.execution_times[i]) / self.num_units for i in range(self.num_tasks)]
        
        # Calculate ranks in reverse topological order (from exit tasks to entry tasks)
        def calculate_rank(task):
            if ranks[task] != 0:  # Already calculated
                return ranks[task]
            
            successors = self.get_successors(task)
            if not successors:  # Exit task
                ranks[task] = avg_execution[task]
                return ranks[task]
            
            max_successor_rank = 0
            for succ in successors:
                # Average communication cost
                comm_cost = self.communication_times[task][succ]
                
                # Recursive calculation of rank for successor
                succ_rank = calculate_rank(succ)
                
                # Update max successor rank
                if comm_cost + succ_rank > max_successor_rank:
                    max_successor_rank = comm_cost + succ_rank
            
            ranks[task] = avg_execution[task] + max_successor_rank
            return ranks[task]
        
        # Calculate ranks for all tasks
        for task in range(self.num_tasks):
            if ranks[task] == 0:  # Not calculated yet
                calculate_rank(task)
        
        return ranks
    
    def schedule_tasks(self):
        """
        Schedule tasks according to HEFT algorithm
        1. Calculate upward ranks for all tasks
        2. Sort tasks by decreasing upward rank
        3. For each task in the sorted list, assign it to the processor that minimizes the finish time
        """
        # Calculate upward ranks
        ranks = self.calculate_upward_rank()
        
        # Sort tasks by decreasing rank
        sorted_tasks = sorted(range(self.num_tasks), key=lambda x: ranks[x], reverse=True)
        
        # Initialize processor available times
        processor_available = [0] * self.num_units
        
        # Schedule each task
        for task in sorted_tasks:
            # Find best processor for this task
            earliest_finish_time = float('inf')
            best_processor = 0
            task_start_time = 0
            
            for processor in range(self.num_units):
                # Calculate earliest start time on this processor
                # (considering predecessor tasks and communication costs)
                ready_time = 0
                for pred in self.get_predecessors(task):
                    pred_processor = self.unit_schedule[self.task_schedule.index(pred)]
                    
                    # If predecessor is on a different processor, add communication cost
                    comm_cost = self.communication_times[pred][task] if pred_processor != processor else 0
                    
                    # Task can start after predecessor finishes + communication
                    pred_finish_with_comm = self.finish_times[pred] + comm_cost
                    ready_time = max(ready_time, pred_finish_with_comm)
                
                # Task can start after the processor becomes available
                possible_start_time = max(ready_time, processor_available[processor])
                
                # Calculate finish time
                execution_time = self.execution_times[task][processor]
                finish_time = possible_start_time + execution_time
                
                # Update best processor if this one provides earlier finish time
                if finish_time < earliest_finish_time:
                    earliest_finish_time = finish_time
                    best_processor = processor
                    task_start_time = possible_start_time
            
            # Schedule task on best processor
            self.task_schedule.append(task)
            self.unit_schedule.append(best_processor)
            self.start_times[task] = task_start_time
            self.finish_times[task] = earliest_finish_time
            
            # Update processor available time
            processor_available[best_processor] = earliest_finish_time
    
    def run(self):
        """Execute the HEFT algorithm"""
        start_time = time.time()
        
        # Schedule tasks
        self.schedule_tasks()
        
        # Calculate makespan
        makespan = max(self.finish_times)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"HEFT completed in {execution_time:.2f} seconds")
        print(f"Makespan: {makespan}")
        print(f"Schedule: {self.task_schedule}")
        print(f"Processor assignments: {self.unit_schedule}")
        
        return {
            "tasks": self.task_schedule,
            "units": self.unit_schedule,
            "makespan": makespan,
            "start_times": self.start_times,
            "finish_times": self.finish_times
        }
    
    def generate_heft_chromosome(self, r2ga_instance):
        """
        Generate a chromosome compatible with R2GA based on the HEFT schedule
        This can be used to seed the R2GA population with a good initial solution
        """
        chromosome = [0] * (2 * self.num_tasks)
        
        # Create a map of task->position in the HEFT schedule
        task_positions = {self.task_schedule[i]: i for i in range(self.num_tasks)}
        
        # Set task selection genes (first half of chromosome)
        # Lower values get higher priority in R2GA's decode method
        for task in range(self.num_tasks):
            position = task_positions[task]
            # Normalize position to 0-1 range, lower position = lower gene value = higher priority
            chromosome[task] = position / self.num_tasks
        
        # Set processor selection genes (second half of chromosome)
        for task in range(self.num_tasks):
            position = task_positions[task]
            assigned_unit = self.unit_schedule[position]
            # Set gene to prioritize the assigned processor
            chromosome[self.num_tasks + task] = assigned_unit / self.num_units
        
        return chromosome


class R2GA:
    def __init__(self, dag, execution_times, communication_times, num_tasks, computing_units, 
                 pop_size=1000, max_iterations=5000, elite_size=100, mutation_prob=0.3,
                 crossover_prob=0.8):
        """Initialize the R2GA algorithm with the given parameters"""
        self.dag = dag  # Workflow DAG - adjacency list format
        self.execution_times = execution_times  # Matrix of task execution times on diff units
        self.communication_times = communication_times  # Matrix of communication times between tasks
        self.num_tasks = num_tasks  # Number of tasks
        self.computing_units = computing_units  # List of computing units
        self.num_units = len(computing_units)
        
        # GA parameters
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.elite_size = elite_size
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        
        # Initialize population
        self.population = self.initialize(self.pop_size, self.num_tasks, self.computing_units)
        
        # Keep track of best solutions
        self.best_solutions = []
        self.best_fitness_history = []
        
    def encode(self, n):
        """
        Algorithm 1: Encoding Algorithm
        Generates a chromosome of length 2n with random values between 0 and 1
        """
        chromosome = []
        for i in range(2 * n):
            g = random.random()  # Generate random number between 0 and 1
            chromosome.append(g)
        return chromosome
    
    def decode(self, chromosome):
        """
        Algorithm 2: Decoding Algorithm
        Decodes a chromosome into a solution (task allocation and scheduling)
        """
        # Initialize candidate task set (CTS), task list (TL), and unit list (UL)
        cts = self.get_entry_tasks()  # Tasks with no predecessors
        task_list = []
        unit_list = []
        
        # Schedule tasks one by one
        for i in range(self.num_tasks):
            # Sort tasks in CTS by their IDs
            cts.sort()
            
            # Get the task and unit genes
            t_gene = chromosome[i]
            u_gene = chromosome[self.num_tasks + i]
            
            # Select task and computing unit
            t_index = int(t_gene * len(cts))
            if t_index >= len(cts):  # Bound check
                t_index = len(cts) - 1
            t = cts[t_index]
            
            u_index = int(u_gene * self.num_units)
            if u_index >= self.num_units:  # Bound check
                u_index = self.num_units - 1
            u = self.computing_units[u_index]
            
            # Add to task and unit lists
            task_list.append(t)
            unit_list.append(u)
            
            # Remove task from CTS
            cts.remove(t)
            
            # Add successor tasks to CTS if they are ready (all predecessors completed)
            for successor in self.get_successors(t):
                if self.is_candidate_task(successor, task_list):
                    cts.append(successor)
        
        return {"tasks": task_list, "units": unit_list}
    
    def get_entry_tasks(self):
        """Get tasks with no predecessors"""
        entry_tasks = []
        for task in range(self.num_tasks):
            if not any(task in self.get_successors(pred) for pred in range(self.num_tasks)):
                entry_tasks.append(task)
        return entry_tasks
    
    def get_successors(self, task):
        """Get successors of a task from the DAG"""
        return self.dag.get(task, [])
    
    def is_candidate_task(self, task, scheduled_tasks):
        """Check if a task is a candidate (all predecessors are scheduled)"""
        for predecessor in range(self.num_tasks):
            if task in self.get_successors(predecessor) and predecessor not in scheduled_tasks:
                return False
        return True
    
    def initialize(self, pop_size, n, computing_units):
        """
        Algorithm 3: Initialize Population
        Creates an initial population of pop_size chromosomes
        """
        population = []
        
        # Generate random chromosomes
        for i in range(pop_size - 1):
            chromosome = self.encode(n)
            population.append(chromosome)
        
        # Add a HEFT-seeded chromosome
        heft = HEFT(self.dag, self.execution_times, self.communication_times, self.num_tasks, self.computing_units)
        heft.run()
        heft_chromosome = heft.generate_heft_chromosome(self)
        population.append(heft_chromosome)
        
        return population
    
    def select(self, population, elite_size):
        """
        Algorithm 4: Selection Operator
        Selects elite_size best chromosomes from the population
        """
        # Evaluate fitness of each chromosome
        fitness_values = []
        for chromosome in population:
            makespan = self.calculate_makespan(chromosome)
            fitness_values.append(makespan)
        
        # Sort chromosomes by fitness (makespan) - lower is better
        sorted_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i])
        
        # Select elite chromosomes
        elites = []
        for i in range(elite_size):
            elites.append(deepcopy(population[sorted_indices[i]]))
        
        # Return the best fitness for tracking
        best_fitness = fitness_values[sorted_indices[0]]
        self.best_fitness_history.append(best_fitness)
        
        return elites, sorted_indices, best_fitness
    
    def crossover(self, population, sorted_indices, n, elite_size, pop_size):
        """
        Algorithm 5: Crossover Operator
        Performs crossover on the population to create new chromosomes
        """
        crossovers = []
        
        # Perform crossover for non-elite chromosomes
        for i in range(0, pop_size - elite_size, 2):
            if i + 1 < pop_size - elite_size:
                # Select parents using tournament selection
                parent1_idx = sorted_indices[i]
                parent2_idx = sorted_indices[i + 1]
                parent1 = deepcopy(population[parent1_idx])
                parent2 = deepcopy(population[parent2_idx])
                
                # Crossover with probability crossover_prob
                if random.random() < self.crossover_prob:
                    # Select crossover points
                    pcrossover1 = random.randint(0, n - 1)
                    pcrossover2 = random.randint(n, 2 * n - 1)
                    
                    # Swap genes between crossover points
                    for j in range(pcrossover1, pcrossover2 + 1):
                        parent1[j], parent2[j] = parent2[j], parent1[j]
                
                crossovers.append(parent1)
                crossovers.append(parent2)
        
        return crossovers
    
    def mutate(self, crossovers, n, elites, pop_size, elite_size):
        """
        Algorithm 6: Mutation Operator
        Performs mutation on the crossover population
        """
        mutants = []
        
        # Perform mutation for each chromosome in crossovers
        for i in range(len(crossovers)):
            chromosome = deepcopy(crossovers[i])
            
            # Mutate with probability mutation_prob
            if random.random() < self.mutation_prob:
                # Select random positions for task and unit genes
                ptask = random.randint(0, n - 1)
                punit = random.randint(n, 2 * n - 1)
                
                # Mutate genes
                chromosome[ptask] = random.random()
                chromosome[punit] = random.random()
            
            mutants.append(chromosome)
        
        # Create new population by combining elites and mutants
        population = elites + mutants
        
        # If population size is less than pop_size, add random chromosomes
        while len(population) < pop_size:
            population.append(self.encode(n))
        
        return population
    
    def calculate_makespan(self, chromosome):
        """
        Calculate the makespan (total execution time) for a given chromosome
        This considers task dependencies, execution times, and communication times
        """
        solution = self.decode(chromosome)
        task_list = solution["tasks"]
        unit_list = solution["units"]
        
        # Initialize finish times for each task
        finish_times = [0] * self.num_tasks
        
        # Initialize available times for each computing unit
        unit_available_times = [0] * self.num_units
        
        # Calculate start and finish time for each task
        for i in range(len(task_list)):
            task = task_list[i]
            unit = unit_list[i]
            unit_idx = self.computing_units.index(unit)
            
            # Earliest time the task can start (after its predecessors complete)
            earliest_start = 0
            for j in range(i):
                pred_task = task_list[j]
                pred_unit = unit_list[j]
                pred_unit_idx = self.computing_units.index(pred_unit)
                
                if pred_task in self.get_predecessors_of(task):
                    # Include communication time if tasks are on different units
                    comm_time = 0
                    if pred_unit != unit:
                        comm_time = self.communication_times[pred_task][task]
                    
                    # Task can start after predecessor finishes + communication
                    potential_start = finish_times[pred_task] + comm_time
                    earliest_start = max(earliest_start, potential_start)
            
            # Task can start after the unit becomes available
            start_time = max(earliest_start, unit_available_times[unit_idx])
            
            # Calculate finish time
            execution_time = self.execution_times[task][unit_idx]
            finish_time = start_time + execution_time
            
            # Update finish time and unit available time
            finish_times[task] = finish_time
            unit_available_times[unit_idx] = finish_time
        
        # Makespan is the maximum finish time
        return max(finish_times)
    
    def get_predecessors_of(self, task):
        """
        Get predecessors of a task from the DAG
        """
        predecessors = []
        for potential_pred in range(self.num_tasks):
            if task in self.get_successors(potential_pred):
                predecessors.append(potential_pred)
        return predecessors
    
    def run(self):
        """
        Algorithm 7: R2GA
        Main algorithm that runs the genetic algorithm
        """
        start_time = time.time()
        
        iteration = 0
        # while iteration < 5000:
        while iteration < self.max_iterations:
            # Select elite chromosomes
            elites, sorted_indices, best_fitness = self.select(self.population, self.elite_size)
            
            # Store best solution
            best_solution = self.decode(self.population[sorted_indices[0]])
            self.best_solutions.append(best_solution)
            
            # Print progress
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Best Makespan: {best_fitness}")
            
            # Perform crossover
            crossovers = self.crossover(self.population, sorted_indices, self.num_tasks, 
                                       self.elite_size, self.pop_size)
            
            # Perform mutation and create new population
            self.population = self.mutate(crossovers, self.num_tasks, elites, 
                                         self.pop_size, self.elite_size)
            
            iteration += 1
        
        # Get the best solution
        elites, sorted_indices, best_fitness = self.select(self.population, self.elite_size)
        best_solution = self.decode(self.population[sorted_indices[0]])
        best_makespan = self.calculate_makespan(self.population[sorted_indices[0]])
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"R2GA completed in {execution_time:.2f} seconds")
        print(f"Best makespan: {best_makespan}")
        print(f"Best solution: {best_solution}")
        
        # Plot convergence curve
        self.plot_convergence()
        
        return best_solution, best_makespan
    
    def plot_convergence(self):
        """Plot the convergence of the algorithm"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.best_fitness_history)), self.best_fitness_history)
        plt.title('R2GA Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Makespan')
        plt.grid(True)
        plt.show()

def compare_algorithms(dag, execution_times, communication_times, num_tasks, computing_units):
    """Compare HEFT and R2GA algorithms on the same workflow"""
    print("\n=== Comparing HEFT and R2GA Algorithms ===\n")
    
    # Run HEFT
    print("Running HEFT...")
    heft = HEFT(dag, execution_times, communication_times, num_tasks, computing_units)
    heft_result = heft.run()
    heft_makespan = heft_result["makespan"]
    
    # Run R2GA
    print("\nRunning R2GA...")
    r2ga = R2GA(dag, execution_times, communication_times, num_tasks, computing_units,
               pop_size=50, max_iterations=100, elite_size=5)
    r2ga_solution, r2ga_makespan = r2ga.run()
    
    # Compare results
    print("\n=== Results Comparison ===")
    print(f"HEFT Makespan: {heft_makespan}")
    print(f"R2GA Makespan: {r2ga_makespan}")
    
    if r2ga_makespan < heft_makespan:
        improvement = ((heft_makespan - r2ga_makespan) / heft_makespan) * 100
        print(f"R2GA improved makespan by {improvement:.2f}% compared to HEFT")
    elif heft_makespan < r2ga_makespan:
        degradation = ((r2ga_makespan - heft_makespan) / heft_makespan) * 100
        print(f"R2GA performed {degradation:.2f}% worse than HEFT")
    else:
        print("Both algorithms produced the same makespan")
    
    # Visualize schedules
    visualize_schedules(heft_result, r2ga_solution, r2ga_makespan, num_tasks, computing_units)
    
    return heft_result, (r2ga_solution, r2ga_makespan)

def visualize_schedules(heft_result, r2ga_solution, r2ga_makespan, num_tasks, computing_units):
    """Visualize HEFT and R2GA schedules as Gantt charts"""
    # Create figure and axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Colors for tasks
    colors = plt.cm.viridis(np.linspace(0, 1, num_tasks))
    
    # Plot HEFT schedule
    ax1.set_title("HEFT Schedule")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Computing Units")
    ax1.set_yticks(range(len(computing_units)))
    ax1.set_yticklabels([f"Unit {u}" for u in computing_units])
    
    for task in range(num_tasks):
        idx = heft_result["tasks"].index(task)
        unit = heft_result["units"][idx]
        start = heft_result["start_times"][task]
        finish = heft_result["finish_times"][task]
        duration = finish - start
        ax1.barh(unit, duration, left=start, color=colors[task], alpha=0.7)
        ax1.text(start + duration/2, unit, f"T{task}", ha='center', va='center')
    
    # Plot R2GA schedule
    # First, calculate start and finish times for R2GA solution
    r2ga_start_times = [0] * num_tasks
    r2ga_finish_times = [0] * num_tasks
    unit_available_times = [0] * len(computing_units)
    
    for i, task in enumerate(r2ga_solution["tasks"]):
        unit = r2ga_solution["units"][i]
        unit_idx = computing_units.index(unit)
        
        # Find predecessors
        earliest_start = 0
        for j in range(i):
            pred_task = r2ga_solution["tasks"][j]
            pred_unit = r2ga_solution["units"][j]
            pred_idx = computing_units.index(pred_unit)
            
            # Check if predecessor
            is_predecessor = False
            for potential_pred in range(num_tasks):
                if task in heft.get_successors(potential_pred) and potential_pred == pred_task:
                    is_predecessor = True
                    break
            
            if is_predecessor:
                # Include communication time if tasks are on different units
                comm_time = 0
                if pred_unit != unit:
                    comm_time = communication_times[pred_task][task]
                
                # Task can start after predecessor finishes + communication
                potential_start = r2ga_finish_times[pred_task] + comm_time
                earliest_start = max(earliest_start, potential_start)
        
        # Task can start after the unit becomes available
        start_time = max(earliest_start, unit_available_times[unit_idx])
        
        # Calculate finish time
        execution_time = execution_times[task][unit_idx]
        finish_time = start_time + execution_time
        
        # Update times
        r2ga_start_times[task] = start_time
        r2ga_finish_times[task] = finish_time
        unit_available_times[unit_idx] = finish_time
    
    ax2.set_title("R2GA Schedule")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Computing Units")
    ax2.set_yticks(range(len(computing_units)))
    ax2.set_yticklabels([f"Unit {u}" for u in computing_units])
    
    for i, task in enumerate(r2ga_solution["tasks"]):
        unit = r2ga_solution["units"][i]
        unit_idx = computing_units.index(unit)
        start = r2ga_start_times[task]
        finish = r2ga_finish_times[task]
        duration = finish - start
        ax2.barh(unit_idx, duration, left=start, color=colors[task], alpha=0.7)
        ax2.text(start + duration/2, unit_idx, f"T{task}", ha='center', va='center')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Define sample inputs
    
    # Number of tasks
    num_tasks = 8
    
    # DAG representation (adjacency list)
    dag = {
        0: [1,2,3,4],  # Task 0 has successors 1, 2, 3, and 4
        1: [5],
        2: [5],
        3: [5,6],
        4: [6],
        5: [7],
        6: [7],
        7: []
    }
    
    # Number of computing units
    computing_units = [0, 1, 2]  # 3 computing units
    
    # Execution time matrix (task x unit)
    execution_times = [
        [11, 13, 9],   # Task 0 execution times on units 0, 1, 2
        [10, 15, 11],  # Task 1 execution times
        [9, 12, 14],   # Task 2 execution times
        [11, 16, 10],  # Task 3 execution times
        [15, 11, 19],  # Task 4 execution times
        [12, 9, 5],    # Task 5 execution times
        [10, 14, 13],  # Task 6 execution times
        [11, 15, 10]   # Task 7 execution times
    ]
    
    # Communication time matrix (task x task)
    communication_times = np.zeros((num_tasks, num_tasks))
    communication_times[0][1] = 11
    communication_times[0][2] = 17
    communication_times[0][3] = 14
    communication_times[0][4] = 11
    communication_times[1][5] = 13
    communication_times[2][5] = 10
    communication_times[3][5] = 19
    communication_times[3][6] = 13
    communication_times[4][6] = 27
    communication_times[5][7] = 21
    communication_times[6][7] = 13
    
    # Option 1: Run just the HEFT algorithm
    print("Running HEFT algorithm...")
    heft = HEFT(dag, execution_times, communication_times, num_tasks, computing_units)
    heft_result = heft.run()
    
    # Option 2: Run just the R2GA algorithm
    print("\nRunning R2GA algorithm...")
    r2ga = R2GA(dag, execution_times, communication_times, num_tasks, computing_units,
               pop_size=50, max_iterations=100, elite_size=5)
    best_solution, best_makespan = r2ga.run()
    
    # Option 3: Compare both algorithms
    print("\nComparing both algorithms...")
    compare_algorithms(dag, execution_times, communication_times, num_tasks, computing_units)