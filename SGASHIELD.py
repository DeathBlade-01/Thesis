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
    
    def generate_heft_chromosome(self, ga_instance):
        """
        Generate a chromosome compatible with the GA based on the HEFT schedule
        This can be used to seed the GA population with a good initial solution
        """
        # For a security-aware GA, we need to include security strengths
        # First, determine the number of messages in the DAG
        num_messages = 0
        for task in range(self.num_tasks):
            num_messages += len(self.get_successors(task))
        
        chromosome = [0] * (2 * self.num_tasks + 3 * num_messages)
        
        # Create a map of task->position in the HEFT schedule
        task_positions = {self.task_schedule[i]: i for i in range(self.num_tasks)}
        
        # Set task selection genes (first half of chromosome)
        # Lower values get higher priority in GA's decode method
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
        
        # For security strengths, set default values (middle security level)
        # In a real implementation, this would be based on security requirements
        msg_index = 0
        for task in range(self.num_tasks):
            for succ in self.get_successors(task):
                # Set medium security level (0.5) for each service
                for service in range(3):  # 3 services: confidentiality, integrity, authentication
                    chromosome[2 * self.num_tasks + 3 * msg_index + service] = 0.5
                msg_index += 1
        
        return chromosome


class SecurityAwareGA:
    def __init__(self, dag, execution_times, communication_times, num_tasks, computing_units, 
                 security_weights=None, security_strengths=None, security_overheads=None,
                 pop_size=100, max_iterations=500, elite_size=10, mutation_prob=0.3,
                 crossover_prob=0.8, alpha=0.7, beta=0.3):
        """Initialize the Security-Aware GA algorithm with the given parameters"""
        self.dag = dag  # Workflow DAG - adjacency list format
        self.execution_times = execution_times  # Matrix of task execution times on diff units
        self.communication_times = communication_times  # Matrix of communication times between tasks
        self.num_tasks = num_tasks  # Number of tasks
        self.computing_units = computing_units  # List of computing units
        self.num_units = len(computing_units)
        
        # Security parameters
        self.security_weights = security_weights  # Weights for security services
        self.security_strengths = security_strengths  # Security strength options
        self.security_overheads = security_overheads  # Security overheads
        
        # Calculate number of messages in the DAG
        self.num_messages = 0
        for task in range(self.num_tasks):
            successors = self.get_successors(task)
            self.num_messages += len(successors)
        
        # Default security settings if not provided
        if security_weights is None:
            # Default equal weights for each security service (confidentiality, integrity, authentication)
            self.security_weights = [1/3, 1/3, 1/3]
        
        if security_strengths is None:
            # Default security strengths (5 levels for each service)
            self.security_strengths = [
                [0.0, 0.25, 0.5, 0.75, 1.0],  # Confidentiality
                [0.0, 0.25, 0.5, 0.75, 1.0],  # Integrity
                [0.0, 0.25, 0.5, 0.75, 1.0]   # Authentication
            ]
        
        if security_overheads is None:
            # Default security overheads (processing and communication overheads)
            # For each security level and service
            self.security_overheads = {
                "processing": [
                    [0.0, 0.05, 0.1, 0.15, 0.2],  # Confidentiality
                    [0.0, 0.03, 0.08, 0.13, 0.18],  # Integrity
                    [0.0, 0.02, 0.07, 0.12, 0.17]   # Authentication
                ],
                "communication": [
                    [0.0, 0.04, 0.09, 0.14, 0.19],  # Confidentiality
                    [0.0, 0.03, 0.08, 0.13, 0.18],  # Integrity
                    [0.0, 0.02, 0.07, 0.12, 0.17]   # Authentication
                ]
            }
        
        # GA parameters
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.elite_size = elite_size
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.alpha = alpha  # Weight for makespan in fitness function
        self.beta = beta    # Weight for security utility in fitness function
        
        # Initialize population
        self.population = self.initialize(self.pop_size)
        
        # Keep track of best solutions
        self.best_solutions = []
        self.best_fitness_history = []
        
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
    
    def encode(self):
        """
        Generate a chromosome with task priorities, processor assignments, and security strengths
        """
        # Task priorities (normalized between 0 and 1)
        task_priorities = [random.random() for _ in range(self.num_tasks)]
        
        # Processor assignments (normalized between 0 and 1)
        processor_assignments = [random.random() for _ in range(self.num_tasks)]
        
        # Security strengths (normalized between 0 and 1)
        # For each message and each security service
        security_strengths = []
        for _ in range(self.num_messages * 3):  # 3 services per message
            security_strengths.append(random.random())
        
        # Combine all parts into a single chromosome
        chromosome = task_priorities + processor_assignments + security_strengths
        
        return chromosome
    
    def decode(self, chromosome):
        """
        Decode a chromosome into a solution (task allocation, scheduling, and security assignments)
        """
        # Split chromosome into its parts
        task_priorities = chromosome[:self.num_tasks]
        processor_assignments = chromosome[self.num_tasks:2*self.num_tasks]
        security_strengths = chromosome[2*self.num_tasks:]
        
        # Initialize candidate task set (CTS), task list (TL), and unit list (UL)
        cts = self.get_entry_tasks()  # Tasks with no predecessors
        task_list = []
        unit_list = []
        security_list = []  # Store security assignments for each message
        
        # Schedule tasks one by one
        while len(task_list) < self.num_tasks:
            if not cts:
                break  # Should not happen in a valid DAG
                
            # Find task with highest priority (lowest value in this case)
            best_task = None
            best_priority = float('inf')
            
            for task in cts:
                priority = task_priorities[task]
                if priority < best_priority:
                    best_priority = priority
                    best_task = task
            
            # Assign task to a processor
            proc_value = processor_assignments[best_task]
            processor = int(proc_value * self.num_units)
            if processor >= self.num_units:
                processor = self.num_units - 1
            
            # Add to task and unit lists
            task_list.append(best_task)
            unit_list.append(processor)
            
            # Assign security strengths to messages from this task
            successors = self.get_successors(best_task)
            for succ in successors:
                msg_index = 0
                # Find the message index for this task-successor pair
                for t in range(self.num_tasks):
                    for s in self.get_successors(t):
                        if t == best_task and s == succ:
                            # Extract security strengths for this message
                            msg_security = []
                            for service in range(3):
                                security_value = security_strengths[3 * msg_index + service]
                                # Convert to discrete security level
                                level = int(security_value * len(self.security_strengths[service]))
                                if level >= len(self.security_strengths[service]):
                                    level = len(self.security_strengths[service]) - 1
                                msg_security.append(level)
                            security_list.append(msg_security)
                            break
                        msg_index += 1
            
            # Remove task from CTS
            cts.remove(best_task)
            
            # Add successor tasks to CTS if they are ready (all predecessors completed)
            for successor in self.get_successors(best_task):
                if self.is_candidate_task(successor, task_list):
                    cts.append(successor)
        
        return {"tasks": task_list, "units": unit_list, "security": security_list}
    
    def get_entry_tasks(self):
        """Get tasks with no predecessors"""
        entry_tasks = []
        for task in range(self.num_tasks):
            if not self.get_predecessors(task):
                entry_tasks.append(task)
        return entry_tasks
    
    def is_candidate_task(self, task, scheduled_tasks):
        """Check if a task is a candidate (all predecessors are scheduled)"""
        for predecessor in self.get_predecessors(task):
            if predecessor not in scheduled_tasks:
                return False
        return True
    
    def initialize(self, pop_size):
        """
        Create an initial population of pop_size chromosomes
        """
        population = []
        
        # Generate random chromosomes
        for _ in range(pop_size - 1):
            chromosome = self.encode()
            population.append(chromosome)
        
        # Add a HEFT-seeded chromosome
        heft = HEFT(self.dag, self.execution_times, self.communication_times, self.num_tasks, self.computing_units)
        heft.run()
        heft_chromosome = heft.generate_heft_chromosome(self)
        population.append(heft_chromosome)
        
        return population
    
    def calculate_makespan(self, solution):
        """
        Calculate makespan for a given solution, considering security overheads
        """
        task_list = solution["tasks"]
        unit_list = solution["units"]
        security_list = solution.get("security", [])
        
        # Initialize finish times for each task
        finish_times = [0] * self.num_tasks
        
        # Initialize available times for each computing unit
        unit_available_times = [0] * self.num_units
        
        # Calculate start and finish time for each task
        for i, task in enumerate(task_list):
            unit = unit_list[i]
            
            # Find the earliest time the task can start (after predecessors complete)
            ready_time = 0
            msg_index = 0
            
            for j in range(i):
                pred_task = task_list[j]
                pred_unit = unit_list[j]
                
                # Check if pred_task is a predecessor of task
                if task in self.get_successors(pred_task):
                    # Include communication time if tasks are on different units
                    base_comm_time = self.communication_times[pred_task][task]
                    
                    # Add security overhead for communication if available
                    comm_overhead = 0
                    if security_list and msg_index < len(security_list):
                        for service in range(3):
                            security_level = security_list[msg_index][service]
                            overhead = self.security_overheads["communication"][service][security_level]
                            comm_overhead += overhead * base_comm_time
                    
                    # Calculate actual communication time
                    comm_time = base_comm_time * (1 + comm_overhead) if pred_unit != unit else 0
                    
                    # Task can start after predecessor finishes + communication
                    pred_finish_with_comm = finish_times[pred_task] + comm_time
                    ready_time = max(ready_time, pred_finish_with_comm)
                    
                    msg_index += 1  # Move to next message
            
            # Task can start after the unit becomes available
            start_time = max(ready_time, unit_available_times[unit])
            
            # Calculate execution time with security overhead
            base_execution_time = self.execution_times[task][unit]
            
            # Add security overhead for processing
            proc_overhead = 0
            for succ in self.get_successors(task):
                if security_list and msg_index < len(security_list):
                    for service in range(3):
                        security_level = security_list[msg_index][service]
                        overhead = self.security_overheads["processing"][service][security_level]
                        proc_overhead += overhead
                    msg_index += 1
            
            # Calculate actual execution time
            execution_time = base_execution_time * (1 + proc_overhead)
            
            # Calculate finish time
            finish_time = start_time + execution_time
            
            # Update finish time and unit available time
            finish_times[task] = finish_time
            unit_available_times[unit] = finish_time
        
        # Makespan is the maximum finish time
        return max(finish_times)
    
    def calculate_security_utility(self, solution):
        """
        Calculate the Total Security Utility (TSU) for a given solution
        TSU = Σ(weight_service * security_strength) for all messages and services
        """
        security_list = solution.get("security", [])
        if not security_list:
            return 0
        
        total_utility = 0
        msg_index = 0
        
        # Calculate utility for each message
        for task in range(self.num_tasks):
            for succ in self.get_successors(task):
                if msg_index < len(security_list):
                    msg_security = security_list[msg_index]
                    
                    # Calculate utility for this message
                    msg_utility = 0
                    for service in range(3):
                        security_level = msg_security[service]
                        security_strength = self.security_strengths[service][security_level]
                        weight = self.security_weights[service]
                        msg_utility += weight * security_strength
                    
                    total_utility += msg_utility
                    msg_index += 1
        
        # Normalize by number of messages
        if self.num_messages > 0:
            total_utility /= self.num_messages
        
        return total_utility
    
    def fitness(self, chromosome):
        """
        Calculate fitness of a chromosome
        Fitness is a weighted combination of makespan and security utility
        Lower fitness is better (minimize makespan, maximize security)
        """
        solution = self.decode(chromosome)
        
        # Calculate makespan
        makespan = self.calculate_makespan(solution)
        
        # Calculate security utility
        security_utility = self.calculate_security_utility(solution)
        
        # Normalize makespan (assuming a reasonable upper bound)
        max_makespan = sum([max(self.execution_times[task]) for task in range(self.num_tasks)])
        norm_makespan = makespan / max_makespan
        
        # Calculate fitness (weighted sum)
        # We want to minimize makespan and maximize security utility
        fitness = self.alpha * norm_makespan + self.beta * (1 - security_utility)
        
        return fitness, solution, makespan, security_utility
    
    def select(self):
        """
        Select elite_size best chromosomes from the population
        """
        # Evaluate fitness of each chromosome
        fitness_data = []
        for chromosome in self.population:
            fitness_value, solution, makespan, security = self.fitness(chromosome)
            fitness_data.append((fitness_value, chromosome, solution, makespan, security))
        
        # Sort chromosomes by fitness (lower is better)
        fitness_data.sort(key=lambda x: x[0])
        
        # Select elite chromosomes
        elites = []
        for i in range(self.elite_size):
            elites.append(deepcopy(fitness_data[i][1]))
        
        # Track best solution
        best_fitness, _, best_solution, best_makespan, best_security = fitness_data[0]
        self.best_solutions.append(best_solution)
        self.best_fitness_history.append(best_fitness)
        
        return elites, fitness_data
    
    def order_preserving_crossover(self, parent1, parent2):
        """
        Perform order-preserving crossover for task priorities
        and standard crossover for processor assignments and security strengths
        """
        # Create empty offspring
        offspring1 = [0] * len(parent1)
        offspring2 = [0] * len(parent2)
        
        # Task priorities part (first part of chromosome)
        task_genes = self.num_tasks
        
        # For task priorities, use Order Crossover (OX)
        # Select random subsequence
        start = random.randint(0, task_genes - 2)
        end = random.randint(start + 1, task_genes - 1)
        
        # Copy subsequence from parent1 to offspring1 and from parent2 to offspring2
        for i in range(start, end + 1):
            offspring1[i] = parent1[i]
            offspring2[i] = parent2[i]
        
        # Fill remaining positions with values from other parent
        for i in range(task_genes):
            if i < start or i > end:
                offspring1[i] = parent2[i]
                offspring2[i] = parent1[i]
        
        # For processor assignments and security strengths, use standard crossover
        # Select crossover point
        crossover_point = random.randint(task_genes, len(parent1) - 1)
        
        # Copy genes from parents to offspring
        for i in range(task_genes, len(parent1)):
            if i <= crossover_point:
                offspring1[i] = parent1[i]
                offspring2[i] = parent2[i]
            else:
                offspring1[i] = parent2[i]
                offspring2[i] = parent1[i]
        
        return offspring1, offspring2
    
    def mutate(self, chromosome):
        """
        Mutate a chromosome with probability mutation_prob
        """
        # Create a copy of the chromosome
        mutated = deepcopy(chromosome)
        
        # Mutate task priorities
        for i in range(self.num_tasks):
            if random.random() < self.mutation_prob:
                mutated[i] = random.random()
        
        # Mutate processor assignments
        for i in range(self.num_tasks, 2 * self.num_tasks):
            if random.random() < self.mutation_prob:
                mutated[i] = random.random()
        
        # Mutate security strengths
        for i in range(2 * self.num_tasks, len(chromosome)):
            if random.random() < self.mutation_prob:
                mutated[i] = random.random()
        
        return mutated
    
    def crossover_and_mutate(self, fitness_data):
        """
        Perform crossover and mutation to create a new population
        """
        # Extract chromosomes and their fitness values
        chromosomes = [data[1] for data in fitness_data]
        fitness_values = [data[0] for data in fitness_data]
        
        # Calculate selection probabilities (inverse of fitness, since lower is better)
        total_inverse_fitness = sum(1/f for f in fitness_values)
        selection_probs = [(1/f)/total_inverse_fitness for f in fitness_values]
        
        # Create new population through crossover and mutation
        new_population = []
        
        # Add elite chromosomes
        for i in range(self.elite_size):
            new_population.append(deepcopy(chromosomes[i]))
        
        # Create rest of population through crossover and mutation
        while len(new_population) < self.pop_size:
            # Select parents using roulette wheel selection
            parent1_idx = random.choices(range(len(chromosomes)), weights=selection_probs)[0]
            parent2_idx = random.choices(range(len(chromosomes)), weights=selection_probs)[0]
            
            parent1 = chromosomes[parent1_idx]
            parent2 = chromosomes[parent2_idx]
            
            # Perform crossover with probability crossover_prob
            if random.random() < self.crossover_prob:
                offspring1, offspring2 = self.order_preserving_crossover(parent1, parent2)
            else:
                offspring1, offspring2 = deepcopy(parent1), deepcopy(parent2)
            
            # Perform mutation
            offspring1 = self.mutate(offspring1)
            offspring2 = self.mutate(offspring2)
            
            # Add offspring to new population
            new_population.append(offspring1)
            if len(new_population) < self.pop_size:
                new_population.append(offspring2)
        
        return new_population
    
    def run(self):
        """
        Execute the Security-Aware GA algorithm
        """
        start_time = time.time()
        
        iteration = 0
        while iteration < self.max_iterations:
            # Select elite chromosomes and evaluate fitness
            elites, fitness_data = self.select()
            
            # Get best solution data
            best_fitness, _, best_solution, best_makespan, best_security = fitness_data[0]
            
            # Print progress
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Best Fitness: {best_fitness:.4f}, "
                      f"Makespan: {best_makespan:.2f}, Security: {best_security:.4f}")
            
            # Create new population through crossover and mutation
            self.population = self.crossover_and_mutate(fitness_data)
            
            iteration += 1
        
        # Get final best solution
        elites, fitness_data = self.select()
        best_fitness, best_chromosome, best_solution, best_makespan, best_security = fitness_data[0]
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"\nSecurity-Aware GA completed in {execution_time:.2f} seconds")
        print(f"Best Fitness: {best_fitness:.4f}")
        print(f"Best Makespan: {best_makespan:.2f}")
        print(f"Security Utility: {best_security:.4f}")
        print(f"Task Schedule: {best_solution['tasks']}")
        print(f"Processor Assignments: {best_solution['units']}")
        
        # Plot convergence curve
        self.plot_convergence()
        
        return best_solution, best_makespan, best_security
    
    def plot_convergence(self):
        """Plot the convergence of the algorithm"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.best_fitness_history)), self.best_fitness_history)
        plt.title('Security-Aware GA Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness (lower is better)')
        plt.grid(True)
        plt.show()


def compare_algorithms(dag, execution_times, communication_times, num_tasks, computing_units):
    """Compare HEFT and Security-Aware GA algorithms on the same workflow"""
    print("\n=== Comparing HEFT and Security-Aware GA Algorithms ===\n")
    
    # Run HEFT
    print("Running HEFT...")
    heft = HEFT(dag, execution_times, communication_times, num_tasks, computing_units)
    heft_result = heft.run()
    heft_makespan = heft_result["makespan"]
    
    # Run Security-Aware GA
    print("\nRunning Security-Aware GA...")
    sa_ga = SecurityAwareGA(dag, execution_times, communication_times, num_tasks, computing_units,
                           pop_size=50, max_iterations=100, elite_size=5)
    ga_solution, ga_makespan, ga_security = sa_ga.run()
    
    # Compare results
    print("\n=== Comparison Results ===")
    print(f"HEFT Makespan: {heft_makespan:.2f}")
    print(f"Security-Aware GA Makespan: {ga_makespan:.2f}")
    print(f"Security-Aware GA Security Utility: {ga_security:.4f}")
    
    makespan_improvement = ((heft_makespan - ga_makespan) / heft_makespan) * 100
    print(f"Makespan Improvement: {makespan_improvement:.2f}%")
    
    # Compare task schedules
    print("\nHEFT Task Schedule:", heft_result["tasks"])
    print("GA Task Schedule:", ga_solution["tasks"])
    
    print("\nHEFT Processor Assignments:", heft_result["units"])
    print("GA Processor Assignments:", ga_solution["units"])
    
    # Visualize schedules (optional)
    plot_schedules(heft_result, ga_solution, computing_units)
    
    return {
        "heft": {
            "makespan": heft_makespan,
            "schedule": heft_result
        },
        "ga": {
            "makespan": ga_makespan,
            "security": ga_security,
            "schedule": ga_solution
        }
    }

def plot_schedules(heft_result, ga_result, computing_units):
    """Plot task schedules for both algorithms"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot HEFT schedule
    plot_schedule(ax1, heft_result, computing_units, "HEFT Schedule")
    
    # Plot GA schedule
    plot_schedule(ax2, ga_result, computing_units, "Security-Aware GA Schedule")
    
    plt.tight_layout()
    plt.show()

def plot_schedule(ax, schedule, computing_units, title):
    """Plot a single schedule on the given axis"""
    colors = plt.cm.tab10.colors
    
    # Extract schedule data
    tasks = schedule["tasks"]
    units = schedule["units"]
    
    # For HEFT, we have start_times and finish_times
    if "start_times" in schedule and "finish_times" in schedule:
        start_times = schedule["start_times"]
        finish_times = schedule["finish_times"]
    else:
        # For GA, we need to recalculate times (this is simplified)
        # In a real implementation, you would get these from the GA
        start_times = [0] * len(tasks)
        finish_times = [10] * len(tasks)  # Placeholder
    
    # Plot tasks as bars on their assigned processors
    for i, task in enumerate(tasks):
        unit = units[i] if i < len(units) else 0
        start = start_times[task] if task < len(start_times) else 0
        finish = finish_times[task] if task < len(finish_times) else 10
        
        ax.barh(unit, finish - start, left=start, height=0.5, 
                color=colors[task % len(colors)], alpha=0.7)
        
        # Add task label
        ax.text(start + (finish - start)/2, unit, f"T{task}", 
                ha='center', va='center', color='black', fontweight='bold')
    
    # Set axis properties
    ax.set_yticks(range(len(computing_units)))
    ax.set_yticklabels([f"P{p}" for p in computing_units])
    ax.set_xlabel("Time")
    ax.set_ylabel("Processor")
    ax.set_title(title)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

if __name__ == "__main__":
    # Define sample inputs
    num_tasks = 8
    
    dag = {
        0: [1, 2, 3, 4],
        1: [5],
        2: [5],
        3: [5, 6],
        4: [6],
        5: [7],
        6: [7],
        7: []
    }
    
    computing_units = [0, 1, 2]
    
    execution_times = [
        [11, 13, 9],
        [10, 15, 11],
        [9, 12, 14],
        [11, 16, 10],
        [15, 11, 19],
        [12, 9, 5],
        [10, 14, 13],
        [11, 15, 10]
    ]
    
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
    
    print("Running HEFT algorithm...")
    heft = HEFT(dag, execution_times, communication_times, num_tasks, computing_units)
    heft_result = heft.run()
    
    print("\nRunning Security-Aware GA algorithm...")
    sa_ga = SecurityAwareGA(dag, execution_times, communication_times, num_tasks, computing_units,
                            pop_size=50, max_iterations=100, elite_size=5)
    best_solution, best_makespan, best_security = sa_ga.run()
    
    print("\nComparing both algorithms...")
    compare_algorithms(dag, execution_times, communication_times, num_tasks, computing_units)
