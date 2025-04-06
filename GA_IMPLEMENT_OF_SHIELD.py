import numpy as np
import matplotlib.pyplot as plt
import copy
import random
from typing import List, Dict, Tuple, Any
import time

class GeneticAlgorithmScheduler:
    def __init__(self, tasks, messages, processors, network, security_service, deadline, 
                 population_size=50, generations=100, elite_size=5, mutation_rate=0.2,
                 crossover_rate=0.8, tournament_size=3):
        self.tasks = tasks
        self.messages = messages
        self.processors = processors
        self.network = network
        self.security = security_service
        self.deadline = deadline
        self.num_tasks = len(tasks)
        self.num_processors = len(processors)
        
        # GA parameters
        self.population_size = population_size
        self.generations = generations
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        
        # Initialize task dependencies and successors
        self.initialize_successors()
        self.compute_task_priorities()
        
        # Initialize security to minimum levels
        self.assign_minimum_security()
        
        # For storing best solutions
        self.best_schedule = None
        self.best_makespan = float('inf')
        self.best_security_utility = 0
        
        # For tracking convergence
        self.stage1_convergence = []
        self.stage2_convergence = []

    def initialize_successors(self):
        # Clear any existing successors
        for task in self.tasks:
            task.successors = []
        
        # Add each task as a successor to its predecessors
        for task in self.tasks:
            for pred_id in task.predecessors:
                pred_task = self.get_task_by_id(pred_id)
                if pred_task:
                    pred_task.successors.append(task.task_id)
    
    def get_task_by_id(self, task_id):
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
    
    def get_message(self, source_id, dest_id):
        for message in self.messages:
            if message.source_id == source_id and message.dest_id == dest_id:
                return message
        return None
    
    def compute_task_priorities(self):
        """Compute priorities for tasks using upward rank method."""
        # First, calculate the average execution time for each task
        for task in self.tasks:
            task.avg_execution = sum(task.execution_times) / len(task.execution_times)
        
        # Calculate priority (upward rank) for each task
        def calculate_upward_rank(task):
            if task.priority is not None:
                return task.priority
            
            if not task.successors:  # Exit task
                task.priority = task.avg_execution
                return task.priority
            
            max_successor_rank = 0
            for succ_id in task.successors:
                succ_task = self.get_task_by_id(succ_id)
                if succ_task:
                    # Get communication cost between task and successor
                    message = self.get_message(task.task_id, succ_id)
                    comm_cost = 0
                    if message:
                        comm_cost = message.size / self.network.bandwidth
                    
                    succ_rank = calculate_upward_rank(succ_task)
                    rank_with_comm = comm_cost + succ_rank
                    max_successor_rank = max(max_successor_rank, rank_with_comm)
            
            task.priority = task.avg_execution + max_successor_rank
            return task.priority
        
        # Calculate upward rank for all tasks
        for task in self.tasks:
            calculate_upward_rank(task)
    
    def assign_minimum_security(self):
        """Assign minimum security levels to all messages."""
        for message in self.messages:
            # For each security service, find the minimum protocol that meets requirements
            for service in ['confidentiality', 'integrity', 'authentication']:
                strengths = self.security.strengths[service]
                min_strength = message.min_security[service]
                
                # Find the protocol with minimum strength that meets the requirement
                protocol_idx = 0
                for i, strength in enumerate(strengths):
                    if strength >= min_strength:
                        protocol_idx = i
                        break
                
                message.assigned_security[service] = protocol_idx
    
    def calc_security_overhead(self, message, source_proc, dest_proc):
        """Calculate the security overhead for a message."""
        total_overhead = 0
        
        # Add overhead for confidentiality and integrity (data-dependent)
        for service in ['confidentiality', 'integrity']:
            protocol_idx = message.assigned_security[service] + 1  # 1-indexed in the overhead table
            overhead_factor = self.security.overheads[service][protocol_idx][source_proc - 1]
            overhead = (message.size / 1024) * overhead_factor  # Convert KB to MB if needed
            total_overhead += overhead
        
        # Add overhead for authentication (data-independent)
        auth_protocol_idx = message.assigned_security['authentication'] + 1
        auth_overhead = self.security.overheads['authentication'][auth_protocol_idx][dest_proc - 1]
        total_overhead += auth_overhead
        
        return total_overhead
    
    # GA Stage 1: Task-Processor Mapping
    def generate_initial_population_stage1(self):
        """Generate initial population for task-processor mapping."""
        population = []
        
        # Sort tasks by priority (descending)
        sorted_tasks = sorted(self.tasks, key=lambda t: -t.priority)
        task_ids = [task.task_id for task in sorted_tasks]
        
        for _ in range(self.population_size):
            # Create a random processor assignment for each task
            chromosome = {}
            for task_id in task_ids:
                processor_id = random.randint(1, self.num_processors)
                chromosome[task_id] = processor_id
            
            population.append({
                'task_order': task_ids,
                'processor_assignment': chromosome,
                'security_levels': self.get_initial_security_levels()
            })
        
        return population
    
    def get_initial_security_levels(self):
        """Get the initial minimum security levels for all messages."""
        security_levels = {}
        for message in self.messages:
            security_levels[message.id] = {
                'confidentiality': message.assigned_security['confidentiality'],
                'integrity': message.assigned_security['integrity'],
                'authentication': message.assigned_security['authentication']
            }
        return security_levels
    
    def evaluate_fitness_stage1(self, individual):
        """Evaluate the fitness of an individual for task-processor mapping."""
        # Create a copy of tasks and processors for simulation
        tasks_copy = copy.deepcopy(self.tasks)
        processors_copy = copy.deepcopy(self.processors)
        
        # Apply the processor assignment
        processor_assignment = individual['processor_assignment']
        for task in tasks_copy:
            task.assigned_processor = processor_assignment[task.task_id]
        
        # Simulate the schedule
        schedule, makespan, local_slack_times = self.simulate_schedule(tasks_copy, processors_copy, individual['security_levels'])
        
        # Calculate the total security utility
        security_utility = self.calculate_security_utility(individual['security_levels'])
        
        # Calculate the total slack time
        global_slack_time = max(0, self.deadline - makespan)
        
        # Calculate the fitness
        if makespan > self.deadline:
            # Penalize schedules that miss the deadline
            fitness = -1000 - (makespan - self.deadline)
        else:
            # Reward schedules that meet the deadline and have slack time
            total_local_slack = sum(local_slack_times.values())
            
            # Identify critical paths and high-security messages
            critical_path_slack = self.identify_critical_path_slack(tasks_copy, local_slack_times)
            
            # Calculate fitness based on makespan, slack times, and critical path
            fitness = (
                1000  # Base score for meeting deadline
                + global_slack_time * 10  # Global slack time
                + total_local_slack * 5  # Total local slack time
                + critical_path_slack * 15  # Critical path slack (higher weight)
                + security_utility * 2  # Security utility
            )
        
        return {
            'fitness': fitness,
            'makespan': makespan,
            'global_slack': global_slack_time,
            'local_slack': local_slack_times,
            'security_utility': security_utility,
            'schedule': schedule
        }
    
    def identify_critical_path_slack(self, tasks, local_slack_times):
        """Identify slack time on critical paths that could be used for security upgrades."""
        # Identify critical paths (longest paths in the graph)
        exit_tasks = [task for task in tasks if not task.successors]
        
        # Calculate slack on paths leading to exit tasks
        critical_path_slack = 0
        for exit_task in exit_tasks:
            # Find the critical path for this exit task
            path_slack = self.calculate_path_slack(exit_task, tasks, local_slack_times)
            critical_path_slack += path_slack
        
        return critical_path_slack
    
    def calculate_path_slack(self, task, tasks, local_slack_times):
        """Calculate the slack time along a path from a root task to the given task."""
        if not task.predecessors:
            return local_slack_times.get(task.task_id, 0)
        
        path_slack = 0
        for pred_id in task.predecessors:
            pred_task = next((t for t in tasks if t.task_id == pred_id), None)
            if pred_task:
                # Add the slack between this task and its predecessor
                task_slack = local_slack_times.get(task.task_id, 0)
                path_slack += task_slack + self.calculate_path_slack(pred_task, tasks, local_slack_times)
        
        return path_slack
    
    def simulate_schedule(self, tasks, processors, security_levels):
        """Simulate the schedule execution and return makespan and local slack times."""
        # Reset processor available times
        for processor in processors:
            processor.available_time = 0
        
        # Reset task scheduled status
        for task in tasks:
            task.is_scheduled = False
            task.start_time = None
            task.finish_time = None
        
        # Apply security levels to messages
        messages_copy = copy.deepcopy(self.messages)
        for message in messages_copy:
            if message.id in security_levels:
                message.assigned_security = security_levels[message.id]
        
        # Process tasks in order of priority
        schedule = []
        local_slack_times = {}
        
        # Sort tasks by priority (descending)
        sorted_tasks = sorted(tasks, key=lambda t: -t.priority)
        
        for task in sorted_tasks:
            # Get the assigned processor
            processor_id = task.assigned_processor
            processor = next((p for p in processors if p.proc_id == processor_id), None)
            
            # Calculate the earliest start time for this task
            est = self.calculate_est(task, processor_id, tasks, messages_copy)
            
            # Calculate the finish time
            finish_time = est + task.execution_times[processor_id - 1]
            
            # Update task and processor status
            task.start_time = est
            task.finish_time = finish_time
            task.is_scheduled = True
            processor.available_time = finish_time
            
            # Calculate local slack time for this task
            next_task_start = float('inf')
            for succ_id in task.successors:
                succ_task = next((t for t in tasks if t.task_id == succ_id), None)
                if succ_task and succ_task.start_time is not None:
                    next_task_start = min(next_task_start, succ_task.start_time)
            
            # If there's no successor, use the deadline
            if next_task_start == float('inf'):
                next_task_start = self.deadline
            
            # Calculate local slack time
            local_slack = next_task_start - finish_time
            local_slack_times[task.task_id] = max(0, local_slack)
            
            # Add to schedule
            schedule.append({
                'task_id': task.task_id,
                'name': task.name,
                'processor': processor_id,
                'start_time': est,
                'finish_time': finish_time
            })
        
        # Calculate the makespan
        makespan = max(task.finish_time for task in tasks if task.is_scheduled)
        
        return schedule, makespan, local_slack_times
    
    def calculate_est(self, task, processor_id, tasks, messages):
        """Calculate Earliest Start Time for a task on a processor."""
        # Get the processor
        processor = next((p for p in processors if p.proc_id == processor_id), None)
        processor_ready_time = processor.available_time
        
        # If the task has no predecessors, it can start at processor ready time
        if not task.predecessors:
            return processor_ready_time
        
        # Calculate the earliest time this task can start considering all predecessors
        max_pred_finish_time = 0
        for pred_id in task.predecessors:
            pred_task = next((t for t in tasks if t.task_id == pred_id), None)
            if not pred_task or not pred_task.is_scheduled:
                return float('inf')  # Predecessor not scheduled yet
            
            message = next((m for m in messages if m.source_id == pred_id and m.dest_id == task.task_id), None)
            if not message:
                # If there's no explicit message, just consider the predecessor's finish time
                max_pred_finish_time = max(max_pred_finish_time, pred_task.finish_time)
                continue
            
            # If predecessor is on the same processor, only consider finish time
            if pred_task.assigned_processor == processor_id:
                comm_finish_time = pred_task.finish_time
            else:
                # Consider communication time and security overhead
                security_overhead = self.calc_security_overhead(
                    message, 
                    pred_task.assigned_processor, 
                    processor_id
                )
                comm_time = self.network.get_communication_time(
                    message, 
                    pred_task.assigned_processor, 
                    processor_id
                )
                comm_finish_time = pred_task.finish_time + comm_time + security_overhead
            
            max_pred_finish_time = max(max_pred_finish_time, comm_finish_time)
        
        return max(processor_ready_time, max_pred_finish_time)
    # def calculate_est(self, task, processor_id, tasks, messages):
    #     """Calculate Earliest Start Time (EST) and Effective Finish Time (EFT) for a task."""
    #     processor = next((p for p in self.processors if p.proc_id == processor_id), None)
    #     processor_ready_time = processor.available_time

    #     if not task.predecessors:
    #         return processor_ready_time  # If no predecessors, EST = processor ready time

    #     # Calculate the latest finish time of all predecessors
    #     max_pred_finish_time = 0
    #     for pred_id in task.predecessors:
    #         pred_task = self.tasks[pred_id]

    #         if not pred_task.is_scheduled:
    #             return float('inf')  # If predecessor isn't scheduled, return invalid time

    #         # Communication delay if assigned to different processors
    #         if pred_task.assigned_processor == processor_id:
    #             comm_time = 0
    #         else:
    #             message = self.get_message(pred_task.task_id, task.task_id)
    #             comm_time = (message.size / self.network.bandwidth) if message else 0
    #             comm_time += self.calc_security_overhead(message, pred_task.assigned_processor, processor_id)

    #         max_pred_finish_time = max(max_pred_finish_time, pred_task.finish_time + comm_time)

    #     # Compute EST
    #     est = max(processor_ready_time, max_pred_finish_time)
    #     return est

    def calculate_security_utility(self, security_levels):
        """Calculate the total security utility based on the assigned security levels."""
        total_utility = 0
        
        for message in self.messages:
            if message.id not in security_levels:
                continue
                
            message_utility = 0
            for service in ['confidentiality', 'integrity', 'authentication']:
                protocol_idx = security_levels[message.id][service]
                strength = self.security.strengths[service][protocol_idx]
                weight = message.weights[service]
                message_utility += weight * strength
            
            total_utility += message_utility
        
        return total_utility
    
    def tournament_selection(self, population, fitness_results):
        """Select individuals using tournament selection."""
        tournament = random.sample(list(range(len(population))), self.tournament_size)
        tournament_fitness = [fitness_results[i]['fitness'] for i in tournament]
        return population[tournament[tournament_fitness.index(max(tournament_fitness))]]
    
    def crossover_stage1(self, parent1, parent2):
        """Perform crossover between two parents for task-processor mapping."""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1)
        
        child = {
            'task_order': parent1['task_order'],  # Order is preserved
            'processor_assignment': {},
            'security_levels': copy.deepcopy(parent1['security_levels'])  # Inherit security levels from parent1
        }
        
        # Single-point crossover for processor assignments
        crossover_point = random.randint(1, len(parent1['task_order']) - 1)
        
        # Take processor assignments from parent1 up to crossover point
        for i in range(crossover_point):
            task_id = parent1['task_order'][i]
            child['processor_assignment'][task_id] = parent1['processor_assignment'][task_id]
        
        # Take processor assignments from parent2 after crossover point
        for i in range(crossover_point, len(parent1['task_order'])):
            task_id = parent1['task_order'][i]
            child['processor_assignment'][task_id] = parent2['processor_assignment'][task_id]
        
        return child
    
    def mutate_stage1(self, individual):
        """Mutate an individual's processor assignments."""
        mutated = copy.deepcopy(individual)
        
        for task_id in mutated['task_order']:
            if random.random() < self.mutation_rate:
                # Randomly assign a new processor
                new_processor = random.randint(1, self.num_processors)
                mutated['processor_assignment'][task_id] = new_processor
        
        return mutated
    
    def ensure_precedence_constraints(self, individual):
        """Ensure that precedence constraints are satisfied."""
        # The precedence constraints are already enforced by the simulation,
        # but we can validate the chromosome here if needed
        return individual
    
    def run_ga_stage1(self):
        """Run the first stage of the genetic algorithm (task-processor mapping)."""
        # Initialize population
        population = self.generate_initial_population_stage1()
        
        # Evaluate initial population
        fitness_results = [self.evaluate_fitness_stage1(individual) for individual in population]
        
        # Track best solution
        best_individual_idx = max(range(len(fitness_results)), key=lambda i: fitness_results[i]['fitness'])
        best_individual = copy.deepcopy(population[best_individual_idx])
        best_fitness = fitness_results[best_individual_idx]
        
        # Record initial best fitness for convergence tracking
        self.stage1_convergence.append(best_fitness['fitness'])
        
        print(f"Initial best fitness: {best_fitness['fitness']}, Makespan: {best_fitness['makespan']}")
        
        # Main GA loop
        for generation in range(self.generations):
            # Select elites
            sorted_indices = sorted(range(len(fitness_results)), key=lambda i: -fitness_results[i]['fitness'])
            elites = [copy.deepcopy(population[i]) for i in sorted_indices[:self.elite_size]]
            
            # Create new population
            new_population = elites.copy()
            
            # Fill the rest of the population with crossover and mutation
            while len(new_population) < self.population_size:
                # Select parents
                parent1 = self.tournament_selection(population, fitness_results)
                parent2 = self.tournament_selection(population, fitness_results)
                
                # Crossover
                child = self.crossover_stage1(parent1, parent2)
                
                # Mutation
                child = self.mutate_stage1(child)
                
                # Ensure precedence constraints
                child = self.ensure_precedence_constraints(child)
                
                new_population.append(child)
            
            # Update population
            population = new_population
            
            # Evaluate new population
            fitness_results = [self.evaluate_fitness_stage1(individual) for individual in population]
            
            # Update best solution
            current_best_idx = max(range(len(fitness_results)), key=lambda i: fitness_results[i]['fitness'])
            if fitness_results[current_best_idx]['fitness'] > best_fitness['fitness']:
                best_individual = copy.deepcopy(population[current_best_idx])
                best_fitness = fitness_results[current_best_idx]
                print(f"Generation {generation}: New best fitness: {best_fitness['fitness']}, Makespan: {best_fitness['makespan']}")
            
            # Record best fitness for convergence tracking
            self.stage1_convergence.append(best_fitness['fitness'])
        
        # Return the best individual
        return best_individual, best_fitness
    
    # GA Stage 2: Security Level Optimization
    def generate_initial_population_stage2(self, base_individual):
        """Generate initial population for security level optimization."""
        population = []
        
        # Always include the base individual
        population.append(copy.deepcopy(base_individual))
        
        # Generate variations of the base individual with different security levels
        for _ in range(self.population_size - 1):
            individual = copy.deepcopy(base_individual)
            
            # Randomly modify some security levels
            for message_id in individual['security_levels']:
                for service in ['confidentiality', 'integrity', 'authentication']:
                    if random.random() < self.mutation_rate:
                        # Get the maximum possible level for this service
                        max_level = len(self.security.strengths[service]) - 1
                        
                        # Randomly select a new level (potentially higher)
                        new_level = random.randint(individual['security_levels'][message_id][service], max_level)
                        individual['security_levels'][message_id][service] = new_level
            
            population.append(individual)
        
        return population
    
    def evaluate_fitness_stage2(self, individual):
        """Evaluate the fitness of an individual for security level optimization."""
        # Create a copy of tasks and processors for simulation
        tasks_copy = copy.deepcopy(self.tasks)
        processors_copy = copy.deepcopy(self.processors)
        
        # Apply the processor assignment
        processor_assignment = individual['processor_assignment']
        for task in tasks_copy:
            task.assigned_processor = processor_assignment[task.task_id]
        
        # Simulate the schedule
        schedule, makespan, _ = self.simulate_schedule(tasks_copy, processors_copy, individual['security_levels'])
        
        # Calculate the total security utility
        security_utility = self.calculate_security_utility(individual['security_levels'])
        
        # Calculate the fitness
        if makespan > self.deadline:
            # Heavily penalize schedules that miss the deadline
            fitness = -1000 - (makespan - self.deadline) * 10
        else:
            # Reward schedules with high security utility
            slack_time = self.deadline - makespan
            fitness = security_utility * 100 + slack_time
        
        return {
            'fitness': fitness,
            'makespan': makespan,
            'security_utility': security_utility,
            'schedule': schedule
        }
    
    def crossover_stage2(self, parent1, parent2):
        """Perform crossover between two parents for security level optimization."""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1)
        
        child = {
            'task_order': parent1['task_order'],
            'processor_assignment': copy.deepcopy(parent1['processor_assignment']),
            'security_levels': {}
        }
        
        # Multiple-point crossover for security levels
        for message_id in parent1['security_levels']:
            child['security_levels'][message_id] = {}
            
            # Randomly select security levels from either parent
            for service in ['confidentiality', 'integrity', 'authentication']:
                if random.random() < 0.5:
                    child['security_levels'][message_id][service] = parent1['security_levels'][message_id][service]
                else:
                    child['security_levels'][message_id][service] = parent2['security_levels'][message_id][service]
        
        return child
    
    def mutate_stage2(self, individual):
        """Mutate an individual's security levels."""
        mutated = copy.deepcopy(individual)
        
        for message_id in mutated['security_levels']:
            for service in ['confidentiality', 'integrity', 'authentication']:
                if random.random() < self.mutation_rate:
                    # Get the maximum possible level for this service
                    max_level = len(self.security.strengths[service]) - 1
                    
                    # Current level
                    current_level = mutated['security_levels'][message_id][service]
                    
                    # Randomly increase or decrease the level
                    if random.random() < 0.7:  # Bias towards increasing security
                        new_level = min(current_level + 1, max_level)
                    else:
                        new_level = max(current_level - 1, 0)
                    
                    mutated['security_levels'][message_id][service] = new_level
        
        return mutated
    
    def run_ga_stage2(self, base_individual):
        """Run the second stage of the genetic algorithm (security level optimization)."""
        # Initialize population
        population = self.generate_initial_population_stage2(base_individual)
        
        # Evaluate initial population
        fitness_results = [self.evaluate_fitness_stage2(individual) for individual in population]
        
        # Track best solution
        best_individual_idx = max(range(len(fitness_results)), key=lambda i: fitness_results[i]['fitness'])
        best_individual = copy.deepcopy(population[best_individual_idx])
        best_fitness = fitness_results[best_individual_idx]
        
        # Record initial best fitness for convergence tracking
        self.stage2_convergence.append(best_fitness['fitness'])
        
        print(f"Stage 2 Initial best fitness: {best_fitness['fitness']}, Security Utility: {best_fitness['security_utility']}")
        
        # Main GA loop
        for generation in range(self.generations):
            # Select elites
            sorted_indices = sorted(range(len(fitness_results)), key=lambda i: -fitness_results[i]['fitness'])
            elites = [copy.deepcopy(population[i]) for i in sorted_indices[:self.elite_size]]
            
            # Create new population
            new_population = elites.copy()
            
            # Fill the rest of the population with crossover and mutation
            while len(new_population) < self.population_size:
                # Select parents
                parent1 = self.tournament_selection(population, fitness_results)
                parent2 = self.tournament_selection(population, fitness_results)
                
                # Crossover
                child = self.crossover_stage2(parent1, parent2)
                
                # Mutation
                child = self.mutate_stage2(child)
                
                new_population.append(child)
            
            # Update population
            population = new_population
            
            # Evaluate new population
            fitness_results = [self.evaluate_fitness_stage2(individual) for individual in population]
            
            # Update best solution
            current_best_idx = max(range(len(fitness_results)), key=lambda i: fitness_results[i]['fitness'])
            if fitness_results[current_best_idx]['fitness'] > best_fitness['fitness']:
                best_individual = copy.deepcopy(population[current_best_idx])
                best_fitness = fitness_results[current_best_idx]
                print(f"Generation {generation}: New best fitness: {best_fitness['fitness']}, Security Utility: {best_fitness['security_utility']}")
            
            # Record best fitness for convergence tracking
            self.stage2_convergence.append(best_fitness['fitness'])
        
        # Return the best individual
        return best_individual, best_fitness
    
    def run(self):
        """Run the complete two-stage genetic algorithm."""
        start_time = time.time()
        
        # Stage 1: Task-Processor Mapping
        print("Starting Stage 1: Task-Processor Mapping")
        best_stage1_individual, best_stage1_fitness = self.run_ga_stage1()
        
        # Check if a valid solution was found
        if best_stage1_fitness['makespan'] > self.deadline:
            print(f"No valid solution found in Stage 1. Best makespan: {best_stage1_fitness['makespan']}, Deadline: {self.deadline}")
            return None, None
        
        # Stage 2: Security Level Optimization
        print("\nStarting Stage 2: Security Level Optimization")
        best_stage2_individual, best_stage2_fitness = self.run_ga_stage2(best_stage1_individual)
        
        # Create the final schedule
        tasks_copy = copy.deepcopy(self.tasks)
        processors_copy = copy.deepcopy(self.processors)
        
        # Apply the processor assignment
        for task in tasks_copy:
            task.assigned_processor = best_stage2_individual['processor_assignment'][task.task_id]
        
        # Simulate the schedule
        schedule, makespan, _ = self.simulate_schedule(tasks_copy, processors_copy, best_stage2_individual['security_levels'])
        
        # Apply security levels to the original messages
        for message in self.messages:
            if message.id in best_stage2_individual['security_levels']:
                for service in ['confidentiality', 'integrity', 'authentication']:
                    message.assigned_security[service] = best_stage2_individual['security_levels'][message.id][service]
        
        # Store the best solution
        self.best_schedule = schedule
        self.best_makespan = makespan
        self.best_security_utility = best_stage2_fitness['security_utility']
        
        elapsed_time = time.time() - start_time
        print(f"\nGA completed in {elapsed_time:.2f} seconds")
        print(f"Final makespan: {self.best_makespan}, Security Utility: {self.best_security_utility}")
        
        # Plot the results
        self.plot_results()
        
        return self.best_makespan, self.best_security_utility
    
    def plot_results(self):
        """Plot the results of the GA."""
        # Plot convergence graphs
        self.plot_convergence()
        
        # Plot the schedule
        self.plot_schedule()
        
        # Plot the security levels
        self.plot_security_levels()
    
    def plot_convergence(self):
        """Plot the convergence of fitness over generations."""
        plt.figure(figsize=(12, 5))
        
        # Stage 1 convergence
        plt.subplot(1, 2, 1)
        plt.plot(range(len(self.stage1_convergence)), self.stage1_convergence, 'b-')
        plt.title('Stage 1: Task-Processor Mapping Convergence')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.grid(True)
        
        # Stage 2 convergence
        plt.subplot(1, 2, 2)
        plt.plot(range(len(self.stage2_convergence)), self.stage2_convergence, 'r-')
        plt.title('Stage 2: Security Level Optimization Convergence')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.grid(True)
        
        plt.tight_layout()