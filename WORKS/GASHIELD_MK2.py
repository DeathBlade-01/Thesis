import numpy as np
import matplotlib.pyplot as plt
import copy
import random
from typing import List, Dict, Tuple, Any
import time

class Task:
    def __init__(self, task_id, name, execution_times, predecessors=None):
        self.task_id = task_id
        self.name = name
        self.execution_times = execution_times  # List of execution times on each processor
        self.predecessors = predecessors if predecessors else []
        self.successors = []
        self.priority = None
        self.assigned_processor = None
        self.start_time = None
        self.finish_time = None
        self.is_scheduled = False

class Message:
    def __init__(self, source_id, dest_id, size):
        self.source_id = source_id
        self.dest_id = dest_id
        self.size = size  # in KB
        
        # Security requirements as specified in Table 7
        self.min_security = {
            'confidentiality': 0.0,
            'integrity': 0.0,
            'authentication': 0.0
        }
        self.weights = {
            'confidentiality': 0.0,
            'integrity': 0.0,
            'authentication': 0.0
        }
        self.assigned_security = {
            'confidentiality': 0,  # Index of the protocol
            'integrity': 0,
            'authentication': 0
        }
        self.id = f"e_{source_id}_{dest_id}"
    
    def set_security_requirements(self, conf_min, integ_min, auth_min, conf_w, integ_w, auth_w):
        self.min_security['confidentiality'] = conf_min
        self.min_security['integrity'] = integ_min
        self.min_security['authentication'] = auth_min
        self.weights['confidentiality'] = conf_w
        self.weights['integrity'] = integ_w
        self.weights['authentication'] = auth_w

class Processor:
    def __init__(self, proc_id):
        self.proc_id = proc_id
        self.available_time = 0

class SecurityService:
    def __init__(self):
        # Security strengths for protocols based on Table 1 from paper
        self.strengths = {
            'confidentiality': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # 8 protocols
            'integrity': [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0],  # 7 protocols
            'authentication': [0.2, 0.5, 1.0]  # 3 protocols
        }
        
        # Security overheads (execution time) for each protocol on each processor
        # Based on Table 4 from paper (simplified for 2 processors)
        self.overheads = {
            'confidentiality': {
                # Format: [χ2 values for data-dependent overhead]
                1: [150, 145],  # Processor 1, 2
                2: [95, 90],
                3: [35, 30],
                4: [30, 32],
                5: [28, 27],
                6: [20, 22],
                7: [14, 15],
                8: [12, 13]
            },
            'integrity': {
                1: [22, 24],
                2: [16, 18],
                3: [11, 12],
                4: [9, 10],
                5: [6, 7],
                6: [5, 6],
                7: [4, 4.5]
            },
            'authentication': {
                # Format: [χ1 values for data-independent overhead]
                1: [80, 85],
                2: [135, 140],
                3: [155, 160]
            }
        }

class CommunicationNetwork:
    def __init__(self, num_processors, bandwidth=500):  # 500 KB/s as mentioned in the paper
        self.bandwidth = bandwidth
        self.num_processors = num_processors
        
    def get_communication_time(self, message, source_proc, dest_proc):
        if source_proc == dest_proc:
            return 0
        else:
            # Calculate comm time including security overhead
            return message.size / self.bandwidth

class GeneticAlgorithmScheduler:
    def __init__(self, tasks, messages, processors, network, security_service, deadline, 
                 population_size=100, generations=1000, elite_size=5, mutation_rate=0.2,
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

        self.stage1_convergence = []
        self.stage2_convergence = []
        self.generationsTrack = []  # Track generation numbers

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
            print(f"Task {task.task_id} ({task.name}) has priority {task.priority}")
    
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

    def calculate_est(self, task, processor_id, tasks, messages):
        """Calculate Earliest Start Time (EST) for a task on a processor."""
        # Get the processor
        processor = next((p for p in self.processors if p.proc_id == processor_id), None)
        if processor is None:
            raise ValueError(f"Processor {processor_id} not found.")
        
        # The earliest start time is at least the processor's available time
        est = processor.available_time
        
        # If the task has no predecessors, it can start at the processor's available time
        if not task.predecessors:
            return est
        
        # Calculate the earliest time this task can start considering all predecessors
        max_pred_finish_time = 0
        for pred_id in task.predecessors:
            pred_task = next((t for t in tasks if t.task_id == pred_id), None)
            if not pred_task or not pred_task.is_scheduled:
                return float('inf')  # Predecessor not scheduled yet
            
            # Get the message between the predecessor and the current task
            message = next((m for m in messages if m.source_id == pred_id and m.dest_id == task.task_id), None)
            
            if not message:
                # If there's no explicit message, just consider the predecessor's finish time
                max_pred_finish_time = max(max_pred_finish_time, pred_task.finish_time)
                continue
            
            # If the predecessor is on the same processor, no communication delay is needed
            if pred_task.assigned_processor == processor_id:
                comm_finish_time = pred_task.finish_time
            else:
                # Calculate communication time and security overhead
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
            
            # Update the maximum predecessor finish time
            max_pred_finish_time = max(max_pred_finish_time, comm_finish_time)
        
        # The earliest start time is the maximum of the processor's available time and the latest predecessor finish time
        return max(est, max_pred_finish_time)
    
    def simulate_schedule(self, tasks, processors, security_levels):
        """Simulate the schedule execution and return makespan and local slack times."""
        # Reset processor availability
        for processor in processors:
            processor.available_time = 0
        
        # Reset task scheduling status
        for task in tasks:
            task.is_scheduled = False
            task.start_time = None
            task.finish_time = None
        
        schedule = []
        local_slack_times = {}
        
        # Sort tasks by priority (descending)
        sorted_tasks = sorted(tasks, key=lambda t: -t.priority)
        
        for task in sorted_tasks:
            processor_id = task.assigned_processor
            processor = next((p for p in processors if p.proc_id == processor_id), None)
            
            if processor is None:
                raise ValueError(f"Processor {processor_id} not found.")
            
            # Calculate the earliest start time (EST) for the task
            est = self.calculate_est(task, processor_id, tasks, self.messages)
            
            # Ensure the task does not start before the processor is available
            est = max(est, processor.available_time)
            
            # Calculate the finish time
            finish_time = est + task.execution_times[processor_id - 1]
            
            # Update task scheduling information
            task.start_time = est
            task.finish_time = finish_time
            task.is_scheduled = True
            
            # Update processor availability
            processor.available_time = finish_time
            
            # Add the task to the schedule
            schedule.append({
                'task_id': task.task_id,
                'name': task.name,
                'processor': processor_id,
                'start_time': est,
                'finish_time': finish_time
            })
            
            # Debugging: Print task details
            print(f"Task {task.task_id} ({task.name}) scheduled on Processor {processor_id} from {est} to {finish_time}")
        
        # Calculate the makespan (total schedule length)
        makespan = max(task.finish_time for task in tasks if task.is_scheduled)
        return schedule, makespan, local_slack_times
    
    def calculate_security_utility(self, security_levels):
        """Calculate the total security utility based on the assigned security levels."""
        total_utility = 0
        
        for message in self.messages:
            if message.id not in security_levels:
                continue
                
            message_utility = 0
            for service in ['confidentiality', 'integrity', 'authentication']:
                # Get the protocol index assigned to this service for the message
                protocol_idx = security_levels[message.id][service]
                
                # Get the security strength of the assigned protocol
                strength = self.security.strengths[service][protocol_idx]
                
                # Get the weight of the service for this message
                weight = message.weights[service]
                
                # Calculate the weighted security utility for this service
                message_utility += weight * strength
            
            # Add the message's security utility to the total
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
        
        # Return the best individual
        self.stage2_convergence.append(best_fitness['fitness'])
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
        
        self.plot_convergence()
        self.plot_schedule()
        return self.best_makespan, self.best_security_utility
        
    def plot_convergence(self):
        """Plot the convergence of fitness over generations."""
        plt.figure(figsize=(12, 5))
        
        # Stage 1 convergence
        plt.subplot(1, 2, 1)
        if isinstance(self.stage1_convergence[0], dict):
            plt.plot(range(len(self.stage1_convergence)), [float(entry['fitness']) for entry in self.stage1_convergence], 'b-')
        else:
            plt.plot(range(len(self.stage1_convergence)), self.stage1_convergence, 'b-')
        
        plt.title('Stage 1: Task-Processor Mapping Convergence')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.grid(True)
        
        # Stage 2 convergence
        if self.stage2_convergence:
            plt.subplot(1, 2, 2)
            plt.plot(range(len(self.stage2_convergence)), self.stage2_convergence, 'r-', label='Security Optimization')
            plt.title('Stage 2: Security Level Optimization Convergence')
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('ga_convergence.png')
        plt.show()

    def plot_schedule(self):
        """Plot the schedule as a Gantt chart."""
        if not self.best_schedule:
            print("No schedule to plot")
            return
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define colors for each processor
        colors = plt.cm.tab10.colors
        
        # Plot each task
        for task in self.best_schedule:
            proc_idx = task['processor'] - 1  # Convert 1-based to 0-based indexing
            color = colors[proc_idx % len(colors)]
            
            # Plot the task as a horizontal bar
            ax.barh(
                task['processor'],
                task['finish_time'] - task['start_time'],
                left=task['start_time'],
                height=0.5,
                color=color,
                edgecolor='black',
                alpha=0.8
            )
            
            # Add task label
            ax.text(
                task['start_time'] + (task['finish_time'] - task['start_time']) / 2,
                task['processor'],
                f"T{task['task_id']}",
                ha='center',
                va='center',
                color='white',
                fontweight='bold'
            )
        
        # Add deadline line
        ax.axvline(x=self.deadline, color='red', linestyle='--', label='Deadline')
        
        # Set labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Processor')
        ax.set_title('Task Schedule Gantt Chart')
        
        # Set y-axis ticks to only show processor numbers
        ax.set_yticks(range(1, self.num_processors + 1))
        ax.set_yticklabels([f'P{i}' for i in range(1, self.num_processors + 1)])
        
        # Add makespan annotation
        ax.text(
            self.best_makespan,
            0.5,
            f'Makespan: {self.best_makespan:.2f}',
            ha='right',
            va='bottom',
            color='blue',
            fontweight='bold'
        )
        
        # Add a legend
        ax.legend()
        
        # Add grid for better readability
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Tight layout
        plt.tight_layout()
        
        # Save the figure
        plt.savefig('task_schedule.png')
        plt.show()

    def visualize_security_levels(self):
        """Visualize the security levels assigned to messages."""
        if not self.messages:
            print("No messages to visualize")
            return
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create data structures for plotting
        message_ids = [msg.id for msg in self.messages]
        conf_levels = [self.security.strengths['confidentiality'][msg.assigned_security['confidentiality']] 
                    for msg in self.messages]
        integ_levels = [self.security.strengths['integrity'][msg.assigned_security['integrity']] 
                    for msg in self.messages]
        auth_levels = [self.security.strengths['authentication'][msg.assigned_security['authentication']] 
                    for msg in self.messages]
        
        # Set up bar width and positions
        bar_width = 0.25
        r1 = np.arange(len(message_ids))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        
        # Create bars
        ax.bar(r1, conf_levels, width=bar_width, color='blue', label='Confidentiality')
        ax.bar(r2, integ_levels, width=bar_width, color='green', label='Integrity')
        ax.bar(r3, auth_levels, width=bar_width, color='red', label='Authentication')
        
        # Add labels and title
        ax.set_xlabel('Messages')
        ax.set_ylabel('Security Strength')
        ax.set_title('Security Levels Assigned to Messages')
        ax.set_xticks([r + bar_width for r in range(len(message_ids))])
        ax.set_xticklabels(message_ids)
        ax.set_ylim(0, 1.2)  # Set y-axis limit
        
        # Add a legend
        ax.legend()
        
        # Add grid for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add security utility annotation
        ax.text(
            len(message_ids) - 1,
            1.1,
            f'Total Security Utility: {self.best_security_utility:.2f}',
            ha='right',
            va='center',
            color='purple',
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="purple", alpha=0.8)
        )
        
        # Tight layout
        plt.tight_layout()
        
        # Save the figure
        plt.savefig('security_levels.png')
        plt.show()

    def generate_report(self):
        """Generate a comprehensive report of the schedule and security assignments."""
        if not self.best_schedule or not self.messages:
            print("No schedule or messages to report")
            return
        
        print("\n" + "="*80)
        print(f"{'SECURITY-AWARE TASK SCHEDULING REPORT':^80}")
        print("="*80)
        
        # Schedule summary
        print(f"\n{'SCHEDULE SUMMARY':^80}")
        print("-"*80)
        print(f"Total Tasks: {len(self.tasks)}")
        print(f"Total Processors: {self.num_processors}")
        print(f"Deadline: {self.deadline}")
        print(f"Achieved Makespan: {self.best_makespan:.2f}")
        print(f"Deadline Met: {'Yes' if self.best_makespan <= self.deadline else 'No'}")
        print(f"Slack Time: {max(0, self.deadline - self.best_makespan):.2f}")
        
        # Task assignments
        print(f"\n{'TASK ASSIGNMENTS':^80}")
        print("-"*80)
        print(f"{'Task ID':<10}{'Task Name':<20}{'Processor':<12}{'Start Time':<15}{'Finish Time':<15}")
        print("-"*80)
        
        # Sort tasks by start time
        sorted_schedule = sorted(self.best_schedule, key=lambda x: x['start_time'])
        
        for task in sorted_schedule:
            print(f"{task['task_id']:<10}{task['name']:<20}{task['processor']:<12}{task['start_time']:<15.2f}{task['finish_time']:<15.2f}")
        
        # Security assignments
        print(f"\n{'SECURITY ASSIGNMENTS':^80}")
        print("-"*80)
        print(f"{'Message ID':<15}{'Confidentiality':<20}{'Integrity':<20}{'Authentication':<20}")
        print("-"*80)
        
        for message in self.messages:
            conf_level = self.security.strengths['confidentiality'][message.assigned_security['confidentiality']]
            integ_level = self.security.strengths['integrity'][message.assigned_security['integrity']]
            auth_level = self.security.strengths['authentication'][message.assigned_security['authentication']]
            
            print(f"{message.id:<15}{conf_level:<20.2f}{integ_level:<20.2f}{auth_level:<20.2f}")
        
        # Security utility
        print(f"\n{'SECURITY UTILITY':^80}")
        print("-"*80)
        print(f"Total Security Utility: {self.best_security_utility:.2f}")
        
        print("\n" + "="*80)
    def visualize_convergence_3d(self):
        """Create a 3D visualization of the convergence of makespan vs security utility."""
        # This function requires stage history which we'll add below
        if not hasattr(self, 'convergence_history'):
            print("No convergence history available")
            return
        
        # Create a 3D figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract data for plotting
        generations = [data['generation'] for data in self.convergence_history]
        makespans = [data['makespan'] for data in self.convergence_history]
        security_utils = [data['security_utility'] for data in self.convergence_history]
        
        # Create scatter plot
        scatter = ax.scatter(generations, makespans, security_utils, 
                            c=security_utils, cmap='viridis', 
                            s=50, alpha=0.6)
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Security Utility')
        
        # Connect points to show evolution
        ax.plot(generations, makespans, security_utils, 'k-', alpha=0.3)
        
        # Add labels for key points
        # Find best makespan and security points
        best_makespan_idx = np.argmin(makespans)
        best_security_idx = np.argmax(security_utils)
        final_idx = len(generations) - 1
        
        # Annotate best points
        ax.text(generations[best_makespan_idx], makespans[best_makespan_idx], security_utils[best_makespan_idx], 
                "Best Makespan", color='red', fontweight='bold')
        ax.text(generations[best_security_idx], makespans[best_security_idx], security_utils[best_security_idx], 
                "Best Security", color='blue', fontweight='bold')
        ax.text(generations[final_idx], makespans[final_idx], security_utils[final_idx], 
                "Final Solution", color='green', fontweight='bold')
        
        # Add deadline plane
        x_deadline = np.array([min(generations), max(generations)])
        y_deadline = np.array([self.deadline, self.deadline])
        z_deadline = np.array([min(security_utils), max(security_utils)])
        x_grid, z_grid = np.meshgrid(x_deadline, z_deadline)
        y_grid = np.ones_like(x_grid) * self.deadline
        
        ax.plot_surface(x_grid, y_grid, z_grid, color='red', alpha=0.2)
        
        # Add labels and title
        ax.set_xlabel('Generation')
        ax.set_ylabel('Makespan')
        ax.set_zlabel('Security Utility')
        ax.set_title('3D Visualization of GA Convergence')
        
        # Save the figure
        plt.savefig('convergence_3d.png')
        plt.show()

    # Add the ability to track convergence history during optimization
    def run_ga_stage1(self):
        """Run the first stage of the genetic algorithm (task-processor mapping)."""
        # Initialize tracking of convergence history if not present
        if not hasattr(self, 'convergence_history'):
            self.convergence_history = []
        
        # Initialize population
        population = self.generate_initial_population_stage1()
        
        # Evaluate initial population
        fitness_results = [self.evaluate_fitness_stage1(individual) for individual in population]
        
        # Track best solution
        best_individual_idx = max(range(len(fitness_results)), key=lambda i: fitness_results[i]['fitness'])
        best_individual = copy.deepcopy(population[best_individual_idx])
        best_fitness = fitness_results[best_individual_idx]
        
        # Record initial best solution
        self.convergence_history.append({
            'generation': 0,
            'stage': 1,
            'fitness': best_fitness['fitness'],
            'makespan': best_fitness['makespan'],
            'security_utility': best_fitness['security_utility']
        })
        
        # Append to stage1_convergence as a dictionary
        self.stage1_convergence.append({
            'fitness': best_fitness['fitness'],
            'makespan': best_fitness['makespan'],
            'security_utility': best_fitness['security_utility']
        })
        
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
            
            # Record convergence data
            self.convergence_history.append({
                'generation': generation + 1,
                'stage': 1,
                'fitness': fitness_results[current_best_idx]['fitness'],
                'makespan': fitness_results[current_best_idx]['makespan'],
                'security_utility': fitness_results[current_best_idx]['security_utility']
            })
            
            # Append to stage1_convergence as a dictionary
            self.stage1_convergence.append({
                'fitness': fitness_results[current_best_idx]['fitness'],
                'makespan': fitness_results[current_best_idx]['makespan'],
                'security_utility': fitness_results[current_best_idx]['security_utility']
            })
            
            if fitness_results[current_best_idx]['fitness'] > best_fitness['fitness']:
                best_individual = copy.deepcopy(population[current_best_idx])
                best_fitness = fitness_results[current_best_idx]
                print(f"Generation {generation}: New best fitness: {best_fitness['fitness']}, Makespan: {best_fitness['makespan']}")
        
        # Return the best individual
        return best_individual, best_fitness

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
        
        # Record initial best solution for stage 2
        self.convergence_history.append({
            'generation': 0,
            'stage': 2,
            'fitness': best_fitness['fitness'],
            'makespan': best_fitness['makespan'],
            'security_utility': best_fitness['security_utility']
        })
        
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
            
            # Record convergence data
            self.convergence_history.append({
                'generation': generation + 1,
                'stage': 2,
                'fitness': fitness_results[current_best_idx]['fitness'],
                'makespan': fitness_results[current_best_idx]['makespan'],
                'security_utility': fitness_results[current_best_idx]['security_utility']
            })
            
            if fitness_results[current_best_idx]['fitness'] > best_fitness['fitness']:
                best_individual = copy.deepcopy(population[current_best_idx])
                best_fitness = fitness_results[current_best_idx]
                print(f"Generation {generation}: New best fitness: {best_fitness['fitness']}, Security Utility: {best_fitness['security_utility']}")
        
        # Return the best individual
        self.stage2_convergence.append(best_fitness['fitness'])
        return best_individual, best_fitness

    # Update the main run function to use the new visualization features
    def run(self):
        """Run the complete two-stage genetic algorithm."""
        start_time = time.time()
        
        # Initialize tracking of convergence history
        self.convergence_history = []
        self.stage1_convergence = []  # Initialize stage1_convergence as an empty list
        self.stage2_convergence = []  # Initialize stage2_convergence as an empty list
        
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
        
        # Generate visualizations
        self.plot_convergence()
        self.plot_schedule()
        self.visualize_security_levels()
        self.visualize_convergence_3d()
        self.generate_report()
        
        return self.best_makespan, self.best_security_utility

# Add a main function to demonstrate the complete algorithm with an example
def main():
    """Run a demonstration of the security-aware task scheduling algorithm."""
    # Create sample tasks, messages, processors, and security services
    # tasks = [
    #     Task(1, "Task1", [14, 16], []),
    #     Task(2, "Task2", [13, 19], []),
    #     Task(3, "Task3", [11, 13], [1]),
    #     Task(4, "Task4", [13, 8], [1]),
    #     Task(5, "Task5", [12, 13], [2]),
    #     Task(6, "Task6", [13, 16], [2]),
    #     Task(7, "Task7", [7, 15], [3, 4]),
    #     Task(8, "Task8", [5, 11], [3, 5]),
    #     Task(9, "Task9", [18, 12], [6]),
    #     Task(10, "Task10", [21, 7], [7, 8, 9])
    # ]

    tasks = [
        Task(1, "Wheel Speed 1", [400, 410]),
        Task(2, "Wheel Speed 2", [400, 410]),
        Task(3, "Wheel Speed 3", [400, 410]),
        Task(4, "Wheel Speed 4", [400, 410]),
        Task(5, "Driver Input", [120, 130]),
        Task(6, "Slip Calculator", [168, 166], [1, 2, 3, 4]),
        Task(7, "Valve Control", [92, 94]),
        Task(8, "Actuator", [192, 190], [5, 6, 7]),
        Task(9, "Brake Control", [84, 86], [8]),
        Task(10, "Throttle Control", [120, 124], [8])
    ]
    
    # Create messages between tasks
    messages = [
        Message(1, 6, 100),
        Message(2, 6, 100),
        Message(3, 6, 100),
        Message(4, 6, 100),
        Message(5, 8, 100),
        Message(6, 8, 100),
        Message(7, 8, 100),
        Message(8, 9, 100),
        Message(8, 10, 100)
    ]
    
    # Set security requirements for messages
    for i, message in enumerate(messages):
        # Set different security requirements for each message
        conf_min = 0.1 + (i % 3) * 0.1
        integ_min = 0.1 + (i % 2) * 0.1
        auth_min = 0.2 + (i % 2) * 0.3
        
        # Set weights for security services
        conf_weight = 0.4 - (i % 3) * 0.1
        integ_weight = 0.3 + (i % 2) * 0.1
        auth_weight = 0.3 - (i % 2) * 0.1
        
        message.set_security_requirements(conf_min, integ_min, auth_min, conf_weight, integ_weight, auth_weight)
    
    # Create processors
    processors = [Processor(1), Processor(2)]
    
    # Create network and security service
    network = CommunicationNetwork(2)
    security_service = SecurityService()
    
    # Set deadline
    deadline = 1600
    
    # Create the GA scheduler
    scheduler = GeneticAlgorithmScheduler(
        tasks, messages, processors, network, security_service, deadline,
        population_size=50, generations=50, elite_size=5, mutation_rate=0.2,
        crossover_rate=0.8, tournament_size=3
    )
    
    # Run the scheduler
    makespan, security_utility = scheduler.run()
    
    if makespan is not None:
        print(f"\nScheduling completed successfully!")
        print(f"Final makespan: {makespan:.2f}")
        print(f"Security utility: {security_utility:.2f}")
        print(f"Deadline satisfied: {'Yes' if makespan <= deadline else 'No'}")
    else:
        print("\nScheduling failed to find a valid solution.")

if __name__ == "__main__":
    main()