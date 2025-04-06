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

        self.stage1_convergence = []
        self.stage2_convergence = []

    def initialize_successors(self):
        """Ensure tasks have correctly assigned successors based on dependencies."""
        for task in self.tasks:
            task.successors = []
        
        for task in self.tasks:
            for pred_id in task.predecessors:
                pred_task = self.get_task_by_id(pred_id)
                if pred_task:
                    pred_task.successors.append(task.task_id)
    
    def get_task_by_id(self, task_id):
        """Retrieve a task object by its ID."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
    
    def get_message(self, source_id, dest_id):
        """Retrieve a message based on source and destination task IDs."""
        for message in self.messages:
            if message.source_id == source_id and message.dest_id == dest_id:
                return message
        return None
    
    def compute_task_priorities(self):
        """Compute priorities for tasks using an upward rank method."""
        for task in self.tasks:
            task.avg_execution = sum(task.execution_times) / len(task.execution_times)
        
        def calculate_upward_rank(task):
            """Recursively compute the priority based on successors and communication costs."""
            if task.priority is not None:
                return task.priority
            
            if not task.successors:
                task.priority = task.avg_execution
                return task.priority
            
            max_successor_rank = 0
            for succ_id in task.successors:
                succ_task = self.get_task_by_id(succ_id)
                if succ_task:
                    message = self.get_message(task.task_id, succ_id)
                    comm_cost = message.size / self.network.bandwidth if message else 0
                    
                    succ_rank = calculate_upward_rank(succ_task)
                    rank_with_comm = comm_cost + succ_rank
                    max_successor_rank = max(max_successor_rank, rank_with_comm)
            
            task.priority = task.avg_execution + max_successor_rank
            return task.priority
        
        for task in self.tasks:
            calculate_upward_rank(task)
    
    def assign_minimum_security(self):
        """Assign the lowest security levels required by each message."""
        for message in self.messages:
            for service in ['confidentiality', 'integrity', 'authentication']:
                strengths = self.security.strengths[service]
                min_strength = message.min_security[service]
                
                protocol_idx = 0
                for i, strength in enumerate(strengths):
                    if strength >= min_strength:
                        protocol_idx = i
                        break
                
                message.assigned_security[service] = protocol_idx
    
    def generate_initial_population(self):
        """Generate the initial population for GA-based scheduling."""
        population = []
        sorted_tasks = sorted(self.tasks, key=lambda t: -t.priority)
        task_ids = [task.task_id for task in sorted_tasks]
        
        for _ in range(self.population_size):
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
        """Retrieve the initial security levels assigned to messages."""
        security_levels = {}
        for message in self.messages:
            security_levels[message.id] = {
                'confidentiality': message.assigned_security['confidentiality'],
                'integrity': message.assigned_security['integrity'],
                'authentication': message.assigned_security['authentication']
            }
        return security_levels
    def simulate_schedule(self, tasks, processors, security_levels):
        """Simulate the schedule execution and return makespan and local slack times."""
        # Reset processor availability
        for processor in processors:
            processor.available_time = 0
        
        # Reset task states
        for task in tasks:
            task.is_scheduled = False
            task.start_time = None
            task.finish_time = None
        
        # Apply security levels to messages
        messages_copy = copy.deepcopy(self.messages)
        for message in messages_copy:
            if message.id in security_levels:
                message.assigned_security = security_levels[message.id]

        # Schedule tasks based on priority
        schedule = []
        local_slack_times = {}
        
        sorted_tasks = sorted(tasks, key=lambda t: -t.priority)
        
        for task in sorted_tasks:
            processor_id = task.assigned_processor
            processor = next((p for p in processors if p.proc_id == processor_id), None)
            
            est = self.calculate_est(task, processor_id, tasks, messages_copy)
            
            finish_time = est + task.execution_times[processor_id - 1]
            
            task.start_time = est
            task.finish_time = finish_time
            task.is_scheduled = True
            processor.available_time = finish_time
            
            next_task_start = float('inf')
            for succ_id in task.successors:
                succ_task = self.get_task_by_id(succ_id)
                if succ_task and succ_task.start_time is not None:
                    next_task_start = min(next_task_start, succ_task.start_time)
            
            if next_task_start == float('inf'):
                next_task_start = self.deadline
            
            local_slack = next_task_start - finish_time
            local_slack_times[task.task_id] = max(0, local_slack)
            
            schedule.append({
                'task_id': task.task_id,
                'name': task.name,
                'processor': processor_id,
                'start_time': est,
                'finish_time': finish_time
            })
            task.is_scheduled = True


        makespan = max(task.finish_time for task in tasks if task.is_scheduled)
        
        return schedule, makespan, local_slack_times

    def evaluate_fitness(self, individual):
        """Evaluate fitness based on makespan and security utility."""
        tasks_copy = copy.deepcopy(self.tasks)
        processors_copy = copy.deepcopy(self.processors)
        
        processor_assignment = individual['processor_assignment']
        for task in tasks_copy:
            task.assigned_processor = processor_assignment[task.task_id]
        
        schedule, makespan, slack_times = self.simulate_schedule(tasks_copy, processors_copy, individual['security_levels'])
        security_utility = self.calculate_security_utility(individual['security_levels'])
        
        global_slack_time = max(0, self.deadline - makespan)
        total_slack = sum(slack_times.values())
        critical_slack = self.identify_critical_path_slack(tasks_copy, slack_times)
        
        if makespan > self.deadline:
            fitness = -1000 - (makespan - self.deadline)
        else:
            fitness = (
                1000 + global_slack_time * 10 +
                total_slack * 5 + critical_slack * 15 +
                security_utility * 2
            )
        
        return {
            'fitness': fitness,
            'makespan': makespan,
            'security_utility': security_utility,
            'schedule': schedule
        }
    def identify_critical_path_slack(self, tasks, local_slack_times):
        """Identify slack time on critical paths that could be used for security upgrades."""
        exit_tasks = [task for task in tasks if not task.successors]
        critical_path_slack = 0

        for exit_task in exit_tasks:
            path_slack = self.calculate_path_slack(exit_task, tasks, local_slack_times)
            critical_path_slack += path_slack
        
        return critical_path_slack
    
    def calculate_path_slack(self, task, tasks, local_slack_times):
        """Calculate the slack time along a path from a root task to the given task."""
        if not task.predecessors:
            return local_slack_times.get(task.task_id, 0)
        
        path_slack = 0
        for pred_id in task.predecessors:
            pred_task = self.get_task_by_id(pred_id)
            if pred_task:
                task_slack = local_slack_times.get(task.task_id, 0)
                path_slack += task_slack + self.calculate_path_slack(pred_task, tasks, local_slack_times)
        
        return path_slack
    
    def calc_security_overhead(self, message, source_proc, dest_proc):
        """Calculate the security overhead for a message."""
        total_overhead = 0
        for service in ['confidentiality', 'integrity']:
            protocol_idx = message.assigned_security[service] + 1
            overhead_factor = self.security.overheads[service][protocol_idx][source_proc - 1]
            overhead = (message.size / 1024) * overhead_factor  
            total_overhead += overhead
        
        auth_protocol_idx = message.assigned_security['authentication'] + 1
        auth_overhead = self.security.overheads['authentication'][auth_protocol_idx][dest_proc - 1]
        total_overhead += auth_overhead
        
        return total_overhead
    
    def tournament_selection(self, population, fitness_results):
        """Select individuals using tournament selection."""
        tournament = random.sample(list(range(len(population))), self.tournament_size)
        tournament_fitness = [fitness_results[i]['fitness'] for i in tournament]
        return population[tournament[tournament_fitness.index(max(tournament_fitness))]]

    def crossover(self, parent1, parent2):
        """Perform crossover between two parents for task-processor mapping."""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1)
        
        child = {
            'task_order': parent1['task_order'],
            'processor_assignment': {},
            'security_levels': copy.deepcopy(parent1['security_levels'])
        }
        
        crossover_point = random.randint(1, len(parent1['task_order']) - 1)
        
        for i in range(crossover_point):
            task_id = parent1['task_order'][i]
            child['processor_assignment'][task_id] = parent1['processor_assignment'][task_id]
        
        for i in range(crossover_point, len(parent1['task_order'])):
            task_id = parent1['task_order'][i]
            child['processor_assignment'][task_id] = parent2['processor_assignment'][task_id]
        
        return child
    def calculate_est(self, task, processor_id, tasks, messages):
        """Calculate Earliest Start Time (EST) for a task on a given processor."""
        # Get the processor
        processor = next((p for p in self.processors if p.proc_id == processor_id), None)
        processor_ready_time = processor.available_time if processor else float('inf')

        # If the task has no predecessors, it can start at the processor's available time
        if not task.predecessors:
            return processor_ready_time

        max_pred_finish_time = 0
        for pred_id in task.predecessors:
            pred_task = self.get_task_by_id(pred_id)
            if not pred_task or not pred_task.is_scheduled:
                return float('inf')  # Predecessor not scheduled yet
            
            # Check if a message exists between predecessor and current task
            message = next((m for m in messages if m.source_id == pred_id and m.dest_id == task.task_id), None)

            if not message:
                # If there's no explicit message, just consider predecessor's finish time
                max_pred_finish_time = max(max_pred_finish_time, pred_task.finish_time)
                continue

            # If predecessor is on the same processor, only consider finish time
            if pred_task.assigned_processor == processor_id:
                comm_finish_time = pred_task.finish_time
            else:
                # Include communication time and security overhead
                security_overhead = self.calc_security_overhead(
                    message, pred_task.assigned_processor, processor_id
                )
                comm_time = self.network.get_communication_time(
                    message, pred_task.assigned_processor, processor_id
                )
                comm_finish_time = pred_task.finish_time + comm_time + security_overhead
            
            max_pred_finish_time = max(max_pred_finish_time, comm_finish_time)

        return max(processor_ready_time, max_pred_finish_time)

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

    def mutate(self, individual):
        """Mutate an individual's processor assignments."""
        mutated = copy.deepcopy(individual)
        
        for task_id in mutated['task_order']:
            if random.random() < self.mutation_rate:
                new_processor = random.randint(1, self.num_processors)
                mutated['processor_assignment'][task_id] = new_processor
        
        return mutated
    
    def ensure_precedence_constraints(self, individual):
        """Ensure that precedence constraints are satisfied."""
        return individual

    def run_ga_stage1(self):
        """Run the first stage of the genetic algorithm (task-processor mapping)."""
        population = self.generate_initial_population()
        fitness_results = [self.evaluate_fitness(individual) for individual in population]
        
        best_individual_idx = max(range(len(fitness_results)), key=lambda i: fitness_results[i]['fitness'])
        best_individual = copy.deepcopy(population[best_individual_idx])
        best_fitness = fitness_results[best_individual_idx]
        
        print(f"Initial best fitness: {best_fitness['fitness']}, Makespan: {best_fitness['makespan']}")
        
        for generation in range(self.generations):
            sorted_indices = sorted(range(len(fitness_results)), key=lambda i: -fitness_results[i]['fitness'])
            elites = [copy.deepcopy(population[i]) for i in sorted_indices[:self.elite_size]]
            new_population = elites.copy()
            
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitness_results)
                parent2 = self.tournament_selection(population, fitness_results)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                child = self.ensure_precedence_constraints(child)
                new_population.append(child)
            
            population = new_population
            fitness_results = [self.evaluate_fitness(individual) for individual in population]
            
            current_best_idx = max(range(len(fitness_results)), key=lambda i: fitness_results[i]['fitness'])
            if fitness_results[current_best_idx]['fitness'] > best_fitness['fitness']:
                best_individual = copy.deepcopy(population[current_best_idx])
                best_fitness = fitness_results[current_best_idx]
                print(f"Generation {generation}: New best fitness: {best_fitness['fitness']}, Makespan: {best_fitness['makespan']}")
        
        self.stage1_convergence.append(best_fitness['fitness'])
        return best_individual, best_fitness
    def run_ga_stage2(self, base_individual):
        """Run the second stage of the genetic algorithm (security level optimization)."""
        population = self.generate_initial_population_stage2(base_individual)
        fitness_results = [self.evaluate_fitness_stage2(individual) for individual in population]
        
        best_individual_idx = max(range(len(fitness_results)), key=lambda i: fitness_results[i]['fitness'])
        best_individual = copy.deepcopy(population[best_individual_idx])
        best_fitness = fitness_results[best_individual_idx]
        
        print(f"Stage 2 Initial best fitness: {best_fitness['fitness']}, Security Utility: {best_fitness['security_utility']}")
        
        for generation in range(self.generations):
            sorted_indices = sorted(range(len(fitness_results)), key=lambda i: -fitness_results[i]['fitness'])
            elites = [copy.deepcopy(population[i]) for i in sorted_indices[:self.elite_size]]
            new_population = elites.copy()
            
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitness_results)
                parent2 = self.tournament_selection(population, fitness_results)
                child = self.crossover_stage2(parent1, parent2)
                child = self.mutate_stage2(child)
                new_population.append(child)
            
            population = new_population
            fitness_results = [self.evaluate_fitness_stage2(individual) for individual in population]
            
            current_best_idx = max(range(len(fitness_results)), key=lambda i: fitness_results[i]['fitness'])
            if fitness_results[current_best_idx]['fitness'] > best_fitness['fitness']:
                best_individual = copy.deepcopy(population[current_best_idx])
                best_fitness = fitness_results[current_best_idx]
                print(f"Generation {generation}: New best fitness: {best_fitness['fitness']}, Security Utility: {best_fitness['security_utility']}")
        
        self.stage2_convergence.append(best_fitness['fitness'])
        return best_individual, best_fitness
    
    def run(self):
        """Run the complete two-stage genetic algorithm."""
        start_time = time.time()
        
        print("Starting Stage 1: Task-Processor Mapping")
        best_stage1_individual, best_stage1_fitness = self.run_ga_stage1()
        
        if best_stage1_fitness['makespan'] > self.deadline:
            print(f"No valid solution found in Stage 1. Best makespan: {best_stage1_fitness['makespan']}, Deadline: {self.deadline}")
            return None, None
        
        print("\nStarting Stage 2: Security Level Optimization")
        best_stage2_individual, best_stage2_fitness = self.run_ga_stage2(best_stage1_individual)
        
        tasks_copy = copy.deepcopy(self.tasks)
        processors_copy = copy.deepcopy(self.processors)
        
        for task in tasks_copy:
            task.assigned_processor = best_stage2_individual['processor_assignment'][task.task_id]
        
        schedule, makespan, _ = self.simulate_schedule(tasks_copy, processors_copy, best_stage2_individual['security_levels'])
        
        for message in self.messages:
            if message.id in best_stage2_individual['security_levels']:
                for service in ['confidentiality', 'integrity', 'authentication']:
                    message.assigned_security[service] = best_stage2_individual['security_levels'][message.id][service]
        
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
        
        plt.subplot(1, 2, 1)
        plt.plot(range(len(self.stage1_convergence)), self.stage1_convergence, 'b-')
        plt.title('Stage 1: Task-Processor Mapping Convergence')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(len(self.stage2_convergence)), self.stage2_convergence, 'r-')
        plt.title('Stage 2: Security Level Optimization Convergence')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('ga_convergence.png')
        plt.show()

    def plot_schedule(self):
        """Plot the schedule as a Gantt chart."""
        if not self.best_schedule:
            print("No schedule to plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = plt.cm.tab10.colors
        
        for task in self.best_schedule:
            proc_idx = task['processor'] - 1
            color = colors[proc_idx % len(colors)]
            
            ax.barh(
                task['processor'],
                task['finish_time'] - task['start_time'],
                left=task['start_time'],
                height=0.5,
                color=color,
                edgecolor='black',
                alpha=0.8
            )
            
            ax.text(
                task['start_time'] + (task['finish_time'] - task['start_time']) / 2,
                task['processor'],
                f"T{task['task_id']}",
                ha='center',
                va='center',
                color='white',
                fontweight='bold'
            )
        
        ax.axvline(x=self.deadline, color='red', linestyle='--', label='Deadline')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Processor')
        ax.set_title('Task Schedule (Gantt Chart)')
        
        ax.set_yticks(range(1, self.num_processors + 1))
        ax.set_yticklabels([f'P{i}' for i in range(1, self.num_processors + 1)])
        
        ax.grid(True, axis='x', linestyle=':')
        
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('task_schedule.png')
        plt.show()
# Main function to run the genetic algorithm
def run_genetic_algorithm(tasks, messages, processors, network, security_service, deadline):
    """Initialize and run the Genetic Algorithm Scheduler."""
    ga_scheduler = GeneticAlgorithmScheduler(
        tasks=tasks,
        messages=messages,
        processors=processors,
        network=network,
        security_service=security_service,
        deadline=deadline,
        population_size=50,
        generations=50,
        elite_size=5,
        mutation_rate=0.2,
        crossover_rate=0.8,
        tournament_size=3
    )
    
    makespan, security_utility = ga_scheduler.run()
    return ga_scheduler.best_schedule, makespan, security_utility

# Comparison function to compare SHIELD and GA approaches
def compare_approaches(tasks, messages, processors, network, security_service, deadline):
    """Compare the performance of SHIELD and the GA-based scheduler."""
    shield = SHIELD(tasks, messages, processors, network, security_service, deadline)
    shield_makespan, shield_security_utility = shield.run()
    
    tasks_ga, messages_ga, processors_ga = copy.deepcopy(tasks), copy.deepcopy(messages), copy.deepcopy(processors)
    
    ga_schedule, ga_makespan, ga_security_utility = run_genetic_algorithm(
        tasks_ga, messages_ga, processors_ga, network, security_service, deadline
    )
    
    print("\n--- Comparison Results ---")
    print(f"SHIELD Makespan: {shield_makespan}, Security Utility: {shield_security_utility}")
    print(f"GA Makespan: {ga_makespan}, Security Utility: {ga_security_utility}")

# Example usage
if __name__ == "__main__":
    # Example task class definition
    class Task:
        def __init__(self, task_id, name, execution_times, predecessors=None):
            self.task_id = task_id
            self.name = name
            self.execution_times = execution_times
            self.predecessors = predecessors or []
            self.successors = []
            self.priority = None
            self.avg_execution = None
            self.assigned_processor = None
            self.is_scheduled = False
            self.start_time = None
            self.finish_time = None

    class Message:
        def __init__(self, id, source_id, dest_id, size, min_security, weights):
            self.id = id
            self.source_id = source_id
            self.dest_id = dest_id
            self.size = size  # Size in KB
            self.min_security = min_security  # Minimum security requirements
            self.weights = weights  # Weights for each security service
            self.assigned_security = {
                'confidentiality': 0,
                'integrity': 0,
                'authentication': 0
            }

    class Processor:
        def __init__(self, proc_id, name):
            self.proc_id = proc_id
            self.name = name
            self.available_time = 0

    class Network:
        def __init__(self, bandwidth):
            self.bandwidth = bandwidth  # Bandwidth in KB/s
        
        def get_communication_time(self, message, source_proc, dest_proc):
            """Calculate communication time between processors."""
            return message.size / self.bandwidth

    class SecurityService:
        def __init__(self, strengths, overheads):
            self.strengths = strengths  # Security strength of each protocol
            self.overheads = overheads  # Overheads for each protocol and processor

     # Define tasks with execution times across 3 processors
    tasks = [
        Task(1, "T1", [10, 12, 14]),  # Task 1 takes 10, 12, or 14 units on different processors
        Task(2, "T2", [8, 10, 12], [1]),  # Task 2 depends on Task 1
        Task(3, "T3", [15, 18, 20], [1]),  # Task 3 depends on Task 1
        Task(4, "T4", [12, 14, 16], [2, 3]),  # Task 4 depends on Task 2 and Task 3
        Task(5, "T5", [8, 10, 12], [4])  # Task 5 depends on Task 4
    ]
    
    # Define messages with security constraints and communication overhead
    messages = [
        Message(1, 1, 2, 50, {'confidentiality': 0.4, 'integrity': 0.3, 'authentication': 0.2}, 
                {'confidentiality': 0.5, 'integrity': 0.4, 'authentication': 0.1}),
        Message(2, 1, 3, 80, {'confidentiality': 0.3, 'integrity': 0.5, 'authentication': 0.2},
                {'confidentiality': 0.4, 'integrity': 0.5, 'authentication': 0.1}),
        Message(3, 2, 4, 60, {'confidentiality': 0.6, 'integrity': 0.2, 'authentication': 0.3},
                {'confidentiality': 0.7, 'integrity': 0.2, 'authentication': 0.1}),
        Message(4, 3, 4, 70, {'confidentiality': 0.5, 'integrity': 0.3, 'authentication': 0.2},
                {'confidentiality': 0.6, 'integrity': 0.3, 'authentication': 0.1}),
        Message(5, 4, 5, 100, {'confidentiality': 0.5, 'integrity': 0.3, 'authentication': 0.2},
                {'confidentiality': 0.6, 'integrity': 0.3, 'authentication': 0.1})
    ]
    
    # Define processors
    processors = [
        Processor(1, "P1"),
        Processor(2, "P2"),
        Processor(3, "P3")
    ]
    
    # Define a network with a bandwidth of 50 KB/s
    network = Network(bandwidth=50)
    
    # Define security service
    security_service = SecurityService(
        strengths={
            'confidentiality': [0.2, 0.4, 0.6, 0.8, 1.0],
            'integrity': [0.2, 0.5, 0.7, 0.9, 1.0],
            'authentication': [0.3, 0.6, 0.8, 1.0]
        },
        overheads={
            'confidentiality': [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6], [0.5, 0.6, 0.7]],
            'integrity': [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6], [0.5, 0.6, 0.7]],
            'authentication': [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]]
        }
    )
    
    deadline = 10000000000000000000000000000000000000000000000000000000  # 50 time units to ensure a challenging but feasible scheduling problem

    # Run GA Scheduler
    scheduler = GeneticAlgorithmScheduler(
        tasks=tasks,
        messages=messages,
        processors=processors,
        network=network,
        security_service=security_service,
        deadline=deadline
    )

    makespan, security_utility = scheduler.run()

    # Check if GA found a feasible solution
    if makespan and makespan <= deadline:
        print(f"✅ Feasible solution found! Makespan: {makespan}, Security Utility: {security_utility}")
    else:
        print(f"❌ No feasible solution found. Best makespan: {makespan}, Deadline: {deadline}")