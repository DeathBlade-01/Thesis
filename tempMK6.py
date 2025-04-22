import numpy as np
import matplotlib.pyplot as plt
import copy
import random
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
        self.task_history = []  # Track which tasks were scheduled on this processor

class SecurityService:
    def __init__(self):
        # Security strengths for protocols based on Table 1 from paper
        self.strengths = {
            'confidentiality': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # 8 protocols
            'integrity': [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0],  # 7 protocols
            'authentication': [0.2, 0.5, 1.0]  # 3 protocols
        }
        
        # Security overheads (execution time) for each protocol on each processor
        self.overheads = {
            'confidentiality': {
                1: [150, 145],
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
                1: [80, 85],
                2: [135, 140],
                3: [155, 160]
            }
        }

class CommunicationNetwork:
    def __init__(self, num_processors, bandwidth=500):
        self.bandwidth = bandwidth
        self.num_processors = num_processors
        
    def get_communication_time(self, message, source_proc, dest_proc):
        if source_proc == dest_proc:
            return 0
        else:
            return message.size / self.bandwidth

class Scheduler:
    def __init__(self, tasks, messages, processors, network, security_service, deadline):
        self.tasks = tasks
        self.messages = messages
        self.processors = processors
        self.network = network
        self.security = security_service
        self.deadline = deadline
        self.schedule = []
        
        # Set up task successors based on predecessors
        self.initialize_successors()
    
    def initialize_successors(self):
        for task in self.tasks:
            task.successors = []
        
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
    def estimate_security_overhead(self, message, source_proc, dest_proc):
        """Calculate the security overhead for a message."""
        total_overhead = 0
        
        # Add overhead for confidentiality and integri
        # ty (data-dependent)
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
    def compute_task_priorities(self):
        for task in self.tasks:
            # Include security overhead in execution time estimates
            base_time = sum(task.execution_times) / len(task.execution_times)
            sec_cost = self.estimate_security_cost(task)
            task.avg_execution = base_time + sec_cost
        
        def calculate_upward_rank(task):
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
                    comm_cost = 0
                    if message:
                        comm_cost = message.size / self.network.bandwidth
                    
                    succ_rank = calculate_upward_rank(succ_task)
                    rank_with_comm = comm_cost + succ_rank
                    max_successor_rank = max(max_successor_rank, rank_with_comm)
            
            task.priority = task.avg_execution + max_successor_rank
            return task.priority
        
        for task in self.tasks:
            calculate_upward_rank(task)
    
    def optimize_security_assignment(self):
        """Select security protocols that minimize overhead while meeting requirements"""
        for message in self.messages:
            for service in ['confidentiality', 'integrity', 'authentication']:
                min_overhead = float('inf')
                best_protocol = 0
                for protocol in range(len(self.security.strengths[service])):
                    if self.security.strengths[service][protocol] >= message.min_security[service]:
                        overhead = sum(
                            self.security.overheads[service][protocol + 1][pid-1]
                            for pid in range(1, len(self.processors)+1)
                        )
                        if overhead < min_overhead:
                            min_overhead = overhead
                            best_protocol = protocol
                message.assigned_security[service] = best_protocol
    
    def calculate_est(self, task, processor):
        processor_ready_time = self.processors[processor - 1].available_time
        
        max_pred_finish_time = 0
        for pred_id in task.predecessors:
            pred_task = self.get_task_by_id(pred_id)
            if not pred_task.is_scheduled:
                return float('inf')
            
            message = self.get_message(pred_id, task.task_id)
            if not message:
                continue
            
            if pred_task.assigned_processor == processor:
                comm_finish_time = pred_task.finish_time
            else:
                comm_time = self.network.get_communication_time(message, pred_task.assigned_processor, processor)
                # Add security overhead for cross-processor communication
                sec_overhead = self.calculate_security_overhead(message, pred_task.assigned_processor, processor)
                comm_finish_time = pred_task.finish_time + comm_time + sec_overhead
            
            max_pred_finish_time = max(max_pred_finish_time, comm_finish_time)
        
        return max(processor_ready_time, max_pred_finish_time)
class HSMS(Scheduler):
    """Heterogeneous Security-aware Makespan minimizing Scheduler"""
    
    def estimate_security_cost(self, task):
        """Estimate security costs for a task based on its message dependencies"""
        total = 0
        for pred_id in task.predecessors:
            message = self.get_message(pred_id, task.task_id)
            if message:
                # Calculate minimum possible security overhead across all processors
                min_cost = min(
                    self.calculate_security_overhead(message, 
                                                   self.get_task_by_id(pred_id).assigned_processor if self.get_task_by_id(pred_id).assigned_processor else 1, 
                                                   pid)
                    for pid in range(1, len(self.processors)+1)
                )
                total += min_cost
        return total / max(1, len(task.predecessors))  # Return average security cost
    
    def schedule_tasks(self):
        """Schedule tasks using security-aware HEFT algorithm."""
        # Calculate task priorities
        self.compute_task_priorities()
        
        # Assign minimum security levels
        self.assign_minimum_security()
        
        # Sort tasks by priority (descending)
        sorted_tasks = sorted(self.tasks, key=lambda t: -t.priority)
        
        # Schedule each task
        for task in sorted_tasks:
            best_processor = None
            earliest_finish_time = float('inf')
            earliest_start_time = 0
            
            # Try each processor
            for processor in self.processors:
                est = self.calculate_est(task, processor.proc_id)
                eft = est + task.execution_times[processor.proc_id - 1]
                
                if eft < earliest_finish_time:
                    earliest_finish_time = eft
                    earliest_start_time = est
                    best_processor = processor
            
            # Assign task to the best processor
            if best_processor:
                task.assigned_processor = best_processor.proc_id
                task.start_time = earliest_start_time
                task.finish_time = earliest_finish_time
                task.is_scheduled = True
                best_processor.available_time = earliest_finish_time
                
                self.schedule.append({
                    'task_id': task.task_id,
                    'name': task.name,
                    'processor': task.assigned_processor,
                    'start_time': task.start_time,
                    'finish_time': task.finish_time
                })
        
        # Get the makespan (the maximum finish time of all tasks)
        self.makespan = max(task.finish_time for task in self.tasks)
        return self.makespan
    
    def calculate_security_utility(self):
        """Calculate the total security utility."""
        total_utility = 0
        for message in self.messages:
            message_utility = 0
            for service in ['confidentiality', 'integrity', 'authentication']:
                protocol_idx = message.assigned_security[service]
                strength = self.security.strengths[service][protocol_idx]
                weight = message.weights[service]
                message_utility += weight * strength
            total_utility += message_utility
        return total_utility
    
    def run(self):
        """Run the HSMS scheduler."""
        makespan = self.schedule_tasks()
        security_utility = self.calculate_security_utility()
        
        if makespan > self.deadline:
            print(f"HSMS failed to meet deadline. Makespan: {makespan}, Deadline: {self.deadline}")
            return None, None
        else:
            print(f"HSMS successful. Makespan: {makespan}, Security Utility: {security_utility}")
            return makespan, security_utility

class SHIELD(Scheduler):
    """Security-aware scHedulIng for rEaL-time Dags on heterogeneous systems"""
    
    def __init__(self, tasks, messages, processors, network, security_service, deadline):
        super().__init__(tasks, messages, processors, network, security_service, deadline)
        # Run HSMS first to get initial schedule
        self.hsms = HSMS(copy.deepcopy(tasks), messages, copy.deepcopy(processors), 
                         network, security_service, deadline)
    
    def calculate_benefits(self, message, source_proc, dest_proc):
        """Calculate security benefit and time penalty for upgrading each security service."""
        benefits = []
        
        for service in ['confidentiality', 'integrity', 'authentication']:
            current_protocol_idx = message.assigned_security[service]
            
            # If already at highest protocol, skip
            if current_protocol_idx >= len(self.security.strengths[service]) - 1:
                continue
            
            next_protocol_idx = current_protocol_idx + 1
            
            # Calculate security benefit
            current_strength = self.security.strengths[service][current_protocol_idx]
            next_strength = self.security.strengths[service][next_protocol_idx]
            strength_diff = next_strength - current_strength
            security_benefit = message.weights[service] * strength_diff
            
            # Calculate time penalty
            current_overhead = 0
            next_overhead = 0
            
            if service in ['confidentiality', 'integrity']:
                current_protocol = current_protocol_idx + 1  # 1-indexed in overhead table
                next_protocol = next_protocol_idx + 1
                
                current_overhead = (message.size / 1024) * self.security.overheads[service][current_protocol][source_proc - 1]
                next_overhead = (message.size / 1024) * self.security.overheads[service][next_protocol][source_proc - 1]
            else:  # authentication
                current_protocol = current_protocol_idx + 1
                next_protocol = next_protocol_idx + 1
                
                current_overhead = self.security.overheads[service][current_protocol][dest_proc - 1]
                next_overhead = self.security.overheads[service][next_protocol][dest_proc - 1]
            
            time_penalty = next_overhead - current_overhead
            
            # Calculate benefit-to-cost ratio
            if time_penalty > 0:
                benefit_cost_ratio = security_benefit / time_penalty
            else:
                benefit_cost_ratio = float('inf')
            
            benefits.append({
                'message': message,
                'service': service,
                'protocol_idx': next_protocol_idx,
                'security_benefit': security_benefit,
                'time_penalty': time_penalty,
                'benefit_cost_ratio': benefit_cost_ratio
            })
        
        return benefits
    
    def can_upgrade_message(self, message, service, next_protocol_idx, schedule):
        """Check if upgrading a message's security level would still meet the deadline."""
        # Create a copy of the tasks to simulate the schedule
        tasks_copy = copy.deepcopy(self.tasks)
        processors_copy = copy.deepcopy(self.processors)
        
        # Update the message's security level in the copy
        message_copy = copy.deepcopy(message)
        message_copy.assigned_security[service] = next_protocol_idx
        
        # Simulate the schedule with the upgraded security
        for entry in schedule:
            task = next((t for t in tasks_copy if t.task_id == entry['task_id']), None)
            if task:
                task.assigned_processor = entry['processor']
                task.start_time = entry['start_time']
                task.finish_time = entry['finish_time']
                task.is_scheduled = True
        
        # Recalculate the makespan with the upgraded security
        makespan = max(task.finish_time for task in tasks_copy if task.is_scheduled)
        
        # Check if the new makespan would meet the deadline
        return makespan <= self.deadline
    
    def run(self):
        """Run the SHIELD scheduler."""
        # Run HSMS first
        start_time = time.time()

        hsms_makespan, hsms_security_utility = self.hsms.run()
        end_time = time.time()
        if hsms_makespan is None:
            print("SHIELD cannot proceed since HSMS failed to meet the deadline.")
            return None, None
        print(f"HSMS Completed Execution in {(end_time - start_time):.4f} seconds")
        # Copy the schedule from HSMS
        self.schedule = copy.deepcopy(self.hsms.schedule)
        
        # Make a copy of tasks with the HSMS schedule
        for task in self.tasks:
            hsms_task = next((t for t in self.hsms.tasks if t.task_id == task.task_id), None)
            if hsms_task:
                task.assigned_processor = hsms_task.assigned_processor
                task.start_time = hsms_task.start_time
                task.finish_time = hsms_task.finish_time
                task.is_scheduled = True
        
        # Initialize the slack time
        slack_time = self.deadline - hsms_makespan
        
        # While there is slack time available, try to enhance security
        while slack_time > 0:
            all_benefits = []
            
            # Calculate benefits for each message and security service
            for message in self.messages:
                source_task = self.get_task_by_id(message.source_id)
                dest_task = self.get_task_by_id(message.dest_id)
                
                if not source_task or not dest_task or not source_task.is_scheduled or not dest_task.is_scheduled:
                    continue
                
                source_proc = source_task.assigned_processor
                dest_proc = dest_task.assigned_processor
                
                benefits = self.calculate_benefits(message, source_proc, dest_proc)
                all_benefits.extend(benefits)
            
            if not all_benefits:
                break
            
            # Sort benefits by benefit-to-cost ratio (descending)
            all_benefits.sort(key=lambda x: -x['benefit_cost_ratio'])
            
            # Try to apply the best upgrade
            best_upgrade = all_benefits[0]
            
            if best_upgrade['time_penalty'] <= slack_time and self.can_upgrade_message(
                best_upgrade['message'], 
                best_upgrade['service'], 
                best_upgrade['protocol_idx'],
                self.schedule
            ):
                # Apply the upgrade
                message = best_upgrade['message']
                service = best_upgrade['service']
                message.assigned_security[service] = best_upgrade['protocol_idx']
                
                # Update slack time
                slack_time -= best_upgrade['time_penalty']
            else:
                # No feasible upgrades left
                break
        
        # Calculate final security utility
        security_utility = 0
        for message in self.messages:
            message_utility = 0
            for service in ['confidentiality', 'integrity', 'authentication']:
                protocol_idx = message.assigned_security[service]
                strength = self.security.strengths[service][protocol_idx]
                weight = message.weights[service]
                message_utility += weight * strength
            security_utility += message_utility
        
        # Calculate final makespan
        makespan = max(task.finish_time for task in self.tasks if task.is_scheduled)
        end_time=time.time()
        print(f"SHIELD successful. Makespan: {makespan}, Security Utility: {security_utility} Time Elapsed: {(end_time-start_time):.4f}")
        return makespan, security_utility

class ImprovedQLearningScheduler(Scheduler):
    def estimate_security_cost(self, task):
        """Estimate potential security costs for a task"""
        total = 0
        for pred_id in task.predecessors:
            message = self.get_message(pred_id, task.task_id)
            if message:
                min_cost = min(
                    self.calculate_security_overhead(message, 
                                                    self.get_task_by_id(pred_id).assigned_processor if self.get_task_by_id(pred_id).assigned_processor else 1, 
                                                    pid)
                    for pid in range(1, len(self.processors)+1)
                )
                total += min_cost
        return total / max(1, len(task.predecessors))
        
    def __init__(self, tasks, messages, processors, network, security_service, deadline, seed=42):
        super().__init__(tasks, messages, processors, network, security_service, deadline)
        # Set random seed for deterministic behavior
        random.seed(seed)
        np.random.seed(seed)
        
        self.q_table = {}  # Initialize Q-table for QLearning
        self.alpha = 0.2  # Lower learning rate for more stable convergence
        self.gamma = 0.9  # Higher discount factor to value future rewards more
        self.initial_epsilon = 0.8  # Start with high exploration
        self.min_epsilon = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.97  # More gradual decay
        self.episodes = 500  # More training episodes
        
        # Track best solution found
        self.best_makespan = float('inf')
        self.best_schedule = None
        self.best_tasks_state = None
        self.best_security_utility = 0
        self.epsilon = self.initial_epsilon
        
        # Initialize HEFT scheduler for baseline comparison
        self.heft_makespan = float('inf')
        
        self.initialize_q_table()
        
    def initialize_q_table(self):
        """Initialize Q-table with default values for all possible state-action pairs"""
        # First compute task priorities if not done
        if any(task.priority is None for task in self.tasks):
            self.compute_task_priorities()
            
        # For each task, initialize Q-values for each processor assignment
        for task in self.tasks:
            for proc_id in range(1, len(self.processors) + 1):
                self.q_table[(task.task_id, proc_id)] = 0.0

    def update_q_value(self, task_id, processor, reward, next_task_id=None):
        # Update Q-value based on the action taken
        current_q = self.q_table.get((task_id, processor), 0)
        
        if next_task_id:
            # Find the best Q-value for the next task on any processor
            next_max_q = max([self.q_table.get((next_task_id, p), 0) 
                             for p in range(1, len(self.processors) + 1)])
            
            # Q-learning update with future rewards
            self.q_table[(task_id, processor)] = (1 - self.alpha) * current_q + \
                                             self.alpha * (reward + self.gamma * next_max_q)
        else:
            # Simple update for terminal state
            self.q_table[(task_id, processor)] = (1 - self.alpha) * current_q + \
                                             self.alpha * reward

    def choose_processor(self, task, scheduled_tasks):
        """Choose a processor based on epsilon-greedy approach with smarter exploitation"""
        if random.random() < self.epsilon:
            # Exploration: choose a random processor with preference for load balancing
            proc_loads = [(p.proc_id, p.available_time) for p in self.processors]
            # Sort by available time (least busy processors first)
            proc_loads.sort(key=lambda x: x[1])
            
            # Weighted random selection favoring less busy processors
            weights = [1/(i+1) for i in range(len(proc_loads))]
            total_weight = sum(weights)
            normalized_weights = [w/total_weight for w in weights]
            
            procs = [p[0] for p in proc_loads]
            selected_proc = random.choices(procs, weights=normalized_weights, k=1)[0]
            return selected_proc
        else:
            # Exploitation: choose processor with highest Q-value
            # But also consider the current state of the system
            best_proc = 1
            best_value = float('-inf')
            
            for proc_id in range(1, len(self.processors) + 1):
                # Get base Q-value
                q_value = self.q_table.get((task.task_id, proc_id), 0)
                
                # Consider additional factors:
                processor = next(p for p in self.processors if p.proc_id == proc_id)
                est = self.calculate_est(task, proc_id)
                
                # Prioritize processors where predecessors are already scheduled
                predecessor_bonus = 0
                for pred_id in task.predecessors:
                    pred_task = self.get_task_by_id(pred_id)
                    if pred_task and pred_task.assigned_processor == proc_id:
                        predecessor_bonus += 50  # Big bonus for locality
                
                # Calculate a processor preference score
                proc_preference = q_value + predecessor_bonus
                
                # Penalize if processor is busy far into the future
                if est > processor.available_time + 50:
                    proc_preference -= (est - processor.available_time) * 0.5
                
                if proc_preference > best_value:
                    best_value = proc_preference
                    best_proc = proc_id
            
            return best_proc

    def run_heft_baseline(self):
        """Run HEFT algorithm to get a baseline schedule for comparison"""
        # Reset processors and tasks
        for processor in self.processors:
            processor.available_time = 0
        
        for task in self.tasks:
            task.is_scheduled = False
            task.assigned_processor = None
            task.start_time = None
            task.finish_time = None
        
        # Compute priorities and assign basic security
        self.compute_task_priorities()
        self.optimize_security_assignment()
        
        # Sort tasks by priority (higher priority first)
        sorted_tasks = sorted(self.tasks, key=lambda x: -x.priority)
        
        # Schedule each task on processor that minimizes EFT
        for task in sorted_tasks:
            best_processor = None
            earliest_finish_time = float('inf')
            earliest_start_time = 0
            
            for processor in self.processors:
                est = self.calculate_est(task, processor.proc_id)
                eft = est + task.execution_times[processor.proc_id - 1]
                
                if eft < earliest_finish_time:
                    earliest_finish_time = eft
                    earliest_start_time = est
                    best_processor = processor
            
            # Assign task to best processor
            task.assigned_processor = best_processor.proc_id
            task.start_time = earliest_start_time
            task.finish_time = earliest_finish_time
            task.is_scheduled = True
            best_processor.available_time = earliest_finish_time
        
        # Calculate makespan
        self.heft_makespan = max(task.finish_time for task in self.tasks)
        return self.heft_makespan

    def schedule_tasks(self):
        """Schedule tasks using improved Q-learning with memory of best solution"""
        # Run HEFT baseline first to get a good starting point
        heft_makespan = self.run_heft_baseline()
        heft_schedule = []
        heft_tasks_state = copy.deepcopy(self.tasks)
        
        for task in self.tasks:
            if task.is_scheduled:
                entry = {
                    'task_id': task.task_id,
                    'name': task.name,
                    'processor': task.assigned_processor,
                    'start': task.start_time,
                    'finish': task.finish_time
                }
                heft_schedule.append(entry)
        
        # Set the HEFT solution as initial best
        self.best_makespan = heft_makespan
        self.best_schedule = heft_schedule
        self.best_tasks_state = heft_tasks_state
        self.best_security_utility = self.calculate_security_utility()
        
        print(f"HEFT baseline: Makespan={heft_makespan}, Security Utility={self.best_security_utility:.2f}")
        
        # Start Q-learning episodes from this point
        for episode in range(self.episodes):
            # Reset processors and task scheduling status
            for processor in self.processors:
                processor.available_time = 0
                processor.task_history = []
            
            for task in self.tasks:
                task.is_scheduled = False
                task.assigned_processor = None
                task.start_time = None
                task.finish_time = None
            
            current_schedule = []
            
            # Optimize security assignments for this episode
            self.optimize_security_assignment()
            
            # Sort tasks by priority (higher priority first)
            sorted_tasks = sorted(self.tasks, key=lambda x: -x.priority)
            
            # List to track tasks that are ready to be scheduled
            ready_tasks = [task for task in sorted_tasks 
                          if not task.predecessors or 
                          all(self.get_task_by_id(pred_id).is_scheduled for pred_id in task.predecessors)]
            
            # List to track scheduled tasks for this episode
            scheduled_tasks = []
            
            # Continue scheduling until all tasks are scheduled
            while ready_tasks:
                # Sort ready tasks by priority
                ready_tasks.sort(key=lambda x: -x.priority)
                
                # Select the highest priority task
                current_task = ready_tasks.pop(0)
                
                # Choose processor using improved Q-learning policy
                proc_id = self.choose_processor(current_task, scheduled_tasks)
                processor = self.processors[proc_id - 1]
                
                # Calculate EST and EFT
                est = self.calculate_est(current_task, proc_id)
                eft = est + current_task.execution_times[proc_id - 1]
                
                # Schedule the task
                current_task.assigned_processor = proc_id
                current_task.start_time = est
                current_task.finish_time = eft
                current_task.is_scheduled = True
                processor.available_time = eft
                processor.task_history.append(current_task.task_id)
                
                # Record schedule entry
                entry = {
                    'task_id': current_task.task_id,
                    'name': current_task.name,
                    'processor': proc_id,
                    'start': current_task.start_time,
                    'finish': current_task.finish_time
                }
                current_schedule.append(entry)
                scheduled_tasks.append(current_task)
                
                # Find the next task that might become ready
                next_task = None
                for task in sorted_tasks:
                    if not task.is_scheduled and task not in ready_tasks:
                        if all(self.get_task_by_id(pred_id).is_scheduled for pred_id in task.predecessors):
                            next_task = task
                            break
                
                # Calculate reward for this scheduling decision
                reward = self.calculate_improved_reward(current_task, proc_id)
                
                # Update Q-value
                if next_task:
                    self.update_q_value(current_task.task_id, proc_id, reward, next_task.task_id)
                else:
                    self.update_q_value(current_task.task_id, proc_id, reward)
                
                # Update ready tasks
                for task in sorted_tasks:
                    if not task.is_scheduled and task not in ready_tasks:
                        if all(self.get_task_by_id(pred_id).is_scheduled for pred_id in task.predecessors):
                            ready_tasks.append(task)
            
            # Calculate makespan and security utility for this episode
            current_makespan = max(task.finish_time for task in self.tasks)
            current_security_utility = self.calculate_security_utility()
            
            # Update best solution if better makespan found
            if current_makespan < self.best_makespan:
                self.best_makespan = current_makespan
                self.best_schedule = current_schedule
                self.best_tasks_state = copy.deepcopy(self.tasks)
                self.best_security_utility = current_security_utility
                print(f"Episode {episode}: New best makespan: {current_makespan}, Security: {current_security_utility:.2f}")
            # Or if same makespan but better security utility
            elif current_makespan == self.best_makespan and current_security_utility > self.best_security_utility:
                self.best_schedule = current_schedule
                self.best_tasks_state = copy.deepcopy(self.tasks)
                self.best_security_utility = current_security_utility
                print(f"Episode {episode}: Same makespan but better security: {current_security_utility:.2f}")
            
            # Gradually reduce exploration rate according to schedule
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            # Every 50 episodes, incorporate best solution found so far into Q-table
            if episode % 50 == 0 and episode > 0:
                self.reinforce_best_solution()
        
        # Set the final schedule with best found solution
        self.schedule = self.best_schedule
        # Restore best task state
        self.tasks = self.best_tasks_state
        
        print(f"Final best solution: Makespan={self.best_makespan}, Security Utility={self.best_security_utility:.2f}")
        
        # After all episodes, enhance security with remaining slack time
        self.enhance_security_with_slack()
        
        return self.best_makespan

    def reinforce_best_solution(self):
        """Reinforce the best solution found so far in the Q-table"""
        for task in self.best_tasks_state:
            if task.is_scheduled:
                # Give a large positive reinforcement to the best choices
                self.q_table[(task.task_id, task.assigned_processor)] += 25

    def calculate_improved_reward(self, task, processor_id):
        """Enhanced reward function with balanced factors"""
        # Calculate slack time (how much before deadline)
        slack = self.deadline - task.finish_time
        
        # Calculate processor load balance
        proc_loads = [p.available_time for p in self.processors]
        avg_load = sum(proc_loads) / len(proc_loads)
        load_balance_score = -abs(task.finish_time - avg_load) / 100
        
        # Calculate dependency locality bonus
        locality_bonus = 0
        for pred_id in task.predecessors:
            pred_task = self.get_task_by_id(pred_id)
            if pred_task and pred_task.assigned_processor == processor_id:
                locality_bonus += 15  # Reward keeping dependent tasks on same processor
        
        # Calculate parallelism bonus
        parallel_bonus = 0
        for other_task in self.tasks:
            if other_task.is_scheduled and other_task.task_id != task.task_id:
                # If tasks can run in parallel on different processors, reward it
                if (other_task.finish_time >= task.start_time and 
                    other_task.start_time <= task.finish_time and
                    other_task.assigned_processor != processor_id):
                    parallel_bonus += 15
        
        # Calculate security overhead
        sec_overhead = 0
        for pred_id in task.predecessors:
            pred_task = self.get_task_by_id(pred_id)
            if pred_task and pred_task.assigned_processor != processor_id:
                message = self.get_message(pred_id, task.task_id)
                if message:
                    sec_overhead += self.calculate_security_overhead(message, pred_task.assigned_processor, processor_id)
        
        # Calculate critical path bonus
        critical_path_bonus = 0
        if task.priority > sum(t.priority for t in self.tasks) / len(self.tasks):
            # Task is on critical path, reward finishing it early
            critical_path_bonus = 25
            
        # Reward for processor ready time
        proc_ready_reward = 0
        processor = self.processors[processor_id - 1]
        if task.start_time == processor.available_time:
            # Task starts exactly when processor becomes available - optimal usage
            proc_ready_reward = 10
        
        # Combine all factors into final reward with better balanced weights
        reward = (
            (slack > 0) * 50 +  # Binary reward for meeting deadline
            min(0, slack) * 0.5 +  # Small penalty for how much over deadline
            load_balance_score +  # Load balancing score
            locality_bonus +  # Reward for keeping related tasks together
            parallel_bonus +  # Reward for parallelism
            critical_path_bonus +  # Reward for critical path tasks
            proc_ready_reward -  # Reward for processor utilization
            sec_overhead * 0.1  # Small penalty for security overhead
        )
        
        return reward

    def calculate_security_overhead(self, message, source_proc, dest_proc):
        """Calculate total security overhead for a message between processors"""
        if source_proc == dest_proc:
            return 0  # No overhead for same processor
            
        overhead = 0
        for service in ['confidentiality', 'integrity', 'authentication']:
            protocol_idx = message.assigned_security[service]
            if service in ['confidentiality', 'integrity']:
                # Data-dependent overheads
                overhead += self.security.overheads[service][protocol_idx + 1][source_proc - 1]
            else:  # authentication
                # Data-independent overheads
                overhead += self.security.overheads[service][protocol_idx + 1][dest_proc - 1]
                
        return overhead * message.size / 100  # Scale by message size
        
    def calculate_security_utility(self):
        """Calculate the total security utility."""
        total_utility = 0
        for message in self.messages:
            message_utility = 0
            for service in ['confidentiality', 'integrity', 'authentication']:
                protocol_idx = message.assigned_security[service]
                strength = self.security.strengths[service][protocol_idx]
                weight = message.weights[service]
                message_utility += weight * strength
            total_utility += message_utility
        return total_utility
        
    def enhance_security_with_slack(self):
        """Use remaining slack time to enhance security levels"""
        # Calculate current makespan
        makespan = max(task.finish_time for task in self.tasks)
        
        # Calculate slack time
        slack_time = self.deadline - makespan
        if slack_time <= 0:
            return  # No slack time available
        
        print(f"Enhancing security with {slack_time:.2f} slack time")
        
        # Keep track of remaining slack
        remaining_slack = slack_time
        
        # Calculate the potential benefit of upgrading each security service for each message
        while remaining_slack > 0:
            best_upgrade = None
            best_benefit_ratio = 0
            
            for message in self.messages:
                source_task = self.get_task_by_id(message.source_id)
                dest_task = self.get_task_by_id(message.dest_id)
                
                if not source_task or not dest_task or not source_task.is_scheduled or not dest_task.is_scheduled:
                    continue
                    
                if source_task.assigned_processor == dest_task.assigned_processor:
                    continue  # Skip intra-processor messages
                
                source_proc = source_task.assigned_processor
                dest_proc = dest_task.assigned_processor
                
                for service in ['confidentiality', 'integrity', 'authentication']:
                    current_level = message.assigned_security[service]
                    if current_level < len(self.security.strengths[service]) - 1:
                        # Can upgrade this service
                        next_level = current_level + 1
                        
                        # Calculate security benefit
                        current_strength = self.security.strengths[service][current_level]
                        next_strength = self.security.strengths[service][next_level]
                        strength_gain = next_strength - current_strength
                        security_benefit = message.weights[service] * strength_gain
                        
                        # Calculate time cost
                        if service in ['confidentiality', 'integrity']:
                            current_overhead = self.security.overheads[service][current_level + 1][source_proc - 1]
                            next_overhead = self.security.overheads[service][next_level + 1][source_proc - 1]
                        else:  # authentication
                            current_overhead = self.security.overheads[service][current_level + 1][dest_proc - 1]
                            next_overhead = self.security.overheads[service][next_level + 1][dest_proc - 1]
                            
                        time_cost = (next_overhead - current_overhead) * message.size / 100
                        
                        # Check if upgrade fits in remaining slack
                        if time_cost <= remaining_slack:
                            benefit_ratio = security_benefit / max(0.001, time_cost)
                            
                            if benefit_ratio > best_benefit_ratio:
                                best_benefit_ratio = benefit_ratio
                                best_upgrade = (message, service, next_level, time_cost, security_benefit)
            
            if best_upgrade:
                # Apply the best upgrade
                message, service, new_level, time_cost, benefit = best_upgrade
                message.assigned_security[service] = new_level
                remaining_slack -= time_cost
                
                print(f"Upgraded {message.id} {service} to level {new_level}, cost {time_cost:.2f}, benefit {benefit:.4f}")
            else:
                # No more beneficial upgrades possible
                break
        
        # Recalculate security utility after enhancements
        self.best_security_utility = self.calculate_security_utility()
        print(f"Enhanced security utility: {self.best_security_utility:.2f}, remaining slack: {remaining_slack:.2f}")

def create_tc_test_case():
    # Create tasks with execution times as in Figure 10(c)
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
    
    # Create messages as in Figure 10(b)
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
    
    # Set security requirements from Table 7
    messages[0].set_security_requirements(0.2, 0.1, 0.4, 0.3, 0.3, 0.4)  # e_1_6
    messages[1].set_security_requirements(0.2, 0.2, 0.4, 0.3, 0.5, 0.2)  # e_2_6
    messages[2].set_security_requirements(0.2, 0.5, 0.3, 0.2, 0.6, 0.2)  # e_3_6
    messages[3].set_security_requirements(0.3, 0.4, 0.2, 0.2, 0.2, 0.6)  # e_4_6
    messages[4].set_security_requirements(0.4, 0.3, 0.1, 0.2, 0.3, 0.5)  # e_5_8
    messages[5].set_security_requirements(0.4, 0.2, 0.4, 0.7, 0.1, 0.2)  # e_6_8
    messages[6].set_security_requirements(0.4, 0.1, 0.3, 0.7, 0.1, 0.2)  # e_7_8
    messages[7].set_security_requirements(0.3, 0.1, 0.2, 0.7, 0.1, 0.2)  # e_8_9
    messages[8].set_security_requirements(0.3, 0.2, 0.1, 0.7, 0.1, 0.2)  # e_8_10
    
    processors = [Processor(1), Processor(2)]
    network = CommunicationNetwork(2)
    security_service = SecurityService()
    deadline = 1600  # As specified in the paper
    
    return tasks, messages, processors, network, security_service, deadline

def plot_gantt_chart(title, schedule, num_processors, makespan):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for entry in schedule:
        task_id = entry['task_id']
        name = entry['name']
        proc = entry['processor']
        start = entry.get('start', entry.get('start_time', 0))
        finish = entry.get('finish', entry.get('finish_time', 0))
        
        color_idx = task_id % len(colors)
        ax.barh(proc - 1, finish - start, left=start, color=colors[color_idx], edgecolor='black')
        ax.text(start + (finish - start) / 2, proc - 1, f"T{task_id}", 
                va='center', ha='center', color='white', fontsize=10, weight='bold')
    
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Processor")
    ax.set_title(f"{title} - Makespan: {makespan} ms")
    ax.set_yticks(range(num_processors))
    ax.set_yticklabels([f"P{p+1}" for p in range(num_processors)])
    ax.set_xlim(0, makespan * 1.1)  # Add some padding
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    return fig

class ImprovedQLearningScheduler(Scheduler):
    def estimate_security_cost(self, task):
        """Estimate potential security costs for a task"""
        total = 0
        for pred_id in task.predecessors:
            message = self.get_message(pred_id, task.task_id)
            if message:
                min_cost = min(
                    self.calculate_security_overhead(message, 
                                                    self.get_task_by_id(pred_id).assigned_processor if self.get_task_by_id(pred_id).assigned_processor else 1, 
                                                    pid)
                    for pid in range(1, len(self.processors)+1)
                )
                total += min_cost
        return total / max(1, len(task.predecessors))
        
    def __init__(self, tasks, messages, processors, network, security_service, deadline, seed=42):
        super().__init__(tasks, messages, processors, network, security_service, deadline)
        # Set random seed for deterministic behavior
        random.seed(seed)
        np.random.seed(seed)
        
        self.q_table = {}  # Initialize Q-table for QLearning
        self.alpha = 0.2  # Lower learning rate for more stable convergence
        self.gamma = 0.9  # Higher discount factor to value future rewards more
        self.initial_epsilon = 0.8  # Start with high exploration
        self.min_epsilon = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.97  # More gradual decay
        self.episodes = 500  # More training episodes
        
        # Track best solution found
        self.best_makespan = float('inf')
        self.best_schedule = None
        self.best_tasks_state = None
        self.best_security_utility = 0
        self.epsilon = self.initial_epsilon
        
        # Initialize HEFT scheduler for baseline comparison
        self.heft_makespan = float('inf')
        
        self.initialize_q_table()
        
    def initialize_q_table(self):
        """Initialize Q-table with default values for all possible state-action pairs"""
        # First compute task priorities if not done
        if any(task.priority is None for task in self.tasks):
            self.compute_task_priorities()
            
        # For each task, initialize Q-values for each processor assignment
        for task in self.tasks:
            for proc_id in range(1, len(self.processors) + 1):
                self.q_table[(task.task_id, proc_id)] = 0.0

    def update_q_value(self, task_id, processor, reward, next_task_id=None):
        # Update Q-value based on the action taken
        current_q = self.q_table.get((task_id, processor), 0)
        
        if next_task_id:
            # Find the best Q-value for the next task on any processor
            next_max_q = max([self.q_table.get((next_task_id, p), 0) 
                             for p in range(1, len(self.processors) + 1)])
            
            # Q-learning update with future rewards
            self.q_table[(task_id, processor)] = (1 - self.alpha) * current_q + \
                                             self.alpha * (reward + self.gamma * next_max_q)
        else:
            # Simple update for terminal state
            self.q_table[(task_id, processor)] = (1 - self.alpha) * current_q + \
                                             self.alpha * reward

    def choose_processor(self, task, scheduled_tasks):
        """Choose a processor based on epsilon-greedy approach with smarter exploitation"""
        if random.random() < self.epsilon:
            # Exploration: choose a random processor with preference for load balancing
            proc_loads = [(p.proc_id, p.available_time) for p in self.processors]
            # Sort by available time (least busy processors first)
            proc_loads.sort(key=lambda x: x[1])
            
            # Weighted random selection favoring less busy processors
            weights = [1/(i+1) for i in range(len(proc_loads))]
            total_weight = sum(weights)
            normalized_weights = [w/total_weight for w in weights]
            
            procs = [p[0] for p in proc_loads]
            selected_proc = random.choices(procs, weights=normalized_weights, k=1)[0]
            return selected_proc
        else:
            # Exploitation: choose processor with highest Q-value
            # But also consider the current state of the system
            best_proc = 1
            best_value = float('-inf')
            
            for proc_id in range(1, len(self.processors) + 1):
                # Get base Q-value
                q_value = self.q_table.get((task.task_id, proc_id), 0)
                
                # Consider additional factors:
                processor = next(p for p in self.processors if p.proc_id == proc_id)
                est = self.calculate_est(task, proc_id)
                
                # Prioritize processors where predecessors are already scheduled
                predecessor_bonus = 0
                for pred_id in task.predecessors:
                    pred_task = self.get_task_by_id(pred_id)
                    if pred_task and pred_task.assigned_processor == proc_id:
                        predecessor_bonus += 50  # Big bonus for locality
                
                # Calculate a processor preference score
                proc_preference = q_value + predecessor_bonus
                
                # Penalize if processor is busy far into the future
                if est > processor.available_time + 50:
                    proc_preference -= (est - processor.available_time) * 0.5
                
                if proc_preference > best_value:
                    best_value = proc_preference
                    best_proc = proc_id
            
            return best_proc

    def run_heft_baseline(self):
        """Run HEFT algorithm to get a baseline schedule for comparison"""
        # Reset processors and tasks
        for processor in self.processors:
            processor.available_time = 0
        
        for task in self.tasks:
            task.is_scheduled = False
            task.assigned_processor = None
            task.start_time = None
            task.finish_time = None
        
        # Compute priorities and assign basic security
        self.compute_task_priorities()
        self.optimize_security_assignment()
        
        # Sort tasks by priority (higher priority first)
        sorted_tasks = sorted(self.tasks, key=lambda x: -x.priority)
        
        # Schedule each task on processor that minimizes EFT
        for task in sorted_tasks:
            best_processor = None
            earliest_finish_time = float('inf')
            earliest_start_time = 0
            
            for processor in self.processors:
                est = self.calculate_est(task, processor.proc_id)
                eft = est + task.execution_times[processor.proc_id - 1]
                
                if eft < earliest_finish_time:
                    earliest_finish_time = eft
                    earliest_start_time = est
                    best_processor = processor
            
            # Assign task to best processor
            task.assigned_processor = best_processor.proc_id
            task.start_time = earliest_start_time
            task.finish_time = earliest_finish_time
            task.is_scheduled = True
            best_processor.available_time = earliest_finish_time
        
        # Calculate makespan
        self.heft_makespan = max(task.finish_time for task in self.tasks)
        return self.heft_makespan

    def schedule_tasks(self):
        """Schedule tasks using improved Q-learning with memory of best solution"""
        # Run HEFT baseline first to get a good starting point
        heft_makespan = self.run_heft_baseline()
        heft_schedule = []
        heft_tasks_state = copy.deepcopy(self.tasks)
        
        for task in self.tasks:
            if task.is_scheduled:
                entry = {
                    'task_id': task.task_id,
                    'name': task.name,
                    'processor': task.assigned_processor,
                    'start_time': task.start_time,
                    'finish_time': task.finish_time
                }
                heft_schedule.append(entry)
        
        # Set the HEFT solution as initial best
        self.best_makespan = heft_makespan
        self.best_schedule = heft_schedule
        self.best_tasks_state = heft_tasks_state
        self.best_security_utility = self.calculate_security_utility()
        
        print(f"HEFT baseline: Makespan={heft_makespan}, Security Utility={self.best_security_utility:.2f}")
        
        # Start Q-learning episodes from this point
        for episode in range(self.episodes):
            # Reset processors and task scheduling status
            for processor in self.processors:
                processor.available_time = 0
                processor.task_history = []
            
            for task in self.tasks:
                task.is_scheduled = False
                task.assigned_processor = None
                task.start_time = None
                task.finish_time = None
            
            current_schedule = []
            
            # Optimize security assignments for this episode
            self.optimize_security_assignment()
            
            # Sort tasks by priority (higher priority first)
            sorted_tasks = sorted(self.tasks, key=lambda x: -x.priority)
            
            # List to track tasks that are ready to be scheduled
            ready_tasks = [task for task in sorted_tasks 
                          if not task.predecessors or 
                          all(self.get_task_by_id(pred_id).is_scheduled for pred_id in task.predecessors)]
            
            # List to track scheduled tasks for this episode
            scheduled_tasks = []
            
            # Continue scheduling until all tasks are scheduled
            while ready_tasks:
                # Sort ready tasks by priority
                ready_tasks.sort(key=lambda x: -x.priority)
                
                # Select the highest priority task
                current_task = ready_tasks.pop(0)
                
                # Choose processor using improved Q-learning policy
                proc_id = self.choose_processor(current_task, scheduled_tasks)
                processor = self.processors[proc_id - 1]
                
                # Calculate EST and EFT
                est = self.calculate_est(current_task, proc_id)
                eft = est + current_task.execution_times[proc_id - 1]
                
                # Schedule the task
                current_task.assigned_processor = proc_id
                current_task.start_time = est
                current_task.finish_time = eft
                current_task.is_scheduled = True
                processor.available_time = eft
                processor.task_history.append(current_task.task_id)
                
                # Record schedule entry
                entry = {
                    'task_id': current_task.task_id,
                    'name': current_task.name,
                    'processor': proc_id,
                    'start_time': current_task.start_time,
                    'finish_time': current_task.finish_time
                }
                current_schedule.append(entry)
                scheduled_tasks.append(current_task)
                
                # Find the next task that might become ready
                next_task = None
                for task in sorted_tasks:
                    if not task.is_scheduled and task not in ready_tasks:
                        if all(self.get_task_by_id(pred_id).is_scheduled for pred_id in task.predecessors):
                            next_task = task
                            break
                
                # Calculate reward for this scheduling decision
                reward = self.calculate_improved_reward(current_task, proc_id)
                
                # Update Q-value
                if next_task:
                    self.update_q_value(current_task.task_id, proc_id, reward, next_task.task_id)
                else:
                    self.update_q_value(current_task.task_id, proc_id, reward)
                
                # Update ready tasks
                for task in sorted_tasks:
                    if not task.is_scheduled and task not in ready_tasks:
                        if all(self.get_task_by_id(pred_id).is_scheduled for pred_id in task.predecessors):
                            ready_tasks.append(task)
            
            # Calculate makespan and security utility for this episode
            current_makespan = max(task.finish_time for task in self.tasks)
            current_security_utility = self.calculate_security_utility()
            
            # Update best solution if better makespan found
            if current_makespan < self.best_makespan:
                self.best_makespan = current_makespan
                self.best_schedule = current_schedule
                self.best_tasks_state = copy.deepcopy(self.tasks)
                self.best_security_utility = current_security_utility
                print(f"Episode {episode}: New best makespan: {current_makespan}, Security: {current_security_utility:.2f}")
            # Or if same makespan but better security utility
            elif current_makespan == self.best_makespan and current_security_utility > self.best_security_utility:
                self.best_schedule = current_schedule
                self.best_tasks_state = copy.deepcopy(self.tasks)
                self.best_security_utility = current_security_utility
                print(f"Episode {episode}: Same makespan but better security: {current_security_utility:.2f}")
            
            # Gradually reduce exploration rate according to schedule
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            # Every 50 episodes, incorporate best solution found so far into Q-table
            if episode % 50 == 0 and episode > 0:
                self.reinforce_best_solution()
        
        # Set the final schedule with best found solution
        self.schedule = self.best_schedule
        # Restore best task state
        self.tasks = self.best_tasks_state
        
        print(f"Final best solution: Makespan={self.best_makespan}, Security Utility={self.best_security_utility:.2f}")
        
        # After all episodes, enhance security with remaining slack time
        self.enhance_security_with_slack()
        
        return self.best_makespan

    def reinforce_best_solution(self):
        """Reinforce the best solution found so far in the Q-table"""
        for task in self.best_tasks_state:
            if task.is_scheduled:
                # Give a large positive reinforcement to the best choices
                self.q_table[(task.task_id, task.assigned_processor)] += 25

    def calculate_improved_reward(self, task, processor_id):
        """Enhanced reward function with balanced factors"""
        # Calculate slack time (how much before deadline)
        slack = self.deadline - task.finish_time
        
        # Calculate processor load balance
        proc_loads = [p.available_time for p in self.processors]
        avg_load = sum(proc_loads) / len(proc_loads)
        load_balance_score = -abs(task.finish_time - avg_load) / 100
        
        # Calculate dependency locality bonus
        locality_bonus = 0
        for pred_id in task.predecessors:
            pred_task = self.get_task_by_id(pred_id)
            if pred_task and pred_task.assigned_processor == processor_id:
                locality_bonus += 15  # Reward keeping dependent tasks on same processor
        
        # Calculate parallelism bonus
        parallel_bonus = 0
        for other_task in self.tasks:
            if other_task.is_scheduled and other_task.task_id != task.task_id:
                # If tasks can run in parallel on different processors, reward it
                if (other_task.finish_time >= task.start_time and 
                    other_task.start_time <= task.finish_time and
                    other_task.assigned_processor != processor_id):
                    parallel_bonus += 15
        
        # Calculate security overhead
        sec_overhead = 0
        for pred_id in task.predecessors:
            pred_task = self.get_task_by_id(pred_id)
            if pred_task and pred_task.assigned_processor != processor_id:
                message = self.get_message(pred_id, task.task_id)
                if message:
                    sec_overhead += self.calculate_security_overhead(message, pred_task.assigned_processor, processor_id)
        
        # Calculate critical path bonus
        critical_path_bonus = 0
        if task.priority > sum(t.priority for t in self.tasks) / len(self.tasks):
            # Task is on critical path, reward finishing it early
            critical_path_bonus = 25
            
        # Reward for processor ready time
        proc_ready_reward = 0
        processor = self.processors[processor_id - 1]
        if task.start_time == processor.available_time:
            # Task starts exactly when processor becomes available - optimal usage
            proc_ready_reward = 10
        
        # Combine all factors into final reward with better balanced weights
        reward = (
            (slack > 0) * 50 +  # Binary reward for meeting deadline
            min(0, slack) * 0.5 +  # Small penalty for how much over deadline
            load_balance_score +  # Load balancing score
            locality_bonus +  # Reward for keeping related tasks together
            parallel_bonus +  # Reward for parallelism
            critical_path_bonus +  # Reward for critical path tasks
            proc_ready_reward -  # Reward for processor utilization
            sec_overhead * 0.1  # Small penalty for security overhead
        )
        
        return reward

    def calculate_security_overhead(self, message, source_proc, dest_proc):
        """Calculate total security overhead for a message between processors"""
        if source_proc == dest_proc:
            return 0  # No overhead for same processor
            
        overhead = 0
        for service in ['confidentiality', 'integrity', 'authentication']:
            protocol_idx = message.assigned_security[service]
            if service in ['confidentiality', 'integrity']:
                # Data-dependent overheads
                overhead += self.security.overheads[service][protocol_idx + 1][source_proc - 1]
            else:  # authentication
                # Data-independent overheads
                overhead += self.security.overheads[service][protocol_idx + 1][dest_proc - 1]
                
        return overhead * message.size / 100  # Scale by message size
        
    def calculate_security_utility(self):
        """Calculate the total security utility."""
        total_utility = 0
        for message in self.messages:
            message_utility = 0
            for service in ['confidentiality', 'integrity', 'authentication']:
                protocol_idx = message.assigned_security[service]
                strength = self.security.strengths[service][protocol_idx]
                weight = message.weights[service]
                message_utility += weight * strength
            total_utility += message_utility
        return total_utility
        
    def enhance_security_with_slack(self):
        """Use remaining slack time to enhance security levels"""
        # Calculate current makespan
        makespan = max(task.finish_time for task in self.tasks)
        
        # Calculate slack time
        slack_time = self.deadline - makespan
        if slack_time <= 0:
            return  # No slack time available
        
        print(f"Enhancing security with {slack_time:.2f} slack time")
        
        # Keep track of remaining slack
        remaining_slack = slack_time
        
        # Calculate the potential benefit of upgrading each security service for each message
        while remaining_slack > 0:
            best_upgrade = None
            best_benefit_ratio = 0
            
            for message in self.messages:
                source_task = self.get_task_by_id(message.source_id)
                dest_task = self.get_task_by_id(message.dest_id)
                
                if not source_task or not dest_task or not source_task.is_scheduled or not dest_task.is_scheduled:
                    continue
                    
                if source_task.assigned_processor == dest_task.assigned_processor:
                    continue  # Skip intra-processor messages
                
                source_proc = source_task.assigned_processor
                dest_proc = dest_task.assigned_processor
                
                for service in ['confidentiality', 'integrity', 'authentication']:
                    current_level = message.assigned_security[service]
                    if current_level < len(self.security.strengths[service]) - 1:
                        # Can upgrade this service
                        next_level = current_level + 1
                        
                        # Calculate security benefit
                        current_strength = self.security.strengths[service][current_level]
                        next_strength = self.security.strengths[service][next_level]
                        strength_gain = next_strength - current_strength
                        security_benefit = message.weights[service] * strength_gain
                        
                        # Calculate time cost
                        if service in ['confidentiality', 'integrity']:
                            current_overhead = self.security.overheads[service][current_level + 1][source_proc - 1]
                            next_overhead = self.security.overheads[service][next_level + 1][source_proc - 1]
                        else:  # authentication
                            current_overhead = self.security.overheads[service][current_level + 1][dest_proc - 1]
                            next_overhead = self.security.overheads[service][next_level + 1][dest_proc - 1]
                            
                        time_cost = (next_overhead - current_overhead) * message.size / 100
                        
                        # Check if upgrade fits in remaining slack
                        if time_cost <= remaining_slack:
                            benefit_ratio = security_benefit / max(0.001, time_cost)
                            
                            if benefit_ratio > best_benefit_ratio:
                                best_benefit_ratio = benefit_ratio
                                best_upgrade = (message, service, next_level, time_cost, security_benefit)
            
            if best_upgrade:
                # Apply the best upgrade
                message, service, new_level, time_cost, benefit = best_upgrade
                message.assigned_security[service] = new_level
                remaining_slack -= time_cost
                
                print(f"Upgraded {message.id} {service} to level {new_level}, cost {time_cost:.2f}, benefit {benefit:.4f}")
            else:
                # No more beneficial upgrades possible
                break
        
        # Recalculate security utility after enhancements
        self.best_security_utility = self.calculate_security_utility()
        print(f"Enhanced security utility: {self.best_security_utility:.2f}, remaining slack: {remaining_slack:.2f}")

    def run(self):
        """Run the ImprovedQLearningScheduler"""
        start_time = time.time()
        makespan = self.schedule_tasks()
        security_utility = self.best_security_utility
        end_time = time.time()
        
        print(f"ImprovedQLearningScheduler: Makespan={makespan}, Security Utility={security_utility:.2f}")
        print(f"Execution time: {end_time - start_time:.4f} seconds")
        
        return makespan, security_utility

def optimize_security_assignment(scheduler):
    """Select security protocols that minimize overhead while meeting requirements"""
    for message in scheduler.messages:
        for service in ['confidentiality', 'integrity', 'authentication']:
            min_overhead = float('inf')
            best_protocol = 0
            for protocol in range(len(scheduler.security.strengths[service])):
                if scheduler.security.strengths[service][protocol] >= message.min_security[service]:
                    overhead = sum(
                        scheduler.security.overheads[service][protocol + 1][pid-1]
                        for pid in range(1, len(scheduler.processors)+1)
                    )
                    if overhead < min_overhead:
                        min_overhead = overhead
                        best_protocol = protocol
            message.assigned_security[service] = best_protocol

def main():
    # Create the TC system test case
    tasks, messages, processors, network, security_service, deadline = create_tc_test_case()
    
    # Run HSMS
    print("Running HSMS scheduler...")
    hsms = HSMS(copy.deepcopy(tasks), copy.deepcopy(messages), copy.deepcopy(processors), 
               network, security_service, deadline)
    hsms_makespan, hsms_security_utility = hsms.run()
    
    if hsms_makespan:
        hsms_fig = plot_gantt_chart("HSMS", hsms.schedule, len(processors), hsms_makespan)
        print(f"HSMS Security Utility: {hsms_security_utility:.2f}")
    
    # Run SHIELD
    print("\nRunning SHIELD scheduler...")
    tasks_shield, messages_shield, processors_shield, network, security_service, deadline = create_tc_test_case()
    
    shield = SHIELD(tasks_shield, messages_shield, processors_shield, network, security_service, deadline)
    shield_makespan, shield_security_utility = shield.run()
    
    if shield_makespan:
        shield_fig = plot_gantt_chart("SHIELD", shield.schedule, len(processors_shield), shield_makespan)
        print(f"SHIELD Security Utility: {shield_security_utility:.2f}")
    
    # Run ImprovedQLearningScheduler
    print("\nRunning ImprovedQLearningScheduler...")
    tasks_q, messages_q, processors_q, network, security_service, deadline = create_tc_test_case()
    
    # Add the optimize_security_assignment method to Scheduler class
    Scheduler.optimize_security_assignment = optimize_security_assignment
    
    q_scheduler = ImprovedQLearningScheduler(tasks_q, messages_q, processors_q, network, security_service, deadline)
    q_makespan, q_security_utility = q_scheduler.run()
    
    if q_makespan:
        q_fig = plot_gantt_chart("ImprovedQLearningScheduler", q_scheduler.schedule, len(processors_q), q_makespan)
        print(f"ImprovedQLearningScheduler Security Utility: {q_security_utility:.2f}")
    
    # Compare results
    print("\nComparison of Results:")
    print(f"{'Scheduler':<25} {'Makespan (ms)':<15} {'Security Utility':<15}")
    print("-" * 55)
    if hsms_makespan:
        print(f"{'HSMS':<25} {hsms_makespan:<15.2f} {hsms_security_utility:<15.2f}")
    if shield_makespan:
        print(f"{'SHIELD':<25} {shield_makespan:<15.2f} {shield_security_utility:<15.2f}")
    if q_makespan:
        print(f"{'ImprovedQLearningScheduler':<25} {q_makespan:<15.2f} {q_security_utility:<15.2f}")
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("HSMS Gantt Chart")
    plt.xlabel("Time (ms)")
    plt.ylabel("Processor")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 2)
    plt.title("SHIELD Gantt Chart")
    
if __name__ == "__main__":
    main()