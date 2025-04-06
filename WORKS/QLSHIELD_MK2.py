import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import time
from collections import defaultdict

# Keep existing Task, Message, Processor, SecurityService, and CommunicationNetwork classes
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
    
    def calculate_est(self, task, processor):
        """Calculate Earliest Start Time for a task on a processor."""
        processor_ready_time = self.processors[processor - 1].available_time
        
        max_pred_finish_time = 0
        for pred_id in task.predecessors:
            pred_task = self.get_task_by_id(pred_id)
            if not pred_task.is_scheduled:
                return float('inf')  # Predecessor not scheduled yet
            
            message = self.get_message(pred_id, task.task_id)
            if not message:
                continue
                
            # If predecessor is on the same processor, only consider finish time
            if pred_task.assigned_processor == processor:
                comm_finish_time = pred_task.finish_time
            else:
                # Consider communication time and security overhead
                security_overhead = self.calc_security_overhead(
                    message, 
                    pred_task.assigned_processor, 
                    processor
                )
                comm_time = self.network.get_communication_time(
                    message, 
                    pred_task.assigned_processor, 
                    processor
                )
                comm_finish_time = pred_task.finish_time + comm_time + security_overhead
            
            max_pred_finish_time = max(max_pred_finish_time, comm_finish_time)
        
        return max(processor_ready_time, max_pred_finish_time)

class QLScheduler(Scheduler):
    """Q-Learning based scheduler for task mapping and security optimization"""
    
    def __init__(self, tasks, messages, processors, network, security_service, deadline, 
                 alpha=0.1, gamma=0.9, epsilon=0.1, episodes=200):
        super().__init__(tasks, messages, processors, network, security_service, deadline)
        # Q-learning parameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.episodes = episodes  # Number of episodes to train
        
        # Q-tables for task mapping and security level selection
        self.task_mapping_q_table = defaultdict(lambda: np.zeros(len(processors)))
        self.security_q_table = defaultdict(lambda: np.zeros(3))  # 3 security dimensions
        
        # Original HSMS for comparison
        self.hsms = HSMS(copy.deepcopy(tasks), messages, copy.deepcopy(processors), 
                         network, security_service, deadline)
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
    
    def get_state_representation(self, task, scheduled_tasks):
        """Convert the current scheduling state to a discrete state representation"""
        # Create a state representation based on:
        # 1. Task priority level (discretized)
        # 2. Number of predecessors scheduled
        # 3. Number of processors with high load
        
        # Discretize task priority into 5 levels
        if hasattr(task, 'priority') and task.priority is not None:
            priority_level = min(4, int(task.priority / 500))
        else:
            priority_level = 0
            
        # Count scheduled predecessors
        scheduled_preds = sum(1 for pred_id in task.predecessors 
                            if any(t['task_id'] == pred_id for t in scheduled_tasks))
        
        # Count processors with high load (>50% of current max time)
        current_max_time = max(p.available_time for p in self.processors) if self.processors else 0
        high_load_threshold = current_max_time * 0.5 if current_max_time > 0 else 0
        high_load_procs = sum(1 for p in self.processors if p.available_time > high_load_threshold)
        
        return (priority_level, scheduled_preds, high_load_procs)
    
    def get_security_state(self, message, current_makespan):
        """Get state representation for security optimization Q-learning"""
        # Slack percentage (discretized)
        slack_percentage = min(9, int((self.deadline - current_makespan) / self.deadline * 10))
        
        # Message priority based on weights (discretized)
        total_weight = sum(message.weights.values())
        priority_level = min(4, int(total_weight * 5))
        
        # Current security level (average of all services)
        current_levels = [message.assigned_security[s] for s in ['confidentiality', 'integrity', 'authentication']]
        avg_level = min(4, int(sum(current_levels) / len(current_levels)))
        
        return (slack_percentage, priority_level, avg_level)
    
    def calculate_reward_task_mapping(self, task, processor, current_schedule):
        """Calculate reward for task mapping decision"""
        # Make a copy of processors to simulate this assignment
        processors_copy = copy.deepcopy(self.processors)
        
        # Calculate EST and EFT for this assignment
        est = self.calculate_est(task, processor)
        if est == float('inf'):
            return -100  # Invalid assignment (predecessors not scheduled)
        
        eft = est + task.execution_times[processor - 1]
        
        # Calculate temporary makespan
        temp_makespan = max([p.available_time for p in processors_copy] + [eft])
        
        # Calculate global slack
        global_slack = max(0, self.deadline - temp_makespan)
        
        # Calculate local slack (difference between this task's finish time and the start time of its successors)
        local_slack = float('inf')
        for succ_id in task.successors:
            succ_task = self.get_task_by_id(succ_id)
            if succ_task and any(s['task_id'] == succ_task.task_id for s in current_schedule):
                succ_entry = next(s for s in current_schedule if s['task_id'] == succ_task.task_id)
                local_slack = min(local_slack, succ_entry['start_time'] - eft)
        
        if local_slack == float('inf'):
            local_slack = global_slack
        
        # Reward formula:
        # - Penalize makespan relative to deadline
        # - Reward global and local slack
        reward = (self.deadline - temp_makespan) * 0.5 + global_slack * 0.3 + local_slack * 0.2
        
        # Add penalty if exceeding deadline
        if temp_makespan > self.deadline:
            reward -= 1000
        
        return reward
    
    def calculate_reward_security(self, message, service, protocol_idx, current_makespan):
        """Calculate reward for security level selection"""
        # Calculate security benefit
        current_protocol_idx = message.assigned_security[service]
        current_strength = self.security.strengths[service][current_protocol_idx]
        new_strength = self.security.strengths[service][protocol_idx]
        strength_diff = new_strength - current_strength
        security_benefit = message.weights[service] * strength_diff
        
        # Calculate time penalty
        source_task = self.get_task_by_id(message.source_id)
        dest_task = self.get_task_by_id(message.dest_id)
        
        if not source_task or not dest_task or not source_task.is_scheduled or not dest_task.is_scheduled:
            return -100  # Invalid upgrade
        
        source_proc = source_task.assigned_processor
        dest_proc = dest_task.assigned_processor
        
        time_penalty = 0
        if service in ['confidentiality', 'integrity']:
            current_protocol = current_protocol_idx + 1  # 1-indexed in overhead table
            new_protocol = protocol_idx + 1
            
            current_overhead = (message.size / 1024) * self.security.overheads[service][current_protocol][source_proc - 1]
            new_overhead = (message.size / 1024) * self.security.overheads[service][new_protocol][source_proc - 1]
            time_penalty = new_overhead - current_overhead
        else:  # authentication
            current_protocol = current_protocol_idx + 1
            new_protocol = protocol_idx + 1
            
            current_overhead = self.security.overheads[service][current_protocol][dest_proc - 1]
            new_overhead = self.security.overheads[service][new_protocol][dest_proc - 1]
            time_penalty = new_overhead - current_overhead
        
        # Calculate new makespan
        new_makespan = current_makespan + time_penalty
        
        # Calculate reward based on security benefit and deadline constraint
        if new_makespan > self.deadline:
            reward = -1000  # Large penalty for exceeding deadline
        else:
            # Reward is security benefit scaled by remaining slack
            slack_ratio = (self.deadline - new_makespan) / self.deadline
            reward = security_benefit * 100 * slack_ratio
        
        return reward
    
    def q_learning_task_mapping(self):
        """Apply Q-learning for task-to-processor mapping"""
        # Sort tasks by priority (upward rank)
        self.compute_task_priorities()
        sorted_tasks = sorted(self.tasks, key=lambda t: -t.priority)
        
        best_makespan = float('inf')
        best_schedule = None
        
        for episode in range(self.episodes):
            # Reset processors for this episode
            for p in self.processors:
                p.available_time = 0
            
            # Reset task scheduling status
            for task in self.tasks:
                task.assigned_processor = None
                task.start_time = None
                task.finish_time = None
                task.is_scheduled = False
            
            current_schedule = []
            
            # Schedule each task using Q-learning
            for task in sorted_tasks:
                # Get state for this task
                state = self.get_state_representation(task, current_schedule)
                
                # Choose action (processor) using epsilon-greedy policy
                if random.random() < self.epsilon:
                    # Exploration: choose random processor
                    processor_idx = random.randint(0, len(self.processors) - 1)
                else:
                    # Exploitation: choose best processor according to Q-table
                    processor_idx = np.argmax(self.task_mapping_q_table[state])
                
                processor = self.processors[processor_idx]
                
                # Calculate earliest start and finish time
                est = self.calculate_est(task, processor.proc_id)
                
                # If EST is infinity, try other processors
                if est == float('inf'):
                    valid_processors = []
                    for p_idx, p in enumerate(self.processors):
                        if self.calculate_est(task, p.proc_id) < float('inf'):
                            valid_processors.append(p_idx)
                    
                    if not valid_processors:
                        continue  # Skip this task for now
                    
                    processor_idx = random.choice(valid_processors)
                    processor = self.processors[processor_idx]
                    est = self.calculate_est(task, processor.proc_id)
                
                eft = est + task.execution_times[processor.proc_id - 1]
                
                # Assign task to processor
                task.assigned_processor = processor.proc_id
                task.start_time = est
                task.finish_time = eft
                task.is_scheduled = True
                processor.available_time = eft
                
                # Add to schedule
                current_schedule.append({
                    'task_id': task.task_id,
                    'name': task.name,
                    'processor': task.assigned_processor,
                    'start_time': task.start_time,
                    'finish_time': task.finish_time
                })
                
                # Calculate reward
                reward = self.calculate_reward_task_mapping(task, processor.proc_id, current_schedule)
                
                # Get next state (for the next task)
                next_task_idx = sorted_tasks.index(task) + 1
                if next_task_idx < len(sorted_tasks):
                    next_task = sorted_tasks[next_task_idx]
                    next_state = self.get_state_representation(next_task, current_schedule)
                    # Update Q-value using Q-learning update rule
                    max_next_q = np.max(self.task_mapping_q_table[next_state])
                    self.task_mapping_q_table[state][processor_idx] += self.alpha * (
                        reward + self.gamma * max_next_q - self.task_mapping_q_table[state][processor_idx]
                    )
                else:
                    # Final task in the episode
                    self.task_mapping_q_table[state][processor_idx] += self.alpha * (
                        reward - self.task_mapping_q_table[state][processor_idx]
                    )
            
            # Check if all tasks are scheduled
            if all(task.is_scheduled for task in self.tasks):
                # Calculate makespan for this episode
                makespan = max(task.finish_time for task in self.tasks)
                
                # Update best schedule if this one is better
                if makespan < best_makespan and makespan <= self.deadline:
                    best_makespan = makespan
                    best_schedule = copy.deepcopy(current_schedule)
                    
                    # Reduce exploration rate gradually
                    self.epsilon = max(0.01, self.epsilon * 0.99)
        
        # Use the best schedule found
        self.schedule = best_schedule if best_schedule else []
        
        # Update task information from the schedule
        for task in self.tasks:
            schedule_entry = next((s for s in self.schedule if s['task_id'] == task.task_id), None)
            if schedule_entry:
                task.assigned_processor = schedule_entry['processor']
                task.start_time = schedule_entry['start_time']
                task.finish_time = schedule_entry['finish_time']
                task.is_scheduled = True
        
        if self.schedule:
            self.makespan = max(task.finish_time for task in self.tasks if task.is_scheduled)
            return self.makespan
        else:
            return None
    
    def q_learning_security_optimization(self):
        """Apply Q-learning for security level optimization"""
        # Assign minimum security levels initially
        self.assign_minimum_security()
        
        # Calculate current makespan
        current_makespan = max(task.finish_time for task in self.tasks if task.is_scheduled)
        
        # Calculate available slack time
        available_slack = max(0, self.deadline - current_makespan)
        
        best_security_utility = self.calculate_security_utility()
        best_security_config = {msg.id: copy.deepcopy(msg.assigned_security) for msg in self.messages}
        
        # Q-learning for security optimization
        for episode in range(self.episodes):
            # Reset security levels to minimum
            for message in self.messages:
                message.assigned_security = copy.deepcopy(best_security_config[message.id])
            
            # Current makespan with security overhead
            episode_makespan = current_makespan
            
            # Randomly select messages to upgrade
            message_order = list(self.messages)
            random.shuffle(message_order)
            
            for message in message_order:
                source_task = self.get_task_by_id(message.source_id)
                dest_task = self.get_task_by_id(message.dest_id)
                
                if not source_task or not dest_task or not source_task.is_scheduled or not dest_task.is_scheduled:
                    continue
                
                # Try to upgrade each security service
                for service in ['confidentiality', 'integrity', 'authentication']:
                    current_idx = message.assigned_security[service]
                    max_idx = len(self.security.strengths[service]) - 1
                    
                    if current_idx >= max_idx:
                        continue  # Already at max level
                    
                    # Get current state
                    state = self.get_security_state(message, episode_makespan)
                    
                    # Choose action (security level) using epsilon-greedy
                    if random.random() < self.epsilon:
                        # Exploration: choose random level (higher than current)
                        new_idx = random.randint(current_idx + 1, max_idx)
                    else:
                        # Exploitation: choose best level according to Q-table
                        service_idx = ['confidentiality', 'integrity', 'authentication'].index(service)
                        q_values = self.security_q_table[state][service_idx]
                        
                        # Only consider levels higher than current
                        candidates = list(range(current_idx + 1, max_idx + 1))
                        if not candidates:
                            continue
                        
                        new_idx = candidates[np.argmax([q_values for _ in candidates])]
                    
                    # Calculate reward for this upgrade
                    reward = self.calculate_reward_security(
                        message, service, new_idx, episode_makespan
                    )
                    
                    # Check if upgrade is feasible
                    source_proc = source_task.assigned_processor
                    dest_proc = dest_task.assigned_processor
                    
                    # Calculate time penalty
                    time_penalty = 0
                    if service in ['confidentiality', 'integrity']:
                        current_protocol = current_idx + 1
                        new_protocol = new_idx + 1
                        
                        current_overhead = (message.size / 1024) * self.security.overheads[service][current_protocol][source_proc - 1]
                        new_overhead = (message.size / 1024) * self.security.overheads[service][new_protocol][source_proc - 1]
                        time_penalty = new_overhead - current_overhead
                    else:  # authentication
                        current_protocol = current_idx + 1
                        new_protocol = new_idx + 1
                        
                        current_overhead = self.security.overheads[service][current_protocol][dest_proc - 1]
                        new_overhead = self.security.overheads[service][new_protocol][dest_proc - 1]
                        time_penalty = new_overhead - current_overhead
                    
                    # Check if upgrade would exceed deadline
                    new_makespan = episode_makespan + time_penalty
                    
                    if new_makespan <= self.deadline:
                        # Apply the upgrade
                        message.assigned_security[service] = new_idx
                        episode_makespan = new_makespan
                        
                        # Get next state
                        next_state = self.get_security_state(message, episode_makespan)
                        
                        # Update Q-value
                        service_idx = ['confidentiality', 'integrity', 'authentication'].index(service)
                        max_next_q = np.max(self.security_q_table[next_state])
                        
                        self.security_q_table[state][service_idx] += self.alpha * (
                            reward + self.gamma * max_next_q - self.security_q_table[state][service_idx]
                        )
            
            # Calculate security utility for this episode
            security_utility = self.calculate_security_utility()
            
            # Update best configuration if this one is better
            if security_utility > best_security_utility and episode_makespan <= self.deadline:
                best_security_utility = security_utility
                best_security_config = {msg.id: copy.deepcopy(msg.assigned_security) for msg in self.messages}
                
                # Reduce exploration rate
                self.epsilon = max(0.01, self.epsilon * 0.99)
        
        # Apply the best security configuration
        for message in self.messages:
            message.assigned_security = copy.deepcopy(best_security_config[message.id])
        
        return best_security_utility
    
    def run(self):
        """Run the QL-SHIELD scheduler"""
        print("Running Q-Learning for Task Mapping...")
        makespan = self.q_learning_task_mapping()
        
        if makespan is None or makespan > self.deadline:
            print(f"QL-SHIELD failed to meet deadline in task mapping phase.")
            return None, None
        
        print(f"Task mapping successful. Makespan: {makespan}, Deadline: {self.deadline}")
        print("Running Q-Learning for Security Optimization...")
        
        security_utility = self.q_learning_security_optimization()
        
        print(f"QL-SHIELD successful. Makespan: {makespan}, Security Utility: {security_utility}")
        return makespan, security_utility
class HSMS(Scheduler):
    """Heterogeneous Security-aware Makespan minimizing Scheduler"""
    
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
        hsms_makespan, hsms_security_utility = self.hsms.run()
        
        if hsms_makespan is None:
            print("SHIELD cannot proceed since HSMS failed to meet the deadline.")
            return None, None
        
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
        
        print(f"SHIELD successful. Makespan: {makespan}, Security Utility: {security_utility}")
        return makespan, security_utility

def plot_gantt_chart(title, schedule, num_processors, makespan):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for entry in schedule:
        task_id = entry['task_id']
        name = entry['name']
        proc = entry['processor']
        start = entry['start_time']
        finish = entry['finish_time']
        
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

# Create the TC system test case as described in the paper
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


# Update the main function to include QL-SHIELD
def main():
    # Create the TC system test case
    tasks, messages, processors, network, security_service, deadline = create_tc_test_case()
    
    # Run HSMS
    hsms = HSMS(tasks, messages, processors, network, security_service, deadline)
    hsms_makespan, hsms_security_utility = hsms.run()
    
    if hsms_makespan:
        plot_gantt_chart("HSMS", hsms.schedule, len(processors), hsms_makespan)
        print(f"HSMS Security Utility: {hsms_security_utility:.2f}")
    
    # Run SHIELD
    tasks_shield, messages, processors_shield, network, security_service, deadline = create_tc_test_case()
    shield = SHIELD(tasks_shield, messages, processors_shield, network, security_service, deadline)
    shield_makespan, shield_security_utility = shield.run()
    
    if shield_makespan:
        plot_gantt_chart("SHIELD", shield.schedule, len(processors_shield), shield_makespan)
        print(f"SHIELD Security Utility: {shield_security_utility:.2f}")
    
    # Run QL-SHIELD
    start_time=time.time()

    tasks_ql, messages_ql, processors_ql, network_ql, security_service_ql, deadline_ql = create_tc_test_case()
    ql_shield = QLScheduler(tasks_ql, messages_ql, processors_ql, network_ql, security_service_ql, deadline_ql)
    ql_shield_makespan, ql_shield_security_utility = ql_shield.run()
    end_time=time.time()
    if ql_shield_makespan:
        plot_gantt_chart("QL-SHIELD", ql_shield.schedule, len(processors_ql), ql_shield_makespan)
        print(f"QL-SHIELD Security Utility: {ql_shield_security_utility:.2f}")
        
        # Compare results
        print("\nComparison:")
        print(f"HSMS: Makespan={hsms_makespan}, Security={hsms_security_utility:.2f}")
        print(f"SHIELD: Makespan={shield_makespan}, Security={shield_security_utility:.2f}")
        print(f"QL-SHIELD: Makespan={ql_shield_makespan}, Security={ql_shield_security_utility:.2f}, Execution time: {(end_time-start_time):.4f}")
    
    # Show the plots
    plt.show()

if __name__ == "__main__":
    main()