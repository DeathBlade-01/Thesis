import numpy as np
import matplotlib.pyplot as plt
import copy
import random

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
                comm_finish_time = pred_task.finish_time + comm_time
            
            max_pred_finish_time = max(max_pred_finish_time, comm_finish_time)
        
        return max(processor_ready_time, max_pred_finish_time)

class QLearningScheduler(Scheduler):
    def estimate_security_cost(self, task):
        """Estimate potential security costs for a task"""
        total = 0
        for pred_id in task.predecessors:
            message = self.get_message(pred_id, task.task_id)
            if message:
                min_cost = min(
                    self.calculate_security_overhead(message, pid)
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
        self.alpha = 0.4  # Learning rate
        self.gamma = 0.8  # Discount factor
        self.epsilon = 0.2  # Exploration rate
        self.episodes = 300  # Learning episodes
        self.initialize_q_table()
        
    def initialize_q_table(self):
        """Initialize Q-table with default values for all possible state-action pairs"""
        # First compute task priorities if not done
        if any(task.priority is None for task in self.tasks):
            self.compute_task_priorities()
            
        # For each task, initialize Q-values for each processor assignment
        for task in self.tasks:
            for proc_id in range(1, len(self.processors) + 1):
                state = (task.task_id, proc_id)
                self.q_table[state] = 0.0

    def update_q_value(self, state, action, reward, next_state=None):
        # Update Q-value based on the action taken
        current_q = self.q_table.get((state, action), 0)
        
        # Simple update without next state (since we're not using traditional Q-learning)
        self.q_table[(state, action)] = current_q + self.alpha * (reward - current_q)

    def choose_processor(self, task):
        """Choose a processor for the given task using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Exploration: choose a random processor
            return random.randint(1, len(self.processors))
        else:
            # Exploitation: choose the processor with highest Q-value
            best_proc = 1
            best_q = float('-inf')
            
            for proc_id in range(1, len(self.processors) + 1):
                state = (task.task_id, proc_id)
                q_value = self.q_table.get(state, 0)
                
                if q_value > best_q:
                    best_q = q_value
                    best_proc = proc_id
            
            return best_proc

    def schedule_tasks(self):
        """Schedule tasks using Q-learning with parallel execution awareness"""
        # Initialize task priorities and security assignments
        self.compute_task_priorities()
        self.optimize_security_assignment()
        
        best_makespan = float('inf')
        best_schedule = None
        
        # Train the Q-table over multiple episodes
        for episode in range(self.episodes):
            # Reset processors and task scheduling status
            for processor in self.processors:
                processor.available_time = 0
            
            for task in self.tasks:
                task.is_scheduled = False
                task.assigned_processor = None
                task.start_time = None
                task.finish_time = None
            
            current_schedule = []
            
            # Sort tasks by priority (higher priority first)
            sorted_tasks = sorted(self.tasks, key=lambda x: -x.priority)
            
            # List to track tasks that are ready to be scheduled
            ready_tasks = [task for task in sorted_tasks 
                          if not task.predecessors or 
                          all(self.get_task_by_id(pred_id).is_scheduled for pred_id in task.predecessors)]
            
            # Continue scheduling until all tasks are scheduled
            while ready_tasks:
                # Sort ready tasks by priority
                ready_tasks.sort(key=lambda x: -x.priority)
                
                # Select the highest priority task
                current_task = ready_tasks.pop(0)
                
                # Choose processor using Q-learning policy
                proc_id = self.choose_processor(current_task)
                processor = self.processors[proc_id - 1]
                
                # Calculate EST and EFT
                est = self.calculate_est(current_task, proc_id)
                eft = est + current_task.execution_times[proc_id - 1]
                
                # Apply security overhead for cross-processor communications
                for pred_id in current_task.predecessors:
                    pred_task = self.get_task_by_id(pred_id)
                    if pred_task.assigned_processor != proc_id:
                        message = self.get_message(pred_id, current_task.task_id)
                        if message:
                            sec_overhead = self.calculate_security_overhead(message, proc_id)
                            eft += sec_overhead
                
                # Schedule the task
                current_task.assigned_processor = proc_id
                current_task.start_time = est
                current_task.finish_time = eft
                current_task.is_scheduled = True
                processor.available_time = eft
                
                # Record schedule entry
                entry = {
                    'task_id': current_task.task_id,
                    'processor': proc_id,
                    'start': current_task.start_time,
                    'finish': current_task.finish_time
                }
                current_schedule.append(entry)
                
                # Calculate reward for this scheduling decision
                reward = self.calculate_reward(current_task, proc_id)
                
                # Update Q-value
                state = (current_task.task_id, proc_id)
                self.update_q_value(state, proc_id, reward)
                
                # Update ready tasks
                for task in sorted_tasks:
                    if not task.is_scheduled and task not in ready_tasks:
                        if all(self.get_task_by_id(pred_id).is_scheduled for pred_id in task.predecessors):
                            ready_tasks.append(task)
            
            # Calculate makespan for this episode
            current_makespan = max(task.finish_time for task in self.tasks)
            
            # Save the best schedule
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_schedule = current_schedule
                
                # Gradually reduce exploration as we find better schedules
                self.epsilon *= 0.98
        
        # Set the final schedule and return the makespan
        self.schedule = best_schedule
        return best_makespan

    def calculate_reward(self, task, processor_id):
        """Enhanced reward function that encourages parallel execution and load balancing"""
        # Calculate slack time (how much before deadline)
        slack = self.deadline - task.finish_time
        
        # Calculate processor load balance (penalize unbalanced loads)
        proc_loads = [p.available_time for p in self.processors]
        max_load = max(proc_loads)
        min_load = min(proc_loads)
        load_imbalance = max_load - min_load
        
        # Calculate dependency parallelism bonus
        parallel_bonus = 0
        for other_task in self.tasks:
            if other_task.is_scheduled and other_task.task_id != task.task_id:
                # If tasks can run in parallel on different processors, reward it
                if (other_task.finish_time >= task.start_time and 
                    other_task.start_time <= task.finish_time and
                    other_task.assigned_processor != processor_id):
                    parallel_bonus += 25
        
        # Calculate security overhead
        sec_overhead = 0
        for pred_id in task.predecessors:
            pred_task = self.get_task_by_id(pred_id)
            if pred_task and pred_task.assigned_processor != processor_id:
                message = self.get_message(pred_id, task.task_id)
                if message:
                    sec_overhead += self.calculate_security_overhead(message, processor_id)
        
        # Combine all factors into final reward
        reward = (slack * 3) - (0.5 * load_imbalance) - (0.2 * sec_overhead) + parallel_bonus
        
        # Add large bonus/penalty based on whether we meet the deadline
        reward += 200 if slack > 0 else -200
        
        return reward

    def calculate_security_overhead(self, message, processor_id):
        """Calculate total security overhead for a message"""
        overhead = 0
        for service in ['confidentiality', 'integrity', 'authentication']:
            protocol_idx = message.assigned_security[service]
            overhead += self.security.overheads[service][protocol_idx + 1][processor_id - 1]
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

def create_tc_test_case():
    # Create tasks and messages as in previous examples
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
    
    processors = [Processor(1), Processor(2)]
    network = CommunicationNetwork(2)
    security_service = SecurityService()
    deadline = 1600  # As specified in the paper
    
    return tasks, messages, processors, network, security_service, deadline

def plot_security_improvements(messages):
    """Visualize security level improvements for each message"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data
    services = ['confidentiality', 'integrity', 'authentication']
    x = np.arange(len(messages))
    width = 0.2
    
    for i, service in enumerate(services):
        min_levels = [m.min_security[service] for m in messages]
        assigned_levels = [m.assigned_security[service] for m in messages]
        
        ax.bar(x + i*width, min_levels, width, label=f'Min {service}', alpha=0.5)
        ax.bar(x + i*width, assigned_levels, width, bottom=min_levels, 
               label=f'Added {service}', alpha=0.8)
    
    ax.set_xlabel('Messages')
    ax.set_ylabel('Security Level')
    ax.set_title('Security Level Improvements')
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.id for m in messages])
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_schedule(schedule, processors, makespan):
    """Visualize the task schedule as a Gantt chart"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create a color map for tasks
    colors = plt.colormaps['tab20'].resampled(len(schedule))
    
    # Plot each processor as a horizontal track
    for i, proc in enumerate(processors):
        ax.broken_barh([(0, makespan)], (i-0.4, 0.8), facecolors='lightgray')
        ax.text(-50, i, f'P{proc.proc_id}', ha='right', va='center')
    
    # Plot each scheduled task
    for entry in schedule:
        proc_idx = entry['processor'] - 1
        duration = entry['finish'] - entry['start']
        rect = patches.Rectangle(
            (entry['start'], proc_idx-0.4),
            duration,
            0.8,
            linewidth=1,
            edgecolor='black',
            facecolor=colors(entry['task_id'] % 20)
        )
        ax.add_patch(rect)
        ax.text(
            entry['start'] + duration/2,
            proc_idx,
            f'T{entry["task_id"]}',
            ha='center',
            va='center',
            color='white'
        )
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Processor')
    ax.set_title(f'Task Schedule (Makespan: {makespan}ms)')
    ax.set_yticks(range(len(processors)))
    ax.set_yticklabels([f'P{p.proc_id}' for p in processors])
    ax.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create the TC system test case
    tasks, messages, processors, network, security_service, deadline = create_tc_test_case()
    
    # Run QLearning Scheduler with fixed seed for deterministic results
    qlearning_scheduler = QLearningScheduler(tasks, messages, processors, network, security_service, deadline, seed=42)
    makespan = qlearning_scheduler.schedule_tasks()
    
    security_utility = qlearning_scheduler.calculate_security_utility()
    print(f"QLSHIELD Scheduler completed. Makespan: {makespan}, Security Utility: {security_utility:.2f}")
    plot_schedule(qlearning_scheduler.schedule, processors, makespan)

if __name__ == "__main__":
    main()
