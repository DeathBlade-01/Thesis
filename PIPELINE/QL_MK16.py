import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import time
from LASTRESORT import Task, Message, Processor, SecurityService, CommunicationNetwork, Scheduler, HSMS, SHIELD

class ImprovedQLearningScheduler(Scheduler):
    def __init__(self, tasks, messages, processors, network, security_service, deadline, seed=42):
        super().__init__(tasks, messages, processors, network, security_service, deadline)
        random.seed(seed)
        np.random.seed(seed)
        self.q_table = {}
        self.alpha = 0.2
        self.gamma = 0.9
        self.epsilon = 0.8
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.97
        self.episodes = 500
        self.best_makespan = float('inf')
        self.best_schedule = None
        self.best_security_utility = 0
        self.initialize_q_table()
        self.security = security_service
        
    def initialize_q_table(self):
        """Initialize Q-table with default values for all state-action pairs."""
        for task in self.tasks:
            for proc_id in range(1, len(self.processors) + 1):
                self.q_table[(task.task_id, proc_id)] = 0.0
                
    def update_q_value(self, task_id, processor, reward, next_task_id=None):
        """Update Q-value using the Q-learning formula."""
        current_q = self.q_table.get((task_id, processor), 0)
        if next_task_id:
            next_max_q = max(self.q_table.get((next_task_id, p), 0) for p in range(1, len(self.processors) + 1))
            self.q_table[(task_id, processor)] = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * next_max_q)
        else:
            self.q_table[(task_id, processor)] = (1 - self.alpha) * current_q + self.alpha * reward
            
    def choose_processor(self, task, scheduled_tasks=None):
        """Choose a processor using an epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            return random.choice(range(1, len(self.processors) + 1))
        else:
            return max(range(1, len(self.processors) + 1), key=lambda p: self.q_table.get((task.task_id, p), 0))
            
    def calculate_improved_reward(self, task, processor_id):
        """Calculate reward for scheduling a task on a processor."""
        processor = self.processors[processor_id - 1]
        slack = self.deadline - task.finish_time
        locality_bonus = sum(10 for pred_id in task.predecessors if self.get_task_by_id(pred_id).assigned_processor == processor_id)
        return max(0, slack) + locality_bonus - processor.available_time
        
    def calculate_est(self, task, processor_id):
        """Calculate earliest start time for a task on a processor with immediate execution."""
        processor = self.processors[processor_id - 1]
        
        # Base EST is when the processor becomes available
        est = processor.available_time
        
        # Check all predecessors
        for pred_id in task.predecessors:
            pred_task = self.get_task_by_id(pred_id)
            if not pred_task.is_scheduled:
                continue
                
            if pred_task.assigned_processor == processor_id:
                # Same processor: just wait for predecessor to finish
                est = max(est, pred_task.finish_time)
            else:
                # Different processor: account for communication time
                message = self.get_message(pred_id, task.task_id)
                if message:
                    # Calculate communication time including security overhead
                    comm_time = self.calculate_communication_time(message, pred_task.assigned_processor, processor_id)
                    est = max(est, pred_task.finish_time + comm_time)
        
        return est
        
    def calculate_communication_time(self, message, source_proc, dest_proc):
        """Calculate communication time including security overhead."""
        if source_proc == dest_proc:
            return 0  # No communication needed for same processor
            
        # Base communication time (proportional to message size)
        base_comm_time = message.size / 100
        
        # Add security overhead
        security_overhead = 0
        for service in ['confidentiality', 'integrity', 'authentication']:
            protocol_idx = message.assigned_security[service]
            if service in ['confidentiality', 'integrity']:
                security_overhead += self.security.overheads[service][protocol_idx + 1][source_proc - 1]
            else:  # authentication
                security_overhead += self.security.overheads[service][protocol_idx + 1][dest_proc - 1]
                
        return base_comm_time + (security_overhead * message.size / 100)
        
    def run_shield_baseline(self):
        """Run SHIELD algorithm to get a baseline schedule for comparison"""
        # Create a new SHIELD instance with copies of the current state
        shield = SHIELD(copy.deepcopy(self.tasks), copy.deepcopy(self.messages), 
                       copy.deepcopy(self.processors), self.network, self.security, self.deadline)
        
        # Run SHIELD algorithm
        shield_makespan, shield_security_utility = shield.run()
        
        # Copy SHIELD schedule back to our tasks
        for task in self.tasks:
            shield_task = next((t for t in shield.tasks if t.task_id == task.task_id), None)
            if shield_task and shield_task.is_scheduled:
                task.assigned_processor = shield_task.assigned_processor
                task.start_time = shield_task.start_time
                task.finish_time = shield_task.finish_time
                task.is_scheduled = True
                
        # Update processor available times
        for proc in self.processors:
            proc_tasks = [task for task in self.tasks if task.is_scheduled and task.assigned_processor == proc.proc_id]
            if proc_tasks:
                proc.available_time = max(task.finish_time for task in proc_tasks)
            else:
                proc.available_time = 0
                
        # Copy security assignments
        for message in self.messages:
            shield_message = next((m for m in shield.messages if m.id == message.id), None)
            if shield_message:
                message.assigned_security = copy.deepcopy(shield_message.assigned_security)
                
        return shield_makespan
    
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
        
    def schedule_tasks(self):
        """Schedule tasks using improved Q-learning with SHIELD seeding"""
        # First compute task priorities before scheduling
        self.compute_task_priorities()
        
        # Run SHIELD baseline first to get a good starting point
        shield_makespan = self.run_shield_baseline()
        shield_schedule = []
        shield_tasks_state = copy.deepcopy(self.tasks)
        
        for task in self.tasks:
            if task.is_scheduled:
                entry = {
                    'task_id': task.task_id,
                    'name': task.name,
                    'processor': task.assigned_processor,
                    'start_time': task.start_time,
                    'finish_time': task.finish_time
                }
                shield_schedule.append(entry)
                
        # Set the SHIELD solution as initial best
        self.best_makespan = shield_makespan
        self.best_schedule = shield_schedule
        self.best_tasks_state = shield_tasks_state
        self.best_security_utility = self.calculate_security_utility()
        
        print(f"SHIELD baseline: Makespan={shield_makespan}, Security Utility={self.best_security_utility:.2f}")
        
        # Start Q-learning episodes from this point
        for episode in range(self.episodes):
            # Add a safety counter to prevent infinite loops
            safety_counter = 0
            max_iterations = 1000  # Set a reasonable maximum number of iterations
            
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
            
            # Create a time-based scheduling approach for better parallelism
            ready_tasks = [task for task in sorted_tasks 
                        if not task.predecessors or 
                        all(self.get_task_by_id(pred_id).is_scheduled for pred_id in task.predecessors)]
            
            current_time = 0
            all_scheduled = False
            
            while not all_scheduled and safety_counter < max_iterations:
                safety_counter += 1
                
                # Find all tasks that are ready at current_time
                for task in sorted_tasks:
                    if not task.is_scheduled and task not in ready_tasks:
                        # Check if all predecessors are scheduled
                        all_preds_scheduled = all(self.get_task_by_id(pred_id).is_scheduled 
                                                for pred_id in task.predecessors)
                        
                        # Only check finish times if all predecessors are scheduled
                        if all_preds_scheduled:
                            # Get the finish times of all predecessors
                            pred_finish_times = [self.get_task_by_id(pred_id).finish_time 
                                            for pred_id in task.predecessors]
                            
                            # Check if all predecessors have finished by current_time
                            if all(finish_time <= current_time for finish_time in pred_finish_times):
                                ready_tasks.append(task)
                
                if ready_tasks:
                    # Sort ready tasks by priority
                    ready_tasks.sort(key=lambda x: -x.priority)
                    
                    # Select the highest priority task
                    current_task = ready_tasks.pop(0)
                    
                    # Choose processor using improved Q-learning policy
                    proc_id = self.choose_processor(current_task)
                    processor = self.processors[proc_id - 1]
                    
                    # Calculate EST and EFT, ensuring we don't start before current_time
                    est = max(current_time, self.calculate_est(current_task, proc_id))
                    eft = est + current_task.execution_times[proc_id - 1]
                    
                    # Schedule the task
                    current_task.assigned_processor = proc_id
                    current_task.start_time = est
                    current_task.finish_time = eft
                    current_task.is_scheduled = True
                    processor.available_time = eft
                    processor.task_history = getattr(processor, 'task_history', [])
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
                    
                    # Calculate reward for this scheduling decision
                    reward = self.calculate_improved_reward(current_task, proc_id)
                    
                    # Find next task that might become ready
                    next_task = None
                    for task in sorted_tasks:
                        if not task.is_scheduled and task not in ready_tasks:
                            if all(self.get_task_by_id(pred_id).is_scheduled for pred_id in task.predecessors):
                                next_task = task
                                break
                                
                    # Update Q-value
                    if next_task:
                        self.update_q_value(current_task.task_id, proc_id, reward, next_task.task_id)
                    else:
                        self.update_q_value(current_task.task_id, proc_id, reward)
                    
                else:
                    # No tasks ready at current time, advance time to next processor availability
                    next_times = []
                    for proc in self.processors:
                        if proc.available_time > current_time:
                            next_times.append(proc.available_time)
                            
                    # Also consider when tasks might become ready due to predecessors finishing
                    for task in sorted_tasks:
                        if not task.is_scheduled and task not in ready_tasks:
                            # Check if all predecessors are scheduled
                            if all(self.get_task_by_id(pred_id).is_scheduled for pred_id in task.predecessors):
                                # Find the maximum finish time of predecessors
                                max_finish = max(self.get_task_by_id(pred_id).finish_time for pred_id in task.predecessors)
                                if max_finish > current_time:
                                    next_times.append(max_finish)
                                
                    if next_times:
                        current_time = min(next_times)
                    else:
                        # If we can't find any upcoming times, increment by a small amount
                        current_time += 1
                        
                        # Debug output for troubleshooting
                        if safety_counter % 100 == 0:
                            print(f"Warning: Episode {episode}, iteration {safety_counter}, time={current_time}")
                            print(f"Scheduled tasks: {sum(1 for t in self.tasks if t.is_scheduled)}/{len(self.tasks)}")
                
                # Check if all tasks are scheduled
                all_scheduled = all(task.is_scheduled for task in self.tasks)
                
            # Check if we hit the safety limit
            if safety_counter >= max_iterations:
                print(f"Warning: Episode {episode} hit maximum iterations ({max_iterations}). Skipping this episode.")
                continue
                
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
                
        # Enhanced parallelism bonus - prioritize this more
        parallel_bonus = 0
        for other_task in self.tasks:
            if other_task.is_scheduled and other_task.task_id != task.task_id:
                # If tasks can run in parallel on different processors, reward it
                if (other_task.finish_time >= task.start_time and
                    other_task.start_time <= task.finish_time and
                    other_task.assigned_processor != processor_id):
                    parallel_bonus += 25  # Increased from 15 to prioritize parallelism
                    
        # Calculate security overhead
        sec_overhead = 0
        for pred_id in task.predecessors:
            pred_task = self.get_task_by_id(pred_id)
            if pred_task and pred_task.assigned_processor != processor_id:
                message = self.get_message(pred_id, task.task_id)
                if message:
                    sec_overhead += self.calc_security_overhead(message, pred_task.assigned_processor, processor_id)
                    
        # Calculate critical path bonus
        critical_path_bonus = 0
        if task.priority > sum(t.priority for t in self.tasks) / len(self.tasks):
            # Task is on critical path, reward finishing it early
            critical_path_bonus = 25
            
        # Immediate start bonus - reward starting tasks with no delay
        # This addresses the issue with T6 and T9 waiting unnecessarily
        immediate_start_bonus = 0
        if task.predecessors:
            max_pred_finish = max(self.get_task_by_id(pred_id).finish_time for pred_id in task.predecessors 
                                if self.get_task_by_id(pred_id).is_scheduled)
            # If task starts soon after its predecessor finishes, give a large bonus
            if task.start_time <= max_pred_finish + 5:  # Small tolerance for scheduling decisions
                immediate_start_bonus = 50  # High reward for minimizing task waiting time
                
        # Combine all factors into final reward with better balanced weights
        reward = (
            (slack > 0) * 50 +  # Binary reward for meeting deadline
            min(0, slack) * 0.5 +  # Small penalty for how much over deadline
            load_balance_score +  # Load balancing score
            locality_bonus +  # Reward for keeping related tasks together
            parallel_bonus +  # Reward for parallelism (increased weight)
            critical_path_bonus +  # Reward for critical path tasks
            immediate_start_bonus -  # Reward for starting tasks immediately after predecessors
            sec_overhead * 0.1  # Small penalty for security overhead
        )
        
        return reward
        
    def calc_security_overhead(self, message, source_proc, dest_proc):
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
class SecurityEnhancedQLearningScheduler(ImprovedQLearningScheduler):
    def __init__(self, tasks, messages, processors, network, security_service, deadline, seed=42):
        super().__init__(tasks, messages, processors, network, security_service, deadline, seed)
        # More balanced parameters with less aggressive exploration
        self.alpha = 0.25  # Moderate learning rate
        self.gamma = 0.9   # Keep discount factor
        self.epsilon = 0.8  # Start with same exploration rate as parent
        self.min_epsilon = 0.05  # Lower minimum for better convergence
        self.epsilon_decay = 0.97  # Same decay rate as parent
        self.episodes = 600  # Slightly more episodes than parent
        # More balanced weights - prioritize makespan more
        self.security_weight = 0.3  # Reduced from 0.4
        self.makespan_weight = 0.7  # Increased from 0.6
        self.security_epsilon = 0.2  # Reduced from 0.3
        
    def calculate_improved_reward(self, task, processor_id):
        """Enhanced reward function with better balance between makespan and security"""
        # Get base reward from parent class
        base_reward = super().calculate_improved_reward(task, processor_id)
        
        # Calculate security utility for current state - scaled down
        current_security_utility = self.calculate_security_utility()
        security_reward = current_security_utility * 0.1  # Reduced from 0.2
        
        # Get dependent messages
        dependent_messages = []
        for msg in self.messages:
            if msg.source_id == task.task_id or msg.dest_id == task.task_id:
                dependent_messages.append(msg)
        
        # Calculate message security reward - with lower scaling
        message_security_reward = 0
        for message in dependent_messages:
            for service in ['confidentiality', 'integrity', 'authentication']:
                protocol_idx = message.assigned_security[service]
                strength = self.security.strengths[service][protocol_idx]
                weight = message.weights[service]
                message_security_reward += weight * strength * 2  # Reduced from 5
        
        # Combined reward with better balance
        return base_reward + security_reward + message_security_reward
    
    def optimize_security_assignment(self):
        """Improved security protocol selection that better balances overhead and utility"""
        for message in self.messages:
            source_task = self.get_task_by_id(message.source_id)
            dest_task = self.get_task_by_id(message.dest_id)
            
            # Skip if tasks not available or on same processor
            if not source_task or not dest_task:
                super().optimize_security_assignment()  # Use parent method
                continue
                
            if source_task.is_scheduled and dest_task.is_scheduled and \
               source_task.assigned_processor == dest_task.assigned_processor:
                # Set minimum security for intra-processor messages
                for service in ['confidentiality', 'integrity', 'authentication']:
                    min_level = 0
                    for i, strength in enumerate(self.security.strengths[service]):
                        if strength >= message.min_security[service]:
                            min_level = i
                            break
                    message.assigned_security[service] = min_level
                continue
            
            # For inter-processor messages, apply more sophisticated logic
            for service in ['confidentiality', 'integrity', 'authentication']:
                # Apply security epsilon with adjusted probability
                if random.random() < self.security_epsilon:
                    # Choose from mid-range protocols, not just highest
                    available_protocols = range(len(self.security.strengths[service]))
                    valid_protocols = [p for p in available_protocols 
                                     if self.security.strengths[service][p] >= message.min_security[service]]
                    
                    if valid_protocols:
                        # Sort by strength in descending order
                        sorted_protocols = sorted(valid_protocols, 
                                               key=lambda p: self.security.strengths[service][p],
                                               reverse=True)
                        
                        # Choose from mid-range protocols (not just top half)
                        mid_start = len(sorted_protocols) // 4
                        mid_end = 3 * len(sorted_protocols) // 4
                        mid_range = sorted_protocols[mid_start:mid_end+1] if mid_start < mid_end else sorted_protocols
                        message.assigned_security[service] = random.choice(mid_range)
                else:
                    # Find balanced protocol
                    best_protocol = 0
                    best_efficiency = 0
                    
                    for protocol in range(len(self.security.strengths[service])):
                        if self.security.strengths[service][protocol] >= message.min_security[service]:
                            # Calculate overhead estimate
                            if source_task.is_scheduled and dest_task.is_scheduled:
                                source_proc = source_task.assigned_processor
                                dest_proc = dest_task.assigned_processor
                                
                                if service in ['confidentiality', 'integrity']:
                                    overhead = self.security.overheads[service][protocol + 1][source_proc - 1]
                                else:  # authentication
                                    overhead = self.security.overheads[service][protocol + 1][dest_proc - 1]
                            else:
                                # If tasks not scheduled yet, use average overhead
                                overhead = sum(self.security.overheads[service][protocol + 1]) / len(self.processors)
                            
                            # Get strength and calculate efficiency
                            strength = self.security.strengths[service][protocol]
                            weight = message.weights[service]
                            
                            # Calculate weighted efficiency (strength/overhead ratio)
                            efficiency = (strength * weight) / (overhead + 0.001)
                            
                            if efficiency > best_efficiency:
                                best_efficiency = efficiency
                                best_protocol = protocol
                    
                    message.assigned_security[service] = best_protocol
    
    def choose_processor(self, task, scheduled_tasks=None):
        """Improved processor selection with better balance between makespan and security"""
        if random.random() < self.epsilon:
            return random.choice(range(1, len(self.processors) + 1))
        else:
            # Find processor that maximizes our objective function
            best_processor = None
            best_value = float('-inf')
            
            for p in range(1, len(self.processors) + 1):
                # Get Q-value for this processor
                q_value = self.q_table.get((task.task_id, p), 0)
                
                # Calculate makespan impact (prioritize this)
                processor = self.processors[p - 1]
                est = self.calculate_est(task, p)
                eft = est + task.execution_times[p - 1]
                
                # Stronger penalty for longer makespans
                makespan_impact = -eft  # Negative because lower is better
                
                # Check if this is on critical path - if so, prioritize speed even more
                if task.priority > sum(t.priority for t in self.tasks) / len(self.tasks) * 1.2:
                    makespan_weight = self.makespan_weight * 1.2  # Boost makespan importance
                    security_weight = self.security_weight * 0.8   # Reduce security importance
                else:
                    makespan_weight = self.makespan_weight
                    security_weight = self.security_weight
                
                # Calculate security impact - but don't let it override makespan
                security_impact = self.estimate_security_impact(task, p)
                
                # Balance with appropriate weights
                combined_value = (
                    makespan_weight * makespan_impact +
                    security_weight * security_impact +
                    q_value
                )
                
                if combined_value > best_value:
                    best_value = combined_value
                    best_processor = p
            
            return best_processor
    
    def estimate_security_impact(self, task, processor_id):
        """Estimate the impact on security utility if task is assigned to processor"""
        security_impact = 0
        
        # Focus only on immediate communication needs, not future ones
        # Check messages coming from predecessors
        for pred_id in task.predecessors:
            pred_task = self.get_task_by_id(pred_id)
            if pred_task and pred_task.is_scheduled:
                if pred_task.assigned_processor != processor_id:
                    # Communication needed - check security levels
                    message = self.get_message(pred_id, task.task_id)
                    if message:
                        for service in ['confidentiality', 'integrity', 'authentication']:
                            protocol_idx = message.assigned_security[service]
                            strength = self.security.strengths[service][protocol_idx]
                            weight = message.weights[service]
                            
                            # Scale security impact by message size
                            security_impact += (weight * strength) / (1 + message.size / 50)
        
        return security_impact
    
    def get_security_efficiency(self, message):
        """Calculate the efficiency of security protocols for a message
        
        Returns a value indicating how efficiently the current security protocols
        provide security benefits relative to their overhead.
        """
        efficiency = 0
        source_task = self.get_task_by_id(message.source_id)
        dest_task = self.get_task_by_id(message.dest_id)
        
        if not source_task or not dest_task or not source_task.is_scheduled or not dest_task.is_scheduled:
            return 0
        
        source_proc = source_task.assigned_processor
        dest_proc = dest_task.assigned_processor
        
        for service in ['confidentiality', 'integrity', 'authentication']:
            current_level = message.assigned_security[service]
            
            # Calculate security benefit of current level
            strength = self.security.strengths[service][current_level]
            weight = message.weights[service]
            security_benefit = weight * strength
            
            # Calculate overhead of current level
            if service in ['confidentiality', 'integrity']:
                overhead = self.security.overheads[service][current_level + 1][source_proc - 1]
            else:  # authentication
                overhead = self.security.overheads[service][current_level + 1][dest_proc - 1]
            
            time_cost = overhead * message.size / 100
            
            # Add small constant to avoid division by zero
            service_efficiency = security_benefit / (time_cost + 0.001)
            
            # Higher weights indicate more room for improvement
            potential_improvement = 1
            if current_level < len(self.security.strengths[service]) - 1:
                # Calculate potential improvement if upgraded
                next_level = current_level + 1
                next_strength = self.security.strengths[service][next_level]
                potential_strength_gain = next_strength - strength
                potential_improvement = potential_strength_gain / (strength + 0.001)
            
            # Combine current efficiency with potential for improvement
            efficiency += service_efficiency * potential_improvement
        
        return efficiency
    
    def enhance_security_with_slack(self):
        """More careful security enhancement with slack time"""
        # Calculate current makespan
        makespan = max(task.finish_time for task in self.tasks)
        
        # Calculate slack time with safety margin
        slack_time = (self.deadline - makespan) * 0.95  # Keep 5% safety margin
        if slack_time <= 0:
            print("No slack time available for security improvements")
            return  # No slack time available
        
        print(f"Enhancing security with {slack_time:.2f} slack time (with safety margin)")
        
        # Keep track of remaining slack
        remaining_slack = slack_time
        
        # First, identify the critical path and avoid adding overhead to those messages
        critical_path_tasks = self.identify_critical_path_tasks()
        print(f"Critical path tasks (avoiding security upgrades): {sorted(list(critical_path_tasks))}")
        
        # Keep track of security improvements
        security_improvements = []
        
        # Sort messages by security potential and criticality
        prioritized_messages = []
        for message in self.messages:
            source_task = self.get_task_by_id(message.source_id)
            dest_task = self.get_task_by_id(message.dest_id)
            
            if not source_task or not dest_task or not source_task.is_scheduled or not dest_task.is_scheduled:
                continue
            
            if source_task.assigned_processor == dest_task.assigned_processor:
                continue  # Skip intra-processor messages
            
            # Skip messages on critical path
            if source_task.task_id in critical_path_tasks and dest_task.task_id in critical_path_tasks:
                continue
            
            # Calculate security potential
            security_potential = self.get_security_efficiency(message)
            prioritized_messages.append((message, security_potential))
        
        # Sort by security potential (highest first)
        prioritized_messages.sort(key=lambda x: x[1], reverse=True)
        
        # Enhance security in order of priority
        for message, potential in prioritized_messages:
            source_task = self.get_task_by_id(message.source_id)
            dest_task = self.get_task_by_id(message.dest_id)
            source_proc = source_task.assigned_processor
            dest_proc = dest_task.assigned_processor
            
            message_improvements = []
            
            # Try to upgrade all services in order of weight (importance)
            for service in sorted(['confidentiality', 'integrity', 'authentication'], 
                                key=lambda s: message.weights[s], reverse=True):
                current_level = message.assigned_security[service]
                
                # Only try to upgrade one level at a time for more balanced improvements
                if current_level < len(self.security.strengths[service]) - 1:
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
                    
                    # If this upgrade fits in our remaining slack, apply it
                    if time_cost <= remaining_slack:
                        # Store the previous level for reporting
                        prev_level = message.assigned_security[service]
                        
                        # Apply the upgrade
                        message.assigned_security[service] = next_level
                        remaining_slack -= time_cost
                        
                        # Record this improvement
                        improvement = {
                            'service': service,
                            'from_level': prev_level,
                            'to_level': next_level,
                            'time_cost': time_cost,
                            'security_benefit': security_benefit
                        }
                        message_improvements.append(improvement)
            
            # If this message had any improvements, add to our tracking
            if message_improvements:
                security_improvements.append({
                    'message_id': message.id,
                    'source_task': source_task.task_id,
                    'dest_task': dest_task.task_id,
                    'improvements': message_improvements
                })
        
        # Recalculate security utility after enhancements
        self.best_security_utility = self.calculate_security_utility()
        
        # Print detailed summary of all security improvements
        print("\n===== SECURITY IMPROVEMENTS SUMMARY =====")
        if not security_improvements:
            print("No security improvements were made.")
        else:
            print(f"Total messages improved: {len(security_improvements)}")
            print(f"Final security utility: {self.best_security_utility:.4f}")
            print(f"Remaining slack time: {remaining_slack:.2f}")
            print("\nDetailed improvements:")
            
            for msg_imp in security_improvements:
                print(f"\nMessage from Task {msg_imp['source_task']} to Task {msg_imp['dest_task']} (ID: {msg_imp['message_id']}):")
                
                for imp in msg_imp['improvements']:
                    service_name = imp['service'].capitalize()
                    print(f"  - {service_name}: Level {imp['from_level']} → Level {imp['to_level']} " +
                        f"(Cost: {imp['time_cost']:.2f}, Benefit: {imp['security_benefit']:.4f})")
        
        print("========================================")
    def identify_critical_path_tasks(self):
        """Identify tasks on the critical path"""
        critical_path = set()
        current_task = None
        
        # Find the task that finishes last
        max_finish_time = 0
        for task in self.tasks:
            if task.is_scheduled and task.finish_time > max_finish_time:
                max_finish_time = task.finish_time
                current_task = task
        
        # Trace backwards from the last task to find the critical path
        while current_task:
            critical_path.add(current_task.task_id)
            
            # Find the predecessor that contributes to the critical path
            critical_pred = None
            critical_start = current_task.start_time
            
            for pred_id in current_task.predecessors:
                pred_task = self.get_task_by_id(pred_id)
                if not pred_task or not pred_task.is_scheduled:
                    continue
                
                # Calculate when this predecessor's output is available
                if pred_task.assigned_processor == current_task.assigned_processor:
                    # Same processor - no communication time
                    pred_finish = pred_task.finish_time
                else:
                    # Different processor - account for communication time
                    message = self.get_message(pred_id, current_task.task_id)
                    if message:
                        comm_time = self.calculate_communication_time(
                            message, pred_task.assigned_processor, current_task.assigned_processor)
                        pred_finish = pred_task.finish_time + comm_time
                    else:
                        pred_finish = pred_task.finish_time
                
                # If this predecessor's finish time matches current task's start time
                # (with small tolerance), it's on the critical path
                if abs(pred_finish - critical_start) < 0.001:
                    critical_pred = pred_task
                    break
            
            current_task = critical_pred
        
        return critical_path
    
    def schedule_tasks(self):
        """Modified task scheduling to prioritize makespan while still improving security"""
        # First compute task priorities before scheduling
        self.compute_task_priorities()
        
        # Run SHIELD baseline first to get a good starting point
        shield_makespan = self.run_shield_baseline()
        shield_schedule = []
        shield_tasks_state = copy.deepcopy(self.tasks)
        
        # Store SHIELD security protocols
        shield_security_assignments = {}
        for message in self.messages:
            shield_security_assignments[message.id] = copy.deepcopy(message.assigned_security)
        
        for task in self.tasks:
            if task.is_scheduled:
                entry = {
                    'task_id': task.task_id,
                    'name': task.name,
                    'processor': task.assigned_processor,
                    'start_time': task.start_time,
                    'finish_time': task.finish_time
                }
                shield_schedule.append(entry)
        
        # Set the SHIELD solution as initial best
        self.best_makespan = shield_makespan
        self.best_schedule = shield_schedule
        self.best_tasks_state = shield_tasks_state
        self.best_security_utility = self.calculate_security_utility()
        self.best_security_assignments = copy.deepcopy(shield_security_assignments)
        
        print(f"SHIELD baseline: Makespan={shield_makespan}, Security Utility={self.best_security_utility:.2f}")
        
        # Start Q-learning episodes
        for episode in range(self.episodes):
            # Add a safety counter to prevent infinite loops
            safety_counter = 0
            max_iterations = 1000
            
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
            
            # More realistic progression of exploration parameters
            if episode < self.episodes * 0.2:
                # Early episodes: focus on finding good schedules
                self.security_epsilon = 0.1
                self.security_weight = 0.2
                self.makespan_weight = 0.8
            elif episode < self.episodes * 0.6:
                # Middle episodes: balance exploration
                self.security_epsilon = 0.2
                self.security_weight = 0.3
                self.makespan_weight = 0.7
            else:
                # Later episodes: slight increase in security focus
                self.security_epsilon = 0.25
                self.security_weight = 0.35
                self.makespan_weight = 0.65
            
            # Optimize security assignments for this episode
            self.optimize_security_assignment()
            
            # Sort tasks by priority (higher priority first)
            sorted_tasks = sorted(self.tasks, key=lambda x: -x.priority)
            
            # Create a time-based scheduling approach for better parallelism
            ready_tasks = [task for task in sorted_tasks 
                        if not task.predecessors or 
                        all(self.get_task_by_id(pred_id).is_scheduled for pred_id in task.predecessors)]
            
            current_time = 0
            all_scheduled = False
            
            # Main scheduling loop
            while not all_scheduled and safety_counter < max_iterations:
                safety_counter += 1
                
                # Find all tasks that are ready at current_time
                for task in sorted_tasks:
                    if not task.is_scheduled and task not in ready_tasks:
                        # Check if all predecessors are scheduled
                        all_preds_scheduled = all(self.get_task_by_id(pred_id).is_scheduled 
                                                for pred_id in task.predecessors)
                        
                        # Only check finish times if all predecessors are scheduled
                        if all_preds_scheduled:
                            # Get the finish times of all predecessors
                            pred_finish_times = [self.get_task_by_id(pred_id).finish_time 
                                            for pred_id in task.predecessors]
                            
                            # Check if all predecessors have finished by current_time
                            if all(finish_time <= current_time for finish_time in pred_finish_times):
                                ready_tasks.append(task)
                
                if ready_tasks:
                    # Sort ready tasks by priority
                    ready_tasks.sort(key=lambda x: -x.priority)
                    
                    # Select the highest priority task
                    current_task = ready_tasks.pop(0)
                    
                    # Choose processor using our enhanced decision making
                    proc_id = self.choose_processor(current_task)
                    processor = self.processors[proc_id - 1]
                    
                    # Calculate EST and EFT, ensuring we don't start before current_time
                    est = max(current_time, self.calculate_est(current_task, proc_id))
                    eft = est + current_task.execution_times[proc_id - 1]
                    
                    # Schedule the task
                    current_task.assigned_processor = proc_id
                    current_task.start_time = est
                    current_task.finish_time = eft
                    current_task.is_scheduled = True
                    processor.available_time = eft
                    processor.task_history = getattr(processor, 'task_history', [])
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
                    
                    # Calculate reward for this scheduling decision
                    reward = self.calculate_improved_reward(current_task, proc_id)
                    
                    # Find next task that might become ready
                    next_task = None
                    for task in sorted_tasks:
                        if not task.is_scheduled and task not in ready_tasks:
                            if all(self.get_task_by_id(pred_id).is_scheduled for pred_id in task.predecessors):
                                next_task = task
                                break
                    
                    # Update Q-value
                    if next_task:
                        self.update_q_value(current_task.task_id, proc_id, reward, next_task.task_id)
                    else:
                        self.update_q_value(current_task.task_id, proc_id, reward)
                    
                else:
                    # No tasks ready at current time, advance time
                    next_times = []
                    for proc in self.processors:
                        if proc.available_time > current_time:
                            next_times.append(proc.available_time)
                    
                    # Also consider when tasks might become ready
                    for task in sorted_tasks:
                        if not task.is_scheduled and task not in ready_tasks:
                            if all(self.get_task_by_id(pred_id).is_scheduled for pred_id in task.predecessors):
                                max_finish = max(self.get_task_by_id(pred_id).finish_time 
                                               for pred_id in task.predecessors)
                                if max_finish > current_time:
                                    next_times.append(max_finish)
                    
                    if next_times:
                        current_time = min(next_times)
                    else:
                        current_time += 1
                
                # Check if all tasks are scheduled
                all_scheduled = all(task.is_scheduled for task in self.tasks)
            
            # Skip if we hit safety limit
            if safety_counter >= max_iterations:
                print(f"Warning: Episode {episode} hit maximum iterations. Skipping this episode.")
                continue
            
            # Calculate makespan and security utility
            current_makespan = max(task.finish_time for task in self.tasks)
            current_security_utility = self.calculate_security_utility()
            
            # Save security assignments
            current_security_assignments = {}
            for message in self.messages:
                current_security_assignments[message.id] = copy.deepcopy(message.assigned_security)
            
            # Stricter decision logic - prioritize makespan first
            update_solution = False
            update_reason = ""
            
            # Case 1: Better makespan (highest priority)
            if current_makespan < self.best_makespan:
                update_solution = True
                update_reason = "better makespan"
            
            # Case 2: Same makespan but better security
            elif abs(current_makespan - self.best_makespan) < 0.001 and current_security_utility > self.best_security_utility:
                update_solution = True
                update_reason = "same makespan with better security"
            
            # Case 3: Slightly worse makespan but significantly better security, and within deadline
            elif current_makespan <= min(self.best_makespan * 1.02, self.deadline) and current_security_utility > self.best_security_utility * 1.15:
                update_solution = True
                update_reason = "significantly better security with acceptable makespan"
            
            # Update best solution if needed
            if update_solution:
                self.best_makespan = current_makespan
                self.best_schedule = current_schedule
                self.best_tasks_state = copy.deepcopy(self.tasks)
                self.best_security_utility = current_security_utility
                self.best_security_assignments = current_security_assignments
                print(f"Episode {episode}: New best solution - {update_reason}")
                print(f"  Makespan: {current_makespan}, Security: {current_security_utility:.2f}")
            
            # Gradually reduce exploration rate
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            # Every 50 episodes, reinforce best solution
            if episode % 50 == 0 and episode > 0:
                self.reinforce_best_solution()
        
        # Set the final schedule with best found solution
        self.schedule = self.best_schedule
        
        # Restore best task state
        self.tasks = self.best_tasks_state
        
        # Restore best security assignments
        for message in self.messages:
            if message.id in self.best_security_assignments:
                message.assigned_security = copy.deepcopy(self.best_security_assignments[message.id])
        
        print(f"Final best solution: Makespan={self.best_makespan}, Security Utility={self.best_security_utility:.2f}")
        
        # After all episodes, enhance security with remaining slack time
        self.enhance_security_with_slack()
        
        return self.best_makespan

class UnifiedQLearningScheduler(Scheduler):
    def __init__(self, tasks, messages, processors, network, security_service, deadline, seed=42):
        super().__init__(tasks, messages, processors, network, security_service, deadline)
        random.seed(seed)
        np.random.seed(seed)

        # Learning parameters
        self.q_table = {}
        self.alpha = 0.25
        self.gamma = 0.9
        self.epsilon = 0.8
        self.min_epsilon = 0.05
        self.epsilon_decay = 0.97
        self.episodes = 600

        # Optimization weights
        self.security_weight = 0.3
        self.makespan_weight = 0.7
        self.security_epsilon = 0.2

        # Best solution tracking
        self.best_makespan = float('inf')
        self.best_schedule = None
        self.best_security_utility = 0

        self.initialize_q_table()

    def initialize_q_table(self):
        for task in self.tasks:
            for proc_id in range(1, len(self.processors) + 1):
                self.q_table[(task.task_id, proc_id)] = 0.0

    def update_q_value(self, task_id, proc_id, reward, next_task_id=None):
        current_q = self.q_table.get((task_id, proc_id), 0)
        if next_task_id:
            next_max_q = max(self.q_table.get((next_task_id, p), 0) for p in range(1, len(self.processors) + 1))
            self.q_table[(task_id, proc_id)] = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * next_max_q)
        else:
            self.q_table[(task_id, proc_id)] = (1 - self.alpha) * current_q + self.alpha * reward

    def choose_processor(self, task):
        if random.random() < self.epsilon:
            return random.choice(range(1, len(self.processors) + 1))
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
        best_processor = None
        best_value = float('-inf')
        for p in range(1, len(self.processors) + 1):
            q_value = self.q_table.get((task.task_id, p), 0)
            est = self.calculate_est(task, p)
            eft = est + task.execution_times[p - 1]
            makespan_impact = -eft
            security_impact = self.estimate_security_impact(task, p)
            value = self.makespan_weight * makespan_impact + self.security_weight * security_impact + q_value
            if value > best_value:
                best_value = value
                best_processor = p
        return best_processor

    def estimate_security_impact(self, task, proc_id):
        impact = 0
        for pred_id in task.predecessors:
            pred_task = self.get_task_by_id(pred_id)
            if pred_task and pred_task.is_scheduled and pred_task.assigned_processor != proc_id:
                message = self.get_message(pred_id, task.task_id)
                if message:
                    for service in ['confidentiality', 'integrity', 'authentication']:
                        idx = message.assigned_security[service]
                        strength = self.security.strengths[service][idx]
                        weight = message.weights[service]
                        impact += weight * strength / (1 + message.size / 50)
        return impact

    def calculate_reward(self, task, proc_id):
        slack = self.deadline - task.finish_time
        proc_loads = [p.available_time for p in self.processors]
        avg_load = sum(proc_loads) / len(proc_loads)
        load_score = -abs(task.finish_time - avg_load) / 100

        locality_bonus = sum(15 for pred_id in task.predecessors
                             if self.get_task_by_id(pred_id).assigned_processor == proc_id)

        parallel_bonus = sum(25 for other in self.tasks
                             if other.is_scheduled and other.task_id != task.task_id and
                             other.assigned_processor != proc_id and
                             other.start_time <= task.finish_time and other.finish_time >= task.start_time)

        sec_overhead = sum(self.calc_security_overhead(self.get_message(pred_id, task.task_id),
                                                           self.get_task_by_id(pred_id).assigned_processor,
                                                           proc_id)
                           for pred_id in task.predecessors
                           if self.get_message(pred_id, task.task_id))

        crit_bonus = 25 if task.priority > np.mean([t.priority for t in self.tasks]) else 0

        max_pred_finish = max((self.get_task_by_id(pred_id).finish_time for pred_id in task.predecessors), default=0)
        immediate_bonus = 50 if task.start_time <= max_pred_finish + 5 else 0

        reward = ((slack > 0) * 50 + min(0, slack) * 0.5 + load_score + locality_bonus +
                  parallel_bonus + crit_bonus + immediate_bonus - sec_overhead * 0.1)
        return reward

    def schedule_tasks(self):
        self.compute_task_priorities()
        for episode in range(self.episodes):
            for proc in self.processors:
                proc.available_time = 0
            for task in self.tasks:
                task.is_scheduled = False
                task.assigned_processor = None

            current_schedule = []
            self.optimize_security_assignment()
            sorted_tasks = sorted(self.tasks, key=lambda t: -t.priority)
            ready = [t for t in sorted_tasks if not t.predecessors]
            current_time = 0

            while not all(t.is_scheduled for t in self.tasks):
                for task in sorted_tasks:
                    if not task.is_scheduled and task not in ready:
                        if all(self.get_task_by_id(pid).is_scheduled for pid in task.predecessors):
                            if all(self.get_task_by_id(pid).finish_time <= current_time for pid in task.predecessors):
                                ready.append(task)

                if ready:
                    ready.sort(key=lambda t: -t.priority)
                    task = ready.pop(0)
                    proc_id = self.choose_processor(task)
                    est = max(current_time, self.calculate_est(task, proc_id))
                    eft = est + task.execution_times[proc_id - 1]

                    task.assigned_processor = proc_id
                    task.start_time = est
                    task.finish_time = eft
                    task.is_scheduled = True
                    self.processors[proc_id - 1].available_time = eft

                    reward = self.calculate_reward(task, proc_id)
                    next_task = next((t for t in sorted_tasks if not t.is_scheduled), None)
                    self.update_q_value(task.task_id, proc_id, reward, next_task.task_id if next_task else None)

                    current_schedule.append({
                        'task_id': task.task_id,
                        'name': task.name,
                        'processor': proc_id,
                        'start_time': task.start_time,
                        'finish_time': task.finish_time
                    })
                else:
                    next_times = [p.available_time for p in self.processors if p.available_time > current_time]
                    pred_ready_times = [
                        max(self.get_task_by_id(pid).finish_time for pid in t.predecessors)
                        for t in sorted_tasks if not t.is_scheduled and all(self.get_task_by_id(pid).is_scheduled for pid in t.predecessors)
                    ]
                    all_times = next_times + pred_ready_times
                    current_time = min(all_times) if all_times else current_time + 1

            makespan = max(t.finish_time for t in self.tasks)
            security_utility = self.calculate_security_utility()
            if makespan < self.best_makespan or (makespan == self.best_makespan and security_utility > self.best_security_utility):
                self.best_makespan = makespan
                self.best_schedule = current_schedule
                self.best_security_utility = security_utility

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        self.schedule = self.best_schedule
        self.enhance_security_with_slack()
        return self.best_makespan

    def run(self):
        print("Running UnifiedQLearningScheduler...")
        makespan = self.schedule_tasks()
        print(f"Final Makespan: {makespan}, Security Utility: {self.best_security_utility:.2f}")
        return makespan, self.best_security_utility

    # Reuse calculate_est, calc_security_overhead, calculate_security_utility, and enhance_security_with_slack from your base class
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
    
    # Run Original Q-Learning Scheduler
    print("\nRunning Original Q-Learning Scheduler...")
    tasks_q, messages_q, processors_q, network, security_service, deadline = create_tc_test_case()
    
    # Add the optimize_security_assignment method to Scheduler class
    Scheduler.optimize_security_assignment = ImprovedQLearningScheduler.optimize_security_assignment
    q_scheduler = ImprovedQLearningScheduler(tasks_q, messages_q, processors_q, network, security_service, deadline)
    q_makespan, q_security_utility = q_scheduler.run()
    
    
    if q_makespan:
        q_fig = plot_gantt_chart("ImprovedQLearningScheduler", q_scheduler.schedule, len(processors_q), q_makespan)
        print(f"ImprovedQLearningScheduler Security Utility: {q_security_utility:.2f}")
    
    # Run Security Enhanced Q-Learning Scheduler
    print("\nRunning Security Enhanced Q-Learning Scheduler...")
    tasks_sq, messages_sq, processors_sq, network, security_service, deadline = create_tc_test_case()
    
    # Run the enhanced scheduler
    sq_scheduler = SecurityEnhancedQLearningScheduler(tasks_sq, messages_sq, processors_sq, network, security_service, deadline)
    sq_makespan, sq_security_utility = sq_scheduler.run()
    

    
    if sq_makespan:
        sq_fig = plot_gantt_chart("SecurityEnhancedQLearningScheduler", sq_scheduler.schedule, len(processors_sq), sq_makespan)
        print(f"SecurityEnhancedQLearningScheduler Security Utility: {sq_security_utility:.2f}")
    cq_scheduler = UnifiedQLearningScheduler(tasks_sq, messages_sq, processors_sq, network, security_service, deadline)
    cq_makespan, sq_security_utility = cq_scheduler.run()
    # Compare results
    if cq_makespan:
        cq_fig = plot_gantt_chart("SecurityEnhancedQLearningScheduler", sq_scheduler.schedule, len(processors_sq), sq_makespan)
        print(f"SecurityEnhancedQLearningScheduler Security Utility: {sq_security_utility:.2f}")
    print("\nComparison of Results:")
    print(f"{'Scheduler':<35} {'Makespan (ms)':<15} {'Security Utility':<15} {'Security Improvement':<20}")
    print("-" * 85)
    if hsms_makespan:
        print(f"{'HSMS':<35} {hsms_makespan:<15.2f} {hsms_security_utility:<15.2f} {'Baseline':<20}")
    if shield_makespan:
        shield_improvement = ((shield_security_utility - hsms_security_utility) / hsms_security_utility * 100) if hsms_security_utility > 0 else 0
        print(f"{'SHIELD':<35} {shield_makespan:<15.2f} {shield_security_utility:<15.2f} {shield_improvement:+.2f}%")
    if q_makespan:
        q_improvement = ((q_security_utility - hsms_security_utility) / hsms_security_utility * 100) if hsms_security_utility > 0 else 0
        print(f"{'ImprovedQLearningScheduler':<35} {q_makespan:<15.2f} {q_security_utility:<15.2f} {q_improvement:+.2f}%")
    if sq_makespan:
        sq_improvement = ((sq_security_utility - hsms_security_utility) / hsms_security_utility * 100) if hsms_security_utility > 0 else 0
        sq_vs_original = ((sq_security_utility - q_security_utility) / q_security_utility * 100) if q_security_utility > 0 else 0
        print(f"{'SecurityEnhancedQLearningScheduler':<35} {sq_makespan:<15.2f} {sq_security_utility:<15.2f} {sq_improvement:+.2f}%")
        print(f"{'vs Original Q-Learning':<35} {sq_vs_original:+.2f}%")
    plt.show()

if __name__ == "__main__":
    main()