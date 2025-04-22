import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import time
from LASTRESORT import Task, Message, Processor, SecurityService, CommunicationNetwork, Scheduler, HSMS, SHIELD

class CombinedSecurityQLearningScheduler(Scheduler):
    def __init__(self, tasks, messages, processors, network, security_service, deadline, seed=42):
        super().__init__(tasks, messages, processors, network, security_service, deadline)
        random.seed(seed)
        np.random.seed(seed)
        # Q-learning parameters
        self.q_table = {}
        self.alpha = 0.25  # Balanced learning rate
        self.gamma = 0.9   # Discount factor
        self.epsilon = 0.8  # Initial exploration rate
        self.min_epsilon = 0.05  # Minimum exploration rate
        self.epsilon_decay = 0.97  # Decay rate
        self.episodes = 600  # Number of episodes
        # Balance weights
        self.security_weight = 0.3
        self.makespan_weight = 0.7
        self.security_epsilon = 0.2
        # Best solutions tracking
        self.best_makespan = float('inf')
        self.best_schedule = None
        self.best_security_utility = 0
        self.best_tasks_state = None
        self.best_security_assignments = {}
        self.security_service = security_service
        # Initialize Q-table
        self.initialize_q_table()
        
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
        """Choose a processor using an epsilon-greedy strategy with balanced objective function."""
        if random.random() < self.epsilon:
            return random.choice(range(1, len(self.processors) + 1))
        else:
            # Find processor that maximizes our objective function
            best_processor = None
            best_value = float('-inf')
            
            for p in range(1, len(self.processors) + 1):
                # Get Q-value for this processor
                q_value = self.q_table.get((task.task_id, p), 0)
                
                # Calculate makespan impact
                processor = self.processors[p - 1]
                est = self.calculate_est(task, p)
                eft = est + task.execution_times[p - 1]
                
                # Penalize longer makespans
                makespan_impact = -eft  # Negative because lower is better
                
                # Adjust weights based on critical path
                if task.priority > sum(t.priority for t in self.tasks) / len(self.tasks) * 1.2:
                    makespan_weight = self.makespan_weight * 1.2  # Boost makespan importance
                    security_weight = self.security_weight * 0.8   # Reduce security importance
                else:
                    makespan_weight = self.makespan_weight
                    security_weight = self.security_weight
                
                # Calculate security impact
                security_impact = self.estimate_security_impact(task, p)
                
                # Combined value with appropriate weights
                combined_value = (
                    makespan_weight * makespan_impact +
                    security_weight * security_impact +
                    q_value
                )
                
                if combined_value > best_value:
                    best_value = combined_value
                    best_processor = p
            
            return best_processor
            
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
                security_overhead += self.security_service.overheads[service][protocol_idx + 1][source_proc - 1]
            else:  # authentication
                security_overhead += self.security_service.overheads[service][protocol_idx + 1][dest_proc - 1]
                
        return base_comm_time + (security_overhead * message.size / 100)
    
    # def calculate_improved_reward(self, task, processor_id):
    #     """Enhanced reward function balancing makespan, parallelism, and security."""
    #     # Calculate slack time (how much before deadline)
    #     slack = self.deadline - task.finish_time
        
    #     # Calculate processor load balance
    #     proc_loads = [p.available_time for p in self.processors]
    #     avg_load = sum(proc_loads) / len(proc_loads)
    #     load_balance_score = -abs(task.finish_time - avg_load) / 100
        
    #     # Calculate dependency locality bonus
    #     locality_bonus = 0
    #     for pred_id in task.predecessors:
    #         pred_task = self.get_task_by_id(pred_id)
    #         if pred_task and pred_task.assigned_processor == processor_id:
    #             locality_bonus += 15  # Reward keeping dependent tasks on same processor
                
    #     # Enhanced parallelism bonus
    #     parallel_bonus = 0
    #     for other_task in self.tasks:
    #         if other_task.is_scheduled and other_task.task_id != task.task_id:
    #             # If tasks can run in parallel on different processors, reward it
    #             if (other_task.finish_time >= task.start_time and
    #                 other_task.start_time <= task.finish_time and
    #                 other_task.assigned_processor != processor_id):
    #                 parallel_bonus += 25
                    
    #     # Calculate security overhead
    #     sec_overhead = 0
    #     for pred_id in task.predecessors:
    #         pred_task = self.get_task_by_id(pred_id)
    #         if pred_task and pred_task.assigned_processor != processor_id:
    #             message = self.get_message(pred_id, task.task_id)
    #             if message:
    #                 sec_overhead += self.calculate_security_overhead(message, pred_task.assigned_processor, processor_id)
                    
    #     # Calculate critical path bonus
    #     critical_path_bonus = 0
    #     if task.priority > sum(t.priority for t in self.tasks) / len(self.tasks):
    #         # Task is on critical path, reward finishing it early
    #         critical_path_bonus = 25
            
    #     # Immediate start bonus - reward starting tasks with no delay
    #     immediate_start_bonus = 0
    #     if task.predecessors:
    #         max_pred_finish = max(self.get_task_by_id(pred_id).finish_time for pred_id in task.predecessors 
    #                             if self.get_task_by_id(pred_id).is_scheduled)
    #         # If task starts soon after its predecessor finishes, give a large bonus
    #         if task.start_time <= max_pred_finish + 5:  # Small tolerance for scheduling decisions
    #             immediate_start_bonus = 50
        
    #     # Calculate security utility for current state
    #     security_utility = self.calculate_partial_security_utility(task)
    #     scaled_security_reward = security_utility * 0.1
        
    #     # Combine all factors into final reward
    #     reward = (
    #         (slack > 0) * 50 +  # Binary reward for meeting deadline
    #         min(0, slack) * 0.5 +  # Small penalty for how much over deadline
    #         load_balance_score +  # Load balancing score
    #         locality_bonus +  # Reward for keeping related tasks together
    #         parallel_bonus +  # Reward for parallelism
    #         critical_path_bonus +  # Reward for critical path tasks
    #         immediate_start_bonus +  # Reward for starting tasks immediately after predecessors
    #         scaled_security_reward -  # Reward for security
    #         sec_overhead * 0.1  # Small penalty for security overhead
    #     )
        
    #     return reward
    
    # def calculate_security_overhead(self, message, source_proc, dest_proc):
    #     """Calculate total security overhead for a message between processors"""
    #     if source_proc == dest_proc:
    #         return 0  # No overhead for same processor
            
    #     overhead = 0
    #     for service in ['confidentiality', 'integrity', 'authentication']:
    #         protocol_idx = message.assigned_security[service]
    #         if service in ['confidentiality', 'integrity']:
    #             # Data-dependent overheads
    #             overhead += self.security_service.overheads[service][protocol_idx + 1][source_proc - 1]
    #         else:  # authentication
    #             # Data-independent overheads
    #             overhead += self.security_service.overheads[service][protocol_idx + 1][dest_proc - 1]
                
    #     return overhead * message.size / 100  # Scale by message size
    
    def calculate_partial_security_utility(self, task):
        """Calculate security utility for messages related to a specific task."""
        security_utility = 0
        
        # Find all messages connected to this task
        related_messages = []
        for message in self.messages:
            if message.source_id == task.task_id or message.dest_id == task.task_id:
                related_messages.append(message)
        
        # Calculate security utility for these messages
        for message in related_messages:
            for service in ['confidentiality', 'integrity', 'authentication']:
                protocol_idx = message.assigned_security[service]
                strength = self.security_service.strengths[service][protocol_idx]
                weight = message.weights[service]
                security_utility += weight * strength
                
        return security_utility
        
    # def calculate_security_utility(self):
    #     """Calculate the total security utility."""
    #     total_utility = 0
    #     for message in self.messages:
    #         message_utility = 0
    #         for service in ['confidentiality', 'integrity', 'authentication']:
    #             protocol_idx = message.assigned_security[service]
    #             strength = self.security_service.strengths[service][protocol_idx]
    #             weight = message.weights[service]
    #             message_utility += weight * strength
    #         total_utility += message_utility
    #     return total_utility
    
    def estimate_security_impact(self, task, processor_id):
        """Estimate the impact on security utility if task is assigned to processor"""
        security_impact = 0
        
        # Focus on immediate communication needs
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
                            strength = self.security_service.strengths[service][protocol_idx]
                            weight = message.weights[service]
                            
                            # Scale security impact by message size
                            security_impact += (weight * strength) / (1 + message.size / 50)
        
        return security_impact
    
    def optimize_security_assignment(self):
        """Improved security protocol selection that balances overhead and utility"""
        for message in self.messages:
            source_task = self.get_task_by_id(message.source_id)
            dest_task = self.get_task_by_id(message.dest_id)
            
            # Skip if tasks not available or on same processor
            if not source_task or not dest_task:
                # Fallback to minimum security levels that meet requirements
                for service in ['confidentiality', 'integrity', 'authentication']:
                    min_level = 0
                    for i, strength in enumerate(self.security_service.strengths[service]):
                        if strength >= message.min_security[service]:
                            min_level = i
                            break
                    message.assigned_security[service] = min_level
                continue
                
            if source_task.is_scheduled and dest_task.is_scheduled and \
               source_task.assigned_processor == dest_task.assigned_processor:
                # Set minimum security for intra-processor messages
                for service in ['confidentiality', 'integrity', 'authentication']:
                    min_level = 0
                    for i, strength in enumerate(self.security_service.strengths[service]):
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
                    available_protocols = range(len(self.security_service.strengths[service]))
                    valid_protocols = [p for p in available_protocols 
                                     if self.security_service.strengths[service][p] >= message.min_security[service]]
                    
                    if valid_protocols:
                        # Sort by strength in descending order
                        sorted_protocols = sorted(valid_protocols, 
                                               key=lambda p: self.security_service.strengths[service][p],
                                               reverse=True)
                        
                        # Choose from mid-range protocols
                        mid_start = len(sorted_protocols) // 4
                        mid_end = 3 * len(sorted_protocols) // 4
                        mid_range = sorted_protocols[mid_start:mid_end+1] if mid_start < mid_end else sorted_protocols
                        message.assigned_security[service] = random.choice(mid_range)
                else:
                    # Find balanced protocol
                    best_protocol = 0
                    best_efficiency = 0
                    
                    for protocol in range(len(self.security_service.strengths[service])):
                        if self.security_service.strengths[service][protocol] >= message.min_security[service]:
                            # Calculate overhead estimate
                            if source_task.is_scheduled and dest_task.is_scheduled:
                                source_proc = source_task.assigned_processor
                                dest_proc = dest_task.assigned_processor
                                
                                if service in ['confidentiality', 'integrity']:
                                    overhead = self.security_service.overheads[service][protocol + 1][source_proc - 1]
                                else:  # authentication
                                    overhead = self.security_service.overheads[service][protocol + 1][dest_proc - 1]
                            else:
                                # If tasks not scheduled yet, use average overhead
                                overhead = sum(self.security_service.overheads[service][protocol + 1]) / len(self.processors)
                            
                            # Get strength and calculate efficiency
                            strength = self.security_service.strengths[service][protocol]
                            weight = message.weights[service]
                            
                            # Calculate weighted efficiency (strength/overhead ratio)
                            efficiency = (strength * weight) / (overhead + 0.001)
                            
                            if efficiency > best_efficiency:
                                best_efficiency = efficiency
                                best_protocol = protocol
                    
                    message.assigned_security[service] = best_protocol
    
    def run_baselines(self):
        """Run both SHIELD and HSMS algorithms for comparison"""
        results = {}
        
        # Run SHIELD
        shield = SHIELD(copy.deepcopy(self.tasks), copy.deepcopy(self.messages), 
                        copy.deepcopy(self.processors), self.network, self.security_service, self.deadline)
        shield_makespan, shield_security = shield.run()
        results['SHIELD'] = {
            'makespan': shield_makespan,
            'security': shield_security,
            'tasks': copy.deepcopy(shield.tasks),
            'messages': copy.deepcopy(shield.messages)
        }
        print(f"SHIELD baseline: Makespan={shield_makespan}, Security Utility={shield_security:.2f}")
        
        # Run HSMS
        hsms = HSMS(copy.deepcopy(self.tasks), copy.deepcopy(self.messages), 
                   copy.deepcopy(self.processors), self.network, self.security_service, self.deadline)
        hsms_makespan, hsms_security = hsms.run()
        results['HSMS'] = {
            'makespan': hsms_makespan,
            'security': hsms_security,
            'tasks': copy.deepcopy(hsms.tasks),
            'messages': copy.deepcopy(hsms.messages)
        }
        print(f"HSMS baseline: Makespan={hsms_makespan}, Security Utility={hsms_security:.2f}")
        
        return results
    
    def run_shield_baseline(self):
        """Run SHIELD algorithm to get a baseline schedule for comparison"""
        # Create a new SHIELD instance with copies of the current state
        shield = SHIELD(copy.deepcopy(self.tasks), copy.deepcopy(self.messages), 
                       copy.deepcopy(self.processors), self.network, self.security_service, self.deadline)
        
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
        
    # def reinforce_best_solution(self):
    #     """Reinforce the best solution found so far in the Q-table"""
    #     for task in self.best_tasks_state:
    #         if task.is_scheduled:
    #             # Give a large positive reinforcement to the best choices
    #             self.q_table[(task.task_id, task.assigned_processor)] += 25
    
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
        
    # def enhance_security_with_slack(self):
    #     """More careful security enhancement with slack time"""
    #     # Calculate current makespan
    #     makespan = max(task.finish_time for task in self.tasks)
        
    #     # Calculate slack time with safety margin
    #     slack_time = (self.deadline - makespan) * 0.95  # Keep 5% safety margin
    #     if slack_time <= 0:
    #         print("No slack time available for security improvements")
    #         return  # No slack time available
        
    #     print(f"Enhancing security with {slack_time:.2f} slack time (with safety margin)")
        
    #     # Keep track of remaining slack
    #     remaining_slack = slack_time
        
    #     # Identify critical path tasks - avoid upgrading their security
    #     critical_path_tasks = self.identify_critical_path_tasks()
    #     print(f"Critical path tasks (avoiding security upgrades): {sorted(list(critical_path_tasks))}")
        
    #     # Keep track of security improvements
    #     security_improvements = []
        
    #     # Sort messages by security potential and criticality
    #     prioritized_messages = []
    #     for message in self.messages:
    #         source_task = self.get_task_by_id(message.source_id)
    #         dest_task = self.get_task_by_id(message.dest_id)
            
    #         if not source_task or not dest_task or not source_task.is_scheduled or not dest_task.is_scheduled:
    #             continue
            
    #         if source_task.assigned_processor == dest_task.assigned_processor:
    #             continue  # Skip intra-processor messages
            
    #         # Skip messages on critical path
    #         if source_task.task_id in critical_path_tasks and dest_task.task_id in critical_path_tasks:
    #             continue
            
    #         # Calculate security potential
    #         security_potential = self.get_security_efficiency(message)
    #         prioritized_messages.append((message, security_potential))
        
    #     # Sort by security potential (highest first)
    #     prioritized_messages.sort(key=lambda x: x[1], reverse=True)
        
    #     # Enhance security in order of priority
    #     for message, potential in prioritized_messages:
    #         source_task = self.get_task_by_id(message.source_id)
    #         dest_task = self.get_task_by_id(message.dest_id)
    #         source_proc = source_task.assigned_processor
    #         dest_proc = dest_task.assigned_processor
            
    #         message_improvements = []
            
    #         # Try to upgrade all services in order of weight (importance)
    #         for service in sorted(['confidentiality', 'integrity', 'authentication'], 
    #                             key=lambda s: message.weights[s], reverse=True):
    #             current_level = message.assigned_security[service]
                
    #             # Only try to upgrade one level at a time for more balanced improvements
    #             if current_level < len(self.security_service.strengths[service]) - 1:
    #                 next_level = current_level + 1
                    
    #                 # Calculate security benefit
    #                 current_strength = self.security_service.strengths[service][current_level]
    #                 next_strength = self.security_service.strengths[service][next_level]
    #                 strength_gain = next_strength - current_strength
    #                 security_benefit = message.weights[service] * strength_gain
                    
    #                 # Calculate time cost
    #                 if service in ['confidentiality', 'integrity']:
    #                     current_overhead = self.security_service.overheads[service][current_level + 1][source_proc - 1]
    #                     next_overhead = self.security_service.overheads[service][next_level + 1][source_proc - 1]
    #                 else:  # authentication
    #                     current_overhead = self.security_service.overheads[service][current_level + 1][dest_proc - 1]
    #                     next_overhead = self.security_service.overheads[service][next_level + 1][dest_proc - 1]
                    
    #                 time_cost = (next_overhead - current_overhead) * message.size / 100
                    
    #                 # If this upgrade fits in our remaining slack, apply it
    #                 if time_cost <= remaining_slack:
    #                     # Store the previous level for reporting
    #                     prev_level = message.assigned_security[service]
                        
    #                     # Apply the upgrade
    #                     message.assigned_security[service] = next_level
    #                     remaining_slack -= time_cost
                        
    #                     # Record this improvement
    #                     improvement = {
    #                         'service': service,
    #                         'from_level': prev_level,
    #                         'to_level': next_level,
    #                         'time_cost': time_cost,
    #                         'security_benefit': security_benefit
    #                     }
    #                     message_improvements.append(improvement)
            
    #         # If this message had any improvements, add to our tracking
    #         if message_improvements:
    #             security_improvements.append({
    #                 'message_id': message.id,
    #                 'source_task': source_task.task_id,
    #                 'dest_task': dest_task.task_id,
    #                 'improvements': message_improvements
    #             })
        
    #     # Recalculate security utility after enhancements
    #     self.best_security_utility = self.calculate_security_utility()
        
    #     # Print detailed summary of all security improvements
    #     print("\n===== SECURITY IMPROVEMENTS SUMMARY =====")
    #     if not security_improvements:
    #         print("No security improvements were made.")
    #     else:
    #         print(f"Total messages improved: {len(security_improvements)}")
    #         print(f"Final security utility: {self.best_security_utility:.4f}")
    #         print(f"Remaining slack time: {remaining_slack:.2f}")
    #         print("\nDetailed improvements:")
            
    #         for msg_imp in security_improvements:
    #             print(f"\nMessage from Task {msg_imp['source_task']} to Task {msg_imp['dest_task']} (ID: {msg_imp['message_id']}):")
                
    #             for imp in msg_imp['improvements']:
    #                 service_name = imp['service'].capitalize()
    #                 print(f"  - {service_name}: Level {imp['from_level']} → Level {imp['to_level']} " +
    #                     f"(Cost: {imp['time_cost']:.2f}, Benefit: {imp['security_benefit']:.4f})")
        
    #     print("========================================")
    
    def get_security_efficiency(self, message):
        """Calculate the efficiency of security protocols for a message"""
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
            strength = self.security_service.strengths[service][current_level]
            weight = message.weights[service]
            security_benefit = weight * strength
            
            # Calculate overhead of current level
            if service in ['confidentiality', 'integrity']:
                overhead = self.security_service.overheads[service][current_level + 1][source_proc - 1]
            else:  # authentication
                overhead = self.security_service.overheads[service][current_level + 1][dest_proc - 1]
            
            time_cost = overhead * message.size / 100
            
            # Add small constant to avoid division by zero
            service_efficiency = security_benefit / (time_cost + 0.001)
            
            # Higher weights indicate more room for improvement
            potential_improvement = 1
            if current_level < len(self.security_service.strengths[service]) - 1:
                # Calculate potential improvement if upgraded
                next_level = current_level + 1
                next_strength = self.security_service.strengths[service][next_level]
                potential_strength_gain = next_strength - strength
                potential_improvement = potential_strength_gain / (strength + 0.001)
            
            # Combine current efficiency with potential for improvement
            efficiency += service_efficiency * potential_improvement
        
        return efficiency

    # def schedule_tasks(self):
    #     """Schedule tasks using combined Q-learning approach with adaptive parameters"""
    #     # First compute task priorities before scheduling
    #     self.compute_task_priorities()
        
    #     # Run baseline algorithms for comparison
    #     baseline_results = self.run_baselines()
        
    #     # Use SHIELD as initial best solution
    #     shield_makespan = self.run_shield_baseline()
    #     shield_schedule = []
    #     shield_tasks_state = copy.deepcopy(self.tasks)
        
    #     # Store SHIELD security protocols
    #     shield_security_assignments = {}
    #     for message in self.messages:
    #         # Store SHIELD security protocols
    #         shield_security_assignments = {}    
    #         for message in self.messages:
    #             shield_security_assignments[message.id] = copy.deepcopy(message.assigned_security)
                
    #     # Calculate security utility of SHIELD solution
    #         shield_security_utility = self.calculate_security_utility()
                
    #         # Set SHIELD as initial best solution
    #         self.best_makespan = shield_makespan
    #         self.best_security_utility = shield_security_utility
    #         self.best_schedule = shield_schedule
    #         self.best_tasks_state = shield_tasks_state
    #         self.best_security_assignments = shield_security_assignments
                
    #         print(f"Initial SHIELD solution: Makespan={shield_makespan}, Security Utility={shield_security_utility:.2f}")
                
    #         # Reset processor states
    #         for processor in self.processors:
    #             processor.available_time = 0
                
    #         # Reset task states
    #         for task in self.tasks:
    #             task.is_scheduled = False
    #             task.assigned_processor = None
    #             task.start_time = 0
    #             task.finish_time = 0
                
    #         # Main Q-learning loop
    #         for episode in range(1, self.episodes + 1):
    #             # Reset processor states for this episode
    #             for processor in self.processors:
    #                 processor.available_time = 0
                    
    #             # Reset task states for this episode
    #             for task in self.tasks:
    #                 task.is_scheduled = False
    #                 task.assigned_processor = None
    #                 task.start_time = 0
    #                 task.finish_time = 0
                
    #             # For detailed logging
    #             if episode % 100 == 0:
    #                 print(f"Episode {episode}/{self.episodes}, Epsilon: {self.epsilon:.3f}")
                    
    #             # Initially assign security protocols
    #             self.optimize_security_assignment()
                    
    #             # Get ready tasks (tasks with all predecessors scheduled)
    #             ready_tasks = self.get_ready_tasks()
                
    #             # Schedule tasks using Q-learning
    #             tasks_scheduled = 0
    #             while ready_tasks:
    #                 # Sort ready tasks by priority (highest first)
    #                 ready_tasks.sort(key=lambda t: t.priority, reverse=True)
                    
    #                 # Schedule highest priority task
    #                 current_task = ready_tasks.pop(0)
                    
    #                 # Choose processor using Q-learning
    #                 processor_id = self.choose_processor(current_task)
                    
    #                 # Schedule task on chosen processor
    #                 processor = self.processors[processor_id - 1]
    #                 est = self.calculate_est(current_task, processor_id)
    #                 eft = est + current_task.execution_times[processor_id - 1]
                    
    #                 # Update task state
    #                 current_task.assigned_processor = processor_id
    #                 current_task.start_time = est
    #                 current_task.finish_time = eft
    #                 current_task.is_scheduled = True
                    
    #                 # Update processor availability
    #                 processor.available_time = eft
                    
    #                 # Calculate reward for this action
    #                 reward = self.calculate_improved_reward(current_task, processor_id)
                    
    #                 # Update Q-value for this state-action pair
    #                 self.update_q_value(current_task.task_id, processor_id, reward)
                    
    #                 # Re-optimize security protocols
    #                 self.optimize_security_assignment()
                    
    #                 # Update ready tasks
    #                 ready_tasks = self.get_ready_tasks()
                    
    #                 tasks_scheduled += 1
                
    #             # All tasks scheduled - evaluate makespan and security utility
    #             makespan = max(task.finish_time for task in self.tasks)
    #             security_utility = self.calculate_security_utility()
                
    #             # Calculate combined score with appropriate weighting
    #             combined_score = self.makespan_weight * (-makespan) + self.security_weight * security_utility
                
    #             # Update best solution if current solution is better
    #             if makespan <= self.deadline and (combined_score > 
    #                 (self.makespan_weight * (-self.best_makespan) + self.security_weight * self.best_security_utility)):
    #                 self.best_makespan = makespan
    #                 self.best_security_utility = security_utility
    #                 self.best_tasks_state = copy.deepcopy(self.tasks)
                    
    #                 # Store best security assignments
    #                 self.best_security_assignments = {}
    #                 for message in self.messages:
    #                     self.best_security_assignments[message.id] = copy.deepcopy(message.assigned_security)
                        
    #                 # Special reinforcement for this good solution
    #                 self.reinforce_best_solution()
                    
    #                 print(f"New best solution found in episode {episode}:")
    #                 print(f"  Makespan: {makespan} (deadline: {self.deadline})")
    #                 print(f"  Security Utility: {security_utility:.2f}")
                
    #             # Decay epsilon for exploration-exploitation balance
    #             self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
                
    #             # Adaptive learning rate adjustment
    #             if episode % 100 == 0:
    #                 # Reduce learning rate slowly
    #                 self.alpha = max(0.05, self.alpha * 0.95)
                    
    #                 # Adjust security epsilon based on progress
    #                 progress = episode / self.episodes
    #                 if progress > 0.5:
    #                     self.security_epsilon = max(0.05, self.security_epsilon * 0.9)
                
    #             # If we've been improving for a while, slightly boost security weight
    #             if episode > self.episodes * 0.5 and security_utility > self.best_security_utility * 0.8:
    #                 if self.security_weight < 0.5:  # Ensure we don't overweight security
    #                     self.security_weight = min(0.5, self.security_weight * 1.05)
    #                     self.makespan_weight = 1 - self.security_weight
                
    #             # End early if we've found a good solution and have done enough exploration
    #             if episode > self.episodes * 0.9 and self.best_makespan < self.deadline * 0.9:
    #                 print(f"Ending early at episode {episode} with good solution")
    #                 break
                
    #     # After Q-learning, restore best solution found
    #         if self.best_tasks_state:
    #             print("\n===== FINAL SOLUTION =====")
    #             print(f"Best makespan: {self.best_makespan} (deadline: {self.deadline})")
    #             print(f"Best security utility: {self.best_security_utility:.2f}")
                
    #             # Restore best task schedule
    #             for task in self.tasks:
    #                 best_task = next((t for t in self.best_tasks_state if t.task_id == task.task_id), None)
    #                 if best_task:
    #                     task.assigned_processor = best_task.assigned_processor
    #                     task.start_time = best_task.start_time
    #                     task.finish_time = best_task.finish_time
    #                     task.is_scheduled = best_task.is_scheduled
                
    #             # Restore best security assignments
    #             for message in self.messages:
    #                 if message.id in self.best_security_assignments:
    #                     message.assigned_security = copy.deepcopy(self.best_security_assignments[message.id])
                
    #             # Update processor available times
    #             for proc in self.processors:
    #                 proc_tasks = [task for task in self.tasks if task.is_scheduled and task.assigned_processor == proc.proc_id]
    #                 if proc_tasks:
    #                     proc.available_time = max(task.finish_time for task in proc_tasks)
    #                 else:
    #                     proc.available_time = 0
                
    #             # Try to enhance security with remaining slack time
    #             self.enhance_security_with_slack()
                
    #             # Compare with baseline methods
    #             print("\n===== COMPARISON WITH BASELINES =====")
    #             print(f"Our solution: Makespan={self.best_makespan}, Security Utility={self.best_security_utility:.2f}")
                
    #             for method in baseline_results:
    #                 makespan = baseline_results[method]['makespan']
    #                 security = baseline_results[method]['security']
    #                 makespan_imp = (makespan - self.best_makespan) / makespan * 100
    #                 security_imp = (self.best_security_utility - security) / security * 100
                    
    #                 print(f"{method}: Makespan={makespan}, Security Utility={security:.2f}")
    #                 print(f"  Improvement: Makespan {makespan_imp:.2f}%, Security {security_imp:.2f}%")
                
    #             # Print task schedule details
    #             self.print_schedule()
            
    #     return self.best_makespan, self.best_security_utility

    def schedule_tasks(self):
        """
        Combined task scheduling method that prioritizes makespan while 
        enhancing security, using improved Q-learning with SHIELD seeding
        """
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
            
            # Dynamic exploration parameters based on episode progression
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
            
            # Save security assignments
            current_security_assignments = {}
            for message in self.messages:
                current_security_assignments[message.id] = copy.deepcopy(message.assigned_security)
            
            # Enhanced decision logic for updating the best solution
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
            
            # Gradually reduce exploration rate according to schedule
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            # Every 50 episodes, incorporate best solution found so far into Q-table
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
                    parallel_bonus += 25  # Prioritize parallelism
        
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
        
        # Immediate start bonus - reward starting tasks with no delay
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
                    if current_level < len(self.security_service.strengths[service]) - 1:
                        # Can upgrade this service
                        next_level = current_level + 1
                        
                        # Calculate security benefit
                        current_strength = self.security_service.strengths[service][current_level]
                        next_strength = self.security_service.strengths[service][next_level]
                        strength_gain = next_strength - current_strength
                        security_benefit = message.weights[service] * strength_gain
                        
                        # Calculate time cost
                        if service in ['confidentiality', 'integrity']:
                            current_overhead = self.security_service.overheads[service][current_level + 1][source_proc - 1]
                            next_overhead = self.security_service.overheads[service][next_level + 1][source_proc - 1]
                        else:  # authentication
                            current_overhead = self.security_service.overheads[service][current_level + 1][dest_proc - 1]
                            next_overhead = self.security_service.overheads[service][next_level + 1][dest_proc - 1]
                        
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

    def calculate_security_overhead(self, message, source_proc, dest_proc):
        """Calculate total security overhead for a message between processors"""
        if source_proc == dest_proc:
            return 0  # No overhead for same processor
        
        overhead = 0
        for service in ['confidentiality', 'integrity', 'authentication']:
            protocol_idx = message.assigned_security[service]
            if service in ['confidentiality', 'integrity']:
                # Data-dependent overheads
                overhead += self.security_service.overheads[service][protocol_idx + 1][source_proc - 1]
            else:  # authentication
                # Data-independent overheads
                overhead += self.security_service.overheads[service][protocol_idx + 1][dest_proc - 1]
        
        return overhead * message.size / 100  # Scale by message size

    def calculate_security_utility(self):
        """Calculate the total security utility."""
        total_utility = 0
        for message in self.messages:
            message_utility = 0
            for service in ['confidentiality', 'integrity', 'authentication']:
                protocol_idx = message.assigned_security[service]
                strength = self.security_service.strengths[service][protocol_idx]
                weight = message.weights[service]
                message_utility += weight * strength
            total_utility += message_utility
        return total_utility

    def run(self):
        """Run the ImprovedQLearningScheduler"""
        start_time = time.time()
        makespan = self.schedule_tasks()
        security_utility = self.best_security_utility
        end_time = time.time()
        
        print(f"ImprovedQLearningScheduler: Makespan={makespan}, Security Utility={security_utility:.2f}")
        print(f"Execution time: {end_time - start_time:.4f} seconds")
        
        return makespan, security_utility
            

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

# def create_tc_test_case():
    # tasks = [
    #     Task(1, "Sensor Read", [100, 110]),
    #     Task(2, "Preprocessing", [200, 190], [1]),
    #     Task(3, "Feature Extraction", [250, 240], [2]),
    #     Task(4, "Decision", [150, 160], [3]),
    #     Task(5, "Actuator Control", [120, 110], [4])
    # ]
    # messages = [
    #     Message(1, 2, 50),
    #     Message(2, 3, 50),
    #     Message(3, 4, 50),
    #     Message(4, 5, 50)
    # ]
    # for msg in messages:
    #     msg.set_security_requirements(0.1, 0.1, 0.2, 0.2, 0.2, 0.2)
    # processors = [Processor(1), Processor(2)]
    # network = CommunicationNetwork(2)
    # security_service = SecurityService()
    # deadline = 1000
    # return tasks, messages, processors, network, security_service, deadline
    
    # MEETS THE DEADLINE 
    
    
    # tasks = [
    #     Task(1, "Input A", [180, 160]),
    #     Task(2, "Input B", [190, 170]),
    #     Task(3, "Fusion", [300, 310], [1, 2]),
    #     Task(4, "Analysis", [400, 420], [3]),
    #     Task(5, "Control Signal", [250, 260], [4])
    # ]
    # messages = [
    #     Message(1, 3, 100),
    #     Message(2, 3, 100),
    #     Message(3, 4, 120),
    #     Message(4, 5, 100)
    # ]
    # for msg in messages:
    #     msg.set_security_requirements(0.8, 0.9, 0.9, 0.5, 0.5, 0.6)
    # processors = [Processor(1), Processor(2)]
    # network = CommunicationNetwork(2)
    # security_service = SecurityService()
    # deadline = 1800
    # return tasks, messages, processors, network, security_service, deadline
    
    # HIGHER PARALLELLISM
    
    
    # tasks = [
    #     Task(1, "Input 1", [150, 140, 130]),
    #     Task(2, "Input 2", [160, 150, 140]),
    #     Task(3, "Input 3", [170, 160, 150]),
    #     Task(4, "Aggregate", [300, 280, 260], [1, 2, 3])
    # ]
    # messages = [
    #     Message(1, 4, 80),
    #     Message(2, 4, 80),
    #     Message(3, 4, 80)
    # ]
    # for msg in messages:
    #     msg.set_security_requirements(0.3, 0.3, 0.3, 0.2, 0.2, 0.2)
    # processors = [Processor(1), Processor(2)]
    # network = CommunicationNetwork(2)
    # security_service = SecurityService()
    # deadline = 1500
    # return tasks, messages, processors, network, security_service, deadline
    
    # BS SCHEDULE AS SEQUENTIAL
    
    # tasks = [
    #     Task(i, f"T{i}", [80 + i * 2, 90 + i * 2], [i - 1] if i > 1 else []) for i in range(1, 11)
    # ]
    # messages = [Message(i, i + 1, 150) for i in range(1, 10)]
    # for msg in messages:
    #     msg.set_security_requirements(0.5, 0.5, 0.5, 0.4, 0.4, 0.4)
    # processors = [Processor(1), Processor(2)]
    # network = CommunicationNetwork(2)
    # security_service = SecurityService()
    # deadline = 1600
    # return tasks, messages, processors, network, security_service, deadline
    
    # tasks = [
    #     Task(1, "Sensor A", [100, 110]),
    #     Task(2, "Sensor B", [120, 130]),
    #     Task(3, "Sensor C", [110, 115]),
    #     Task(4, "Fusion AB", [200, 180], [1, 2]),
    #     Task(5, "Fusion BC", [210, 190], [2, 3]),
    #     Task(6, "Final Decision", [300, 310], [4, 5])
    # ]
    
    # messages = [
    #     Message(1, 4, 80),  # 1 → 4
    #     Message(2, 4, 80),  # 2 → 4
    #     Message(2, 5, 80),  # 2 → 5
    #     Message(3, 5, 80),  # 3 → 5
    #     Message(4, 6, 100), # 4 → 6
    #     Message(5, 6, 100)  # 5 → 6
    # ]
    
    # # Moderate and diverse security requirements
    # for msg in messages:
    #     msg.set_security_requirements(0.4, 0.4, 0.5, 0.5, 0.5, 0.5)
    
    # processors = [Processor(1), Processor(2)]
    # network = CommunicationNetwork(2)
    # security_service = SecurityService()
    # deadline = 1500
    
    # return tasks, messages, processors, network, security_service, deadline\
        
    # tasks = [
    #     Task(1, "Sensor A", [100, 110]),
    #     Task(2, "Sensor B", [120, 130]),
    #     Task(3, "Sensor C", [110, 120]),
    #     Task(4, "Sensor D", [100, 105]),
    #     Task(5, "Fusion AB", [200, 180], [1, 2]),
    #     Task(6, "Fusion CD", [210, 190], [3, 4]),
    #     Task(7, "Mid Analysis", [220, 200], [5]),
    #     Task(8, "Mid Decision", [230, 210], [6]),
    #     Task(9, "Aggregation", [250, 240], [7, 8]),
    #     Task(10, "Final Control", [300, 310], [9])
    # ]
    
    # messages = [
    #     Message(1, 5, 60),   # Sensor A → Fusion AB
    #     Message(2, 5, 60),   # Sensor B → Fusion AB
    #     Message(3, 6, 60),   # Sensor C → Fusion CD
    #     Message(4, 6, 60),   # Sensor D → Fusion CD
    #     Message(5, 7, 80),   # Fusion AB → Mid Analysis
    #     Message(6, 8, 80),   # Fusion CD → Mid Decision
    #     Message(7, 9, 100),  # Mid Analysis → Aggregation
    #     Message(8, 9, 100),  # Mid Decision → Aggregation
    #     Message(9, 10, 120)  # Aggregation → Final Control
    # ]
    
    # # Assign unique and varied security requirements
    # security_settings = [
    #     (0.2, 0.3, 0.4, 0.4, 0.5, 0.6),
    #     (0.4, 0.2, 0.3, 0.5, 0.3, 0.4),
    #     (0.3, 0.4, 0.5, 0.6, 0.4, 0.3),
    #     (0.5, 0.3, 0.4, 0.4, 0.5, 0.5),
    #     (0.4, 0.5, 0.2, 0.3, 0.6, 0.2),
    #     (0.3, 0.4, 0.4, 0.4, 0.3, 0.5),
    #     (0.2, 0.2, 0.2, 0.6, 0.4, 0.3),
    #     (0.6, 0.5, 0.3, 0.3, 0.2, 0.4),
    #     (0.5, 0.6, 0.4, 0.5, 0.3, 0.5)
    # ]
    
    # for msg, reqs in zip(messages, security_settings):
    #     msg.set_security_requirements(*reqs)
    
    # processors = [Processor(1), Processor(2)]
    # network = CommunicationNetwork(2)
    # security_service = SecurityService()
    # deadline = 1300  # Somewhat tight to enforce smarter scheduling
    
    # return tasks, messages, processors, network, security_service, deadline


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
    Scheduler.optimize_security_assignment = CombinedSecurityQLearningScheduler.optimize_security_assignment
    q_scheduler = CombinedSecurityQLearningScheduler(tasks_q, messages_q, processors_q, network, security_service, deadline)
    q_makespan, q_security_utility = q_scheduler.run()
    
    if q_makespan:
        q_fig = plot_gantt_chart("ImprovedQLearningScheduler", q_scheduler.schedule, len(processors_q), q_makespan)
        print(f"ImprovedQLearningScheduler Security Utility: {q_security_utility:.2f}")
    
    # Run Security Enhanced Q-Learning Scheduler
    print("\nRunning Security Enhanced Q-Learning Scheduler...")
    tasks_sq, messages_sq, processors_sq, network, security_service, deadline = create_tc_test_case()
    
    # Run the enhanced scheduler

    # Compare results
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
    plt.show()

if __name__ == "__main__":
    main()