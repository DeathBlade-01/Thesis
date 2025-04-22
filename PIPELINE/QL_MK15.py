import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import time
from LASTRESORT import Task, Message, Processor, SecurityService, CommunicationNetwork, Scheduler, HSMS, SHIELD

class CombinedQLearningScheduler(Scheduler):
    def __init__(self, tasks, messages, processors, network, security_service, deadline, seed=42):
        super().__init__(tasks, messages, processors, network, security_service, deadline)
        random.seed(seed)
        np.random.seed(seed)
        self.q_table = {}
        # Parameters from SecurityEnhancedQLearningScheduler for better balance
        self.alpha = 0.25  # Moderate learning rate
        self.gamma = 0.9   # Discount factor
        self.epsilon = 0.8  # Initial exploration rate
        self.min_epsilon = 0.05  # Lower minimum for better convergence
        self.epsilon_decay = 0.97  # Decay rate
        self.episodes = 650  # More episodes for better convergence
        # Balance weights - adaptive based on episode
        self.security_weight = 0.3
        self.makespan_weight = 0.7
        self.security_epsilon = 0.2
        self.best_makespan = float('inf')
        self.best_schedule = None
        self.best_security_utility = 0
        self.best_security_assignments = {}
        self.best_tasks_state = None
        self.initialize_q_table()
        self.security_service = security_service
        
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
        """Enhanced processor selection balancing makespan and security."""
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
                
                # Stronger penalty for longer makespans
                makespan_impact = -eft  # Negative because lower is better
                
                # Check if this is on critical path - if so, prioritize speed
                if task.priority > sum(t.priority for t in self.tasks) / len(self.tasks) * 1.2:
                    makespan_weight = self.makespan_weight * 1.2  # Boost makespan importance
                    security_weight = self.security_weight * 0.8   # Reduce security importance
                else:
                    makespan_weight = self.makespan_weight
                    security_weight = self.security_weight
                
                # Calculate security impact - but don't let it override makespan
                security_impact = self.estimate_security_impact(task, p)
                
                # Add immediate start bonus from ImprovedQLearningScheduler
                immediate_start_bonus = 0
                if task.predecessors:
                    pred_tasks = [self.get_task_by_id(pred_id) for pred_id in task.predecessors]
                    scheduled_preds = [t for t in pred_tasks if t and t.is_scheduled]
                    
                    if scheduled_preds:
                        max_pred_finish = max(t.finish_time for t in scheduled_preds)
                        # If task starts soon after its predecessor finishes, give bonus
                        if est <= max_pred_finish + 5:  # Small tolerance
                            immediate_start_bonus = 50
                
                # Check for locality (predecessor on same processor)
                locality_bonus = sum(10 for pred_id in task.predecessors 
                                    if self.get_task_by_id(pred_id).is_scheduled and 
                                    self.get_task_by_id(pred_id).assigned_processor == p)
                
                # Balance with appropriate weights plus bonuses
                combined_value = (
                    makespan_weight * makespan_impact +
                    security_weight * security_impact +
                    immediate_start_bonus +
                    locality_bonus +
                    q_value
                )
                
                if combined_value > best_value:
                    best_value = combined_value
                    best_processor = p
            
            return best_processor
    
    def estimate_security_impact(self, task, processor_id):
        """Estimate the impact on security utility if task is assigned to processor."""
        security_impact = 0
        
        # Focus only on immediate communication needs
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
    
    def optimize_security_assignment(self):
        """Enhanced security protocol selection that better balances overhead and utility"""
        for message in self.messages:
            source_task = self.get_task_by_id(message.source_id)
            dest_task = self.get_task_by_id(message.dest_id)
            
            # Skip if tasks not available or on same processor
            if not source_task or not dest_task:
                # Fallback to basic optimization
                for service in ['confidentiality', 'integrity', 'authentication']:
                    min_overhead = float('inf')
                    best_protocol = 0
                    
                    for protocol in range(len(self.security_service.strengths[service])):
                        if self.security_service.strengths[service][protocol] >= message.min_security[service]:
                            overhead = sum(
                                self.security_service.overheads[service][protocol + 1][pid-1]
                                for pid in range(1, len(self.processors)+1)
                            )
                            if overhead < min_overhead:
                                min_overhead = overhead
                                best_protocol = protocol
                                
                    message.assigned_security[service] = best_protocol
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
            
            # For inter-processor messages, apply sophisticated logic
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
    
    def calculate_improved_reward(self, task, processor_id):
        """Enhanced reward function combining the best of both approaches."""
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
                
        # Enhanced parallelism bonus
        parallel_bonus = 0
        for other_task in self.tasks:
            if other_task.is_scheduled and other_task.task_id != task.task_id:
                # If tasks can run in parallel on different processors, reward it
                if (other_task.finish_time >= task.start_time and
                    other_task.start_time <= task.finish_time and
                    other_task.assigned_processor != processor_id):
                    parallel_bonus += 25  # High reward for parallelism
                    
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
            pred_tasks = [self.get_task_by_id(pred_id) for pred_id in task.predecessors]
            scheduled_preds = [t for t in pred_tasks if t and t.is_scheduled]
            
            if scheduled_preds:
                max_pred_finish = max(t.finish_time for t in scheduled_preds)
                # If task starts soon after its predecessor finishes, give a large bonus
                if task.start_time <= max_pred_finish + 5:  # Small tolerance
                    immediate_start_bonus = 50  # High reward for minimizing waiting time
                
        # Security utility for current task's messages
        security_reward = 0
        dependent_messages = []
        for msg in self.messages:
            if msg.source_id == task.task_id or msg.dest_id == task.task_id:
                dependent_messages.append(msg)
        
        for message in dependent_messages:
            for service in ['confidentiality', 'integrity', 'authentication']:
                protocol_idx = message.assigned_security[service]
                strength = self.security_service.strengths[service][protocol_idx]
                weight = message.weights[service]
                security_reward += weight * strength * 2
                
        # Combine all factors into final reward with balanced weights
        reward = (
            (slack > 0) * 50 +  # Binary reward for meeting deadline
            min(0, slack) * 0.5 +  # Small penalty for how much over deadline
            load_balance_score +  # Load balancing score
            locality_bonus +  # Reward for keeping related tasks together
            parallel_bonus +  # Reward for parallelism
            critical_path_bonus +  # Reward for critical path tasks
            immediate_start_bonus +  # Reward for starting tasks immediately
            security_reward -  # Reward for security
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

    def enhance_security_with_slack(self):
        """Enhanced security optimization with slack time respecting critical path"""
        # Calculate current makespan
        makespan = max(task.finish_time for task in self.tasks)
        
        # Calculate slack time with safety margin
        slack_time = (self.deadline - makespan) * 0.95  # Keep 5% safety margin
        if slack_time <= 0:
            print("No slack time available for security improvements")
            return  # No slack time available
        
        print(f"\n===== ENHANCING SECURITY WITH {slack_time:.2f} SLACK TIME =====")
        
        # Identify critical path tasks
        critical_path_tasks = self.identify_critical_path_tasks()
        print(f"Critical path tasks (avoiding security upgrades): {sorted(list(critical_path_tasks))}")
        
        # Remaining slack
        remaining_slack = slack_time
        
        # Track security improvements
        security_improvements = []
        
        # Prioritize messages by security potential
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
            
            # Calculate security efficiency
            security_efficiency = self.get_security_efficiency(message)
            prioritized_messages.append((message, security_efficiency))
        
        # Sort by security efficiency (highest first)
        prioritized_messages.sort(key=lambda x: x[1], reverse=True)
        
        # Enhance security in order of priority
        for message, efficiency in prioritized_messages:
            source_task = self.get_task_by_id(message.source_id)
            dest_task = self.get_task_by_id(message.dest_id)
            source_proc = source_task.assigned_processor
            dest_proc = dest_task.assigned_processor
            
            message_improvements = []
            
            # Try to upgrade services in order of weight (importance)
            services_by_weight = sorted(['confidentiality', 'integrity', 'authentication'], 
                                      key=lambda s: message.weights[s], reverse=True)
            
            for service in services_by_weight:
                current_level = message.assigned_security[service]
                
                # Only try to upgrade one level at a time
                if current_level < len(self.security_service.strengths[service]) - 1:
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
                    
                    # If this upgrade fits in our remaining slack, apply it
                    if time_cost <= remaining_slack:
                        # Store previous level for reporting
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

    def reinforce_best_solution(self):
        """Reinforce the best solution found so far in the Q-table"""
        for task in self.best_tasks_state:
            if task.is_scheduled:
                # Give a large positive reinforcement to the best choices
                self.q_table[(task.task_id, task.assigned_processor)] += 25
    
    def schedule_tasks(self):
        """Main task scheduling method combining both approaches"""
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
        
        # Store initial SHIELD solution
        for task in self.tasks:
            if task.is_scheduled:
                shield_schedule.append((task.task_id, task.assigned_processor))
        
        # Calculate security utility of SHIELD solution
        shield_security_utility = self.calculate_security_utility()
        
        print(f"SHIELD baseline makespan: {shield_makespan}")
        print(f"SHIELD baseline security utility: {shield_security_utility}")
        
        # Initialize best solution with SHIELD results
        self.best_makespan = shield_makespan
        self.best_security_utility = shield_security_utility
        self.best_schedule = shield_schedule
        self.best_security_assignments = shield_security_assignments
        self.best_tasks_state = shield_tasks_state
        
        # Reset tasks and processors for Q-learning
        for task in self.tasks:
            task.is_scheduled = False
            task.assigned_processor = None
            task.start_time = 0
            task.finish_time = 0
        
        for proc in self.processors:
            proc.available_time = 0
        
        # Reset security assignments
        for message in self.messages:
            message.assigned_security = {'confidentiality': 0, 'integrity': 0, 'authentication': 0}
        
        # Prepare episode metrics tracking
        makespan_history = []
        security_history = []
        
        # Main Q-learning training loop
        ready_tasks = self.get_ready_tasks()
        ordered_tasks = []
        
        print("\nStarting Q-learning optimization...")
        
        for episode in range(self.episodes):
            # Reset tasks and processors for this episode
            for task in self.tasks:
                task.is_scheduled = False
                task.assigned_processor = None
                task.start_time = 0
                task.finish_time = 0
            
            for proc in self.processors:
                proc.available_time = 0
            
            # Reset security assignments but with 10% chance to use best known assignments
            if random.random() < 0.1 and self.best_security_assignments:
                for message in self.messages:
                    if message.id in self.best_security_assignments:
                        message.assigned_security = copy.deepcopy(self.best_security_assignments[message.id])
                    else:
                        message.assigned_security = {'confidentiality': 0, 'integrity': 0, 'authentication': 0}
            else:
                for message in self.messages:
                    message.assigned_security = {'confidentiality': 0, 'integrity': 0, 'authentication': 0}
            
            # Optimize security assignments
            self.optimize_security_assignment()
            
            # Get ready tasks and schedule them
            scheduled_task_ids = set()
            ready_tasks = self.get_ready_tasks()
            
            # Calculate task priorities for this episode
            if episode % 50 == 0:  # Recalculate every 50 episodes to reduce computation
                self.compute_task_priorities()
            
            # Sort tasks by priority (critical path) if requested
            ordered_tasks = sorted(self.tasks, key=lambda t: t.priority, reverse=True)
            
            # Process tasks until all are scheduled
            while len(scheduled_task_ids) < len(self.tasks):
                ready_tasks = [t for t in ordered_tasks if 
                            t.task_id not in scheduled_task_ids and 
                            all(pred_id in scheduled_task_ids for pred_id in t.predecessors)]
                
                if not ready_tasks:
                    break  # No ready tasks but not all scheduled - something's wrong
                
                # Process all currently ready tasks
                for task in ready_tasks:
                    # Choose processor using Q-learning
                    processor_id = self.choose_processor(task, scheduled_task_ids)
                    
                    # Calculate earliest start time
                    est = self.calculate_est(task, processor_id)
                    
                    # Calculate finish time
                    eft = est + task.execution_times[processor_id - 1]
                    
                    # Schedule the task
                    task.assigned_processor = processor_id
                    task.start_time = est
                    task.finish_time = eft
                    task.is_scheduled = True
                    
                    # Update processor available time
                    self.processors[processor_id - 1].available_time = eft
                    
                    # Add to scheduled tasks
                    scheduled_task_ids.add(task.task_id)
                    
                    # Get next task for Q-value update (any ready task will do)
                    next_ready_tasks = [t for t in ordered_tasks if 
                                    t.task_id not in scheduled_task_ids and 
                                    all(pred_id in scheduled_task_ids for pred_id in t.predecessors)]
                    next_task_id = next_ready_tasks[0].task_id if next_ready_tasks else None
                    
                    # Calculate reward
                    reward = self.calculate_improved_reward(task, processor_id)
                    
                    # Update Q-value
                    self.update_q_value(task.task_id, processor_id, reward, next_task_id)
            
            # Calculate makespan and security utility for this episode
            if all(task.is_scheduled for task in self.tasks):
                makespan = max(task.finish_time for task in self.tasks)
                security_utility = self.calculate_security_utility()
                
                makespan_history.append(makespan)
                security_history.append(security_utility)
                
                # Check if this solution is better
                if makespan <= self.deadline and (
                makespan < self.best_makespan or 
                (abs(makespan - self.best_makespan) < 0.001 and security_utility > self.best_security_utility)):
                    
                    self.best_makespan = makespan
                    self.best_security_utility = security_utility
                    self.best_schedule = [(task.task_id, task.assigned_processor) for task in self.tasks]
                    self.best_tasks_state = copy.deepcopy(self.tasks)
                    
                    # Store best security assignments
                    self.best_security_assignments = {}
                    for message in self.messages:
                        self.best_security_assignments[message.id] = copy.deepcopy(message.assigned_security)
                    
                    print(f"Episode {episode}: Found new best solution - Makespan: {makespan:.2f}, Security: {security_utility:.4f}")
                    
                    # Reinforce this solution in the Q-table
                    self.reinforce_best_solution()
            
            # Decay epsilon for more exploitation in later episodes
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay
        
        # Restore best found solution
        if self.best_tasks_state:
            for task in self.tasks:
                best_task = next((t for t in self.best_tasks_state if t.task_id == task.task_id), None)
                if best_task:
                    task.assigned_processor = best_task.assigned_processor
                    task.start_time = best_task.start_time
                    task.finish_time = best_task.finish_time
                    task.is_scheduled = True
            
            for proc in self.processors:
                proc_tasks = [task for task in self.tasks 
                            if task.is_scheduled and task.assigned_processor == proc.proc_id]
                if proc_tasks:
                    proc.available_time = max(task.finish_time for task in proc_tasks)
                else:
                    proc.available_time = 0
        
        # Restore best security assignments
        if self.best_security_assignments:
            for message in self.messages:
                if message.id in self.best_security_assignments:
                    message.assigned_security = copy.deepcopy(self.best_security_assignments[message.id])
        
        # Final security enhancement with slack time
        if self.best_makespan < self.deadline:
            self.enhance_security_with_slack()
        
        # Print final results
        makespan = max(task.finish_time for task in self.tasks)
        security_utility = self.calculate_security_utility()
        
        print("\n===== FINAL SCHEDULE =====")
        print(f"Final makespan: {makespan:.2f}")
        print(f"Final security utility: {security_utility:.4f}")
        print(f"Deadline: {self.deadline}")
        print(f"Makespan meets deadline: {'Yes' if makespan <= self.deadline else 'No'}")
        print(f"Makespan improvement over SHIELD: {((shield_makespan - makespan) / shield_makespan) * 100:.2f}%")
        print(f"Security improvement over SHIELD: {((security_utility - shield_security_utility) / shield_security_utility) * 100:.2f}%")
        
        # Print detailed schedule
        print("\nTask schedule:")
        sorted_tasks = sorted(self.tasks, key=lambda t: t.task_id)
        for task in sorted_tasks:
            print(f"Task {task.task_id}: Processor {task.assigned_processor}, " + 
                f"Start: {task.start_time:.2f}, Finish: {task.finish_time:.2f}")
        
        return makespan, security_utility

    def get_ready_tasks(self):
        """Get tasks that are ready to be scheduled (all predecessors completed)"""
        scheduled_task_ids = set(task.task_id for task in self.tasks if task.is_scheduled)
        ready_tasks = []
        
        for task in self.tasks:
            if not task.is_scheduled and all(pred_id in scheduled_task_ids for pred_id in task.predecessors):
                ready_tasks.append(task)
        
        return ready_tasks

    def get_task_by_id(self, task_id):
        """Find and return task by its ID"""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def get_message(self, source_id, dest_id):
        """Find message between source and destination tasks"""
        for message in self.messages:
            if message.source_id == source_id and message.dest_id == dest_id:
                return message
        return None

    def compute_task_priorities(self):
        """Compute task priorities based on their position in the DAG (critical path)"""
        # Initialize upward ranks (priorities) to zero
        for task in self.tasks:
            task.priority = 0
        
        # Calculate average execution time for each task
        for task in self.tasks:
            task.avg_execution = sum(task.execution_times) / len(task.execution_times)
        
        # Find tasks with no successors (exit tasks)
        exit_tasks = []
        all_task_ids = set(task.task_id for task in self.tasks)
        all_pred_ids = set()
        for task in self.tasks:
            all_pred_ids.update(task.predecessors)
        exit_task_ids = all_task_ids - all_pred_ids
        
        for task in self.tasks:
            if task.task_id in exit_task_ids:
                exit_tasks.append(task)
        
        # Compute upward rank starting from exit tasks
        for exit_task in exit_tasks:
            self.compute_upward_rank(exit_task)
        
        # Calculate downward rank for each task
        self.compute_downward_ranks()
        
        # Final priority is upward rank + downward rank
        self.compute_task_priorities()
        
        # Sort tasks by priority
        self.tasks.sort(key=lambda task: task.priority, reverse=True)
        
        # Set priority for each task
        for i, task in enumerate(self.tasks ):
            task.priority = i + 1

    def compute_upward_rank(self, task):
        """Recursively compute upward rank (priority) of tasks"""
        # If already computed, return
        if hasattr(task, "upward_rank") and task.upward_rank > 0:
            return task.upward_rank
        
        # For exit tasks, rank is just average execution time
        if not self.get_successors(task):
            task.upward_rank = task.avg_execution
            return task.upward_rank
        
        # Calculate maximum rank of successors
        max_successor_rank = 0
        for succ_task in self.get_successors(task):
            # Calculate communication cost
            message = self.get_message(task.task_id, succ_task.task_id)
            comm_cost = message.size / 100 if message else 0
            
            # Recursive call to get successor rank
            succ_rank = self.compute_upward_rank(succ_task)
            
            # Update max successor rank
            max_successor_rank = max(max_successor_rank, comm_cost + succ_rank)
        
        # Task rank is its execution time + max successor rank
        task.upward_rank = task.avg_execution + max_successor_rank
        return task.upward_rank

    def compute_downward_ranks(self):
        """Compute downward ranks for all tasks"""
        # Find entry tasks (tasks with no predecessors)
        entry_tasks = [task for task in self.tasks if not task.predecessors]
        
        # Set downward rank of entry tasks to 0
        for task in entry_tasks:
            task.downward_rank = 0
        
        # Sort tasks by upward rank in descending order for topological traversal
        sorted_tasks = sorted(self.tasks, key=lambda t: getattr(t, "upward_rank", 0), reverse=True)
        
        # Calculate downward rank for each task
        for task in sorted_tasks:
            if task in entry_tasks:
                continue
                
            # Calculate downward rank for each predecessor
            for pred_id in task.predecessors:
                pred_task = self.get_task_by_id(pred_id)
                if not pred_task:
                    continue
                    
                # Calculate communication cost
                message = self.get_message(pred_id, task.task_id)
                comm_cost = message.size / 100 if message else 0
                
                # Calculate new downward rank
                new_downward_rank = getattr(task, "downward_rank", 0) + pred_task.avg_execution + comm_cost
                
                # Update if higher
                if not hasattr(pred_task, "downward_rank") or new_downward_rank > pred_task.downward_rank:
                    pred_task.downward_rank = new_downward_rank

    def get_successors(self, task):
        """Find all successors of a task"""
        successors = []
        for other_task in self.tasks:
            if task.task_id in other_task.predecessors:
                successors.append(other_task)
        return successors

    def run(self):
        """Main algorithm entry point"""
        # Schedule tasks using Q-learning
        makespan, security_utility = self.schedule_tasks()
        
        # Return results
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
    Scheduler.optimize_security_assignment = CombinedQLearningScheduler.optimize_security_assignment
    q_scheduler = CombinedQLearningScheduler(tasks_q, messages_q, processors_q, network, security_service, deadline)
    q_makespan, q_security_utility = q_scheduler.run()
    
    if q_makespan:
        q_fig = plot_gantt_chart("ImprovedQLearningScheduler", q_scheduler.schedule, len(processors_q), q_makespan)
        print(f"ImprovedQLearningScheduler Security Utility: {q_security_utility:.2f}")
    
    
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