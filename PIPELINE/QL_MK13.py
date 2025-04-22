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
            
    def choose_processor(self, task, available_procs, current_time):
        """Choose a processor using epsilon-greedy strategy with constraints."""
        # Get processors that can start the task earliest
        viable_procs = []
        earliest_start = float('inf')
        
        for proc_id in available_procs:
            est = max(self.processors[proc_id-1].available_time, current_time)
            
            # Check predecessors' finish times
            for pred_id in task.predecessors:
                pred_task = self.get_task_by_id(pred_id)
                if pred_task.assigned_processor == proc_id:
                    # Same processor - just wait for the predecessor to finish
                    est = max(est, pred_task.finish_time)
                else:
                    # Different processor - add communication time
                    message = self.get_message(pred_id, task.task_id)
                    if message:
                        comm_time = self.calculate_communication_time(
                            message, pred_task.assigned_processor, proc_id
                        )
                        est = max(est, pred_task.finish_time + comm_time)
            
            # If this processor allows earlier start than current best, reset viable list
            if est < earliest_start:
                earliest_start = est
                viable_procs = [proc_id]
            # If this processor matches current earliest start time, add it to viable list
            elif est == earliest_start:
                viable_procs.append(proc_id)
        
        # Exploration: with probability epsilon, choose randomly from viable processors
        if random.random() < self.epsilon:
            return random.choice(viable_procs)
        # Exploitation: choose processor with highest Q-value among viable processors
        else:
            return max(viable_procs, key=lambda p: self.q_table.get((task.task_id, p), 0))
    
    def get_available_processors(self, current_time):
        """Get processors that are available at the current time."""
        available_procs = []
        for proc in self.processors:
            if proc.available_time <= current_time:
                available_procs.append(proc.proc_id)
        return available_procs
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
    def calculate_priority_level(self):
        """Calculate priority level for all tasks using critical path method."""
        # Initialize all priority levels to 0
        for task in self.tasks:
            task.priority = 0
            
        # Start with tasks that have no successors (exit tasks)
        exit_tasks = []
        for task in self.tasks:
            has_successors = False
            for other_task in self.tasks:
                if task.task_id in other_task.predecessors:
                    has_successors = True
                    break
            if not has_successors:
                exit_tasks.append(task)
                
        # Set priority for exit tasks based on execution time
        for task in exit_tasks:
            task.priority = max(task.execution_times)
            
        # Traverse the task graph in reverse topological order to calculate priorities
        changed = True
        while changed:
            changed = False
            for task in self.tasks:
                # Check all successors to calculate priority
                for other_task in self.tasks:
                    if task.task_id in other_task.predecessors:
                        # Successor found, update priority
                        new_priority = other_task.priority + max(task.execution_times)
                        if new_priority > task.priority:
                            task.priority = new_priority
                            changed = True
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
    
    def schedule_tasks(self):
        """Schedule tasks using improved Q-learning with SHIELD seeding."""
        # Calculate task priorities using critical path method
        self.calculate_priority_level()
        
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
        
        # Seed the Q-table with the SHIELD solution to provide a strong baseline
        self.seed_q_table_with_shield()
        
        # Start Q-learning episodes
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
            
            # Create ready and scheduled task lists
            ready_tasks = []
            scheduled_tasks = []
            
            # Initialize ready tasks (those with no predecessors)
            for task in self.tasks:
                if not task.predecessors:
                    ready_tasks.append(task)
            
            # Sort ready tasks by priority (higher priority first)
            ready_tasks.sort(key=lambda x: -x.priority)
            
            # Event-driven scheduling
            current_time = 0
            
            # Main scheduling loop
            while len(scheduled_tasks) < len(self.tasks):
                # If there are no ready tasks, advance time to next processor availability
                if not ready_tasks:
                    earliest_proc_time = float('inf')
                    # Find the earliest time a processor becomes available
                    for proc in self.processors:
                        if proc.available_time > current_time and proc.available_time < earliest_proc_time:
                            earliest_proc_time = proc.available_time
                    
                    # If no processor becomes available, there must be a waiting task
                    # whose predecessors will finish
                    if earliest_proc_time == float('inf'):
                        # Find the earliest time any unscheduled task can become ready
                        for task in self.tasks:
                            if not task.is_scheduled and task not in ready_tasks:
                                # Check if all but one predecessor are scheduled
                                unscheduled_preds = [p for p in task.predecessors 
                                                  if not self.get_task_by_id(p).is_scheduled]
                                if len(unscheduled_preds) == 1:
                                    # This task will become ready when its last predecessor finishes
                                    for other in scheduled_tasks:
                                        if other.task_id == unscheduled_preds[0]:
                                            if other.finish_time < earliest_proc_time:
                                                earliest_proc_time = other.finish_time
                    
                    # Update current time
                    if earliest_proc_time < float('inf'):
                        current_time = earliest_proc_time
                    else:
                        # This shouldn't happen - increment time slightly as a safeguard
                        current_time += 1
                
                # Check for newly ready tasks based on current time
                newly_ready = []
                for task in self.tasks:
                    if not task.is_scheduled and task not in ready_tasks:
                        # Check if all predecessors are scheduled
                        if all(self.get_task_by_id(pred_id).is_scheduled for pred_id in task.predecessors):
                            # Check if all predecessors have finished by current time
                            all_preds_finished = True
                            for pred_id in task.predecessors:
                                pred_task = self.get_task_by_id(pred_id)
                                if pred_task.finish_time > current_time:
                                    all_preds_finished = False
                                    break
                            if all_preds_finished:
                                newly_ready.append(task)
                
                # Add newly ready tasks and sort by priority
                ready_tasks.extend(newly_ready)
                ready_tasks.sort(key=lambda x: -x.priority)
                
                # Schedule tasks if there are ready tasks and available processors
                if ready_tasks:
                    current_task = ready_tasks[0]
                    
                    # Get available processors at current time
                    available_procs = [p.proc_id for p in self.processors 
                                      if p.available_time <= current_time]
                    
                    if available_procs:
                        # Remove the task from ready list
                        ready_tasks.pop(0)
                        
                        # Choose processor using Q-learning strategy
                        proc_id = self.choose_processor(current_task, available_procs, current_time)
                        processor = self.processors[proc_id - 1]
                        
                        # Calculate actual start and finish times
                        start_time = max(current_time, processor.available_time)
                        
                        # Ensure we respect predecessor dependencies
                        for pred_id in current_task.predecessors:
                            pred_task = self.get_task_by_id(pred_id)
                            if pred_task.assigned_processor == proc_id:
                                # Same processor - just wait for predecessor
                                start_time = max(start_time, pred_task.finish_time)
                            else:
                                # Different processor - add communication time
                                message = self.get_message(pred_id, current_task.task_id)
                                if message:
                                    comm_time = self.calculate_communication_time(
                                        message, pred_task.assigned_processor, proc_id
                                    )
                                    start_time = max(start_time, pred_task.finish_time + comm_time)
                        
                        # Calculate finish time
                        finish_time = start_time + current_task.execution_times[proc_id - 1]
                        
                        # Schedule the task
                        current_task.assigned_processor = proc_id
                        current_task.start_time = start_time
                        current_task.finish_time = finish_time
                        current_task.is_scheduled = True
                        processor.available_time = finish_time
                        scheduled_tasks.append(current_task)
                        
                        # Add to schedule
                        entry = {
                            'task_id': current_task.task_id,
                            'name': current_task.name,
                            'processor': proc_id,
                            'start_time': start_time,
                            'finish_time': finish_time
                        }
                        current_schedule.append(entry)
                        
                        # Calculate reward based on how well it minimizes waiting time
                        reward = self.calculate_minimal_wait_reward(current_task, proc_id)
                        
                        # Update Q-table
                        if ready_tasks:
                            self.update_q_value(current_task.task_id, proc_id, reward, ready_tasks[0].task_id)
                        else:
                            self.update_q_value(current_task.task_id, proc_id, reward)
                            
                        # Check if any new tasks become ready after scheduling this task
                        for task in self.tasks:
                            if not task.is_scheduled and task not in ready_tasks:
                                # Check if all predecessors are scheduled
                                if all(self.get_task_by_id(pred_id).is_scheduled for pred_id in task.predecessors):
                                    # Check if all predecessors have finished
                                    all_preds_finished = True
                                    for pred_id in task.predecessors:
                                        pred_task = self.get_task_by_id(pred_id)
                                        if pred_task.finish_time > finish_time:
                                            all_preds_finished = False
                                            break
                                    if all_preds_finished:
                                        ready_tasks.append(task)
                        
                        # Re-sort ready tasks by priority
                        ready_tasks.sort(key=lambda x: -x.priority)
                    else:
                        # No available processors, advance time to next processor availability
                        earliest_proc_time = min(proc.available_time for proc in self.processors 
                                              if proc.available_time > current_time)
                        current_time = earliest_proc_time
                
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
    
    def seed_q_table_with_shield(self):
        """Seed the Q-table with values from the SHIELD solution."""
        for task in self.best_tasks_state:
            if task.is_scheduled:
                # Give a high Q-value to the processor assignments from SHIELD
                self.q_table[(task.task_id, task.assigned_processor)] = 50.0
                
                # Give lower values to other processor assignments
                for proc_id in range(1, len(self.processors) + 1):
                    if proc_id != task.assigned_processor:
                        self.q_table[(task.task_id, proc_id)] = 10.0
    
    def calculate_minimal_wait_reward(self, task, processor_id):
        """Calculate reward with a strong emphasis on minimizing waiting time."""
        # Calculate the earliest possible start time based on predecessors
        earliest_possible_start = 0
        for pred_id in task.predecessors:
            pred_task = self.get_task_by_id(pred_id)
            if pred_task and pred_task.is_scheduled:
                if pred_task.assigned_processor == processor_id:
                    # Same processor - direct dependency
                    earliest_possible_start = max(earliest_possible_start, pred_task.finish_time)
                else:
                    # Different processor - add communication time
                    message = self.get_message(pred_id, task.task_id)
                    if message:
                        comm_time = self.calculate_communication_time(
                            message, pred_task.assigned_processor, processor_id
                        )
                        earliest_possible_start = max(
                            earliest_possible_start, 
                            pred_task.finish_time + comm_time
                        )
        
        # Calculate processor availability
        processor = self.processors[processor_id - 1]
        processor_availability = processor.available_time
        
        # The actual start time will be the maximum of earliest possible start 
        # and processor availability
        actual_start = max(earliest_possible_start, processor_availability)
        
        # Calculate waiting time (negative reward)
        waiting_time = actual_start - earliest_possible_start
        waiting_penalty = -100 * waiting_time if waiting_time > 0 else 50
        
        # Calculate makespan impact (how close to deadline)
        finish_time = actual_start + task.execution_times[processor_id - 1]
        makespan_impact = self.deadline - finish_time
        makespan_reward = 20 if makespan_impact > 0 else -10 * abs(makespan_impact)
        
        # Critical path bonus
        critical_path_bonus = 30 if task.priority > self.tasks[0].priority / 2 else 0
        
        # Processor locality bonus (reward keeping related tasks on same processor)
        locality_bonus = 0
        for pred_id in task.predecessors:
            pred_task = self.get_task_by_id(pred_id)
            if pred_task and pred_task.assigned_processor == processor_id:
                locality_bonus += 20
        
        # Combine factors with stronger weight on waiting time
        reward = (
            waiting_penalty +      # Strong penalty/reward for waiting time
            makespan_reward +      # Reward for meeting deadline
            critical_path_bonus +  # Bonus for critical path tasks
            locality_bonus         # Bonus for locality
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
                    sec_overhead += self.calculate_security_overhead(message, pred_task.assigned_processor, processor_id)
                    
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
        
    def run(self):
        """Run the ImprovedQLearningScheduler"""
        start_time = time.time()
        makespan = self.schedule_tasks()
        security_utility = self.best_security_utility
        end_time = time.time()
        
        print(f"ImprovedQLearningScheduler: Makespan={makespan}, Security Utility={security_utility:.2f}")
        print(f"Execution time: {end_time - start_time:.4f} seconds")
        
        return makespan, security_utility
    def optimize_security_assignment(self):
        """Select security protocols that minimize overhead while meeting requirements"""
        for message in self.messages:
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
    
    # Run ImprovedQLearningScheduler
    print("\nRunning ImprovedQLearningScheduler...")
    tasks_q, messages_q, processors_q, network, security_service, deadline = create_tc_test_case()
    
    # Add the optimize_security_assignment method to Scheduler class
    Scheduler.optimize_security_assignment = ImprovedQLearningScheduler.optimize_security_assignment
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
    
    plt.subplot(1, 2, 2)
    plt.title("ImprovedQLearningScheduler Gantt Chart")
    plt.xlabel("Time (ms)")
    plt.ylabel("Processor")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.show()
    plt.savefig("scheduler_comparison.png")
    plt.close()
    plt.show()
    print("Scheduler comparison plot saved as 'scheduler_comparison.png'.")
if __name__ == "__main__":
    main()