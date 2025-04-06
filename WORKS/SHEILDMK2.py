import numpy as np
import matplotlib.pyplot as plt
import copy
import time
from tabulate import tabulate  # For prettier console output

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
        self.est = None  # Added EST
        self.eft = None  # Added EFT
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
        self.security_overhead = 0  # Added to track security overhead
    
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
            # Calculate comm time without security overhead
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
        
        # Print task priorities
        print("\n=== Task Priorities (Upward Ranks) ===")
        ranks_data = []
        for task in sorted(self.tasks, key=lambda t: -t.priority):
            ranks_data.append([
                task.task_id, 
                task.name, 
                f"{task.avg_execution:.2f}", 
                f"{task.priority:.2f}"
            ])
        print(tabulate(ranks_data, headers=["ID", "Name", "Avg Execution Time", "Priority (Rank)"]))
    
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
        
        # Print assigned security levels
        print("\n=== Assigned Minimum Security Levels ===")
        security_data = []
        for message in self.messages:
            conf_idx = message.assigned_security['confidentiality']
            integ_idx = message.assigned_security['integrity']
            auth_idx = message.assigned_security['authentication']
            
            conf_str = self.security.strengths['confidentiality'][conf_idx]
            integ_str = self.security.strengths['integrity'][integ_idx]
            auth_str = self.security.strengths['authentication'][auth_idx]
            
            security_data.append([
                message.id,
                f"{conf_str} (level {conf_idx+1})",
                f"{integ_str} (level {integ_idx+1})",
                f"{auth_str} (level {auth_idx+1})"
            ])
        print(tabulate(security_data, headers=["Message", "Confidentiality", "Integrity", "Authentication"]))
    
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
        
        # Store the overhead for later use in visualization
        message.security_overhead = total_overhead
        
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
        
        est = max(processor_ready_time, max_pred_finish_time)
        return est

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
        print("\n=== HSMS Scheduling Process ===")
        scheduling_data = []
        
        for task in sorted_tasks:
            best_processor = None
            earliest_finish_time = float('inf')
            earliest_start_time = 0
            task_est_data = {}
            task_eft_data = {}
            
            # Try each processor
            for processor in self.processors:
                est = self.calculate_est(task, processor.proc_id)
                eft = est + task.execution_times[processor.proc_id - 1]
                
                task_est_data[processor.proc_id] = est
                task_eft_data[processor.proc_id] = eft
                
                if eft < earliest_finish_time:
                    earliest_finish_time = eft
                    earliest_start_time = est
                    best_processor = processor
            
            # Store EST and EFT for the task
            task.est = earliest_start_time
            task.eft = earliest_finish_time
            
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
                    'finish_time': task.finish_time,
                    'est': task.est,
                    'eft': task.eft,
                    'execution_time': task.finish_time - task.start_time
                })
                
                # Add data for console output
                est_values = [f"P{p.proc_id}: {task_est_data.get(p.proc_id, float('inf')):.2f}" for p in self.processors]
                eft_values = [f"P{p.proc_id}: {task_eft_data.get(p.proc_id, float('inf')):.2f}" for p in self.processors]
                
                scheduling_data.append([
                    task.task_id,
                    task.name,
                    f"{task.priority:.2f}",
                    ", ".join(est_values),
                    ", ".join(eft_values),
                    task.assigned_processor,
                    f"{task.start_time:.2f}",
                    f"{task.finish_time:.2f}",
                    f"{task.finish_time - task.start_time:.2f}"
                ])
        
        # Print scheduling data
        print(tabulate(scheduling_data, headers=[
            "Task ID", "Name", "Priority", "EST", "EFT", 
            "Assigned Processor", "Start Time", "Finish Time", "Execution Time"
        ]))
        
        # Get the makespan (the maximum finish time of all tasks)
        self.makespan = max(task.finish_time for task in self.tasks)
        return self.makespan
    
    def calculate_security_utility(self):
        """Calculate the total security utility."""
        total_utility = 0
        print("\n=== Security Utility Calculation ===")
        utility_data = []
        
        for message in self.messages:
            message_utility = 0
            message_utilities = {}
            
            for service in ['confidentiality', 'integrity', 'authentication']:
                protocol_idx = message.assigned_security[service]
                strength = self.security.strengths[service][protocol_idx]
                weight = message.weights[service]
                service_utility = weight * strength
                message_utility += service_utility
                message_utilities[service] = service_utility
            
            total_utility += message_utility
            
            # For console output
            utility_data.append([
                message.id,
                f"{message_utilities.get('confidentiality', 0):.4f}",
                f"{message_utilities.get('integrity', 0):.4f}",
                f"{message_utilities.get('authentication', 0):.4f}",
                f"{message_utility:.4f}"
            ])
        
        print(tabulate(utility_data, headers=[
            "Message", "Confidentiality Utility", "Integrity Utility", 
            "Authentication Utility", "Total Utility"
        ]))
        print(f"Overall Security Utility: {total_utility:.4f}")
        
        return total_utility
    
    def run(self):
        """Run the HSMS scheduler."""
        print("\n========== RUNNING HSMS SCHEDULER ==========")
        start_time = time.time()
        makespan = self.schedule_tasks()
        security_utility = self.calculate_security_utility()
        end_time = time.time()
        
        if makespan > self.deadline:
            print(f"HSMS failed to meet deadline. Makespan: {makespan}, Deadline: {self.deadline}")
            return None, None
        else:
            print(f"HSMS successful. Makespan: {makespan:.2f}, Security Utility: {security_utility:.4f}")
            print(f"HSMS Execution Time: {(end_time - start_time):.4f} seconds")
            print(f"Slack Time Available: {(self.deadline - makespan):.2f} ms")
            return makespan, security_utility

class SHIELD(Scheduler):
    """Security-aware scHedulIng for rEaL-time Dags on heterogeneous systems"""
    
    def __init__(self, tasks, messages, processors, network, security_service, deadline):
        super().__init__(tasks, messages, processors, network, security_service, deadline)
        # Create a new HSMS instance but don't run it yet
        self.hsms = HSMS(copy.deepcopy(tasks), copy.deepcopy(messages), copy.deepcopy(processors), 
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
        print("\n========== RUNNING SHIELD SCHEDULER ==========")
        start_time = time.time()
        
        # Run HSMS first
        hsms_makespan, hsms_security_utility = self.hsms.run()
        
        if hsms_makespan is None:
            print("SHIELD cannot proceed since HSMS failed to meet the deadline.")
            return None, None
        
        # Copy the schedule and tasks from HSMS
        self.schedule = copy.deepcopy(self.hsms.schedule)
        self.messages = copy.deepcopy(self.hsms.messages)  # Copy messages with their security levels
        
        # Make a copy of tasks with the HSMS schedule
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
        initial_slack = slack_time
        print(f"\nInitial Slack Time: {slack_time:.2f} ms")
        
        upgrade_history = []  # Track security upgrades for visualization
        
        # While there is slack time available, try to enhance security
        iteration = 1
        while slack_time > 0 and slack_time < initial_slack:
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
                print("No more upgrades possible.")
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
                old_protocol_idx = message.assigned_security[service]
                message.assigned_security[service] = best_upgrade['protocol_idx']
                
                # Update slack time
                old_slack = slack_time
                slack_time -= best_upgrade['time_penalty']  # Corrected to properly reduce slack time
                
                # Log the upgrade for console output
                print(f"\nIteration {iteration}:")
                print(f"  Upgraded {message.id} {service} from level {old_protocol_idx+1} to {best_upgrade['protocol_idx']+1}")
                print(f"  Security Benefit: {best_upgrade['security_benefit']:.4f}")
                print(f"  Time Penalty: {best_upgrade['time_penalty']:.2f} ms")
                print(f"  Benefit-Cost Ratio: {best_upgrade['benefit_cost_ratio']:.4f}")
                print(f"  Remaining Slack: {slack_time:.2f} ms")
                
                # Save for visualization
                upgrade_history.append({
                    'iteration': iteration,
                    'message': message.id,
                    'service': service,
                    'old_level': old_protocol_idx + 1,
                    'new_level': best_upgrade['protocol_idx'] + 1,
                    'security_benefit': best_upgrade['security_benefit'],
                    'time_penalty': best_upgrade['time_penalty'],
                    'slack_before': old_slack,
                    'slack_after': slack_time
                })
                
                iteration += 1
            else:
                # No feasible upgrades left
                print("No more feasible upgrades.")
                break

        
        # Calculate final security utility
        print("\n=== Final Security Configuration after SHIELD ===")
        security_data = []
        for message in self.messages:
            conf_idx = message.assigned_security['confidentiality']
            integ_idx = message.assigned_security['integrity']
            auth_idx = message.assigned_security['authentication']
            
            conf_str = self.security.strengths['confidentiality'][conf_idx]
            integ_str = self.security.strengths['integrity'][integ_idx]
            auth_str = self.security.strengths['authentication'][auth_idx]
            
            security_data.append([
                message.id,
                f"{conf_str} (level {conf_idx+1})",
                f"{integ_str} (level {integ_idx+1})",
                f"{auth_str} (level {auth_idx+1})"
            ])
        print(tabulate(security_data, headers=["Message", "Confidentiality", "Integrity", "Authentication"]))
        
        # Calculate final security utility
        security_utility = 0
        utility_data = []
        
        for message in self.messages:
            message_utility = 0
            message_utilities = {}
            
            for service in ['confidentiality', 'integrity', 'authentication']:
                protocol_idx = message.assigned_security[service]
                strength = self.security.strengths[service][protocol_idx]
                weight = message.weights[service]
                service_utility = weight * strength
                message_utility += service_utility
                message_utilities[service] = service_utility
            
            security_utility += message_utility
            
            # For console output
            utility_data.append([
                message.id,
                f"{message_utilities.get('confidentiality', 0):.4f}",
                f"{message_utilities.get('integrity', 0):.4f}",
                f"{message_utilities.get('authentication', 0):.4f}",
                f"{message_utility:.4f}"
            ])
        
        print("\n=== Final Security Utility ===")
        print(tabulate(utility_data, headers=[
            "Message", "Confidentiality Utility", "Integrity Utility", 
            "Authentication Utility", "Total Utility"
        ]))
        
        # Calculate final makespan
        makespan = max(task.finish_time for task in self.tasks if task.is_scheduled)
        end_time = time.time()
        
        print(f"\nSHIELD successful. Makespan: {makespan:.2f}, Security Utility: {security_utility:.4f}")
        print(f"Security Improvement: {security_utility - hsms_security_utility:.4f}")
        print(f"Slack Used: {initial_slack - slack_time:.2f} ms")
        print(f"Remaining Slack: {slack_time:.2f} ms")
        print(f"SHIELD Execution Time: {(end_time - start_time):.4f} seconds")
        
        # Save upgrade history for visualization
        self.upgrade_history = upgrade_history
        
        return makespan, security_utility

def plot_gantt_chart(title, schedule, num_processors, makespan, deadline=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = plt.gcf()
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Enhanced to show task details in the chart
    for entry in schedule:
        task_id = entry['task_id']
        name = entry['name']
        proc = entry['processor']
        start = entry['start_time']
        finish = entry['finish_time']
        
        color_idx = task_id % len(colors)
        bar = ax.barh(proc - 1, finish - start, left=start, color=colors[color_idx], edgecolor='black')
        
        # Add more detailed text label
        ax.text(start + (finish - start) / 2, proc - 1, f"T{task_id}", 
                va='center', ha='center', color='white', fontsize=10, weight='bold')
    
    # Create a detailed legend for task information
    legend_elements = []
    for entry in schedule:
        task_id = entry['task_id']
        color_idx = task_id % len(colors)
        legend_elements.append(plt.Rectangle((0,0), 1, 1, color=colors[color_idx], 
                               label=f"T{task_id}: EST={entry['est']:.1f}, EFT={entry['eft']:.1f}, Exe={entry['execution_time']:.1f}"))
    
    # Add deadline line if provided
    if deadline:
        ax.axvline(x=deadline, color='red', linestyle='--', linewidth=2, label='Deadline')
        
        # Show slack time as a filled area - improved visualization
        if makespan < deadline:
            slack_time = deadline - makespan
            slack_patch = plt.Rectangle((makespan, -0.5), slack_time, num_processors, 
                                      color='lightgreen', alpha=0.3)
            ax.add_patch(slack_patch)
            
            # Add text to show slack time amount
            ax.text(makespan + slack_time/2, num_processors/2, 
                   f"Slack: {slack_time:.2f} ms", 
                   ha='center', va='center', color='green', 
                   fontsize=12, weight='bold', bbox=dict(facecolor='white', alpha=0.7))
            
            # Add to legend
            legend_elements.append(plt.Rectangle((0,0), 1, 1, color='lightgreen', alpha=0.3, label=f'Slack: {slack_time:.2f} ms'))
    
    # Add deadline to legend if applicable
    if deadline:
        legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label=f'Deadline: {deadline} ms'))
    
    # Place the legend outside the plot
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), 
              fancybox=True, shadow=True, ncol=3)
    
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Processor")
    ax.set_title(f"{title} - Makespan: {makespan:.2f} ms")
    ax.set_yticks(range(num_processors))
    ax.set_yticklabels([f"P{p+1}" for p in range(num_processors)])
    
    max_x = deadline * 1.05 if deadline else makespan * 1.1
    ax.set_xlim(0, max_x)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Adjust layout to make room for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    
    return fig, ax

class HSMS(Scheduler):
    """Heterogeneous Security-aware Makespan minimizing Scheduler"""
    
    def schedule_tasks(self):
        """Schedule tasks using security-aware HEFT algorithm."""

        self.compute_task_priorities()

        self.assign_minimum_security()
        
        # Sort tasks by priority (descending)
        sorted_tasks = sorted(self.tasks, key=lambda t: -t.priority)
        
        # Schedule each task
        print("\n=== HSMS Scheduling Process ===")
        scheduling_data = []
        
        for task in sorted_tasks:
            best_processor = None
            earliest_finish_time = float('inf')
            earliest_start_time = 0
            task_est_data = {}
            task_eft_data = {}
            
            # Try each processor
            for processor in self.processors:
                est = self.calculate_est(task, processor.proc_id)
                eft = est + task.execution_times[processor.proc_id - 1]
                
                task_est_data[processor.proc_id] = est
                task_eft_data[processor.proc_id] = eft
                
                if eft < earliest_finish_time:
                    earliest_finish_time = eft
                    earliest_start_time = est
                    best_processor = processor
            
            # Store EST and EFT for the task
            task.est = earliest_start_time
            task.eft = earliest_finish_time
            
            # Assign task to the best processor
            if best_processor:
                task.assigned_processor = best_processor.proc_id
                task.start_time = earliest_start_time
                task.finish_time = earliest_finish_time
                task.is_scheduled = True
                best_processor.available_time = earliest_finish_time
                
                # Calculate security overhead for each predecessor
                total_security_overhead = 0
                for pred_id in task.predecessors:
                    pred_task = self.get_task_by_id(pred_id)
                    if pred_task and pred_task.is_scheduled and pred_task.assigned_processor != task.assigned_processor:
                        message = self.get_message(pred_id, task.task_id)
                        if message:
                            sec_overhead = self.calc_security_overhead(
                                message, 
                                pred_task.assigned_processor, 
                                task.assigned_processor
                            )
                            total_security_overhead += sec_overhead
                
                # Enhanced schedule entry with more data
                self.schedule.append({
                    'task_id': task.task_id,
                    'name': task.name,
                    'processor': task.assigned_processor,
                    'start_time': task.start_time,
                    'finish_time': task.finish_time,
                    'est': task.est,
                    'eft': task.eft,
                    'execution_time': task.execution_times[task.assigned_processor - 1],
                    'priority': task.priority,
                    'security_overhead': total_security_overhead
                })
                
                # Add data for console output
                est_values = [f"P{p.proc_id}: {task_est_data.get(p.proc_id, float('inf')):.2f}" for p in self.processors]
                eft_values = [f"P{p.proc_id}: {task_eft_data.get(p.proc_id, float('inf')):.2f}" for p in self.processors]
                
                scheduling_data.append([
                    task.task_id,
                    task.name,
                    f"{task.priority:.2f}",
                    ", ".join(est_values),
                    ", ".join(eft_values),
                    task.assigned_processor,
                    f"{task.start_time:.2f}",
                    f"{task.finish_time:.2f}",
                    f"{task.execution_times[task.assigned_processor - 1]:.2f}",
                    f"{total_security_overhead:.2f}"
                ])
        
        # Print scheduling data with enhanced information
        print(tabulate(scheduling_data, headers=[
            "Task ID", "Name", "Priority(Rank)", "EST", "EFT", 
            "Assigned Processor", "Start Time", "Finish Time", "Execution Time", "Security Overhead"
        ]))
        
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
        print("\n========== RUNNING HSMS SCHEDULER ==========")
        start_time = time.time()
        makespan = self.schedule_tasks()
        security_utility = self.calculate_security_utility()
        end_time = time.time()
        
        if makespan > self.deadline:
            print(f"HSMS failed to meet deadline. Makespan: {makespan}, Deadline: {self.deadline}")
            return None, None
        else:
            print(f"HSMS successful. Makespan: {makespan:.2f}, Security Utility: {security_utility:.4f}")
            print(f"HSMS Execution Time: {(end_time - start_time):.4f} seconds")
            print(f"Slack Time Available: {(self.deadline - makespan):.2f} ms")
            return makespan, security_utility

class SHIELD(Scheduler):
    """Security-aware scHedulIng for rEaL-time Dags on heterogeneous systems"""
    
    def __init__(self, tasks, messages, processors, network, security_service, deadline):
        super().__init__(tasks, messages, processors, network, security_service, deadline)
        # Create a new HSMS instance but don't run it yet
        self.hsms = HSMS(copy.deepcopy(tasks), copy.deepcopy(messages), copy.deepcopy(processors), 
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
        print("\n========== RUNNING SHIELD SCHEDULER ==========")
        start_time = time.time()
        
        # Run HSMS first
        hsms_makespan, hsms_security_utility = self.hsms.run()
        
        if hsms_makespan is None:
            print("SHIELD cannot proceed since HSMS failed to meet the deadline.")
            return None, None
        
        # Copy the schedule and messages from HSMS
        self.schedule = copy.deepcopy(self.hsms.schedule)
        self.messages = copy.deepcopy(self.hsms.messages)  # Copy messages with their security levels
        
        # Make a copy of tasks with the HSMS schedule
        for task in self.tasks:
            hsms_task = next((t for t in self.hsms.tasks if t.task_id == task.task_id), None)
            if hsms_task:
                task.assigned_processor = hsms_task.assigned_processor
                task.start_time = hsms_task.start_time
                task.finish_time = hsms_task.finish_time
                task.est = hsms_task.est
                task.eft = hsms_task.eft
                task.is_scheduled = True
        
        # Initialize the slack time
        slack_time = self.deadline - hsms_makespan
        initial_slack = slack_time
        print(f"\nInitial Slack Time: {slack_time:.2f} ms")
        
        upgrade_history = []  # Track security upgrades for visualization
        
        # While there is slack time available, try to enhance security
        iteration = 1
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
                print("No more upgrades possible.")
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
                old_protocol_idx = message.assigned_security[service]
                message.assigned_security[service] = best_upgrade['protocol_idx']
                
                # Update slack time
                old_slack = slack_time
                slack_time -= best_upgrade['time_penalty']
                
                # Log the upgrade for console output
                print(f"\nIteration {iteration}:")
                print(f"  Upgraded {message.id} {service} from level {old_protocol_idx+1} to {best_upgrade['protocol_idx']+1}")
                print(f"  Security Benefit: {best_upgrade['security_benefit']:.4f}")
                print(f"  Time Penalty: {best_upgrade['time_penalty']:.2f} ms")
                print(f"  Benefit-Cost Ratio: {best_upgrade['benefit_cost_ratio']:.4f}")
                print(f"  Remaining Slack: {slack_time:.2f} ms")
                
                # Save for visualization
                upgrade_history.append({
                    'iteration': iteration,
                    'message': message.id,
                    'service': service,
                    'old_level': old_protocol_idx + 1,
                    'new_level': best_upgrade['protocol_idx'] + 1,
                    'security_benefit': best_upgrade['security_benefit'],
                    'time_penalty': best_upgrade['time_penalty'],
                    'slack_before': old_slack,
                    'slack_after': slack_time
                })
                
                iteration += 1
            else:
                # No feasible upgrades left
                print("No more feasible upgrades.")
                break
        
        # Calculate final security utility
        print("\n=== Final Security Configuration after SHIELD ===")
        security_data = []
        for message in self.messages:
            conf_idx = message.assigned_security['confidentiality']
            integ_idx = message.assigned_security['integrity']
            auth_idx = message.assigned_security['authentication']
            
            conf_str = self.security.strengths['confidentiality'][conf_idx]
            integ_str = self.security.strengths['integrity'][integ_idx]
            auth_str = self.security.strengths['authentication'][auth_idx]
            
            security_data.append([
                message.id,
                f"{conf_str} (level {conf_idx+1})",
                f"{integ_str} (level {integ_idx+1})",
                f"{auth_str} (level {auth_idx+1})"
            ])
        print(tabulate(security_data, headers=["Message", "Confidentiality", "Integrity", "Authentication"]))
        
        # Calculate final security utility
        security_utility = 0
        utility_data = []
        
        for message in self.messages:
            message_utility = 0
            message_utilities = {}
            
            for service in ['confidentiality', 'integrity', 'authentication']:
                protocol_idx = message.assigned_security[service]
                strength = self.security.strengths[service][protocol_idx]
                weight = message.weights[service]
                service_utility = weight * strength
                message_utility += service_utility
                message_utilities[service] = service_utility
            
            security_utility += message_utility
            
            # For console output
            utility_data.append([
                message.id,
                f"{message_utilities.get('confidentiality', 0):.4f}",
                f"{message_utilities.get('integrity', 0):.4f}",
                f"{message_utilities.get('authentication', 0):.4f}",
                f"{message_utility:.4f}"
            ])
        
        print("\n=== Final Security Utility ===")
        print(tabulate(utility_data, headers=[
            "Message", "Confidentiality Utility", "Integrity Utility", 
            "Authentication Utility", "Total Utility"
        ]))
        
        # Calculate final makespan
        makespan = max(task.finish_time for task in self.tasks if task.is_scheduled)
        end_time = time.time()
        
        print(f"\nSHIELD successful. Makespan: {makespan:.2f}, Security Utility: {security_utility:.4f}")
        print(f"Security Improvement: {security_utility - hsms_security_utility:.4f}")
        print(f"Slack Used: {initial_slack - slack_time:.2f} ms")
        print(f"Remaining Slack: {slack_time:.2f} ms")
        print(f"SHIELD Execution Time: {(end_time - start_time):.4f} seconds")
        
        # Save upgrade history for visualization
        self.upgrade_history = upgrade_history
        
        return makespan, security_utility

# Add visualization for security upgrades and slack time
def plot_shield_improvements(shield, deadline, hsms_makespan, hsms_utility, shield_makespan, shield_utility):
    """Visualize the security improvements and slack time usage from SHIELD."""
    if not hasattr(shield, 'upgrade_history') or not shield.upgrade_history:
        print("No upgrade history available to visualize.")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Slack Time Usage
    iterations = list(range(len(shield.upgrade_history) + 1))
    slack_values = [deadline - hsms_makespan]  # Initial slack
    
    for upgrade in shield.upgrade_history:
        slack_values.append(upgrade['slack_after'])
    
    ax1.plot(iterations, slack_values, 'bo-', linewidth=2)
    ax1.fill_between(iterations, slack_values, alpha=0.3, color='lightblue')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Remaining Slack Time (ms)')
    ax1.set_title('Slack Time Consumption During SHIELD')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Annotate points with message and service info
    for i, upgrade in enumerate(shield.upgrade_history):
        ax1.annotate(f"{upgrade['message']}\n{upgrade['service']}: {upgrade['old_level']}→{upgrade['new_level']}",
                    (i + 1, upgrade['slack_after']),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center')
    
    # Plot 2: Security Utility Improvement
    utility_values = [hsms_utility]
    cumulative_benefit = hsms_utility
    
    for upgrade in shield.upgrade_history:
        cumulative_benefit += upgrade['security_benefit']
        utility_values.append(cumulative_benefit)
    
    ax2.plot(iterations, utility_values, 'ro-', linewidth=2)
    ax2.fill_between(iterations, utility_values, min(utility_values), alpha=0.3, color='lightcoral')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cumulative Security Utility')
    ax2.set_title('Security Utility Improvement During SHIELD')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Annotate points with details
    for i, upgrade in enumerate(shield.upgrade_history):
        ax2.annotate(f"{upgrade['message']}\n+{upgrade['security_benefit']:.4f}",
                    (i + 1, utility_values[i + 1]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center')
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Function to visualize comparative results
def plot_comparison(hsms_scheduler, shield_scheduler, deadline):
    """Create a comparative visualization of HSMS and SHIELD results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Get makespans
    hsms_makespan = max(task.finish_time for task in hsms_scheduler.tasks if hasattr(task, 'finish_time') and task.finish_time)
    shield_makespan = max(task.finish_time for task in shield_scheduler.tasks if hasattr(task, 'finish_time') and task.finish_time)
    
    # Plot HSMS schedule
    plot_gantt_chart("HSMS Schedule", hsms_scheduler.schedule, len(hsms_scheduler.processors), hsms_makespan, deadline, ax1)
    
    # Plot SHIELD schedule 
    plot_gantt_chart("SHIELD Schedule", shield_scheduler.schedule, len(shield_scheduler.processors), shield_makespan, deadline, ax2)
    
    plt.tight_layout()
    plt.show()
    
    return fig
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
    # Note: We need to create fresh copies of the tasks and processors
    tasks_shield, messages, processors_shield, network, security_service, deadline = create_tc_test_case()
    
    shield = SHIELD(tasks_shield, messages, processors_shield, network, security_service, deadline)
    shield_makespan, shield_security_utility = shield.run()
    
    if shield_makespan:
        plot_gantt_chart("SHIELD", shield.schedule, len(processors_shield), shield_makespan)
        print(f"SHIELD Security Utility: {shield_security_utility:.2f}")

        plot_shield_improvements(shield, deadline, hsms_makespan, hsms_security_utility, shield_makespan, shield_security_utility)
        plot_comparison(hsms, shield, deadline)



    
    # Show the plots
    plt.show()

if __name__ == "__main__":
    main()

