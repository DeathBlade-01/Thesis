import numpy as np
import matplotlib.pyplot as plt
import copy
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
       # In the SecurityService.__init__ method:
        self.strengths = {
            'confidentiality': [0.1, 0.25, 0.4, 0.55, 0.7, 0.8, 0.9, 1.0],  # Was [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            'integrity': [0.1, 0.25, 0.4, 0.6, 0.75, 0.9, 1.0],  # Was [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
            'authentication': [0.2, 0.6, 1.0]  # Was [0.2, 0.5, 1.0]
        }
        
        # Security overheads (execution time) for each protocol on each processor
        # Based on Table 4 from paper (simplified for 2 processors)
        self.overheads = {
            'confidentiality': {
                # Scale down values by factor of ~10
                1: [8, 8.5],  # Was [150, 145]
                2: [5.5, 5.0],  # Was [95, 90]
                3: [3.5, 3.0],  # Was [35, 30]
                4: [3.0, 3.2],  # Was [30, 32]
                5: [2.8, 2.7],  # Was [28, 27]
                6: [2.0, 2.2],  # Was [20, 22]
                7: [1.4, 1.5],  # Was [14, 15]
                8: [1.2, 1.3]   # Was [12, 13]
            },
            'integrity': {
                # Scale down values by factor of ~5
                1: [4.4, 4.8],  # Was [22, 24]
                2: [3.2, 3.6],  # Was [16, 18]
                3: [2.2, 2.4],  # Was [11, 12]
                4: [1.8, 2.0],  # Was [9, 10]
                5: [1.2, 1.4],  # Was [6, 7]
                6: [1.0, 1.2],  # Was [5, 6]
                7: [0.8, 0.9]   # Was [4, 4.5]
            },
            'authentication': {
                # Keep these similar, they're already small
                1: [0.080, 0.085],
                2: [0.135, 0.140],
                3: [0.155, 0.160]
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
        """Compute priorities for tasks using upward rank method with security overhead."""
        # First, calculate the average execution time for each task
        for task in self.tasks:
            task.avg_execution = sum(task.execution_times) / len(task.execution_times)
        
        # Calculate minimum security overhead for each task
        def calculate_min_security_overhead(task):
            total_min_overhead = 0
            
            # For incoming messages
            for pred_id in task.predecessors:
                message = self.get_message(pred_id, task.task_id)
                if message:
                    # Calculate minimum security overhead for this message
                    min_conf_overhead = float('inf')
                    min_integ_overhead = float('inf')
                    min_auth_overhead = float('inf')
                    
                    # Find minimum confidentiality overhead that meets requirements
                    for i, strength in enumerate(self.security.strengths['confidentiality']):
                        if strength >= message.min_security['confidentiality']:
                            # Average overhead across processors
                            avg_overhead = sum(self.security.overheads['confidentiality'][i+1]) / len(self.security.overheads['confidentiality'][i+1])
                            min_conf_overhead = min(min_conf_overhead, avg_overhead)
                    
                    # Find minimum integrity overhead that meets requirements
                    for i, strength in enumerate(self.security.strengths['integrity']):
                        if strength >= message.min_security['integrity']:
                            # Average overhead across processors
                            avg_overhead = sum(self.security.overheads['integrity'][i+1]) / len(self.security.overheads['integrity'][i+1])
                            min_integ_overhead = min(min_integ_overhead, avg_overhead)
                    
                    # Find minimum authentication overhead that meets requirements
                    for i, strength in enumerate(self.security.strengths['authentication']):
                        if strength >= message.min_security['authentication']:
                            # Average overhead across processors
                            avg_overhead = sum(self.security.overheads['authentication'][i+1]) / len(self.security.overheads['authentication'][i+1])
                            min_auth_overhead = min(min_auth_overhead, avg_overhead)
                    
                    # Add processing overhead plus transmission overhead
                    if min_conf_overhead != float('inf'):
                        total_min_overhead += min_conf_overhead + message.size / (self.security.strengths['confidentiality'][0] * self.network.bandwidth)
                    if min_integ_overhead != float('inf'):
                        total_min_overhead += min_integ_overhead + message.size / (self.security.strengths['integrity'][0] * self.network.bandwidth)
                    if min_auth_overhead != float('inf'):
                        total_min_overhead += min_auth_overhead + message.size / (self.security.strengths['authentication'][0] * self.network.bandwidth)
            
            # For outgoing messages
            for succ_id in task.successors:
                message = self.get_message(task.task_id, succ_id)
                if message:
                    # Calculate minimum security overhead for this message (same approach as above)
                    min_conf_overhead = float('inf')
                    min_integ_overhead = float('inf')
                    min_auth_overhead = float('inf')
                    
                    # Find minimum confidentiality overhead that meets requirements
                    for i, strength in enumerate(self.security.strengths['confidentiality']):
                        if strength >= message.min_security['confidentiality']:
                            # Average overhead across processors
                            avg_overhead = sum(self.security.overheads['confidentiality'][i+1]) / len(self.security.overheads['confidentiality'][i+1])
                            min_conf_overhead = min(min_conf_overhead, avg_overhead)
                    
                    # Find minimum integrity overhead that meets requirements
                    for i, strength in enumerate(self.security.strengths['integrity']):
                        if strength >= message.min_security['integrity']:
                            # Average overhead across processors
                            avg_overhead = sum(self.security.overheads['integrity'][i+1]) / len(self.security.overheads['integrity'][i+1])
                            min_integ_overhead = min(min_integ_overhead, avg_overhead)
                    
                    # Find minimum authentication overhead that meets requirements
                    for i, strength in enumerate(self.security.strengths['authentication']):
                        if strength >= message.min_security['authentication']:
                            # Average overhead across processors
                            avg_overhead = sum(self.security.overheads['authentication'][i+1]) / len(self.security.overheads['authentication'][i+1])
                            min_auth_overhead = min(min_auth_overhead, avg_overhead)
                    
                    # Add processing overhead plus transmission overhead
                    if min_conf_overhead != float('inf'):
                        total_min_overhead += min_conf_overhead + message.size / (self.security.strengths['confidentiality'][0] * self.network.bandwidth)
                    if min_integ_overhead != float('inf'):
                        total_min_overhead += min_integ_overhead + message.size / (self.security.strengths['integrity'][0] * self.network.bandwidth)
                    if min_auth_overhead != float('inf'):
                        total_min_overhead += min_auth_overhead + message.size / (self.security.strengths['authentication'][0] * self.network.bandwidth)
            
            return total_min_overhead
        
        # Calculate priority (upward rank) for each task
        def calculate_upward_rank(task):
            if task.priority is not None:
                return task.priority
            
            # Calculate security overhead for this task
            security_overhead = calculate_min_security_overhead(task)
            
            if not task.successors:  # Exit task
                task.priority = task.avg_execution + security_overhead
                return task.priority
            
            max_successor_rank = 0
            for succ_id in task.successors:
                succ_task = self.get_task_by_id(succ_id)
                if succ_task:
                    # Get communication cost between task and successor
                    message = self.get_message(task.task_id, succ_id)
                    comm_cost = 0
                    if message:
                        # Average communication time
                        comm_cost = message.size / self.network.bandwidth
                    
                    succ_rank = calculate_upward_rank(succ_task)
                    rank_with_comm = comm_cost + succ_rank
                    max_successor_rank = max(max_successor_rank, rank_with_comm)
            
            # According to Equation 11: ρ[gj] = C̄j + max(comm̄j,k + ρ[gk]) + S̄Oj
            task.priority = task.avg_execution + max_successor_rank + security_overhead
            return task.priority
        
        # Calculate upward rank for all tasks
        for task in self.tasks:
            calculate_upward_rank(task)
            print(f"Task {task.task_id} priority: {task.priority}")
            
            
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
        # If source and destination are on the same processor, no overhead
        if source_proc == dest_proc:
            return 0
            
        total_overhead = 0
        
        # Add overhead for confidentiality and integrity (data-dependent)
        for service in ['confidentiality', 'integrity']:
            protocol_idx = message.assigned_security[service] + 1  # 1-indexed in the overhead table
            # Get overhead factor for the processor that's encoding (source)
            overhead_factor = self.security.overheads[service][protocol_idx][source_proc - 1]
            overhead = (message.size / 1024) * overhead_factor  # Convert KB to MB if needed
            total_overhead += overhead
        
        # Add overhead for authentication (data-independent)
        auth_protocol_idx = message.assigned_security['authentication'] + 1
        # Authentication overhead applies to the receiving processor
        auth_overhead = self.security.overheads['authentication'][auth_protocol_idx][dest_proc - 1]
        total_overhead += auth_overhead
        
        return total_overhead

    def calculate_est(self, task, processor):
        """Calculate Earliest Start Time for a task on a processor with security overhead."""
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
                # Calculate security overhead for inter-processor messages only
                sender_security_overhead = self.calc_security_overhead(
                    message, 
                    pred_task.assigned_processor, 
                    processor
                )
                
                # Calculate communication time
                comm_time = self.network.get_communication_time(
                    message, 
                    pred_task.assigned_processor, 
                    processor
                )
                
                # Total communication finish time includes:
                # 1. Predecessor task finish time
                # 2. Time for the message to travel over the network
                # 3. Security overhead for inter-processor communication
                comm_finish_time = pred_task.finish_time + comm_time + sender_security_overhead
            
            max_pred_finish_time = max(max_pred_finish_time, comm_finish_time)
        
        return max(processor_ready_time, max_pred_finish_time)
    
class HSMS(Scheduler):
    """Heterogeneous Security-aware Makespan minimizing Scheduler"""
    def finalize_schedule(self):
        """Recalculate task finish times after all task placements are determined."""
        # Sort tasks by start time to process them in order
        sorted_tasks = sorted(self.tasks, key=lambda t: t.start_time if t.start_time is not None else float('inf'))
        
        # Reset processor available times
        for proc in self.processors:
            proc.available_time = 0
        
        # Recalculate start and finish times with actual security overheads
        for task in sorted_tasks:
            if not task.is_scheduled:
                continue
                
            processor_id = task.assigned_processor
            processor = self.processors[processor_id - 1]
            
            # Calculate earliest start time based on predecessors
            est = 0
            for pred_id in task.predecessors:
                pred_task = self.get_task_by_id(pred_id)
                if not pred_task.is_scheduled:
                    continue
                    
                message = self.get_message(pred_id, task.task_id)
                if not message:
                    continue
                    
                # If on same processor, just use predecessor finish time - no security or communication overhead
                if pred_task.assigned_processor == processor_id:
                    pred_finish = pred_task.finish_time
                else:
                    # Calculate actual communication time with security overhead for inter-processor communication
                    comm_time = self.network.get_communication_time(message, pred_task.assigned_processor, processor_id)
                    security_overhead = self.calc_security_overhead(message, pred_task.assigned_processor, processor_id)
                    
                    # Cross-processor communication includes security overhead
                    pred_finish = pred_task.finish_time + comm_time + security_overhead
                
                est = max(est, pred_finish)
            
            # Also consider processor availability
            est = max(est, processor.available_time)
            
            # Calculate execution time including security overhead
            base_exec_time = task.execution_times[processor_id - 1]
            
            # Calculate total security overhead for this task
            total_security_overhead = 0
            
            # Overhead as receiver from predecessors on different processors ONLY
            for pred_id in task.predecessors:
                pred_task = self.get_task_by_id(pred_id)
                if (pred_task and pred_task.is_scheduled and 
                    pred_task.assigned_processor != processor_id):  # Only for inter-processor messages
                    message = self.get_message(pred_id, task.task_id)
                    if message:
                        # Apply receiver authentication overhead
                        auth_protocol_idx = message.assigned_security['authentication'] + 1
                        auth_overhead = self.security.overheads['authentication'][auth_protocol_idx][processor_id - 1]
                        total_security_overhead += auth_overhead
            
            # Overhead as sender to successors on different processors ONLY
            for succ_id in task.successors:
                succ_task = self.get_task_by_id(succ_id)
                if (succ_task and succ_task.is_scheduled and 
                    succ_task.assigned_processor != processor_id):  # Only for inter-processor messages
                    message = self.get_message(task.task_id, succ_id)
                    if message:
                        # Apply sender confidentiality and integrity overhead
                        for service in ['confidentiality', 'integrity']:
                            protocol_idx = message.assigned_security[service] + 1
                            overhead_factor = self.security.overheads[service][protocol_idx][processor_id - 1]
                            overhead = (message.size / 1024) * overhead_factor
                            total_security_overhead += overhead
            
            # Update task timing
            task.start_time = est
            task.finish_time = est + base_exec_time + total_security_overhead
            processor.available_time = task.finish_time
            
            # Update schedule entry
            for entry in self.schedule:
                if entry['task_id'] == task.task_id:
                    entry['start_time'] = task.start_time
                    entry['finish_time'] = task.finish_time
        
        # Update the makespan
        self.makespan = max(task.finish_time for task in self.tasks if task.is_scheduled)
        return self.makespan
    def schedule_tasks(self):
        """Schedule tasks using security-aware HEFT algorithm with proper security overhead accounting."""
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
                
                # Base execution time without security overhead
                base_exec_time = task.execution_times[processor.proc_id - 1]
                
                # Calculate security overhead for this task as a receiver
                receiver_overhead = 0
                for pred_id in task.predecessors:
                    pred_task = self.get_task_by_id(pred_id)
                    if (pred_task and pred_task.is_scheduled and 
                        pred_task.assigned_processor != processor.proc_id):  # Only inter-processor
                        message = self.get_message(pred_id, task.task_id)
                        if message:
                            # Only calculate receiver overhead for cross-processor communication
                            # Specifically for authentication, which is applied at the receiver
                            auth_protocol_idx = message.assigned_security['authentication'] + 1
                            auth_overhead = self.security.overheads['authentication'][auth_protocol_idx][processor.proc_id - 1]
                            receiver_overhead += auth_overhead
                
                # Calculate security overhead for this task as a sender
                sender_overhead = 0
                for succ_id in task.successors:
                    succ_task = self.get_task_by_id(succ_id)
                    # We don't know where successors will be scheduled yet, so we estimate
                    # assuming they might be on a different processor
                    if succ_task:
                        message = self.get_message(task.task_id, succ_id)
                        if message:
                            # For already scheduled successors, we can check if they're on different processors
                            if succ_task.is_scheduled and succ_task.assigned_processor == processor.proc_id:
                                continue  # Skip intra-processor messages
                                
                            # For unscheduled successors, we still estimate overhead
                            # This may overestimate if successors end up on same processor
                            for service in ['confidentiality', 'integrity']:
                                protocol_idx = message.assigned_security[service] + 1
                                overhead_factor = self.security.overheads[service][protocol_idx][processor.proc_id - 1]
                                overhead = (message.size / 1024) * overhead_factor
                                sender_overhead += overhead
                
                # Total execution time includes base time plus security overheads
                total_exec_time = base_exec_time + receiver_overhead + sender_overhead
                
                # Calculate finish time
                eft = est + total_exec_time
                
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
        self.finalize_schedule()
        return self.makespan

    def calculate_security_utility(self):
        """Calculate the total security utility for inter-processor messages only."""
        total_utility = 0
        for message in self.messages:
            # Get source and destination tasks
            source_task = self.get_task_by_id(message.source_id)
            dest_task = self.get_task_by_id(message.dest_id)
            
            # Skip if either task is not scheduled or if they're on the same processor
            if (not source_task or not dest_task or 
                not source_task.is_scheduled or not dest_task.is_scheduled or
                source_task.assigned_processor == dest_task.assigned_processor):
                continue
            
            # Calculate utility only for inter-processor messages
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
    def display_security_information(self, algorithm_name):
        """Display security levels for all message communications"""
        print(f"\n--- {algorithm_name} Security Information ---")
        print("Message ID | Source → Destination | Confidentiality | Integrity | Authentication")
        print("-" * 75)
        
        # Define security protocols - adjust these based on your actual implementation
        security_protocols = {
            'confidentiality': ['C0', 'C1', 'C2', 'C3','C4','C5', 'C6', 'C7'],
            'integrity': ['IO', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6'],
            'authentication': ['A0', 'A1', 'A2']
        }
        
        for message in self.messages:
            source_task = self.get_task_by_id(message.source_id)
            dest_task = self.get_task_by_id(message.dest_id)
            
            if source_task and dest_task:
                # Safe access to security level with bounds checking
                conf_idx = message.assigned_security['confidentiality']
                integ_idx = message.assigned_security['integrity']
                auth_idx = message.assigned_security['authentication']
                
                # Ensure indices are within bounds
                conf_level = security_protocols['confidentiality'][conf_idx] if 0 <= conf_idx < len(security_protocols['confidentiality']) else 'Unknown'
                integ_level = security_protocols['integrity'][integ_idx] if 0 <= integ_idx < len(security_protocols['integrity']) else 'Unknown'
                auth_level = security_protocols['authentication'][auth_idx] if 0 <= auth_idx < len(security_protocols['authentication']) else 'Unknown'
                
                print(f"{message.id:9} | {source_task.name:5} → {dest_task.name:5} | {conf_level:14} | {integ_level:9} | {auth_level:13}")
        print()
    def print_task_schedule(self):
        """
        Prints detailed information about each scheduled task including:
        - Task ID and name
        - Assigned processor
        - Start time
        - Base execution time
        - Time increases due to security measures (confidentiality, integrity, authentication)
        - Total security overhead
        - Actual finish time
        """
        print("\n" + "="*100)
        print("DETAILED TASK SCHEDULE")
        print("="*100)
        print(f"{'ID':<5} {'Task Name':<20} {'Proc':<5} {'Start':<8} {'Exec':<8} {'Conf Δ':<8} {'Integ Δ':<8} {'Auth Δ':<8} {'Overhead':<8} {'Finish':<8}")
        print("-"*100)
        
        # Sort tasks by start time for a chronological view
        sorted_tasks = sorted(self.tasks, key=lambda x: (x.start_time if x.start_time is not None else float('inf')))
        
        total_base_execution = 0
        total_security_overhead = 0
        
        for task in sorted_tasks:
            if not task.is_scheduled:
                continue
                
            # Get base execution time without security overhead
            base_exec_time = task.execution_times[task.assigned_processor - 1]
            total_base_execution += base_exec_time
            
            # Calculate security overheads for each message related to this task
            conf_overhead = 0
            integ_overhead = 0
            auth_overhead = 0
            
            # Find all messages associated with this task
            related_messages = []
            for message in self.messages:
                if message.source_id == task.task_id or message.dest_id == task.task_id:
                    related_messages.append(message)
            
            # Calculate security overheads
            for message in related_messages:
                # Confidentiality overhead is data-dependent (χ2)
                if message.assigned_security['confidentiality'] > 0:
                    protocol_idx = message.assigned_security['confidentiality'] + 1  # Convert to 1-indexed
                    conf_overhead += (self.security.overheads['confidentiality'][protocol_idx][task.assigned_processor - 1] 
                                    * message.size / 1000)  # Adjust for message size
                
                # Integrity overhead is data-dependent (χ2)
                if message.assigned_security['integrity'] > 0:
                    protocol_idx = message.assigned_security['integrity'] + 1  # Convert to 1-indexed
                    integ_overhead += (self.security.overheads['integrity'][protocol_idx][task.assigned_processor - 1] 
                                    * message.size / 1000)  # Adjust for message size
                
                # Authentication overhead is data-independent (χ1)
                if message.assigned_security['authentication'] > 0:
                    protocol_idx = message.assigned_security['authentication'] + 1  # Convert to 1-indexed
                    auth_overhead += self.security.overheads['authentication'][protocol_idx][task.assigned_processor - 1]
            
            # Calculate total security overhead
            total_overhead = conf_overhead + integ_overhead + auth_overhead
            total_security_overhead += total_overhead
            
            # Print task details
            print(f"{task.task_id:<5} {task.name[:20]:<20} {task.assigned_processor:<5} "
                f"{task.start_time:<8.2f} {base_exec_time:<8.2f} "
                f"{conf_overhead:<8.2f} {integ_overhead:<8.2f} {auth_overhead:<8.2f} "
                f"{total_overhead:<8.2f} {task.finish_time:<8.2f}")
        
        print("-"*100)
        # Print summary statistics
        print(f"Total base execution time: {total_base_execution:.2f}")
        print(f"Total security overhead: {total_security_overhead:.2f} ({(total_security_overhead/total_base_execution)*100:.2f}% of base execution)")
        
        # Use makespan instead of best_makespan
        print(f"Makespan: {self.makespan:.2f}")
        print("="*100)
        
        # Return the data if needed for further analysis
        return {
            "total_base_execution": total_base_execution,
            "total_security_overhead": total_security_overhead,
            "makespan": self.makespan
        }
class SHIELD(Scheduler):
    """Security-aware scHedulIng for rEaL-time Dags on heterogeneous systems"""
    
    def __init__(self, tasks, messages, processors, network, security_service, deadline):
        super().__init__(tasks, messages, processors, network, security_service, deadline)
        # Run HSMS first to get initial schedule
        self.hsms = HSMS(copy.deepcopy(tasks), messages, copy.deepcopy(processors), 
                         network, security_service, deadline)
        self.makespan = None
    
    def calculate_benefits(self, message, source_task, dest_task):
        """Calculate security benefit and time penalty for upgrading each security service."""
        benefits = []
        
        source_proc = source_task.assigned_processor
        dest_proc = dest_task.assigned_processor
        
        # Skip intra-processor messages
        if source_proc == dest_proc:
            return benefits
        
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
            # In calculate_benefits, adjust weights for different security services
            if service == 'confidentiality':
                security_benefit = message.weights[service] * strength_diff * 8.0  # Higher priority
            elif service == 'integrity':
                security_benefit = message.weights[service] * strength_diff * 6.0
            else:  # authentication
                security_benefit = message.weights[service] * strength_diff * 4.0 # 5.0 is a scaling factor
            
            # Calculate time penalty - the issue is here
            # We should be calculating the actual overhead difference
            
            if service in ['confidentiality', 'integrity']:
                # These overheads apply to the sender (source task)
                current_protocol = current_protocol_idx + 1  # 1-indexed in overhead table
                next_protocol = next_protocol_idx + 1
                
                # Calculate proper overhead values based on message size
                current_overhead = (message.size / 1024) * self.security.overheads[service][current_protocol][source_proc - 1]
                next_overhead = (message.size / 1024) * self.security.overheads[service][next_protocol][source_proc - 1]
            else:  # authentication
                # Authentication overhead applies to the receiver (dest task)
                current_protocol = current_protocol_idx + 1
                next_protocol = next_protocol_idx + 1
                
                # Authentication is fixed overhead regardless of message size
                current_overhead = self.security.overheads[service][current_protocol][dest_proc - 1]
                next_overhead = self.security.overheads[service][next_protocol][dest_proc - 1]
            
            # The overhead should increase with higher security levels
            # Make sure time_penalty is positive
            time_penalty = next_overhead - current_overhead
            
            # If time_penalty is negative, there's an issue with the overhead table data
            # For the sake of correctness, we'll enforce a minimum penalty
            if time_penalty < 0:
                # Fix: Replace negative penalties with a minimum positive value
                # This ensures stronger security always has some cost
                time_penalty = 0.1  # Small positive value
            
            # Calculate benefit-to-cost ratio
            if time_penalty > 0:
                benefit_cost_ratio = security_benefit / time_penalty
            else:
                benefit_cost_ratio = float('inf')  # Free security upgrade (shouldn't happen now)
            
            benefits.append({
                'message': message,
                'service': service,
                'protocol_idx': next_protocol_idx,
                'security_benefit': security_benefit,
                'time_penalty': time_penalty,
                'benefit_cost_ratio': benefit_cost_ratio
            })
        
        return benefits
    
    def test_upgrade_feasibility(self, upgrade):
        """Test if applying an upgrade still meets the deadline constraint."""
        # Save the current state
        message = upgrade['message']
        service = upgrade['service']
        prev_level = message.assigned_security[service]
        
        # Temporarily apply the upgrade
        message.assigned_security[service] = upgrade['protocol_idx']
        
        try:
            # Recalculate makespan
            new_makespan = self.recalculate_makespan_after_upgrade()
            
            # Calculate the actual time difference to ensure it's not decreasing
            time_difference = new_makespan - self.makespan
            
            # If the new makespan is somehow less than the current makespan,
            # this is likely due to inaccuracies in our estimation or inconsistencies in data
            # To be safe, we'll use the time penalty from the upgrade calculation
            if time_difference < 0:
                # Use the calculated time penalty instead
                new_makespan = self.makespan + upgrade['time_penalty']
            
            is_feasible = new_makespan <= self.deadline
            return is_feasible, new_makespan
        finally:
            # Always restore original security level
            message.assigned_security[service] = prev_level
    # In the SHIELD.test_upgrade_feasibility method:
    # def test_upgrade_feasibility(self, upgrade):
    #     """Test if applying an upgrade still meets the deadline constraint."""
    #     # Simplified approach: just check if the time penalty fits within slack
    #     time_penalty = upgrade['time_penalty']
    #     new_makespan = self.makespan + time_penalty
    #     return new_makespan <= self.deadline, new_makespan
    def recalculate_makespan_after_upgrade(self):
        """Recalculate makespan after security upgrades."""
        # Make a deep copy of tasks to work with
        tasks_copy = copy.deepcopy(self.tasks)
        
        # Reset processor available times
        for proc in self.processors:
            proc.available_time = 0
        
        # Sort tasks by their original start times to process them in order
        sorted_tasks = sorted(tasks_copy, key=lambda t: t.start_time if t.start_time is not None else float('inf'))
        
        # Process tasks in order
        for task in sorted_tasks:
            if not task.is_scheduled:
                continue
            
            processor_id = task.assigned_processor
            processor = self.processors[processor_id - 1]
            
            # Calculate earliest start time based on predecessors
            est = 0
            for pred_id in task.predecessors:
                pred_task = next((t for t in tasks_copy if t.task_id == pred_id), None)
                if not pred_task or not pred_task.is_scheduled:
                    continue
                
                # If on same processor, just use predecessor finish time
                if pred_task.assigned_processor == processor_id:
                    pred_finish = pred_task.finish_time
                else:
                    # Calculate communication and security overhead
                    message = self.get_message(pred_id, task.task_id)
                    if message:
                        comm_time = self.network.get_communication_time(
                            message, pred_task.assigned_processor, processor_id
                        )
                        security_overhead = self.calc_security_overhead(
                            message, pred_task.assigned_processor, processor_id
                        )
                        pred_finish = pred_task.finish_time + comm_time + security_overhead
                    else:
                        pred_finish = pred_task.finish_time
                
                est = max(est, pred_finish)
            
            # Also consider processor availability
            est = max(est, processor.available_time)
            
            # Calculate execution time including security overheads
            base_exec_time = task.execution_times[processor_id - 1]
            
            # Calculate total security overhead for this task
            total_security_overhead = 0
            
            # Overhead as receiver from predecessors on different processors
            for pred_id in task.predecessors:
                pred_task = next((t for t in tasks_copy if t.task_id == pred_id), None)
                if (pred_task and pred_task.is_scheduled and 
                    pred_task.assigned_processor != processor_id):
                    message = self.get_message(pred_id, task.task_id)
                    if message:
                        # Authentication overhead at receiver
                        auth_protocol_idx = message.assigned_security['authentication'] + 1
                        auth_overhead = self.security.overheads['authentication'][auth_protocol_idx][processor_id - 1]
                        total_security_overhead += auth_overhead
            
            # Overhead as sender to successors on different processors
            for succ_id in task.successors:
                succ_task = next((t for t in tasks_copy if t.task_id == succ_id), None)
                if succ_task and succ_task.is_scheduled and succ_task.assigned_processor != processor_id:
                    message = self.get_message(task.task_id, succ_id)
                    if message:
                        # Confidentiality and integrity overhead at sender
                        for service in ['confidentiality', 'integrity']:
                            protocol_idx = message.assigned_security[service] + 1
                            overhead_factor = self.security.overheads[service][protocol_idx][processor_id - 1]
                            overhead = (message.size / 1024) * overhead_factor
                            total_security_overhead += overhead
            
            # Update task timing
            task.start_time = est
            task.finish_time = est + base_exec_time + total_security_overhead
            processor.available_time = task.finish_time
        
        # Determine makespan
        return max(task.finish_time for task in tasks_copy if task.is_scheduled)

    def apply_security_upgrade(self, upgrade):
        """Apply a security upgrade to a message."""
        message = upgrade['message']
        message.assigned_security[upgrade['service']] = upgrade['protocol_idx']
        return message
        
    def calculate_security_utility(self):
        """Modified security utility calculation with higher emphasis on security levels"""
        total_utility = 0
        for message in self.messages:
            source_task = self.get_task_by_id(message.source_id)
            dest_task = self.get_task_by_id(message.dest_id)
            
            if (source_task and dest_task and 
                source_task.is_scheduled and dest_task.is_scheduled and
                source_task.assigned_processor != dest_task.assigned_processor):
                
                # Square the strength values to emphasize higher security levels
                message_utility = 0
                for service in ['confidentiality', 'integrity', 'authentication']:
                    protocol_idx = message.assigned_security[service]
                    strength = self.security.strengths[service][protocol_idx]
                    weight = message.weights[service]
                    # Square the strength to give higher weight to stronger protocols
                    message_utility += weight * (strength ** 2)
                total_utility += message_utility
                
        return total_utility
    
    def run(self):
        """Run the SHIELD scheduler with simplified security upgrade phase."""
        # First run HSMS to get initial schedule
        start_time = time.time()
        hsms_makespan, hsms_security_utility = self.hsms.run()
        hsms_runtime = time.time() - start_time
        
        if hsms_makespan is None:
            print("SHIELD cannot proceed since HSMS failed to meet the deadline.")
            return None, None
        
        print(f"HSMS Completed Execution in {hsms_runtime:.4f} seconds")
        print(f"Initial HSMS makespan: {hsms_makespan}, security utility: {hsms_security_utility}")
        
        # Copy HSMS results to SHIELD
        for task in self.tasks:
            hsms_task = next((t for t in self.hsms.tasks if t.task_id == task.task_id), None)
            if hsms_task:
                task.assigned_processor = hsms_task.assigned_processor
                task.start_time = hsms_task.start_time
                task.finish_time = hsms_task.finish_time
                task.is_scheduled = hsms_task.is_scheduled
        
        # Copy the schedule
        self.schedule = copy.deepcopy(self.hsms.schedule)
        self.makespan = hsms_makespan
        
        # Security upgrade phase
        current_makespan = hsms_makespan
        slack_time = self.deadline - current_makespan
        print(f"Initial slack time: {slack_time}")
        
        # Track security upgrades
        applied_upgrades = []
        upgrade_iterations = 0
        max_iterations = 50  # Reduced safety limit
        
        while slack_time > 0 and upgrade_iterations < max_iterations:
            upgrade_iterations += 1
            print(f"Upgrade iteration {upgrade_iterations}")
            
            # Find all potential security upgrades
            all_benefits = []
            for message in self.messages:
                source_task = self.get_task_by_id(message.source_id)
                dest_task = self.get_task_by_id(message.dest_id)
                
                if (source_task and dest_task and 
                    source_task.is_scheduled and dest_task.is_scheduled and
                    source_task.assigned_processor != dest_task.assigned_processor):
                    
                    benefits = self.calculate_benefits(message, source_task, dest_task)
                    all_benefits.extend(benefits)
            
            print(f"Found {len(all_benefits)} potential upgrades")
            if not all_benefits:
                print("No more security upgrades possible")
                break
                
            # Sort by benefit-to-cost ratio
            all_benefits.sort(key=lambda x: -x['benefit_cost_ratio'])
            
            # Try to apply the best upgrade
            applied_upgrade = False
            for upgrade in all_benefits[:10]:  # Try only top 10 for efficiency
                service_name = upgrade['service']
                protocol_idx = upgrade['protocol_idx']
                benefit = upgrade['security_benefit']
                penalty = upgrade['time_penalty']
                
                print(f"Testing upgrade: {upgrade['message'].id}, {service_name}, level {protocol_idx}")
                print(f"  Benefit: {benefit:.4f}, Penalty: {penalty:.4f}")
                
                # Only consider upgrades that consume less than the available slack
                if penalty > slack_time:
                    print(f"  Skipping - penalty {penalty:.2f} exceeds slack {slack_time:.2f}")
                    continue
                
                # Test feasibility
                is_feasible, new_makespan = self.test_upgrade_feasibility(upgrade)
                print(f"  Feasible: {is_feasible}, New makespan: {new_makespan:.2f}")
                
                if is_feasible:
                    # Apply the upgrade permanently
                    self.apply_security_upgrade(upgrade)
                    
                    # Update makespan and slack
                    current_makespan = new_makespan
                    slack_time = self.deadline - current_makespan
                    
                    # Record the upgrade
                    applied_upgrades.append({
                        'message_id': upgrade['message'].id,
                        'service': service_name,
                        'old_level': protocol_idx - 1,
                        'new_level': protocol_idx,
                        'benefit': benefit,
                        'penalty': penalty
                    })
                    
                    print(f"  Applied upgrade. New makespan: {current_makespan:.2f}, slack: {slack_time:.2f}")
                    applied_upgrade = True
                    break
            
            if not applied_upgrade:
                print("No feasible upgrades found. Stopping.")
                break
        
        # Final calculation
        self.makespan = current_makespan
        security_utility = self.calculate_security_utility()
        
        print(f"SHIELD final makespan: {self.makespan:.2f}, security utility: {security_utility:.4f}")
        print(f"Applied {len(applied_upgrades)} security upgrades")
        
        return self.makespan, security_utility
    def display_security_information(self, algorithm_name):
        """Display security levels for all message communications"""
        print(f"\n--- {algorithm_name} Security Information ---")
        print("Message ID | Source → Destination | Confidentiality | Integrity | Authentication")
        print("-" * 75)
        
        # Define security protocols - adjust these based on your actual implementation
        security_protocols = {
            'confidentiality': ['C0', 'C1', 'C2', 'C3','C4','C5', 'C6', 'C7'],
            'integrity': ['IO', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6'],
            'authentication': ['A0', 'A1', 'A2']
        }
        
        for message in self.messages:
            source_task = self.get_task_by_id(message.source_id)
            dest_task = self.get_task_by_id(message.dest_id)
            
            if source_task and dest_task:
                # Safe access to security level with bounds checking
                conf_idx = message.assigned_security['confidentiality']
                integ_idx = message.assigned_security['integrity']
                auth_idx = message.assigned_security['authentication']
                
                # Ensure indices are within bounds
                conf_level = security_protocols['confidentiality'][conf_idx] if 0 <= conf_idx < len(security_protocols['confidentiality']) else 'Unknown'
                integ_level = security_protocols['integrity'][integ_idx] if 0 <= integ_idx < len(security_protocols['integrity']) else 'Unknown'
                auth_level = security_protocols['authentication'][auth_idx] if 0 <= auth_idx < len(security_protocols['authentication']) else 'Unknown'
                
                print(f"{message.id:9} | {source_task.name:5} → {dest_task.name:5} | {conf_level:14} | {integ_level:9} | {auth_level:13}")
        print()
    def print_task_schedule(self):
        """
        Prints detailed information about each scheduled task including:
        - Task ID and name
        - Assigned processor
        - Start time
        - Base execution time
        - Time increases due to security measures (confidentiality, integrity, authentication)
        - Total security overhead
        - Actual finish time
        """
        print("\n" + "="*100)
        print("DETAILED TASK SCHEDULE")
        print("="*100)
        print(f"{'ID':<5} {'Task Name':<20} {'Proc':<5} {'Start':<8} {'Exec':<8} {'Conf Δ':<8} {'Integ Δ':<8} {'Auth Δ':<8} {'Overhead':<8} {'Finish':<8}")
        print("-"*100)
        
        # Sort tasks by start time for a chronological view
        sorted_tasks = sorted(self.tasks, key=lambda x: (x.start_time if x.start_time is not None else float('inf')))
        
        total_base_execution = 0
        total_security_overhead = 0
        
        for task in sorted_tasks:
            if not task.is_scheduled:
                continue
                
            # Get base execution time without security overhead
            base_exec_time = task.execution_times[task.assigned_processor - 1]
            total_base_execution += base_exec_time
            
            # Calculate security overheads for each message related to this task
            conf_overhead = 0
            integ_overhead = 0
            auth_overhead = 0
            
            # Find all messages associated with this task
            related_messages = []
            for message in self.messages:
                if message.source_id == task.task_id or message.dest_id == task.task_id:
                    related_messages.append(message)
            
            # Calculate security overheads
            for message in related_messages:
                # Confidentiality overhead is data-dependent (χ2)
                if message.assigned_security['confidentiality'] > 0:
                    protocol_idx = message.assigned_security['confidentiality'] + 1  # Convert to 1-indexed
                    conf_overhead += (self.security.overheads['confidentiality'][protocol_idx][task.assigned_processor - 1] 
                                    * message.size / 1000)  # Adjust for message size
                
                # Integrity overhead is data-dependent (χ2)
                if message.assigned_security['integrity'] > 0:
                    protocol_idx = message.assigned_security['integrity'] + 1  # Convert to 1-indexed
                    integ_overhead += (self.security.overheads['integrity'][protocol_idx][task.assigned_processor - 1] 
                                    * message.size / 1000)  # Adjust for message size
                
                # Authentication overhead is data-independent (χ1)
                if message.assigned_security['authentication'] > 0:
                    protocol_idx = message.assigned_security['authentication'] + 1  # Convert to 1-indexed
                    auth_overhead += self.security.overheads['authentication'][protocol_idx][task.assigned_processor - 1]
            
            # Calculate total security overhead
            total_overhead = conf_overhead + integ_overhead + auth_overhead
            total_security_overhead += total_overhead
            
            # Print task details
            print(f"{task.task_id:<5} {task.name[:20]:<20} {task.assigned_processor:<5} "
                f"{task.start_time:<8.2f} {base_exec_time:<8.2f} "
                f"{conf_overhead:<8.2f} {integ_overhead:<8.2f} {auth_overhead:<8.2f} "
                f"{total_overhead:<8.2f} {task.finish_time:<8.2f}")
        
        print("-"*100)
        # Print summary statistics
        print(f"Total base execution time: {total_base_execution:.2f}")
        print(f"Total security overhead: {total_security_overhead:.2f} ({(total_security_overhead/total_base_execution)*100:.2f}% of base execution)")
        
        # Use makespan instead of best_makespan
        print(f"Makespan: {self.makespan:.2f}")
        print("="*100)
        
        # Return the data if needed for further analysis
        return {
            "total_base_execution": total_base_execution,
            "total_security_overhead": total_security_overhead,
            "makespan": self.makespan
        }



# Create the TC system test case as described in the paper
def create_tc_test_case():
    # Create tasks with execution times as in Figure 10(c)
    tasks = [
        Task(1, "Wheel Speed 1", [205, 200]),
        Task(2, "Wheel Speed 2", [207, 200]),
        Task(3, "Wheel Speed 3", [190, 210]),
        Task(4, "Wheel Speed 4", [200, 198]),
        Task(5, "Driver Input", [150, 155]),
        Task(6, "Slip Calculator", [297, 300], [1, 2, 3, 4]),
        Task(7, "Valve Control", [175, 180]),
        Task(8, "Actuator", [405,400], [5, 6, 7]),
        Task(9, "Brake Control", [146, 154], [8]),
        Task(10, "Throttle Control", [199,201], [8])
    ]
    
    # Create messages as in Figure 10(b)
    messages = [
        Message(1, 6, 512),
        Message(2, 6, 512),
        Message(3, 6, 512),
        Message(4, 6, 512),
        Message(5, 8, 128),
        Message(6, 8, 128),
        Message(7, 8, 64),
        Message(8, 9, 512),
        Message(8, 10, 512)
    ]
    
    security_values = [
        (0.2, 0.1, 0.4, 0.3, 0.3, 0.4),
        (0.2, 0.2, 0.4, 0.3, 0.5, 0.2),
        (0.2, 0.5, 0.3, 0.2, 0.6, 0.2),
        (0.3, 0.4, 0.2, 0.2, 0.2, 0.6),
        (0.4, 0.3, 0.1, 0.2, 0.3, 0.5),
        (0.4, 0.2, 0.4, 0.7, 0.1, 0.2),
        (0.4, 0.1, 0.3, 0.7, 0.1, 0.2),
        (0.3, 0.1, 0.2, 0.7, 0.1, 0.2),
        (0.3, 0.2, 0.1, 0.7, 0.1, 0.2)
    ]

    for msg, (conf_min, integ_min, auth_min, conf_w, integ_w, auth_w) in zip(messages, security_values):
        msg.set_security_requirements(conf_min, integ_min, auth_min, conf_w, integ_w, auth_w)

    processors = [Processor(1), Processor(2)]
    network = CommunicationNetwork(2)
    security_service = SecurityService()
    deadline = 1600  # As specified in the paper
    
    # # return tasks, messages, processors, network, security_service, deadline
    # tasks = [
    #     Task(1, "Source 1", [100, 150]),
    #     Task(2, "Source 2", [120, 90]),
    #     Task(3, "Source 3", [80, 110]),
    #     Task(4, "Process A", [200, 150], [1, 2]),
    #     Task(5, "Process B", [180, 220], [2, 3]),
    #     Task(6, "Merge", [150, 130], [4, 5]),
    #     Task(7, "Output 1", [90, 120], [6]),
    #     Task(8, "Output 2", [110, 80], [6])
    # ]
    
    # # Create messages between tasks
    # # Varying message sizes to create different security overhead impacts
    # messages = [
    #     Message(1, 4, 256),   # Source 1 -> Process A
    #     Message(2, 4, 512),   # Source 2 -> Process A
    #     Message(2, 5, 384),   # Source 2 -> Process B
    #     Message(3, 5, 320),   # Source 3 -> Process B
    #     Message(4, 6, 768),   # Process A -> Merge
    #     Message(5, 6, 640),   # Process B -> Merge
    #     Message(6, 7, 896),   # Merge -> Output 1
    #     Message(6, 8, 448)    # Merge -> Output 2
    # ]
    
    # # Set security requirements with varying weights
    # # Format: set_security_requirements(conf_min, integ_min, auth_min, conf_weight, integ_weight, auth_weight)
    
    # # Message 1: Prioritize confidentiality
    # messages[0].set_security_requirements(0.3, 0.1, 0.1, 0.7, 0.2, 0.1)
    
    # # Message 2: Balanced security needs
    # messages[1].set_security_requirements(0.25, 0.25, 0.2, 0.4, 0.3, 0.3)
    
    # # Message 3: Prioritize integrity
    # messages[2].set_security_requirements(0.1, 0.4, 0.1, 0.2, 0.6, 0.2)
    
    # # Message 4: Low security requirements but high weights
    # messages[3].set_security_requirements(0.15, 0.15, 0.15, 0.3, 0.4, 0.3)
    
    # # Message 5: High security requirements (critical path)
    # messages[4].set_security_requirements(0.4, 0.5, 0.3, 0.4, 0.4, 0.2)
    
    # # Message 6: Medium security requirements
    # messages[5].set_security_requirements(0.3, 0.3, 0.2, 0.3, 0.4, 0.3)
    
    # # Message 7: High confidentiality for output
    # messages[6].set_security_requirements(0.5, 0.2, 0.2, 0.6, 0.2, 0.2)
    
    # # Message 8: High authentication for output
    # messages[7].set_security_requirements(0.2, 0.2, 0.5, 0.3, 0.2, 0.5)
    
    # processors = [Processor(1), Processor(2)]
    # network = CommunicationNetwork(2, bandwidth=1000)  # Higher bandwidth
    # security_service = SecurityService()
    
    # # Set deadline with some slack to allow SHIELD to perform upgrades
    # # Calculate a reasonable deadline based on the critical path plus some slack
    # critical_path_time = 100 + 200 + 150 + 110  # Source 1 -> Process A -> Merge -> Output 2
    # deadline = int(critical_path_time * 1.6)  # 60% slack
    
    return tasks, messages, processors, network, security_service, deadline
def run_comparison():
    # Create test case
    tasks, messages, processors, network, security_service, deadline = create_tc_test_case()
    print(f"Test case created with deadline: {deadline}")
    
    # Run HSMS
    print("\n--- Running HSMS ---")
    hsms_tasks = [Task(t.task_id, t.name, t.execution_times, t.predecessors) for t in tasks]
    hsms_processors = [Processor(p.proc_id) for p in processors]
    hsms = HSMS(hsms_tasks, messages, hsms_processors, network, security_service, deadline)
    hsms_makespan, hsms_security_utility = hsms.run()
    
    if hsms_makespan:
        print(f"HSMS completed with makespan: {hsms_makespan:.2f}, security utility: {hsms_security_utility:.2f}")
        print(f"Slack time: {deadline - hsms_makespan:.2f}")
        hsms.display_security_information("HSMS")
        hsms.print_task_schedule()
    else:
        print("HSMS failed to meet deadline")
        return
    
    # Run SHIELD
    print("\n--- Running SHIELD ---")
    shield_tasks = [Task(t.task_id, t.name, t.execution_times, t.predecessors) for t in tasks]
    shield_processors = [Processor(p.proc_id) for p in processors]
    shield = SHIELD(shield_tasks, messages, shield_processors, network, security_service, deadline)
    shield_makespan, shield_security_utility = shield.run()
    
    if shield_makespan:
        print(f"SHIELD completed with makespan: {shield_makespan:.2f}, security utility: {shield_security_utility:.2f}")
        print(f"Slack time: {deadline - shield_makespan:.2f}")
        shield.display_security_information("SHIELD")
        shield.print_task_schedule()
        
        # Calculate improvement
        security_improvement = ((shield_security_utility - hsms_security_utility) / hsms_security_utility) * 100
        print(f"\nSecurity utility improvement: {security_improvement:.2f}%")
    else:
        print("SHIELD failed to meet deadline")
    
    # Visualize the schedules
    if hsms_makespan and shield_makespan:
        fig1 = plot_gantt_chart("HSMS", hsms.schedule, len(processors), hsms_makespan)
        fig1.savefig("hsms_schedule.png")
        
        fig2 = plot_gantt_chart("SHIELD", shield.schedule, len(processors), shield_makespan)
        fig2.savefig("shield_schedule.png")
        
        # Show both plots
        plt.show()
    
    # Compare security levels chosen by each algorithm
    print("\n--- Security Level Comparison ---")
    print("Message ID | HSMS (C/I/A) | SHIELD (C/I/A) | Security Upgrade")
    print("-" * 60)
    
    # Define security protocols for better readability
    security_protocols = {
        'confidentiality': ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'],
        'integrity': ['I0', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6'],
        'authentication': ['A0', 'A1', 'A2']
    }
    
    for msg in messages:
        # Find this message's security assignments in both algorithms
        hsms_msg = next((m for m in messages if m.id == msg.id), None)
        shield_msg = next((m for m in messages if m.id == msg.id), None)
        
        if hsms_msg and shield_msg:
            # Get indices for each security service
            hsms_conf = hsms_msg.assigned_security['confidentiality']
            hsms_integ = hsms_msg.assigned_security['integrity']
            hsms_auth = hsms_msg.assigned_security['authentication']
            
            shield_conf = shield_msg.assigned_security['confidentiality']
            shield_integ = shield_msg.assigned_security['integrity']
            shield_auth = shield_msg.assigned_security['authentication']
            
            # Check if SHIELD upgraded any service
            conf_upgrade = "↑" if shield_conf > hsms_conf else " "
            integ_upgrade = "↑" if shield_integ > hsms_integ else " "
            auth_upgrade = "↑" if shield_auth > hsms_auth else " "
            
            # Convert indices to protocol names for readability
            hsms_conf_name = security_protocols['confidentiality'][hsms_conf]
            hsms_integ_name = security_protocols['integrity'][hsms_integ]
            hsms_auth_name = security_protocols['authentication'][hsms_auth]
            
            shield_conf_name = security_protocols['confidentiality'][shield_conf]
            shield_integ_name = security_protocols['integrity'][shield_integ]
            shield_auth_name = security_protocols['authentication'][shield_auth]
            
            print(f"{msg.id:9} | {hsms_conf_name}/{hsms_integ_name}/{hsms_auth_name:2} | {shield_conf_name}/{shield_integ_name}/{shield_auth_name:2} | {conf_upgrade}  {integ_upgrade}  {auth_upgrade}")

def plot_gantt_chart(title, schedule, num_processors, makespan):
    """Create a Gantt chart for the schedule."""
    import numpy as np
    
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
    ax.set_title(f"{title} - Makespan: {makespan:.2f} ms")
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
    hsms = HSMS(tasks, messages, processors, network, security_service, deadline)
    hsms_makespan, hsms_security_utility = hsms.run()
    
    if hsms_makespan:
        plot_gantt_chart("HSMS", hsms.schedule, len(processors), hsms_makespan)
        print(f"HSMS Security Utility: {hsms_security_utility:.2f}")
    
    # Run SHIELD
    # Note: We need to create fresh copies of the tasks and processors
    tasks_shield, messages, processors_shield, network, security_service, deadline = create_tc_test_case()
    hsms.print_task_schedule()
    
    shield = SHIELD(tasks_shield, messages, processors_shield, network, security_service, deadline)
    shield_makespan, shield_security_utility = shield.run()
    shield.print_task_schedule()
    if shield_makespan:
        plot_gantt_chart("SHIELD", shield.schedule, len(processors_shield), shield_makespan)
        print(f"SHIELD Security Utility: {shield_security_utility:.2f}")
    
    # Show the plots
    plt.show()

if __name__ == "__main__":
    main()