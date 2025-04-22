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
        self.makespan = makespan
        end_time=time.time()
        print(f"SHIELD successful. Makespan: {makespan}, Security Utility: {security_utility} Time Elapsed: {(end_time-start_time):.4f}")
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