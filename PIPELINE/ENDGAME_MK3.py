import numpy as np
import matplotlib.pyplot as plt
import copy
import time
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
    def __init__(self, proc_id,bandwidth):
        self.proc_id = proc_id
        self.available_time = 0
        self.bandwidth = bandwidth  # Bandwidth in KB/s
        
    def get_communication_time(self, message, source_proc, dest_proc):
        """Calculate communication time for a message on this processor."""
        bw = source_proc.bandwidth[dest_proc.proc_id - 1]
        return message.size / bw

class SecurityService:
    def __init__(self):
        # Security strengths for protocols based on Table 1 from paper
        # self.strengths = {
        #     'confidentiality': [0.1, 0.25, 0.4, 0.55, 0.7, 0.8, 0.9, 1.0],  # Was [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        #     'integrity': [0.1, 0.25, 0.4, 0.6, 0.75, 0.9, 1.0],  # Was [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
        #     'authentication': [0.2, 0.6, 1.0]  # Was [0.2, 0.5, 1.0]
        # }
        
        # # Security overheads (execution time) for each protocol on each processor
        # # Based on Table 4 from paper (simplified for 2 processors)
        # self.overheads = {
        #     'confidentiality': {
        #         # Scale down values by factor of ~10
        #         1: [150,145],  # Was [150, 145]
        #         2: [95, 90],  # Was [95, 90]
        #         3: [35, 30],  # Was [35, 30]
        #         4: [30, 32],  # Was [30, 32]
        #         5: [28, 27],  # Was [28, 27]
        #         6: [20, 22],  # Was [20, 22]
        #         7: [14, 15],  # Was [14, 15]
        #         8: [12, 13]   # Was [12, 13]
        #     },
        #     'integrity': {
        #         # Scale down values by factor of ~5
        #         1: [22 , 24],  # Was [22, 24]
        #         2: [16 , 18],  # Was [16, 18]
        #         3: [11 , 12],  # Was [11, 12]
        #         4: [9 , 10],  # Was [9, 10]
        #         5: [6 , 7],  # Was [6, 7]
        #         6: [5 , 6],  # Was [5, 6]
        #         7: [4,4.5]   # Was [4, 4.5]
        #     },
        #     'authentication': {
        #         # Keep these similar, they're already small
        #         1: [80, 85],
        #         2: [135, 140],
        #         3: [155, 160]
        #     }
        # }
        self.strengths = {
            'confidentiality': [0.08, 0.14, 0.36, 0.40, 0.46, 0.64, 0.9, 1.0],  # Was [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            'integrity': [0.18, 0.26, 0.36, 0.45, 0.63, 0.77, 1.0],  # Was [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
            'authentication': [0.55, 0.91, 1.0]  # Was [0.2, 0.5, 1.0]
        }
        
        # Security overheads (execution time) for each protocol on each processor
        # Based on Table 4 from paper (simplified for 2 processors)
        self.overheads = {
            'confidentiality': {
                # Scale down values by factor of ~10
                1: [1012.5,1518.7,1181.2],  # Was [150, 145]
                2: [578.58, 867.87,675.01],  # Was [95, 90]
                3: [225.0, 337.5,262.5],  # Was [35, 30]
                4: [202.50, 303.75,236.25],  # Was [30, 32]
                5: [176.10, 264.15,205.45], # Was [28, 27]
                6: [126.54,189.81,147.63],  # Was [20, 22]
                7: [90.0,135.0,105.0],  # Was [14, 15]
                8: [81.0,121.5,94.5]   # Was [12, 13]
            },
            'integrity': {
                # Scale down values by factor of ~5
                1: [143.40, 215.10,167.30],  # Was [22, 24]
                2: [102.54 , 153.81,119.63],  # Was [16, 18]
                3: [72 , 108, 84.0],  # Was [11, 12]
                4: [58.38 , 87.57, 68.11],  # Was [9, 10]
                5: [41.28 , 61.92, 48.16],  # Was [6, 7]
                6: [34.14 , 51.21, 39.83],  # Was [5, 6]
                7: [26.16,39.24, 30.52]   # Was [4, 4.5]
            },
            'authentication': {
                # Keep these similar, they're already small
                1: [15, 10, 12.86],
                2: [24.67, 16.44, 21.14],
                3: [27.17, 18.11, 23.29]
            }
        }

class CommunicationNetwork:
    def __init__(self, num_processors):  # 500 KB/s as mentioned in the paper
        self.num_processors = num_processors
        
    # def get_communication_time(self, message, source_proc, dest_proc):
    #     if source_proc == dest_proc:
    #         return 0
    #     else:
    #         # Calculate comm time including security overhead
    #         return message.size / self.bandwidth

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
    
    def compute_average_bandwidth(self):
        total_bandwidth = 0
        count = 0
        for proc in self.processors:
            # Exclude 0 (self-communication) from average
            bw_values = [bw for bw in proc.bandwidth]
            total_bandwidth += sum(bw_values)
            count += len(bw_values)
        return total_bandwidth / count if count > 0 else 0

    
    def compute_task_priorities(self):
        """Compute priorities for tasks using upward rank method with security overhead."""
        # First, calculate the average execution time for each task
        print(f"Average Bandwidth is: {self.compute_average_bandwidth()}")
        bw_average = self.compute_average_bandwidth()
        
        for task in self.tasks:
            task.avg_execution = sum(task.execution_times) / len(task.execution_times)
        
        # Calculate minimum security overhead for each task
        def calculate_min_security_overhead(task):
            total_weighted_overhead = 0

            # Incoming messages (predecessors)
            for pred_id in task.predecessors:
                message = self.get_message(pred_id, task.task_id)
                if message:
                    total_weighted_overhead += estimate_message_security_cost(message)

            # Outgoing messages (successors)
            for succ_id in task.successors:
                message = self.get_message(task.task_id, succ_id)
                if message:
                    total_weighted_overhead += estimate_message_security_cost(message)

            return total_weighted_overhead

        def estimate_message_security_cost(message):
            """Estimate weighted security overhead for a message (without processor-specific assumptions)."""
            total_cost = 0
            
            # Confidentiality
            for i, strength in enumerate(self.security.strengths['confidentiality']):
                if strength >= message.min_security['confidentiality']:
                    # Get the protocol index (1-indexed in the overhead table)
                    protocol_idx = i + 1
                    # Calculate average overhead across processors
                    avg_overhead = sum(self.security.overheads['confidentiality'][protocol_idx]) / len(self.security.overheads['confidentiality'][protocol_idx])
                    # Apply weight
                    cost = message.weights['confidentiality'] * message.size / avg_overhead
                    total_cost += cost
                    break
            
            # Integrity
            for i, strength in enumerate(self.security.strengths['integrity']):
                if strength >= message.min_security['integrity']:
                    # Get the protocol index (1-indexed in the overhead table)
                    protocol_idx = i + 1
                    # Calculate average overhead across processors
                    avg_overhead = sum(self.security.overheads['integrity'][protocol_idx]) / len(self.security.overheads['integrity'][protocol_idx])
                    # Apply weight
                    cost = message.weights['integrity'] * message.size / avg_overhead
                    total_cost += cost
                    break
            
            # Authentication
            for i, strength in enumerate(self.security.strengths['authentication']):
                if strength >= message.min_security['authentication']:
                    # Get the protocol index (1-indexed in the overhead table)
                    protocol_idx = i + 1
                    # Calculate average overhead across processors
                    avg_overhead = sum(self.security.overheads['authentication'][protocol_idx]) / len(self.security.overheads['authentication'][protocol_idx])
                    # Apply weight
                    cost = message.weights['authentication'] * avg_overhead
                    total_cost += cost
                    break
            
            return total_cost

        
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
                        comm_cost = message.size / bw_average
                    
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
        
        # Add overhead for all three services at both sender and receiver
        for service in ['confidentiality', 'integrity', 'authentication']:
            protocol_idx = message.assigned_security[service] + 1  # 1-indexed in the overhead table
            
            # Sender overhead (locking)
            sender_overhead_factor = self.security.overheads[service][protocol_idx][source_proc - 1]
            if service in ['confidentiality', 'integrity']:
                # Data-dependent overhead for confidentiality and integrity
                sender_overhead = (message.size / sender_overhead_factor)
            else:
                # Fixed overhead for authentication
                sender_overhead = sender_overhead_factor
            total_overhead += sender_overhead
            
            # Receiver overhead (unlocking) - same calculation as per paper
            receiver_overhead_factor = self.security.overheads[service][protocol_idx][dest_proc - 1]
            if service in ['confidentiality', 'integrity']:
                # Data-dependent overhead for confidentiality and integrity
                receiver_overhead = (message.size / receiver_overhead_factor)
            else:
                # Fixed overhead for authentication
                receiver_overhead = receiver_overhead_factor
            total_overhead += receiver_overhead
        
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
                # Calculate full security overhead (both sender and receiver for all services)
                security_overhead = self.calc_security_overhead(message, pred_task.assigned_processor, processor)
                
                # Calculate communication time
                source_proc = self.processors[pred_task.assigned_processor - 1]
                dest_proc = self.processors[processor - 1]
                comm_time = source_proc.get_communication_time(message, source_proc, dest_proc)
                
                # Total communication finish time includes predecessor finish time, comm time, and security overhead
                comm_finish_time = pred_task.finish_time + comm_time + security_overhead
            
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
        
        # Add debug information
        print("\nDEBUG: Security Overhead Calculation Details")
        print("="*80)
        
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
                    
                # If on same processor, just use predecessor finish time
                if pred_task.assigned_processor == processor_id:
                    pred_finish = pred_task.finish_time
                else:
                    source_proc = self.processors[pred_task.assigned_processor - 1]
                    dest_proc = processor
                    comm_time = source_proc.get_communication_time(message, source_proc, dest_proc)
                    pred_finish = pred_task.finish_time + comm_time
                
                est = max(est, pred_finish)
            
            # Also consider processor availability
            est = max(est, processor.available_time)
            
            # Calculate base execution time
            base_exec_time = task.execution_times[processor_id - 1]
            
            # For debugging
            print(f"\nTask {task.task_id} ({task.name}) on Processor {processor_id}:")
            print(f"  Base execution time: {base_exec_time} ms")
            print(f"  Start time (EST): {est} ms")
            
            # FIX 1: Calculate incoming security overhead explicitly for receiver
            incoming_security_overhead = 0
            for pred_id in task.predecessors:
                pred_task = self.get_task_by_id(pred_id)
                if not pred_task or not pred_task.is_scheduled:
                    continue
                    
                # Skip if on same processor (no security overhead)
                if pred_task.assigned_processor == processor_id:
                    continue
                    
                message = self.get_message(pred_id, task.task_id)
                if not message:
                    continue
                
                # Calculate overhead for each security service (receiver side)
                for service in ['confidentiality', 'integrity']:
                    protocol_idx = message.assigned_security[service] + 1  # 1-indexed
                    overhead_factor = self.security.overheads[service][protocol_idx][processor_id - 1]
                    overhead = message.size / overhead_factor
                    incoming_security_overhead += overhead
                    print(f"    Incoming {service} overhead from Task {pred_id}: {overhead:.2f} ms")
                
                # Authentication verification overhead
                auth_idx = message.assigned_security['authentication'] + 1
                auth_overhead = self.security.overheads['authentication'][auth_idx][processor_id - 1]
                incoming_security_overhead += auth_overhead
                print(f"    Incoming authentication overhead from Task {pred_id}: {auth_overhead:.2f} ms")
            
            print(f"  Total incoming security overhead: {incoming_security_overhead:.2f} ms")
            
            # FIX 2: Calculate outgoing security overhead explicitly for sender
            outgoing_security_overhead = 0
            for succ_id in task.successors:
                succ_task = self.get_task_by_id(succ_id)
                if not succ_task or not succ_task.is_scheduled:
                    continue
                    
                # Skip if on same processor (no security overhead)
                if succ_task.assigned_processor == processor_id:
                    continue
                    
                message = self.get_message(task.task_id, succ_id)
                if not message:
                    continue
                
                # Calculate overhead for each security service (sender side)
                for service in ['confidentiality', 'integrity']:
                    protocol_idx = message.assigned_security[service] + 1  # 1-indexed
                    overhead_factor = self.security.overheads[service][protocol_idx][processor_id - 1]
                    overhead = message.size / overhead_factor
                    outgoing_security_overhead += overhead
                    print(f"    Outgoing {service} overhead to Task {succ_id}: {overhead:.2f} ms")
                
                # IMPORTANT FIX: Include authentication generation overhead
                auth_idx = message.assigned_security['authentication'] + 1
                auth_overhead = self.security.overheads['authentication'][auth_idx][processor_id - 1]
                outgoing_security_overhead += auth_overhead
                print(f"    Outgoing authentication overhead to Task {succ_id}: {auth_overhead:.2f} ms")
            
            print(f"  Total outgoing security overhead: {outgoing_security_overhead:.2f} ms")
            
            # Calculate total security overhead
            total_security_overhead = incoming_security_overhead + outgoing_security_overhead
            print(f"  Total security overhead: {total_security_overhead:.2f} ms")
            
            # Update task timing with security overhead
            task.start_time = est
            task.finish_time = est + base_exec_time + total_security_overhead
            print(f"  Final finish time: {task.finish_time:.2f} ms (start + base_exec + security)")
            
            # Update processor availability
            processor.available_time = task.finish_time
            
            # Update schedule entry
            for entry in self.schedule:
                if entry['task_id'] == task.task_id:
                    entry['start_time'] = task.start_time
                    entry['finish_time'] = task.finish_time
        
        # Update the makespan
        self.makespan = max(task.finish_time for task in self.tasks if task.is_scheduled)
        return self.makespan
    def display_task_security_details(self):
        """
        Displays detailed security information for each task including:
        - All messages sent/received by each task
        - Security levels (confidentiality, integrity, authentication) for each message
        - Detailed overhead calculations for each security dimension
        - Total security overhead impact on task execution time
        """
        print("\n" + "="*120)
        print("TASK SECURITY DETAILS AND OVERHEAD CALCULATIONS")
        print("="*120)
        
        # Security protocol names for better readability
        security_protocols = {
            'confidentiality': ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'],
            'integrity': ['I0', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6'],
            'authentication': ['A0', 'A1', 'A2']
        }
        
        # Process each task
        for task in sorted(self.tasks, key=lambda t: t.task_id):
            if not task.is_scheduled:
                continue
                
            print(f"\nTask {task.task_id} ({task.name}) on Processor {task.assigned_processor}:")
            print("-" * 100)
            
            # Base execution time without security overhead
            base_exec_time = task.execution_times[task.assigned_processor - 1]
            print(f"Base execution time: {base_exec_time:.2f} ms")
            
            # Calculate security overhead for this task
            incoming_overhead = 0
            outgoing_overhead = 0
            
            # Track all security-related messages for this task
            security_messages = []
            
            # 1. Process incoming messages (task as receiver)
            print("\n  INCOMING MESSAGES (Task as Receiver):")
            print("  " + "-" * 98)
            print(f"  {'From Task':<12} {'Protocol':<12} {'Security Level':<20} {'Overhead Calculation':<50} {'Overhead (ms)':<15}")
            print("  " + "-" * 98)
            
            for pred_id in task.predecessors:
                pred_task = self.get_task_by_id(pred_id)
                if not pred_task or not pred_task.is_scheduled:
                    continue
                    
                # Skip if on same processor (no security overhead)
                if pred_task.assigned_processor == task.assigned_processor:
                    print(f"  {pred_task.name:<12} {'N/A':<12} {'N/A':<20} {'Same processor - no security overhead':<50} {0:<15.2f}")
                    continue
                    
                message = self.get_message(pred_id, task.task_id)
                if not message:
                    continue
                    
                # Authentication overhead calculation (receiver side)
                auth_idx = message.assigned_security['authentication']
                auth_protocol_idx = auth_idx + 1  # Convert to 1-indexed for overhead table
                auth_protocol = security_protocols['authentication'][auth_idx]
                auth_strength = self.security.strengths['authentication'][auth_idx]
                auth_overhead = self.security.overheads['authentication'][auth_protocol_idx][task.assigned_processor - 1]
                
                # Display the authentication overhead calculation
                calc_desc = f"Authentication: Protocol {auth_protocol} (strength {auth_strength:.2f})"
                calc_formula = f"Overhead = {auth_overhead:.2f} ms"
                
                print(f"  {pred_task.name:<12} {auth_protocol:<12} {auth_strength:<20.2f} {calc_formula:<50} {auth_overhead:<15.2f}")
                
                incoming_overhead += auth_overhead
                
                # Store message details for summary
                security_messages.append({
                    'direction': 'incoming',
                    'other_task': pred_task.name,
                    'message': message,
                    'service': 'authentication',
                    'protocol': auth_protocol,
                    'strength': auth_strength,
                    'overhead': auth_overhead
                })
            
            if not task.predecessors:
                print("  No incoming messages")
            
            # 2. Process outgoing messages (task as sender)
            print("\n  OUTGOING MESSAGES (Task as Sender):")
            print("  " + "-" * 98)
            print(f"  {'To Task':<12} {'Service':<12} {'Protocol':<12} {'Security Level':<15} {'Overhead Calculation':<40} {'Overhead (ms)':<15}")
            print("  " + "-" * 98)
            
            for succ_id in task.successors:
                succ_task = self.get_task_by_id(succ_id)
                if not succ_task or not succ_task.is_scheduled:
                    continue
                    
                # Skip if on same processor (no security overhead)
                if succ_task.assigned_processor == task.assigned_processor:
                    print(f"  {succ_task.name:<12} {'N/A':<12} {'N/A':<12} {'N/A':<15} {'Same processor - no security overhead':<40} {0:<15.2f}")
                    continue
                    
                message = self.get_message(task.task_id, succ_id)
                if not message:
                    continue
                    
                # Process each security service for outgoing messages
                for service in ['confidentiality', 'integrity']:
                    service_idx = message.assigned_security[service]
                    protocol_idx = service_idx + 1  # Convert to 1-indexed for overhead table
                    protocol = security_protocols[service][service_idx]
                    strength = self.security.strengths[service][service_idx]
                    
                    # Calculate overhead (data-dependent)
                    overhead_factor = self.security.overheads[service][protocol_idx][task.assigned_processor - 1]
                    overhead = (message.size / 1024) * overhead_factor
                    
                    # Display the calculation
                    calc_desc = f"{service.capitalize()}: Protocol {protocol} (strength {strength:.2f})"
                    calc_formula = f"({message.size} KB / 1024) * {overhead_factor} = {overhead:.2f} ms"
                    
                    print(f"  {succ_task.name:<12} {service.capitalize():<12} {protocol:<12} {strength:<15.2f} {calc_formula:<40} {overhead:<15.2f}")
                    
                    outgoing_overhead += overhead
                    
                    # Store message details for summary
                    security_messages.append({
                        'direction': 'outgoing',
                        'other_task': succ_task.name,
                        'message': message,
                        'service': service,
                        'protocol': protocol,
                        'strength': strength,
                        'overhead': overhead
                    })
            
            if not task.successors:
                print("  No outgoing messages")
            
            # 3. Summary for the task
            total_security_overhead = incoming_overhead + outgoing_overhead
            actual_overhead = task.finish_time - task.start_time - base_exec_time
            
            print("\n  SECURITY OVERHEAD SUMMARY:")
            print("  " + "-" * 98)
            print(f"  Incoming communication security overhead: {incoming_overhead:.2f} ms")
            print(f"  Outgoing communication security overhead: {outgoing_overhead:.2f} ms")
            print(f"  Total calculated security overhead: {total_security_overhead:.2f} ms")
            print(f"  Actual execution overhead (finish - start - base): {actual_overhead:.2f} ms")
            print(f"  Task start time: {task.start_time:.2f} ms")
            print(f"  Task finish time: {task.finish_time:.2f} ms")
            print(f"  Task execution with security: {(task.finish_time - task.start_time):.2f} ms")
            print(f"  Security overhead percentage: {(total_security_overhead / base_exec_time * 100):.2f}% of base execution time")
        
        print("\n" + "="*120)
        
        # Display overall security statistics
        total_base_execution = sum(task.execution_times[task.assigned_processor - 1] for task in self.tasks if task.is_scheduled)
        total_actual_execution = sum((task.finish_time - task.start_time) for task in self.tasks if task.is_scheduled)
        total_security_overhead = total_actual_execution - total_base_execution
        
        print("\nOVERALL SECURITY STATISTICS:")
        print("-" * 60)
        print(f"Total base execution time (all tasks): {total_base_execution:.2f} ms")
        print(f"Total actual execution time (all tasks): {total_actual_execution:.2f} ms")
        print(f"Total security overhead (all tasks): {total_security_overhead:.2f} ms")
        print(f"Security overhead percentage: {(total_security_overhead / total_base_execution * 100):.2f}%")
        print("=" * 120)
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
                
                # Calculate task execution security overhead (if any)
                task_security_overhead = 0  # Add task-specific security overhead if needed
                
                # Total execution time
                total_exec_time = base_exec_time + task_security_overhead
                
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
        
        # Recalculate the schedule with full security overhead consideration
        self.finalize_schedule()
        
        # Get the makespan
        self.makespan = max(task.finish_time for task in self.tasks)
        return self.makespan
    def calculate_and_print_security_details(self):
        """
        Calculates and prints detailed security information for each message and its overhead impact.
        Shows actual security protocols applied and their overhead impact on task execution.
        """
        print("\n" + "="*120)
        print("DETAILED SECURITY INFORMATION AND OVERHEAD ANALYSIS")
        print("="*120)
        
        # Define security protocols for better readability
        security_protocols = {
            'confidentiality': ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'],
            'integrity': ['I0', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6'],
            'authentication': ['A0', 'A1', 'A2']
        }
        
        print(f"{'Message ID':<12} {'Source→Dest':<15} {'Protocol (C/I/A)':<20} {'Min Security (C/I/A)':<25} {'Applied Security (C/I/A)':<30} {'Overhead (ms)':<15}")
        print("-"*120)
        
        total_overhead = 0
        
        for message in self.messages:
            source_task = self.get_task_by_id(message.source_id)
            dest_task = self.get_task_by_id(message.dest_id)
            
            if not source_task or not dest_task or not source_task.is_scheduled or not dest_task.is_scheduled:
                continue
                
            # Skip if on same processor (no security overhead)
            if source_task.assigned_processor == dest_task.assigned_processor:
                continue
                
            # Get security protocols
            conf_idx = message.assigned_security['confidentiality']
            integ_idx = message.assigned_security['integrity']
            auth_idx = message.assigned_security['authentication']
            
            conf_protocol = security_protocols['confidentiality'][conf_idx]
            integ_protocol = security_protocols['integrity'][integ_idx]
            auth_protocol = security_protocols['authentication'][auth_idx]
            
            # Get security strengths
            conf_strength = self.security.strengths['confidentiality'][conf_idx]
            integ_strength = self.security.strengths['integrity'][integ_idx]
            auth_strength = self.security.strengths['authentication'][auth_idx]
            
            # Calculate actual overhead for this message
            source_proc = source_task.assigned_processor
            dest_proc = dest_task.assigned_processor
            
            # Calculate each component of security overhead
            conf_overhead = 0
            integ_overhead = 0
            auth_overhead = 0
            
            # Calculate confidentiality overhead (applied at source)
            if conf_idx > 0:  # Only if a security protocol is applied
                protocol_idx = conf_idx + 1  # Convert to 1-indexed
                overhead_factor = self.security.overheads['confidentiality'][protocol_idx][source_proc - 1]
                conf_overhead = (message.size / 1000) * overhead_factor
                
            # Calculate integrity overhead (applied at source)
            if integ_idx > 0:  # Only if a security protocol is applied
                protocol_idx = integ_idx + 1  # Convert to 1-indexed
                overhead_factor = self.security.overheads['integrity'][protocol_idx][source_proc - 1]
                integ_overhead = (message.size / 1000) * overhead_factor
                
            # Calculate authentication overhead (applied at destination)
            if auth_idx > 0:  # Only if a security protocol is applied
                protocol_idx = auth_idx + 1  # Convert to 1-indexed
                auth_overhead = self.security.overheads['authentication'][protocol_idx][dest_proc - 1]
                
            # Total overhead for this message
            message_overhead = conf_overhead + integ_overhead + auth_overhead
            total_overhead += message_overhead
            
            # Format and print the information
            source_dest = f"{source_task.name}→{dest_task.name}"
            protocols = f"{conf_protocol}/{integ_protocol}/{auth_protocol}"
            min_security = f"{message.min_security['confidentiality']:.2f}/{message.min_security['integrity']:.2f}/{message.min_security['authentication']:.2f}"
            applied_security = f"{conf_strength:.2f}/{integ_strength:.2f}/{auth_strength:.2f}"
            
            print(f"{message.id:<12} {source_dest:<15} {protocols:<20} {min_security:<25} {applied_security:<30} {message_overhead:<15.2f}")
        
        print("-"*120)
        print(f"Total security overhead: {total_overhead:.2f} ms")
        print("="*120)
        
        # Analysis of task execution times vs security overhead
        print("\n" + "="*120)
        print("SECURITY OVERHEAD VS EXECUTION TIME ANALYSIS")
        print("="*120)
        print(f"{'Task ID':<8} {'Task Name':<15} {'Processor':<10} {'Basic Exec (ms)':<15} {'Actual Security Overhead (ms)':<30} {'Start Time':<12} {'Finish Time':<12} {'Expected Finish':<15} {'Diff':<8}")
        print("-"*120)
        
        for task in self.tasks:
            if not task.is_scheduled:
                continue
                
            # Get the base execution time
            base_exec = task.execution_times[task.assigned_processor - 1]
            
            # Calculate the actual security overhead (finish - start - base_exec)
            actual_overhead = task.finish_time - task.start_time - base_exec
            
            # Calculate the expected finish time based on our calculations
            expected_finish = task.start_time + base_exec + actual_overhead
            
            # Difference between expected and actual finish time
            diff = task.finish_time - expected_finish
            
            print(f"{task.task_id:<8} {task.name:<15} {task.assigned_processor:<10} {base_exec:<15.2f} {actual_overhead:<30.2f} {task.start_time:<12.2f} {task.finish_time:<12.2f} {expected_finish:<15.2f} {diff:<8.2f}")
        
        print("="*120)
        return total_overhead
    
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
        - Actual security overhead (finish - start - base_exec)
        - Finish time
        """
        print("\n" + "="*100)
        print("DETAILED TASK SCHEDULE")
        print("="*100)
        print(f"{'ID':<5} {'Task Name':<20} {'Proc':<5} {'Start':<8} {'Exec':<8} {'Actual Overhead':<15} {'Finish':<8}")
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
            
            # Calculate actual security overhead
            actual_overhead = task.finish_time - task.start_time - base_exec_time
            total_security_overhead += actual_overhead
            
            # Print task details
            print(f"{task.task_id:<5} {task.name[:20]:<20} {task.assigned_processor:<5} "
                f"{task.start_time:<8.2f} {base_exec_time:<8.2f} "
                f"{actual_overhead:<15.2f} {task.finish_time:<8.2f}")
        
        print("-"*100)
        # Print summary statistics
        print(f"Total base execution time: {total_base_execution:.2f}")
        print(f"Total actual security overhead: {total_security_overhead:.2f} ({(total_security_overhead/total_base_execution)*100:.2f}% of base execution)")
        
        print(f"Makespan: {self.makespan:.2f}")
        print("="*100)
        
        # Return the data if needed for further analysis
        return {
            "total_base_execution": total_base_execution,
            "total_security_overhead": total_security_overhead,
            "makespan": self.makespan
        }
        
class SHIELD(HSMS):
    """Security-aware scHedulIng with gLobal and local slacK timE Distribution"""
    
    def __init__(self, tasks, messages, processors, network, security_service, deadline):
        super().__init__(tasks, messages, processors, network, security_service, deadline)
        self.slack_times = {}  # To store slack times between tasks
    
    def calculate_slack_times(self):
        """Calculate both global and local slack times in the schedule."""
        print("\n=== Calculating Slack Times ===")
        
        # First get a valid schedule using HSMS
        self.schedule_tasks()
        self.finalize_schedule()
        
        # Calculate global slack time based on deadline
        global_slack = self.deadline - self.makespan
        print(f"Global slack time: {global_slack:.2f} ms")
        
        # Sort tasks by finish time
        sorted_tasks = sorted(self.tasks, key=lambda t: t.finish_time if t.is_scheduled else float('inf'))
        
        # Calculate local slack times
        for i in range(len(sorted_tasks) - 1):
            current_task = sorted_tasks[i]
            
            # For each successor of the current task
            for succ_id in current_task.successors:
                succ_task = self.get_task_by_id(succ_id)
                if not succ_task or not succ_task.is_scheduled:
                    continue
                
                # Calculate slack time between current task and its successor
                slack = succ_task.start_time - current_task.finish_time
                
                if slack > 0:
                    key = f"{current_task.task_id}→{succ_task.task_id}"
                    self.slack_times[key] = slack
                    print(f"Local slack: {current_task.name}→{succ_task.name} = {slack:.2f} ms")
        
        return global_slack
    
    def can_improve_security(self, message, current_protocol, service):
        """Check if security can be improved for a specific service."""
        current_idx = message.assigned_security[service]
        strengths = self.security.strengths[service]
        
        # Check if there's a stronger protocol available
        return current_idx < len(strengths) - 1
    
    def calculate_security_upgrade_overhead(self, message, source_proc, dest_proc, service, new_protocol_idx):
        """Calculate the additional overhead from upgrading security protocol."""
        current_idx = message.assigned_security[service]
        current_protocol_idx = current_idx + 1  # Convert to 1-indexed
        new_protocol_idx_1indexed = new_protocol_idx + 1  # Convert to 1-indexed
        
        # Calculate current overhead
        current_overhead = 0
        if service in ['confidentiality', 'integrity']:
            # Data-dependent overhead for source
            source_overhead_factor = self.security.overheads[service][current_protocol_idx][source_proc - 1]
            current_overhead += message.size / source_overhead_factor
            
            # Data-dependent overhead for destination
            dest_overhead_factor = self.security.overheads[service][current_protocol_idx][dest_proc - 1]
            current_overhead += message.size / dest_overhead_factor
        else:  # authentication
            # Fixed overhead for source
            current_overhead += self.security.overheads[service][current_protocol_idx][source_proc - 1]
            # Fixed overhead for destination
            current_overhead += self.security.overheads[service][current_protocol_idx][dest_proc - 1]
        
        # Calculate new overhead
        new_overhead = 0
        if service in ['confidentiality', 'integrity']:
            # Data-dependent overhead for source
            source_overhead_factor = self.security.overheads[service][new_protocol_idx_1indexed][source_proc - 1]
            new_overhead += message.size / source_overhead_factor
            
            # Data-dependent overhead for destination
            dest_overhead_factor = self.security.overheads[service][new_protocol_idx_1indexed][dest_proc - 1]
            new_overhead += message.size / dest_overhead_factor
        else:  # authentication
            # Fixed overhead for source
            new_overhead += self.security.overheads[service][new_protocol_idx_1indexed][source_proc - 1]
            # Fixed overhead for destination
            new_overhead += self.security.overheads[service][new_protocol_idx_1indexed][dest_proc - 1]
        
        # Return additional overhead
        return new_overhead - current_overhead
    
    def security_utility_gain(self, message, service, new_protocol_idx):
        """Calculate the security utility gain for upgrading a protocol."""
        current_idx = message.assigned_security[service]
        current_strength = self.security.strengths[service][current_idx]
        new_strength = self.security.strengths[service][new_protocol_idx]
        
        # Calculate utility gain
        gain = (new_strength - current_strength) * message.weights[service]
        return gain
    
    def shift_tasks(self, task_id, shift_amount):
        """Shift a task and all its successors by the specified amount."""
        task = self.get_task_by_id(task_id)
        if not task or not task.is_scheduled:
            return
        
        # Create a queue for BFS
        from collections import deque
        queue = deque()
        visited = set()
        queue.append(task_id)
        visited.add(task_id)
        
        # Keep track of affected processors to update availability times
        affected_processors = set()
        
        while queue:
            current_id = queue.popleft()
            current_task = self.get_task_by_id(current_id)
            
            # Shift this task
            if current_task and current_task.is_scheduled:
                old_finish_time = current_task.finish_time
                current_task.start_time += shift_amount
                current_task.finish_time += shift_amount
                
                # Track affected processor
                affected_processors.add(current_task.assigned_processor - 1)
                
                # Update the schedule
                for entry in self.schedule:
                    if entry['task_id'] == current_id:
                        entry['start_time'] += shift_amount
                        entry['finish_time'] += shift_amount
                        break
                
                # Add unvisited successors to the queue
                for succ_id in current_task.successors:
                    if succ_id not in visited:
                        succ_task = self.get_task_by_id(succ_id)
                        if succ_task and succ_task.is_scheduled:
                            # Only add to queue if this task must be shifted
                            # Either because it depends on the current task 
                            # or its current start time would create a conflict
                            if succ_task.start_time < current_task.finish_time:
                                visited.add(succ_id)
                                queue.append(succ_id)
        
        # Update processor availability times for all affected processors
        for proc_idx in affected_processors:
            processor = self.processors[proc_idx]
            # Find the new last task on this processor
            last_finish_time = 0
            for task in self.tasks:
                if task.is_scheduled and task.assigned_processor == proc_idx + 1:
                    last_finish_time = max(last_finish_time, task.finish_time)
            processor.available_time = last_finish_time
        
        # Also update slack times for affected tasks
        for i in range(len(self.tasks)):
            for j in range(len(self.tasks)):
                task_i = self.tasks[i]
                task_j = self.tasks[j]
                if (task_i.is_scheduled and task_j.is_scheduled and 
                    task_j.task_id in task_i.successors):
                    key = f"{task_i.task_id}→{task_j.task_id}"
                    self.slack_times[key] = max(0, task_j.start_time - task_i.finish_time)
    def optimize_security(self):
        """Optimize security using slack times."""
        print("\n=== Optimizing Security Using Slack Times ===")
        
        # First, calculate all slack times
        global_slack = self.calculate_slack_times()
        
        # Keep track of used slack to ensure we don't exceed the deadline
        used_global_slack = 0
        
        # Process messages in order of weight (highest weight first)
        weighted_messages = []
        for message in self.messages:
            source_task = self.get_task_by_id(message.source_id)
            dest_task = self.get_task_by_id(message.dest_id)
            
            if not source_task or not dest_task or not source_task.is_scheduled or not dest_task.is_scheduled:
                continue
                
            # Skip if on same processor (no security needed)
            if source_task.assigned_processor == dest_task.assigned_processor:
                continue
                
            # Calculate total weight for this message
            total_weight = sum(message.weights.values())
            weighted_messages.append((message, total_weight))
        
        # Sort messages by total weight (descending)
        weighted_messages.sort(key=lambda x: x[1], reverse=True)
        
        # Security services in order of processing priority
        security_services = ['confidentiality', 'integrity', 'authentication']
        
        # Process each message
        for message, _ in weighted_messages:
            source_task = self.get_task_by_id(message.source_id)
            dest_task = self.get_task_by_id(message.dest_id)
            
            if not source_task.is_scheduled or not dest_task.is_scheduled:
                continue
                
            source_proc = source_task.assigned_processor - 1
            dest_proc = dest_task.assigned_processor - 1
            
            # Check local slack first
            local_slack_key = f"{source_task.task_id}→{dest_task.task_id}"
            available_slack = self.slack_times.get(local_slack_key, 0)
            
            # Also consider global slack (but limit to ensure we don't exceed deadline)
            remaining_global_slack = global_slack - used_global_slack
            
            # Determine total available slack
            total_available_slack = available_slack + remaining_global_slack
            
            if total_available_slack <= 0:
                continue
                
            print(f"\nProcessing message {message.id} from {source_task.name} to {dest_task.name}")
            print(f"Available slack: {total_available_slack:.2f} ms")
            
            # Try to improve each security service
            for service in security_services:
                current_idx = message.assigned_security[service]
                
                # Check if we can improve security for this service
                if self.can_improve_security(message, current_idx, service):
                    # Try the next better protocol
                    new_idx = current_idx + 1
                    
                    # Calculate additional overhead
                    additional_overhead = self.calculate_security_upgrade_overhead(
                        message, source_proc, dest_proc, service, new_idx)
                    
                    # Calculate utility gain
                    utility_gain = self.security_utility_gain(message, service, new_idx)
                    
                    # If we can accommodate this upgrade within available slack
                    if additional_overhead <= total_available_slack:
                        print(f"  Upgrading {service} from level {current_idx} to {new_idx}")
                        print(f"  Additional overhead: {additional_overhead:.2f} ms")
                        print(f"  Utility gain: {utility_gain:.4f}")
                        
                        # Apply the upgrade
                        message.assigned_security[service] = new_idx
                        
                        # Update available slack
                        if additional_overhead <= available_slack:
                            available_slack -= additional_overhead
                            self.slack_times[local_slack_key] = available_slack
                        else:
                            # Use some global slack
                            global_slack_used = additional_overhead - available_slack
                            used_global_slack += global_slack_used
                            available_slack = 0
                            self.slack_times[local_slack_key] = 0
                        
                        total_available_slack -= additional_overhead
                        
                        # Shift successor tasks if needed
                        self.shift_tasks(dest_task.task_id, additional_overhead)
                        
                        print(f"  Remaining slack: {total_available_slack:.2f} ms")
                    else:
                        print(f"  Cannot upgrade {service} - requires {additional_overhead:.2f} ms but only {total_available_slack:.2f} ms available")
                else:
                    print(f"  {service} already at maximum level or no stronger protocol available")
        
        # Update makespan after security optimizations
        self.makespan = max(task.finish_time for task in self.tasks if task.is_scheduled)
        return self.makespan
    
    def run(self):
        """Run the SHIELD scheduler."""
        # First run HSMS to get a base schedule
        makespan = self.schedule_tasks()
        
        if makespan > self.deadline:
            print(f"SHIELD failed to meet deadline. Initial makespan: {makespan}, Deadline: {self.deadline}")
            return None, None
        
        # Now optimize security using slack time
        optimized_makespan = self.optimize_security()
        
        # Calculate final security utility
        security_utility = self.calculate_security_utility()
        
        if optimized_makespan > self.deadline:
            print(f"SHIELD security optimization exceeded deadline. Final makespan: {optimized_makespan}, Deadline: {self.deadline}")
            # Revert to original HSMS schedule if needed
            return None, None
        else:
            print(f"SHIELD successful. Final makespan: {optimized_makespan}, Security Utility: {security_utility}")
            return optimized_makespan, security_utility
    
    def print_task_schedule(self):
        """Print the final task schedule with task details."""
        print("\n--- Task Schedule ---")
        print("Task ID | Task Name | Processor | Start Time | Finish Time | Duration")
        print("-" * 70)
        
        # Sort schedule by start time for clearer presentation
        sorted_schedule = sorted(self.schedule, key=lambda x: x['start_time'])
        
        for entry in sorted_schedule:
            task_id = entry['task_id']
            name = entry['name']
            processor = entry['processor']
            start_time = entry['start_time']
            finish_time = entry['finish_time']
            duration = finish_time - start_time
            
            print(f"{task_id:7} | {name:9} | {processor:9} | {start_time:10.2f} | {finish_time:11.2f} | {duration:8.2f}")
        
        print(f"\nMakespan: {self.makespan:.2f}")
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
    # print(f"Total security overhead: {total_security_overhead:.2f} ({(total_security_overhead/total_base_execution)*100:.2f}% of base execution)")
    
    # Use makespan instead of best_makespan
    # print(f"Makespan: {self.makespan:.2f}")
    print("="*100)
    
    # Return the data if needed for further analysis
    return {
        "total_base_execution": total_base_execution,
        "total_security_overhead": total_security_overhead,
        "makespan": self.makespan
    }

class QLSecurityScheduler(HSMS):
    """
    Q-Learning Security-Aware Scheduler that improves upon HSMS by learning
    better security protocol assignments for inter-processor messages.
    """
    def __init__(self, tasks, messages, processors, network, security_service, deadline, 
                 alpha=0.1, gamma=0.9, epsilon=0.3, episodes=2000, security_priority=0.5):
        super().__init__(tasks, messages, processors, network, security_service, deadline)
        
        # Q-learning parameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.episodes = episodes  # Number of episodes to train
        
        # Security priority parameter (0-1)
        # 0: Focus on makespan reduction
        # 1: Focus on security utility improvement
        self.security_priority = max(0, min(1, security_priority))  # Clamp between 0 and 1
        
        # Initialize Q-table
        # Structure: {message_id: {action: Q-value}}
        # where action is a tuple (conf_idx, integ_idx, auth_idx)
        self.q_table = {}
        
        # Store base HSMS schedule
        self.base_schedule = None
        self.base_makespan = None
        self.base_security_utility = None
        
        # For visualization
        self.training_history = {
            'episodes': [],
            'makespan': [],
            'security_utility': [],
            'best_makespan': [],
            'best_security_utility': []
        }

    def initialize_q_table(self):
        """Initialize the Q-table for each message"""
        for message in self.messages:
            # Get source and destination tasks
            source_task = self.get_task_by_id(message.source_id)
            dest_task = self.get_task_by_id(message.dest_id)
            
            # Only consider inter-processor messages
            if (not source_task or not dest_task or 
                not source_task.is_scheduled or not dest_task.is_scheduled or
                source_task.assigned_processor == dest_task.assigned_processor):
                continue
            
            # The message ID is the key in the Q-table
            message_id = message.id
            self.q_table[message_id] = {}
            
            # Generate all possible security protocol combinations that meet minimum requirements
            actions = self.get_valid_security_actions(message)
            
            # Initialize Q-values to 0
            for action in actions:
                self.q_table[message_id][action] = 0.0

    def get_valid_security_actions(self, message):
        """
        Generate all valid security protocol combinations for a message
        that meet the minimum security requirements.
        """
        valid_actions = []
        
        # Find all valid confidentiality protocols
        valid_conf_indices = []
        for i, strength in enumerate(self.security.strengths['confidentiality']):
            if strength >= message.min_security['confidentiality']:
                valid_conf_indices.append(i)
        
        # Find all valid integrity protocols
        valid_integ_indices = []
        for i, strength in enumerate(self.security.strengths['integrity']):
            if strength >= message.min_security['integrity']:
                valid_integ_indices.append(i)
        
        # Find all valid authentication protocols
        valid_auth_indices = []
        for i, strength in enumerate(self.security.strengths['authentication']):
            if strength >= message.min_security['authentication']:
                valid_auth_indices.append(i)
        
        # Generate all combinations
        for conf_idx in valid_conf_indices:
            for integ_idx in valid_integ_indices:
                for auth_idx in valid_auth_indices:
                    valid_actions.append((conf_idx, integ_idx, auth_idx))
        
        return valid_actions

    def choose_action(self, message_id, explore=True):
        """
        Choose an action for a message using epsilon-greedy policy.
        """
        if message_id not in self.q_table:
            return None
        
        if explore and np.random.random() < self.epsilon:
            # Exploration: choose random action
            return random.choice(list(self.q_table[message_id].keys()))
        else:
            # Exploitation: choose best action
            actions = list(self.q_table[message_id].keys())
            q_values = [self.q_table[message_id][a] for a in actions]
            
            # Find max Q-value
            max_q = max(q_values)
            # Find all actions with max Q-value
            best_actions = [actions[i] for i, q in enumerate(q_values) if q == max_q]
            # Randomly choose one of the best actions
            return random.choice(best_actions)

    def apply_security_action(self, message, action):
        """Apply security protocols specified by the action to the message."""
        conf_idx, integ_idx, auth_idx = action
        message.assigned_security['confidentiality'] = conf_idx
        message.assigned_security['integrity'] = integ_idx
        message.assigned_security['authentication'] = auth_idx

    def compute_reward(self, old_makespan, new_makespan, old_utility, new_utility):
        """
        Compute the reward for an action based on makespan and security utility.
        Uses security_priority parameter to balance between makespan and security utility.
        
        - If makespan increases beyond deadline, apply large negative reward
        - Otherwise, balance reward between makespan improvement and security utility based on priority
        """
        # Check if deadline is violated
        if new_makespan > self.deadline:
            return -100  # Large negative reward for violating deadline
        
        # Normalize security utility improvement (typically small values)
        utility_improvement = new_utility - old_utility
        utility_reward = utility_improvement * 10
        
        # Normalize makespan improvement (lower is better)
        makespan_improvement = old_makespan - new_makespan
        makespan_reward = makespan_improvement * 5
        
        # Apply security priority weighting
        weighted_reward = (self.security_priority * utility_reward + 
                          (1 - self.security_priority) * makespan_reward)
        
        return weighted_reward

    def train(self):
        """Train the Q-learning algorithm"""
        print("\nStarting Q-Learning training...")
        print(f"Security Priority: {self.security_priority:.2f} - {'Security focused' if self.security_priority > 0.5 else 'Makespan focused'}")
        
        # First run HSMS to get base schedule
        print("Running HSMS to get base schedule...")
        self.base_makespan = self.schedule_tasks()
        self.base_security_utility = self.calculate_security_utility()
        
        # Store the base schedule for later use
        self.base_schedule = []
        for task in self.tasks:
            if task.is_scheduled:
                self.base_schedule.append({
                    'task_id': task.task_id,
                    'name': task.name,
                    'processor': task.assigned_processor,
                    'start_time': task.start_time,
                    'finish_time': task.finish_time,
                    'security': {}  # Will store security settings for each message
                })
        
        # Print the base schedule with security information
        self.print_schedule_with_security("Base HSMS Schedule")
        print(f"Base HSMS - Makespan: {self.base_makespan:.2f}, Security Utility: {self.base_security_utility:.4f}")
        
        # Initialize Q-table
        self.initialize_q_table()
        
        # Skip training if no inter-processor messages
        if not self.q_table:
            print("No inter-processor messages to optimize. Using base HSMS schedule.")
            return self.base_makespan, self.base_security_utility
        
        best_makespan = self.base_makespan
        best_security_utility = self.base_security_utility
        best_actions = {}
        
        # Save the initial state of all messages
        initial_message_states = {}
        for message in self.messages:
            initial_message_states[message.id] = {
                'confidentiality': message.assigned_security['confidentiality'],
                'integrity': message.assigned_security['integrity'],
                'authentication': message.assigned_security['authentication']
            }
        
        # Store task processor assignments from HSMS
        task_assignments = {}
        for task in self.tasks:
            if task.is_scheduled:
                task_assignments[task.task_id] = task.assigned_processor
        
        # Training loop
        for episode in range(self.episodes):
            # Reset to base HSMS schedule
            for task in self.tasks:
                if task.task_id in task_assignments:
                    task.assigned_processor = task_assignments[task.task_id]
                    task.is_scheduled = True
            
            # Reset all messages to initial security levels
            for message in self.messages:
                if message.id in initial_message_states:
                    state = initial_message_states[message.id]
                    message.assigned_security['confidentiality'] = state['confidentiality']
                    message.assigned_security['integrity'] = state['integrity'] 
                    message.assigned_security['authentication'] = state['authentication']
            
            # Reset processor available times
            for proc in self.processors:
                proc.available_time = 0
            
            # Exploration decreases over time
            current_epsilon = max(0.01, self.epsilon * (1 - episode / self.episodes))
            
            # Target messages for exploration (only inter-processor messages)
            target_messages = [m for m in self.messages if m.id in self.q_table]
            
            # Shuffle messages to add randomness to learning
            random.shuffle(target_messages)
            
            # Track actions taken in this episode
            episode_actions = {}
            
            # For each message, choose and apply a security action
            for message in target_messages:
                # Choose an action based on Q-table and exploration rate
                action = self.choose_action(message.id, explore=(np.random.random() < current_epsilon))
                
                if action:
                    # Store old security settings
                    old_conf = message.assigned_security['confidentiality']
                    old_integ = message.assigned_security['integrity']
                    old_auth = message.assigned_security['authentication']
                    
                    # Apply the new security protocol
                    self.apply_security_action(message, action)
                    
                    # Store the action taken
                    episode_actions[message.id] = action
            
            # Recalculate the schedule with the new security protocols
            self.finalize_schedule()
            current_makespan = max(task.finish_time for task in self.tasks if task.is_scheduled)
            current_utility = self.calculate_security_utility()
            
            # Calculate rewards and update Q-values for each message
            for message in target_messages:
                if message.id in episode_actions:
                    action = episode_actions[message.id]
                    reward = self.compute_reward(self.base_makespan, current_makespan, 
                                               self.base_security_utility, current_utility)
                    
                    # Update Q-value using Q-learning formula
                    old_q = self.q_table[message.id][action]
                    # Simplified Q-learning update (no next state max Q-value since each message is independent)
                    new_q = old_q + self.alpha * (reward - old_q)
                    self.q_table[message.id][action] = new_q
            
            # Track the best solution found based on the priority
            # With high security_priority, favor security utility improvements
            # With low security_priority, favor makespan reductions
            if current_makespan <= self.deadline:
                if self.security_priority > 0.5:
                    # Security-focused: Prioritize security utility
                    if (current_utility > best_security_utility or 
                        (current_utility == best_security_utility and current_makespan < best_makespan)):
                        best_makespan = current_makespan
                        best_security_utility = current_utility
                        best_actions = {m.id: (m.assigned_security['confidentiality'], 
                                             m.assigned_security['integrity'], 
                                             m.assigned_security['authentication']) 
                                       for m in self.messages if m.id in self.q_table}
                else:
                    # Makespan-focused: Prioritize makespan reduction
                    if (current_makespan < best_makespan or 
                        (current_makespan == best_makespan and current_utility > best_security_utility)):
                        best_makespan = current_makespan
                        best_security_utility = current_utility
                        best_actions = {m.id: (m.assigned_security['confidentiality'], 
                                             m.assigned_security['integrity'], 
                                             m.assigned_security['authentication']) 
                                       for m in self.messages if m.id in self.q_table}
            
            # Store history for visualization (store every 10 episodes to reduce data volume)
            if episode % 10 == 0 or episode == self.episodes - 1:
                self.training_history['episodes'].append(episode)
                self.training_history['makespan'].append(current_makespan)
                self.training_history['security_utility'].append(current_utility)
                self.training_history['best_makespan'].append(best_makespan)
                self.training_history['best_security_utility'].append(best_security_utility)
            
            # Print progress every 100 episodes
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}/{self.episodes} - "
                      f"Epsilon: {current_epsilon:.2f}, "
                      f"Makespan: {current_makespan:.2f}, "
                      f"Security Utility: {current_utility:.4f}, "
                      f"Best: {best_security_utility:.4f} / {best_makespan:.2f}")
        
        # Apply the best actions found
        for message in self.messages:
            if message.id in best_actions:
                conf_idx, integ_idx, auth_idx = best_actions[message.id]
                message.assigned_security['confidentiality'] = conf_idx
                message.assigned_security['integrity'] = integ_idx
                message.assigned_security['authentication'] = auth_idx
        
        # Recalculate the final schedule
        self.finalize_schedule()
        final_makespan = max(task.finish_time for task in self.tasks if task.is_scheduled)
        final_security_utility = self.calculate_security_utility()
        
        # Print final schedule with security information
        self.print_schedule_with_security("Final Q-Learning Schedule")
        
        print(f"\nQ-Learning completed:")
        print(f"Initial HSMS - Makespan: {self.base_makespan:.2f}, Security Utility: {self.base_security_utility:.4f}")
        print(f"Final Q-Learning - Makespan: {final_makespan:.2f}, Security Utility: {final_security_utility:.4f}")
        print(f"Improvement - Makespan: {((self.base_makespan - final_makespan) / self.base_makespan) * 100:.2f}%, "
              f"Security: {((final_security_utility - self.base_security_utility) / self.base_security_utility) * 100:.2f}%")
        
        # Generate visualizations
        self.visualize_2d_plots()
        self.visualize_3d_plot()
        
        return final_makespan, final_security_utility
    
    def print_schedule_with_security(self, title="Schedule"):
        """
        Print schedule with security details (C: confidentiality, I: integrity, A: authentication)
        """
        try:
            # Print header with larger text
            print("\n" + "="*80)
            print(f"\033[1m{title}\033[0m".center(80))  # Bold text
            print("="*80)
            
            # Create header
            header = f"{'Task ID':<8} {'Task Name':<20} {'Processor':<10} {'Start':<10} {'Finish':<10} {'Security':<20}"
            print(f"\033[1m{header}\033[0m")  # Bold header
            print("-"*80)
            
            # Sort tasks by start time
            scheduled_tasks = [t for t in self.tasks if t.is_scheduled]
            scheduled_tasks.sort(key=lambda x: x.start_time)
            
            # Print each task with its security details
            for task in scheduled_tasks:
                # Find all messages associated with this task
                task_messages = []
                for message in self.messages:
                    if message.source_id == task.task_id or message.dest_id == task.task_id:
                        task_messages.append(message)
                
                # Get security notation for this task
                security_str = ""
                if task_messages:
                    # Find most secure protocol used for this task
                    max_conf = max([m.assigned_security['confidentiality'] for m in task_messages], default=0)
                    max_integ = max([m.assigned_security['integrity'] for m in task_messages], default=0)
                    max_auth = max([m.assigned_security['authentication'] for m in task_messages], default=0)
                    security_str = f"C{max_conf}, I{max_integ}, A{max_auth}"
                else:
                    security_str = "N/A"
                
                # Format and print task row
                task_row = (f"{task.task_id:<8} {task.name:<20} {task.assigned_processor:<10} "
                          f"{task.start_time:<10.2f} {task.finish_time:<10.2f} {security_str:<20}")
                print(task_row)
            
            print("-"*80)
            print(f"Makespan: {max(task.finish_time for task in scheduled_tasks):.2f}")
            print("="*80 + "\n")
            
        except Exception as e:
            print(f"Error printing schedule: {e}")
    
    def visualize_2d_plots(self):
        """
        Visualize training progress with 2D plots for makespan and security utility
        """
        try:
            import matplotlib.pyplot as plt
            
            # Enhance plot style
            plt.style.use('ggplot')
            
            # Create figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
            
            # 1. Security Utility vs Episode
            ax1.plot(self.training_history['episodes'], self.training_history['security_utility'], 
                    'b-', alpha=0.5, label='Current')
            ax1.plot(self.training_history['episodes'], self.training_history['best_security_utility'], 
                    'b-', linewidth=2, label='Best')
            ax1.set_title('Security Utility vs Episode', fontsize=18)
            ax1.set_xlabel('Episode', fontsize=16)
            ax1.set_ylabel('Security Utility', fontsize=16)
            ax1.tick_params(axis='both', which='major', labelsize=14)
            ax1.legend(fontsize=14)
            ax1.grid(True)
            
            # Add annotations for initial and final values
            ax1.annotate(f"Initial: {self.base_security_utility:.4f}",
                       xy=(0, self.base_security_utility),
                       xytext=(10, -20),
                       textcoords="offset points",
                       fontsize=14,
                       arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
            
            ax1.annotate(f"Final: {self.training_history['best_security_utility'][-1]:.4f}",
                       xy=(self.training_history['episodes'][-1], self.training_history['best_security_utility'][-1]),
                       xytext=(-70, 20),
                       textcoords="offset points",
                       fontsize=14,
                       arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
            
            # 2. Makespan vs Episode
            ax2.plot(self.training_history['episodes'], self.training_history['makespan'], 
                    'r-', alpha=0.5, label='Current')
            ax2.plot(self.training_history['episodes'], self.training_history['best_makespan'], 
                    'r-', linewidth=2, label='Best')
            # Add horizontal line for deadline
            ax2.axhline(y=self.deadline, color='k', linestyle='--', alpha=0.7, 
                       label=f'Deadline ({self.deadline})')
            ax2.set_title('Makespan vs Episode', fontsize=18)
            ax2.set_xlabel('Episode', fontsize=16)
            ax2.set_ylabel('Makespan', fontsize=16)
            ax2.tick_params(axis='both', which='major', labelsize=14)
            ax2.legend(fontsize=14)
            ax2.grid(True)
            
            # Add annotations for initial and final values
            ax2.annotate(f"Initial: {self.base_makespan:.2f}",
                       xy=(0, self.base_makespan),
                       xytext=(10, 20),
                       textcoords="offset points",
                       fontsize=14,
                       arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
            
            ax2.annotate(f"Final: {self.training_history['best_makespan'][-1]:.2f}",
                       xy=(self.training_history['episodes'][-1], self.training_history['best_makespan'][-1]),
                       xytext=(-70, -20),
                       textcoords="offset points",
                       fontsize=14,
                       arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
            
            # Add annotation with priority value
            plt.figtext(0.02, 0.02, f"Security Priority: {self.security_priority:.2f} - "
                                  f"{'Security focused' if self.security_priority > 0.5 else 'Makespan focused'}", 
                      fontsize=16, bbox=dict(facecolor='white', alpha=0.8))
            
            # Layout and save
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.3)
            
            # Save figure
            filename = f"ql_training_2d_plots_priority_{self.security_priority:.2f}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"2D training visualization saved as '{filename}'")
            
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Skipping 2D visualization.")
        except Exception as e:
            print(f"Error creating 2D visualization: {e}")
    
    def visualize_3d_plot(self):
        """
        Create a 3D visualization of security utility vs makespan vs episode
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib as mpl
            from mpl_toolkits.mplot3d import Axes3D
            
            # Enhance plot style
            plt.style.use('ggplot')
            
            # Create figure
            fig = plt.figure(figsize=(14, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            # Create a colormap based on episode numbers
            episodes = self.training_history['episodes']
            norm = mpl.colors.Normalize(vmin=min(episodes), vmax=max(episodes))
            cmap = plt.get_cmap('viridis')
            colors = cmap(norm(episodes))
            
            # Add scatter plot for evolution
            scatter = ax.scatter(self.training_history['security_utility'], 
                       self.training_history['makespan'], 
                       self.training_history['episodes'],
                       c=episodes, cmap='viridis', marker='o', s=40, alpha=0.7)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label('Episode', fontsize=14)
            cbar.ax.tick_params(labelsize=12)
            
            # Add trajectory line
            ax.plot(self.training_history['security_utility'], 
                  self.training_history['makespan'], 
                  self.training_history['episodes'],
                  'k-', alpha=0.3)
            
            # Add best values trajectory
            ax.plot(self.training_history['best_security_utility'], 
                  self.training_history['best_makespan'], 
                  self.training_history['episodes'],
                  'r-', linewidth=2, alpha=0.7, label='Best Solutions')
            
            # Add deadline plane
            x_min, x_max = min(self.training_history['security_utility']) * 0.95, max(self.training_history['security_utility']) * 1.05
            y_min, y_max = min(self.training_history['episodes']), max(self.training_history['episodes'])
            x_plane, y_plane = np.meshgrid(
                np.linspace(x_min, x_max, 10),
                np.linspace(y_min, y_max, 10)
            )
            z_plane = np.ones_like(x_plane) * self.deadline
            ax.plot_surface(x_plane, z_plane, y_plane, color='r', alpha=0.2, label='Deadline')
            
            # Set labels with larger font size
            ax.set_xlabel('Security Utility', fontsize=16)
            ax.set_ylabel('Makespan', fontsize=16)
            ax.set_zlabel('Episode', fontsize=16)
            ax.set_title('Security Utility vs Makespan vs Episode', fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            # Mark the beginning and end points
            ax.scatter([self.training_history['security_utility'][0]], 
                     [self.training_history['makespan'][0]], 
                     [self.training_history['episodes'][0]], 
                     color='blue', marker='s', s=150, label='Start')
            ax.scatter([self.training_history['security_utility'][-1]], 
                     [self.training_history['makespan'][-1]], 
                     [self.training_history['episodes'][-1]], 
                     color='green', marker='*', s=150, label='End')
            
            # Enhance best values
            ax.scatter([self.training_history['best_security_utility'][-1]], 
                     [self.training_history['best_makespan'][-1]], 
                     [self.training_history['episodes'][-1]], 
                     color='darkred', marker='o', s=200, label='Best Final')
            
            # Add legend
            ax.legend(fontsize=14)
            
            # Add text annotation for security priority
            ax.text2D(0.02, 0.02, f"Security Priority: {self.security_priority:.2f} - "
                               f"{'Security focused' if self.security_priority > 0.5 else 'Makespan focused'}", 
                   transform=ax.transAxes, fontsize=16, 
                   bbox=dict(facecolor='white', alpha=0.8))
            
            # Add annotations for initial and final values
            ax.text(self.training_history['security_utility'][0], 
                  self.training_history['makespan'][0], 
                  self.training_history['episodes'][0],
                  f"Initial: Util={self.base_security_utility:.4f}, MS={self.base_makespan:.2f}", 
                  fontsize=12)
            
            ax.text(self.training_history['best_security_utility'][-1], 
                  self.training_history['best_makespan'][-1], 
                  self.training_history['episodes'][-1],
                  f"Best: Util={self.training_history['best_security_utility'][-1]:.4f}, MS={self.training_history['best_makespan'][-1]:.2f}", 
                  fontsize=12)
            
            # Set viewing angle
            ax.view_init(elev=30, azim=45)
            
            # Save figure
            filename = f"ql_training_3d_plot_priority_{self.security_priority:.2f}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"3D training visualization saved as '{filename}'")
            
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Skipping 3D visualization.")
        except Exception as e:
            print(f"Error creating 3D visualization: {e}")

    def run(self):
        """Run the Q-Learning scheduler."""
        print("\nRunning Q-Learning Security-Aware Scheduler...")
        print(f"Security Priority: {self.security_priority:.2f}")
        makespan, security_utility = self.train()
        
        if makespan > self.deadline:
            print(f"Q-Learning failed to meet deadline. Makespan: {makespan}, Deadline: {self.deadline}")
            return None, None
        else:
            print(f"Q-Learning successful. Makespan: {makespan}, Security Utility: {security_utility}")
            return makespan, security_utility
# Create the TC system test case as described in the paper
# def create_tc_test_case():
#     # Create tasks with execution times as in Figure 10(c)
#     tasks = [
#         Task(1, "Wheel Speed 1", [205, 200]),
#         Task(2, "Wheel Speed 2", [207, 200]),
#         Task(3, "Wheel Speed 3", [190, 210]),
#         Task(4, "Wheel Speed 4", [200, 198]),
#         Task(5, "Driver Input", [150, 155]),
#         Task(6, "Slip Calculator", [297, 300], [1, 2, 3, 4]),
#         Task(7, "Valve Control", [175, 180]),
#         Task(8, "Actuator", [405,400], [5, 6, 7]),
#         Task(9, "Brake Control", [146, 154], [8]),
#         Task(10, "Throttle Control", [199,201], [8])
#     ]
    
#     # Create messages as in Figure 10(b)
#     messages = [
#         Message(1, 6, 512),
#         Message(2, 6, 512),
#         Message(3, 6, 512),
#         Message(4, 6, 512),
#         Message(5, 8, 128),
#         Message(6, 8, 128),
#         Message(7, 8, 64),
#         Message(8, 9, 512),
#         Message(8, 10, 512)
#     ]
    
#     security_values = [
#         (0.2, 0.1, 0.4, 0.3, 0.3, 0.4),
#         (0.2, 0.2, 0.4, 0.3, 0.5, 0.2),
#         (0.2, 0.5, 0.3, 0.2, 0.6, 0.2),
#         (0.3, 0.4, 0.2, 0.2, 0.2, 0.6),
#         (0.4, 0.3, 0.1, 0.2, 0.3, 0.5),
#         (0.4, 0.2, 0.4, 0.7, 0.1, 0.2),
#         (0.4, 0.1, 0.3, 0.7, 0.1, 0.2),
#         (0.3, 0.1, 0.2, 0.7, 0.1, 0.2),
#         (0.3, 0.2, 0.1, 0.7, 0.1, 0.2)
#     ]

#     for msg, (conf_min, integ_min, auth_min, conf_w, integ_w, auth_w) in zip(messages, security_values):
#         msg.set_security_requirements(conf_min, integ_min, auth_min, conf_w, integ_w, auth_w)

#     processors = [Processor(1,[0,5]), Processor(2,[10,0])]
#     network = CommunicationNetwork(2)
#     security_service = SecurityService()
#     deadline = 2200  # As specified in the paper
    
#     # # return tasks, messages, processors, network, security_service, deadline
#     # tasks = [
#     #     Task(1, "Source 1", [100, 150]),
#     #     Task(2, "Source 2", [120, 90]),
#     #     Task(3, "Source 3", [80, 110]),
#     #     Task(4, "Process A", [200, 150], [1, 2]),
#     #     Task(5, "Process B", [180, 220], [2, 3]),
#     #     Task(6, "Merge", [150, 130], [4, 5]),
#     #     Task(7, "Output 1", [90, 120], [6]),
#     #     Task(8, "Output 2", [110, 80], [6])
#     # ]
    
#     # # Create messages between tasks
#     # # Varying message sizes to create different security overhead impacts
#     # messages = [
#     #     Message(1, 4, 256),   # Source 1 -> Process A
#     #     Message(2, 4, 512),   # Source 2 -> Process A
#     #     Message(2, 5, 384),   # Source 2 -> Process B
#     #     Message(3, 5, 320),   # Source 3 -> Process B
#     #     Message(4, 6, 768),   # Process A -> Merge
#     #     Message(5, 6, 640),   # Process B -> Merge
#     #     Message(6, 7, 896),   # Merge -> Output 1
#     #     Message(6, 8, 448)    # Merge -> Output 2
#     # ]
    
#     # # Set security requirements with varying weights
#     # # Format: set_security_requirements(conf_min, integ_min, auth_min, conf_weight, integ_weight, auth_weight)
    
#     # # Message 1: Prioritize confidentiality
#     # messages[0].set_security_requirements(0.3, 0.1, 0.1, 0.7, 0.2, 0.1)
    
#     # # Message 2: Balanced security needs
#     # messages[1].set_security_requirements(0.25, 0.25, 0.2, 0.4, 0.3, 0.3)
    
#     # # Message 3: Prioritize integrity
#     # messages[2].set_security_requirements(0.1, 0.4, 0.1, 0.2, 0.6, 0.2)
    
#     # # Message 4: Low security requirements but high weights
#     # messages[3].set_security_requirements(0.15, 0.15, 0.15, 0.3, 0.4, 0.3)
    
#     # # Message 5: High security requirements (critical path)
#     # messages[4].set_security_requirements(0.4, 0.5, 0.3, 0.4, 0.4, 0.2)
    
#     # # Message 6: Medium security requirements
#     # messages[5].set_security_requirements(0.3, 0.3, 0.2, 0.3, 0.4, 0.3)
    
#     # # Message 7: High confidentiality for output
#     # messages[6].set_security_requirements(0.5, 0.2, 0.2, 0.6, 0.2, 0.2)
    
#     # # Message 8: High authentication for output
#     # messages[7].set_security_requirements(0.2, 0.2, 0.5, 0.3, 0.2, 0.5)
    
#     # processors = [Processor(1), Processor(2)]
#     # network = CommunicationNetwork(2, bandwidth=1000)  # Higher bandwidth
#     # security_service = SecurityService()
    
#     # # Set deadline with some slack to allow SHIELD to perform upgrades
#     # # Calculate a reasonable deadline based on the critical path plus some slack
#     # critical_path_time = 100 + 200 + 150 + 110  # Source 1 -> Process A -> Merge -> Output 2
#     # deadline = int(critical_path_time * 1.6)  # 60% slack
    
#     return tasks, messages, processors, network, security_service, deadline

def create_testcase():
    # Define tasks
    tasks=[
        Task(1, 'T1', [64,56,78],[]), 
        Task(2, 'T2', [46,36,58],[1]),
        Task(3, 'T3', [42,63,49],[1]),
        Task(4, 'T4', [36,54,42],[1]),
        Task(5, 'T5', [91,45,55],[2]),
        Task(6, 'T6', [58,72,84],[2,3,4]),
        Task(7, 'T7', [54,81,63],[3,4]),
        Task(8, 'T8', [75,42,61],[5,6,7]),
    ]
    messages = [
        Message(1, 2, 15),
        Message(1, 3, 27),
        Message(1, 4, 12),
        
        Message(2, 5, 34),
        Message(2, 6, 21),
        
        Message(3, 6, 40),
        Message(3, 7, 25),
        
        Message(4, 6, 28),
        Message(4, 7, 51),
        
        Message(5, 8, 27),
        Message(6, 8, 35),
        Message(7, 8, 27),
    ]
    security_values = [
        (0.3,0.2,0.4,0.2,0.3,0.5),
        (0.1,0.2,0.2,0.5,0.3,0.2),
        (0.1,0.4,0.3,0.3,0.6,0.1),
        
        (0.2,0.2,0.4,0.3,0.5,0.2),
        (0.4,0.2,0.3,0.7,0.1,0.2),
        
        (0.3,0.1,0.1,0.2,0.4,0.4),
        (0.3,0.2,0.1,0.1,0.3,0.6),
        
        (0.2,0.5,0.3,0.2,0.6,0.2),
        (0.3,0.1,0.4,0.2,0.4,0.4),
        
        (0.3,0.1,0.4,0.2,0.4,0.4),
        
        (0.3,0.2,0.3,0.3,0.6,0.1),
        
        (0.4,0.3,0.4,0.2,0.3,0.5),
        
    ]

    for msg, (conf_min, integ_min, auth_min, conf_w, integ_w, auth_w) in zip(messages, security_values):
        msg.set_security_requirements(conf_min, integ_min, auth_min, conf_w, integ_w, auth_w)
        
    processors = [Processor(1,[0,1,2]), Processor(2,[1,0,3]) , Processor(3,[2,3,0])]
    network = CommunicationNetwork(3)
    security_service = SecurityService()
    deadline = 500
    
    return tasks, messages, processors, network, security_service, deadline
        
def run_comparison():
    # Create test case
    tasks, messages, processors, network, security_service, deadline = create_testcase()
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
def compare_schedulers(tasks, messages, processors, network, security_service, deadline):
    """Compare HSMS and QL-based schedulers."""
    print("\n" + "="*80)
    print("COMPARING SCHEDULERS")
    print("="*80)
    
    # Run HSMS
    print("\nRunning HSMS scheduler...")
    hsms = HSMS(tasks, messages, processors, network, security_service, deadline)
    hsms_start = time.time()
    hsms_makespan, hsms_utility = hsms.run()
    hsms_time = time.time() - hsms_start
    
    if hsms_makespan is None:
        print("HSMS failed to meet deadline.")
        return
    
    # Display HSMS results
    print(f"\nHSMS Results:")
    print(f"Makespan: {hsms_makespan:.2f} ms")
    print(f"Security Utility: {hsms_utility:.4f}")
    print(f"Runtime: {hsms_time:.2f} seconds")
    
    # Display HSMS security information
    hsms.display_security_information("HSMS")
    
    # Run Q-Learning scheduler
    print("\nRunning Q-Learning Security-Aware scheduler...")
    ql = QLSecurityScheduler(tasks, messages, processors, network, security_service, deadline)
    ql_start = time.time()
    ql_makespan, ql_utility = ql.run()
    ql_time = time.time() - ql_start
    
    if ql_makespan is None:
        print("Q-Learning scheduler failed to meet deadline.")
        return
    
    # Display Q-Learning results
    print(f"\nQ-Learning Results:")
    print(f"Makespan: {ql_makespan:.2f} ms")
    print(f"Security Utility: {ql_utility:.4f}")
    print(f"Runtime: {ql_time:.2f} seconds")
    
    # Display Q-Learning security information
    ql.display_security_information("Q-Learning")
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Metric':<20} {'HSMS':<15} {'Q-Learning':<15} {'Improvement':<15}")
    print("-"*80)
    
    makespan_imp = ((hsms_makespan - ql_makespan) / hsms_makespan) * 100
    utility_imp = ((ql_utility - hsms_utility) / hsms_utility) * 100
    
    print(f"{'Makespan (ms)':<20} {hsms_makespan:<15.2f} {ql_makespan:<15.2f} {makespan_imp:+<15.2f}%")
    print(f"{'Security Utility':<20} {hsms_utility:<15.4f} {ql_utility:<15.4f} {utility_imp:+<15.2f}%")
    print(f"{'Runtime (s)':<20} {hsms_time:<15.2f} {ql_time:<15.2f} {'-':<15}")
    
    # Additional analysis
    ql.print_task_schedule()
    ql.display_task_security_details()
    
    return {
        "hsms": {
            "makespan": hsms_makespan,
            "security_utility": hsms_utility,
            "runtime": hsms_time
        },
        "ql": {
            "makespan": ql_makespan,
            "security_utility": ql_utility,
            "runtime": ql_time
        }
    }
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

# def main():
#     # Create the TC system test case
#     tasks, messages, processors, network, security_service, deadline = create_tc_test_case()
    
#     # Run HSMS
#     hsms = HSMS(tasks, messages, processors, network, security_service, deadline)
#     hsms_makespan, hsms_security_utility = hsms.run()
    
#     if hsms_makespan:
#         plot_gantt_chart("HSMS", hsms.schedule, len(processors), hsms_makespan)
#         print(f"HSMS Security Utility: {hsms_security_utility:.2f}")
    
#     # Run SHIELD
#     # Note: We need to create fresh copies of the tasks and processors
#     tasks_shield, messages, processors_shield, network, security_service, deadline = create_tc_test_case()
#     hsms.print_task_schedule()
    
#     shield = SHIELD(tasks_shield, messages, processors_shield, network, security_service, deadline)
#     shield_makespan, shield_security_utility = shield.run()
#     shield.print_task_schedule()
#     if shield_makespan:
#         plot_gantt_chart("SHIELD", shield.schedule, len(processors_shield), shield_makespan)
#         print(f"SHIELD Security Utility: {shield_security_utility:.2f}")
    
#     # Show the plots
#     plt.show()

def main():
    print("Creating test case...")
    tasks, messages, processors, network, security_service, deadline = create_testcase()
    
    # Run HSMS
    print("\nRunning HSMS...")
    hsms = HSMS(copy.deepcopy(tasks), copy.deepcopy(messages), 
                copy.deepcopy(processors), network, security_service, deadline)
    hsms_makespan, hsms_security_utility = hsms.run()
    
    if hsms_makespan:
        hsms.display_security_information("HSMS")
        hsms.print_task_schedule()
        hsms.display_task_security_details()
        hsms.calculate_and_print_security_details()  # Add this line
        plot_gantt_chart("HSMS Schedule", hsms.schedule, len(processors), hsms_makespan)
        plt.savefig("hsms_schedule.png")
    # Run SHIELD
    # print("\nRunning SHIELD...")
    # shield = SHIELD(copy.deepcopy(tasks), copy.deepcopy(messages), 
    #                 copy.deepcopy(processors), network, security_service, deadline)
    # shield_makespan, shield_security_utility = shield.run()
    
    # if shield_makespan:
    #     shield.display_security_information("SHIELD")
    #     shield.print_task_schedule()
    #     plot_gantt_chart("SHIELD Schedule", shield.schedule, len(processors), shield_makespan)
    #     plt.savefig("shield_schedule.png")
    
    # # Compare results
    
    ql_shield = QLSecurityScheduler(copy.deepcopy(tasks), copy.deepcopy(messages),
                                    copy.deepcopy(processors), network, security_service, deadline,0.1,0.9,0.3,2000,1)
    ql_makespan, ql_security_utility = ql_shield.run()
    plot_gantt_chart("Q-Learning Schedule", ql_shield.schedule, len(processors), ql_makespan)
    ql_shield.display_task_security_details()
    
    # if ql_makespan:
    #     ql_shield.display_security_information("Q-Learning")
    #     ql_shield.print_task_schedule()
    #     plot_gantt_chart("Q-Learning Schedule", ql_shield.schedule, len(processors), ql_makespan)
    #     plt.savefig("ql_schedule.png")
    #     ql_shield.display_task_security_details()
    #     ql_shield.calculate_and_print_security_details()  # Add this line
    #     ql_shield.visualize_2d_plots()
    #     ql_shield.visualize_3d_plot()
    # plt.show()
    # print(f"HSMS Makespan: {hsms_makespan:.2f} seconds | Security Utility: {hsms_security_utility:.2f}")
    # print(f"Q-Learning Makespan: {ql_makespan:.2f} seconds | Security Utility: {ql_security_utility:.2f}")

    # compare_schedulers  (tasks, messages, processors, network, security_service, deadline)


if __name__ == "__main__":
    main()