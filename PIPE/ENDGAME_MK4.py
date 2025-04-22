import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import random
import types
import heapq

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
        """Calculate the security overhead for a message regardless of processor assignment."""
        # Modified to always apply security overhead
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
                
            # Always calculate security overhead regardless of processor assignment
            security_overhead = self.calc_security_overhead(message, pred_task.assigned_processor, processor)
            
            # Calculate communication time
            source_proc = self.processors[pred_task.assigned_processor - 1]
            dest_proc = self.processors[processor - 1]
            comm_time = source_proc.get_communication_time(message, source_proc, dest_proc) if source_proc != dest_proc else 0
            
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
                    
                # Apply communication time regardless of processor assignment
                source_proc = self.processors[pred_task.assigned_processor - 1]
                dest_proc = processor
                comm_time = source_proc.get_communication_time(message, source_proc, dest_proc) if source_proc != dest_proc else 0
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
            
            # Calculate incoming security overhead explicitly for receiver
            # MODIFIED: Apply security overhead regardless of processor assignment
            incoming_security_overhead = 0
            for pred_id in task.predecessors:
                pred_task = self.get_task_by_id(pred_id)
                if not pred_task or not pred_task.is_scheduled:
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
            
            # Calculate outgoing security overhead explicitly for sender
            # MODIFIED: Apply security overhead regardless of processor assignment
            outgoing_security_overhead = 0
            for succ_id in task.successors:
                succ_task = self.get_task_by_id(succ_id)
                if not succ_task or not succ_task.is_scheduled:
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
                
                # Include authentication generation overhead
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
        """Schedule tasks using security-aware HEFT algorithm with all security overhead applied during initial scheduling."""
        # Calculate task priorities
        self.compute_task_priorities()
        
        # Assign minimum security levels to all messages
        self.assign_minimum_security()
        
        # Sort tasks by priority (descending)
        sorted_tasks = sorted(self.tasks, key=lambda t: -t.priority)
        
        # Schedule each task with full security overhead calculation
        for task in sorted_tasks:
            best_processor = None
            earliest_finish_time = float('inf')
            earliest_start_time = 0
            
            # Try each processor
            for proc_idx, processor in enumerate(self.processors):
                proc_id = processor.proc_id
                
                # Calculate earliest start time based on predecessors
                est = 0
                predecessor_overhead = 0  # Security overhead from incoming messages
                
                for pred_id in task.predecessors:
                    pred_task = self.get_task_by_id(pred_id)
                    if not pred_task.is_scheduled:
                        # Cannot schedule this task yet if predecessors aren't scheduled
                        est = float('inf')
                        break
                    
                    # Get the message between predecessor and current task
                    message = self.get_message(pred_id, task.task_id)
                    if not message:
                        continue
                    
                    # Calculate communication time
                    source_proc = self.processors[pred_task.assigned_processor - 1]
                    dest_proc = processor
                    comm_time = source_proc.get_communication_time(message, source_proc, dest_proc) if source_proc != dest_proc else 0
                    
                    # Calculate security overhead for this message (receiver side)
                    message_security_overhead = 0
                    
                    # Apply security overhead for each service (receiver side)
                    for service in ['confidentiality', 'integrity']:
                        protocol_idx = message.assigned_security[service] + 1  # 1-indexed
                        overhead_factor = self.security.overheads[service][protocol_idx][proc_id - 1]
                        overhead = message.size / overhead_factor
                        message_security_overhead += overhead
                    
                    # Add authentication verification overhead
                    auth_idx = message.assigned_security['authentication'] + 1
                    auth_overhead = self.security.overheads['authentication'][auth_idx][proc_id - 1]
                    message_security_overhead += auth_overhead
                    
                    # Update predecessor overhead
                    predecessor_overhead += message_security_overhead
                    
                    # Update EST based on predecessor finish plus communication time
                    pred_finish_with_comm = pred_task.finish_time + comm_time
                    est = max(est, pred_finish_with_comm)
                
                # If any predecessor is not scheduled, skip this processor
                if est == float('inf'):
                    continue
                
                # Account for processor availability
                est = max(est, processor.available_time)
                
                # Base execution time without security overhead
                base_exec_time = task.execution_times[proc_id - 1]
                
                # Calculate outgoing security overhead (sender side)
                successor_overhead = 0
                for succ_id in task.successors:
                    message = self.get_message(task.task_id, succ_id)
                    if not message:
                        continue
                    
                    # Calculate overhead for each security service (sender side)
                    for service in ['confidentiality', 'integrity']:
                        protocol_idx = message.assigned_security[service] + 1  # 1-indexed
                        overhead_factor = self.security.overheads[service][protocol_idx][proc_id - 1]
                        overhead = message.size / overhead_factor
                        successor_overhead += overhead
                    
                    # Include authentication generation overhead
                    auth_idx = message.assigned_security['authentication'] + 1
                    auth_overhead = self.security.overheads['authentication'][auth_idx][proc_id - 1]
                    successor_overhead += auth_overhead
                
                # Total security overhead for this task on this processor
                total_security_overhead = predecessor_overhead + successor_overhead
                
                # Calculate finish time with all overheads included
                eft = est + base_exec_time + total_security_overhead
                
                # Keep track of the best processor assignment
                if eft < earliest_finish_time:
                    earliest_finish_time = eft
                    earliest_start_time = est
                    best_processor = processor
                    # Store the overhead calculations for the best processor
                    best_total_overhead = total_security_overhead
            
            # Assign task to the best processor with all overhead calculations included
            if best_processor:
                task.assigned_processor = best_processor.proc_id
                task.start_time = earliest_start_time
                
                # Base execution time for the assigned processor
                base_exec_time = task.execution_times[task.assigned_processor - 1]
                
                # Set finish time with security overhead included
                task.finish_time = earliest_start_time + base_exec_time + best_total_overhead
                task.is_scheduled = True
                
                # Update processor availability time
                best_processor.available_time = task.finish_time
                
                # Add to schedule
                self.schedule.append({
                    'task_id': task.task_id,
                    'name': task.name,
                    'processor': task.assigned_processor,
                    'start_time': task.start_time,
                    'finish_time': task.finish_time,
                    'security_overhead': best_total_overhead
                })
                
                print(f"Scheduled Task {task.task_id} ({task.name}) on Processor {task.assigned_processor}")
                print(f"  Start time: {task.start_time:.2f} ms")
                print(f"  Base execution: {base_exec_time:.2f} ms")
                print(f"  Security overhead: {best_total_overhead:.2f} ms")
                print(f"  Finish time: {task.finish_time:.2f} ms")
        
        # Calculate the makespan
        self.makespan = max(task.finish_time for task in self.tasks if task.is_scheduled)
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
        """Calculate the total security utility for all messages regardless of processor assignment."""
        total_utility = 0
        for message in self.messages:
            # Get source and destination tasks
            source_task = self.get_task_by_id(message.source_id)
            dest_task = self.get_task_by_id(message.dest_id)
            
            # Skip if either task is not scheduled 
            if (not source_task or not dest_task or 
                not source_task.is_scheduled or not dest_task.is_scheduled):
                continue
            
            # Calculate utility for all messages
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
    """
    SHIELD: Security-aware scHedulIng with sEcurity utiLity maximization unDer real-time constraints
    
    This algorithm extends HSMS by enhancing the security levels of messages 
    while maintaining the task schedule within deadline constraints.
    """
    
    def __init__(self, tasks, messages, processors, network, security_service, deadline):
        super().__init__(tasks, messages, processors, network, security_service, deadline)
        self.original_makespan = None
        self.bcr_heap = []  # Benefit-to-Cost Ratio heap
        self.upgrade_history = []  # Track security upgrades for reporting
    
    def run(self):
        """Execute the SHIELD algorithm in phases."""
        # Phase 1 & 2: Run HSMS first to get initial schedule
        makespan, security_utility = super().run()
        
        if makespan is None or makespan > self.deadline:
            print("SHIELD cannot proceed: HSMS failed to meet deadline requirements")
            return None, None
        
        self.original_makespan = makespan
        print(f"\nSHIELD Phase 1-2 complete: Initial makespan={makespan}, Security utility={security_utility}")
        
        # Phase 3: Security Enhancement
        print("\nSHIELD Phase 3: Starting Security Enhancement...")
        self.enhance_security()
        
        # Calculate final results
        final_makespan = self.finalize_schedule()
        final_security_utility = self.calculate_security_utility()
        
        print(f"SHIELD complete: Final makespan={final_makespan}, Final security utility={final_security_utility}")
        print(f"Security utility improvement: {(final_security_utility - security_utility) / security_utility * 100:.2f}%")
        
        return final_makespan, final_security_utility
    
    def enhance_security(self):
        """Phase 3 of SHIELD: Enhance security levels using available slack time."""
        # Initialize BCR heap with all possible upgrades
        self.initialize_bcr_heap()
        
        upgrade_count = 0
        rejected_count = 0
        
        # Process upgrades in order of highest benefit-to-cost ratio
        while self.bcr_heap:
            # Get the upgrade with the highest BCR
            bcr, (message_id, service, current_level) = heapq.heappop(self.bcr_heap)
            bcr = -bcr  # Negate back to positive value (heap is min-heap)
            
            # Find the message
            message = None
            for msg in self.messages:
                if msg.id == message_id:
                    message = msg
                    break
            
            if not message:
                continue
                
            # Check if we can upgrade further
            next_level = current_level + 1
            max_level = len(self.security.strengths[service]) - 1
            
            if next_level > max_level:
                continue  # Already at max level
            
            print(f"\nTrying to upgrade {message_id} {service} from level {current_level} to {next_level} (BCR: {bcr:.4f})")
            
            # Try to apply the upgrade
            success = self.try_security_upgrade(message, service, next_level)
            
            if success:
                upgrade_count += 1
                # Record this upgrade
                self.upgrade_history.append({
                    'message_id': message_id,
                    'service': service,
                    'from_level': current_level,
                    'to_level': next_level,
                    'bcr': bcr
                })
                
                # If we can upgrade further, calculate new BCR and add back to heap
                if next_level < max_level:
                    new_bcr = self.calculate_bcr(message, service, next_level)
                    if new_bcr > 0:
                        heapq.heappush(self.bcr_heap, (-new_bcr, (message_id, service, next_level)))
                
                print(f"✅ Upgrade successful! New {service} level: {next_level}")
            else:
                rejected_count += 1
                print(f"❌ Upgrade rejected: insufficient slack")
        
        print(f"\nSecurity enhancement complete: {upgrade_count} upgrades applied, {rejected_count} rejected")
    
    def initialize_bcr_heap(self):
        """Initialize the benefit-to-cost ratio heap with all possible upgrades."""
        self.bcr_heap = []
        
        for message in self.messages:
            source_task = self.get_task_by_id(message.source_id)
            dest_task = self.get_task_by_id(message.dest_id)
            
            # Skip if either task is not scheduled
            if not source_task or not dest_task or not source_task.is_scheduled or not dest_task.is_scheduled:
                continue
            
            # For each security service, calculate BCR for upgrading to next level
            for service in ['confidentiality', 'integrity', 'authentication']:
                current_level = message.assigned_security[service]
                max_level = len(self.security.strengths[service]) - 1
                
                # Skip if already at maximum level
                if current_level >= max_level:
                    continue
                
                bcr = self.calculate_bcr(message, service, current_level)
                if bcr > 0:
                    # Use negative BCR for max-heap in a min-heap implementation
                    heapq.heappush(self.bcr_heap, (-bcr, (message.id, service, current_level)))
        
        print(f"Initialized BCR heap with {len(self.bcr_heap)} potential upgrades")
    
    def calculate_bcr(self, message, service, current_level):
        """
        Calculate Benefit-to-Cost Ratio for upgrading from current_level to next_level
        
        BCR = [F_G × (B_G,next - B_G,current)] / [overhead(next) - overhead(current)]
        """
        next_level = current_level + 1
        max_level = len(self.security.strengths[service]) - 1
        
        if next_level > max_level:
            return 0  # Cannot upgrade further
        
        # Get security service weights and strengths
        service_weight = message.weights[service]
        current_strength = self.security.strengths[service][current_level]
        next_strength = self.security.strengths[service][next_level]
        
        # Calculate security utility gain
        utility_gain = service_weight * (next_strength - current_strength)
        
        # Calculate additional overhead
        source_task = self.get_task_by_id(message.source_id)
        dest_task = self.get_task_by_id(message.dest_id)
        
        if not source_task or not dest_task or not source_task.is_scheduled or not dest_task.is_scheduled:
            return 0
            
        source_proc = source_task.assigned_processor
        dest_proc = dest_task.assigned_processor
        
        # Calculate current and next level overhead
        current_overhead = self.calculate_service_overhead(message, service, current_level, source_proc, dest_proc)
        next_overhead = self.calculate_service_overhead(message, service, next_level, source_proc, dest_proc)
        
        overhead_increase = next_overhead - current_overhead
        
        if overhead_increase <= 0:
            return float('inf')  # Special case: benefit with no cost
            
        # Calculate BCR
        bcr = utility_gain / overhead_increase
        
        return bcr
    
    def calculate_service_overhead(self, message, service, level, source_proc, dest_proc):
        """Calculate the overhead for a specific security service at a given level."""
        protocol_idx = level + 1  # Convert to 1-indexed for overhead table
        
        if service in ['confidentiality', 'integrity']:
            # Data-dependent overhead
            source_overhead = message.size / self.security.overheads[service][protocol_idx][source_proc - 1]
            dest_overhead = message.size / self.security.overheads[service][protocol_idx][dest_proc - 1]
            return source_overhead + dest_overhead
        else:  # Authentication
            # Fixed overhead
            source_overhead = self.security.overheads[service][protocol_idx][source_proc - 1]
            dest_overhead = self.security.overheads[service][protocol_idx][dest_proc - 1]
            return source_overhead + dest_overhead
    
    def try_security_upgrade(self, message, service, new_level):
        """
        Try to upgrade a message's security level and check if it breaks the schedule.
        If successful, update the schedule; otherwise, revert changes.
        """
        # Save current state for rollback if needed
        old_level = message.assigned_security[service]
        old_task_states = self.save_task_states()
        old_processor_states = self.save_processor_states()
        
        # Apply the upgrade
        message.assigned_security[service] = new_level
        
        # Update the schedule with the new security level
        self.update_schedule_with_upgrade(message)
        
        # Check if we still meet the deadline
        makespan = max(task.finish_time for task in self.tasks if task.is_scheduled)
        
        if makespan <= self.deadline:
            return True  # Upgrade successful
        else:
            # Rollback the changes
            message.assigned_security[service] = old_level
            self.restore_task_states(old_task_states)
            self.restore_processor_states(old_processor_states)
            return False  # Upgrade rejected
    
    def save_task_states(self):
        """Save current task states for potential rollback."""
        states = {}
        for task in self.tasks:
            states[task.task_id] = {
                'start_time': task.start_time,
                'finish_time': task.finish_time
            }
        return states
    
    def save_processor_states(self):
        """Save current processor states for potential rollback."""
        states = {}
        for i, proc in enumerate(self.processors):
            states[i] = proc.available_time
        return states
    
    def restore_task_states(self, states):
        """Restore task states from saved states."""
        for task in self.tasks:
            if task.task_id in states:
                saved = states[task.task_id]
                task.start_time = saved['start_time']
                task.finish_time = saved['finish_time']
    
    def restore_processor_states(self, states):
        """Restore processor states from saved states."""
        for i, proc in enumerate(self.processors):
            if i in states:
                proc.available_time = states[i]
    
    def update_schedule_with_upgrade(self, message):
        """
        Update task schedule after a security upgrade.
        This propagates changes through the task graph to ensure consistency.
        """
        # Find source and destination tasks
        source_task = self.get_task_by_id(message.source_id)
        dest_task = self.get_task_by_id(message.dest_id)
        
        if not source_task or not dest_task:
            return
            
        # Reset processor available times
        for proc in self.processors:
            proc.available_time = 0
            
        # Create a topologically sorted list of tasks
        sorted_tasks = self.topological_sort()
        
        # Recalculate all task times with the new security settings
        for task in sorted_tasks:
            if not task.is_scheduled:
                continue
                
            processor_id = task.assigned_processor
            processor = self.processors[processor_id - 1]
            
            # Calculate earliest start time based on predecessors
            est = 0
            for pred_id in task.predecessors:
                pred_task = self.get_task_by_id(pred_id)
                if not pred_task or not pred_task.is_scheduled:
                    continue
                    
                message = self.get_message(pred_id, task.task_id)
                if not message:
                    continue
                    
                # Calculate communication time
                source_proc = self.processors[pred_task.assigned_processor - 1]
                dest_proc = processor
                comm_time = source_proc.get_communication_time(message, source_proc, dest_proc) if pred_task.assigned_processor != task.assigned_processor else 0
                
                # Calculate security overhead
                security_overhead = self.calc_security_overhead(message, pred_task.assigned_processor, task.assigned_processor)
                
                # Update EST based on predecessor finish plus communication and security overhead
                pred_finish_with_comm = pred_task.finish_time + comm_time + security_overhead
                est = max(est, pred_finish_with_comm)
            
            # Also consider processor availability
            est = max(est, processor.available_time)
            
            # Calculate base execution time
            base_exec_time = task.execution_times[processor_id - 1]
            
            # Calculate outgoing security overhead
            outgoing_security_overhead = 0
            for succ_id in task.successors:
                succ_task = self.get_task_by_id(succ_id)
                if not succ_task or not succ_task.is_scheduled:
                    continue
                    
                message = self.get_message(task.task_id, succ_id)
                if not message:
                    continue
                
                # Only calculate security overhead if it's not already accounted for at destination
                source_side_overhead = self.calculate_sender_security_overhead(message, task.assigned_processor)
                outgoing_security_overhead += source_side_overhead
            
            # Update task timing
            task.start_time = est
            task.finish_time = est + base_exec_time + outgoing_security_overhead
            
            # Update processor availability
            processor.available_time = task.finish_time
            
            # Update schedule entry
            for entry in self.schedule:
                if entry['task_id'] == task.task_id:
                    entry['start_time'] = task.start_time
                    entry['finish_time'] = task.finish_time
                    entry['security_overhead'] = outgoing_security_overhead
    
    def calculate_sender_security_overhead(self, message, processor_id):
        """Calculate only the sender-side security overhead for a message."""
        total_overhead = 0
        
        # Add overhead for all three services at the sender side only
        for service in ['confidentiality', 'integrity']:
            protocol_idx = message.assigned_security[service] + 1  # 1-indexed in the overhead table
            overhead_factor = self.security.overheads[service][protocol_idx][processor_id - 1]
            overhead = message.size / overhead_factor
            total_overhead += overhead
        
        # Add authentication overhead (sender side)
        auth_idx = message.assigned_security['authentication'] + 1
        auth_overhead = self.security.overheads['authentication'][auth_idx][processor_id - 1]
        total_overhead += auth_overhead
        
        return total_overhead
    
    def topological_sort(self):
        """Return tasks in topological order (prerequisite tasks before dependent tasks)."""
        # Create a dictionary to track visited nodes
        visited = {task.task_id: False for task in self.tasks}
        temp = {task.task_id: False for task in self.tasks}
        order = []
        
        def dfs(task_id):
            if temp[task_id]:
                # Cycle detected
                return False
            if visited[task_id]:
                return True
                
            task = self.get_task_by_id(task_id)
            if not task:
                return True
                
            # Mark as temporarily visited
            temp[task_id] = True
            
            # Visit all successors
            for succ_id in task.successors:
                if not dfs(succ_id):
                    return False
            
            # Mark as permanently visited
            temp[task_id] = False
            visited[task_id] = True
            
            # Add to result
            order.append(task)
            return True
        
        # Visit all nodes
        for task in self.tasks:
            if not visited[task.task_id]:
                if not dfs(task.task_id):
                    # Cycle detected
                    print("Error: Cycle detected in task graph")
                    return sorted(self.tasks, key=lambda t: t.start_time if t.start_time is not None else float('inf'))
        
        # Reverse to get correct order
        return list(reversed(order))
    
    def print_upgrade_history(self):
        """Print the history of security upgrades applied by SHIELD."""
        print("\n" + "="*100)
        print("SHIELD SECURITY UPGRADE HISTORY")
        print("="*100)
        print(f"{'#':<5} {'Message':<12} {'Service':<15} {'From Level':<12} {'To Level':<12} {'From Strength':<15} {'To Strength':<15} {'BCR':<10}")
        print("-"*100)
        
        for i, upgrade in enumerate(self.upgrade_history, 1):
            message_id = upgrade['message_id']
            service = upgrade['service']
            from_level = upgrade['from_level']
            to_level = upgrade['to_level']
            from_strength = self.security.strengths[service][from_level]
            to_strength = self.security.strengths[service][to_level]
            bcr = upgrade['bcr']
            
            print(f"{i:<5} {message_id:<12} {service.capitalize():<15} {from_level:<12} {to_level:<12} "
                 f"{from_strength:<15.3f} {to_strength:<15.3f} {bcr:<10.4f}")
        
        if not self.upgrade_history:
            print("No security upgrades were applied")
        
        print("-"*100)
        print(f"Total upgrades: {len(self.upgrade_history)}")
        print("="*100)
    
    def security_improvement_summary(self):
        """Generate a summary of security improvements after SHIELD enhancement."""
        # Group messages by service type to see improvements
        service_improvements = {
            'confidentiality': {'count': 0, 'avg_before': 0, 'avg_after': 0},
            'integrity': {'count': 0, 'avg_before': 0, 'avg_after': 0},
            'authentication': {'count': 0, 'avg_before': 0, 'avg_after': 0}
        }
        
        # Track all messages that were modified
        modified_messages = set()
        for upgrade in self.upgrade_history:
            modified_messages.add(upgrade['message_id'])
        
        # Count messages by service and security level
        message_counts = {
            'confidentiality': {},
            'integrity': {},
            'authentication': {}
        }
        
        # Calculate average security levels before and after
        for message in self.messages:
            source_task = self.get_task_by_id(message.source_id)
            dest_task = self.get_task_by_id(message.dest_id)
            
            if not source_task or not dest_task or not source_task.is_scheduled or not dest_task.is_scheduled:
                continue
                
            # Track original security levels (we know from the modified list)
            was_modified = message.id in modified_messages
            
            for service in ['confidentiality', 'integrity', 'authentication']:
                level = message.assigned_security[service]
                strength = self.security.strengths[service][level]
                
                # Count at the current level
                if level not in message_counts[service]:
                    message_counts[service][level] = 0
                message_counts[service][level] += 1
                
                # Track for averaging
                service_improvements[service]['count'] += 1
                service_improvements[service]['avg_after'] += strength
                
                # For modified messages, find original level from history
                if was_modified:
                    original_level = level
                    for upgrade in self.upgrade_history:
                        if upgrade['message_id'] == message.id and upgrade['service'] == service:
                            original_level = upgrade['from_level']
                            break
                    original_strength = self.security.strengths[service][original_level]
                    service_improvements[service]['avg_before'] += original_strength
                else:
                    # Not modified, so before = after
                    service_improvements[service]['avg_before'] += strength
        
        # Calculate averages
        for service in service_improvements:
            if service_improvements[service]['count'] > 0:
                service_improvements[service]['avg_before'] /= service_improvements[service]['count']
                service_improvements[service]['avg_after'] /= service_improvements[service]['count']
        
        # Print summary
        print("\n" + "="*100)
        print("SHIELD SECURITY IMPROVEMENT SUMMARY")
        print("="*100)
        print(f"{'Service':<15} {'Avg Before':<15} {'Avg After':<15} {'Improvement':<15} {'% Increase':<15}")
        print("-"*100)
        
        for service, data in service_improvements.items():
            if data['count'] > 0:
                avg_before = data['avg_before']
                avg_after = data['avg_after']
                improvement = avg_after - avg_before
                pct_increase = (improvement / avg_before * 100) if avg_before > 0 else 0
                
                print(f"{service.capitalize():<15} {avg_before:<15.4f} {avg_after:<15.4f} "
                     f"{improvement:<15.4f} {pct_increase:<15.2f}%")
        
        print("\nSecurity Level Distribution After SHIELD:")
        for service in ['confidentiality', 'integrity', 'authentication']:
            levels = sorted(message_counts[service].items())
            print(f"\n{service.capitalize()} Levels:")
            for level, count in levels:
                strength = self.security.strengths[service][level]
                print(f"  Level {level} (strength {strength:.3f}): {count} messages")
        
        print("="*100)
        
        return service_improvements


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

class GeneticScheduler:
    """
    A genetic algorithm-based scheduler that enhances HSMS and SHIELD algorithms
    using evolutionary search techniques.
    """
    
    def __init__(self, tasks, messages, processors, network, security_service, deadline,
                 pop_size=50, generations=100, elite_size=5, crossover_rate=0.8, 
                 mutation_rate=0.2, stagnation_limit=15):
        # Core components
        self.tasks = tasks
        self.messages = messages
        self.processors = processors
        self.network = network
        self.security = security_service
        self.deadline = deadline
        
        # Seed the initial schedules
        self.hsms_solution = None
        self.shield_solution = None
        
        # GA parameters
        self.population_size = pop_size
        self.generations = generations
        self.elite_size = elite_size
        self.initial_crossover_rate = crossover_rate
        self.initial_mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.stagnation_limit = stagnation_limit
        
        # Runtime tracking
        self.stagnation_counter = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_makespan_history = []
        self.best_security_history = []
        self.diversity_history = []
        
        # Current population
        self.population = []
        self.best_solution = None
        
        # Number of tasks and processors
        self.num_tasks = len(tasks)
        self.num_processors = len(processors)
        
        # Dependency graph
        self.successors = self._build_successor_graph()
    
    def _build_successor_graph(self):
        """Build a graph of task successors for precedence constraints."""
        successors = {task.task_id: [] for task in self.tasks}
        
        for task in self.tasks:
            for pred_id in task.predecessors:
                successors[pred_id].append(task.task_id)
        
        return successors
    
    def run(self):
        """Execute the genetic algorithm scheduler."""
        start_time = time.time()
        
        # Generate seed solutions
        self._generate_seed_solutions()
        
        # Initialize population
        self._initialize_population()
        
        # Evaluate initial population
        self._evaluate_population()
        
        # Print initial best solution
        best_solution = self._get_best_solution()
        print(f"Initial best solution: Makespan={best_solution['makespan']:.2f}, Security={best_solution['security_utility']:.4f}")
        
        # Main GA loop
        for gen in range(self.generations):
            # Adapt parameters
            self._adapt_parameters(gen)
            
            # Create new population
            new_population = self._create_new_population()
            
            # Replace population
            self.population = new_population
            
            # Evaluate new population
            self._evaluate_population()
            
            # Update best solution
            current_best = self._get_best_solution()
            
            if self.best_solution is None or current_best['fitness'] > self.best_solution['fitness']:
                self.best_solution = current_best
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
            
            # Track history
            self.best_fitness_history.append(current_best['fitness'])
            avg_fitness = sum(ind['fitness'] for ind in self.population) / len(self.population)
            self.avg_fitness_history.append(avg_fitness)
            self.best_makespan_history.append(current_best['makespan'])
            self.best_security_history.append(current_best['security_utility'])
            
            # Calculate diversity
            diversity = self._calculate_diversity()
            self.diversity_history.append(diversity)
            
            # Print progress
            if gen % 10 == 0 or gen == self.generations - 1:
                print(f"Generation {gen}: Best Makespan={current_best['makespan']:.2f}, "
                      f"Security={current_best['security_utility']:.4f}, "
                      f"Fitness={current_best['fitness']:.4f}, "
                      f"Diversity={diversity:.4f}")
            
            # Check for cataclysmic mutation
            if self.stagnation_counter >= self.stagnation_limit:
                print(f"Stagnation detected at generation {gen}. Applying cataclysmic mutation.")
                self._apply_cataclysmic_mutation()
                self.stagnation_counter = 0
            
            # Termination condition
            if gen > 20 and max(self.best_fitness_history[-20:]) <= self.best_fitness_history[-20]:
                print(f"Early termination at generation {gen}: No improvement for 20 generations.")
                break
        
        end_time = time.time()
        
        # Print final results
        print("\nGenetic Algorithm completed.")
        print(f"Total runtime: {end_time - start_time:.2f} seconds")
        print(f"Best solution: Makespan={self.best_solution['makespan']:.2f}, "
              f"Security={self.best_solution['security_utility']:.4f}")
        
        return self.best_solution['makespan'], self.best_solution['security_utility']
    
    def _generate_seed_solutions(self):
        """Generate HSMS and SHIELD solutions to seed the population."""
        # Create HSMS solution
        print("Generating HSMS seed solution...")
        hsms = HSMS(copy.deepcopy(self.tasks), copy.deepcopy(self.messages), 
                    copy.deepcopy(self.processors), self.network, self.security, self.deadline)
        
        hsms_makespan, hsms_security = hsms.run()
        
        if hsms_makespan:
            self.hsms_solution = {
                'task_seq': self._extract_task_sequence(hsms),
                'proc_map': self._extract_processor_map(hsms),
                'security_levels': self._extract_security_levels(hsms),
                'makespan': hsms_makespan,
                'security_utility': hsms_security,
                'fitness': self._calculate_fitness(hsms_makespan, hsms_security)
            }
            print(f"HSMS seed solution: Makespan={hsms_makespan:.2f}, Security={hsms_security:.4f}")
        else:
            print("HSMS failed to generate a valid solution.")
        
        # Create SHIELD solution
        print("Generating SHIELD seed solution...")
        shield = SHIELD(copy.deepcopy(self.tasks), copy.deepcopy(self.messages), 
                       copy.deepcopy(self.processors), self.network, self.security, self.deadline)
        
        shield_makespan, shield_security = shield.run()
        
        if shield_makespan:
            self.shield_solution = {
                'task_seq': self._extract_task_sequence(shield),
                'proc_map': self._extract_processor_map(shield),
                'security_levels': self._extract_security_levels(shield),
                'makespan': shield_makespan,
                'security_utility': shield_security,
                'fitness': self._calculate_fitness(shield_makespan, shield_security)
            }
            print(f"SHIELD seed solution: Makespan={shield_makespan:.2f}, Security={shield_security:.4f}")
        else:
            print("SHIELD failed to generate a valid solution.")
    
    def _extract_task_sequence(self, scheduler):
        """Extract task sequence from a scheduler solution."""
        # Sort tasks by their start time
        sorted_tasks = sorted(scheduler.tasks, key=lambda t: t.start_time if t.start_time is not None else float('inf'))
        
        # Extract task IDs in sequence
        task_seq = [task.task_id for task in sorted_tasks if task.is_scheduled]
        
        return task_seq
    
    def _extract_processor_map(self, scheduler):
        """Extract processor assignment from a scheduler solution."""
        proc_map = {}
        for task in scheduler.tasks:
            if task.is_scheduled:
                proc_map[task.task_id] = task.assigned_processor
        
        return proc_map
    
    def _extract_security_levels(self, scheduler):
        """Extract security levels from a scheduler solution."""
        security_levels = {}
        for message in scheduler.messages:
            security_levels[message.id] = {
                'confidentiality': message.assigned_security['confidentiality'],
                'integrity': message.assigned_security['integrity'],
                'authentication': message.assigned_security['authentication']
            }
        
        return security_levels
    
    def _initialize_population(self):
        """Initialize the GA population with diverse solutions."""
        self.population = []
        
        # Always include HSMS and SHIELD solutions if they exist
        if self.hsms_solution:
            self.population.append(self.hsms_solution)
        
        if self.shield_solution:
            self.population.append(self.shield_solution)
        
        # Generate the rest of the population
        while len(self.population) < self.population_size:
            # Create a new individual using CTS (Candidate Task Set) approach
            individual = self._generate_individual_with_cts()
            self.population.append(individual)
    
    def _generate_individual_with_cts(self):
        """Generate a valid individual using Candidate Task Set approach."""
        # Start with empty sequences
        task_seq = []
        proc_map = {}
        
        # Track tasks that have been scheduled
        scheduled = set()
        
        # Create a copy of all tasks and their predecessors
        remaining_predecessors = {task.task_id: set(task.predecessors) for task in self.tasks}
        
        # Continue until all tasks are scheduled
        while len(scheduled) < self.num_tasks:
            # Find the candidate task set (tasks whose predecessors are all scheduled)
            candidate_tasks = []
            for task in self.tasks:
                if task.task_id not in scheduled and remaining_predecessors[task.task_id].issubset(scheduled):
                    candidate_tasks.append(task.task_id)
            
            # If no candidate tasks, there's a problem (should not happen with valid inputs)
            if not candidate_tasks:
                break
            
            # Randomly select a task from the candidate set
            selected_task = random.choice(candidate_tasks)
            
            # Add the task to the sequence
            task_seq.append(selected_task)
            
            # Randomly assign a processor
            proc_map[selected_task] = random.randint(1, self.num_processors)
            
            # Mark the task as scheduled
            scheduled.add(selected_task)
        
        # Generate random security levels for all messages
        security_levels = self._generate_random_security_levels()
        
        # Create the individual
        individual = {
            'task_seq': task_seq,
            'proc_map': proc_map,
            'security_levels': security_levels,
            'makespan': None,
            'security_utility': None,
            'fitness': None
        }
        
        return individual
    
    def _generate_random_security_levels(self):
        """Generate random security levels for all messages."""
        security_levels = {}
        
        for message in self.messages:
            # Get the max levels for each service
            max_confidentiality = len(self.security.strengths['confidentiality']) - 1
            max_integrity = len(self.security.strengths['integrity']) - 1
            max_authentication = len(self.security.strengths['authentication']) - 1
            
            # Find minimum required levels
            min_conf_level = 0
            for i, strength in enumerate(self.security.strengths['confidentiality']):
                if strength >= message.min_security['confidentiality']:
                    min_conf_level = i
                    break
            
            min_int_level = 0
            for i, strength in enumerate(self.security.strengths['integrity']):
                if strength >= message.min_security['integrity']:
                    min_int_level = i
                    break
            
            min_auth_level = 0
            for i, strength in enumerate(self.security.strengths['authentication']):
                if strength >= message.min_security['authentication']:
                    min_auth_level = i
                    break
            
            # Generate random levels at or above the minimum
            security_levels[message.id] = {
                'confidentiality': random.randint(min_conf_level, max_confidentiality),
                'integrity': random.randint(min_int_level, max_integrity),
                'authentication': random.randint(min_auth_level, max_authentication)
            }
        
        return security_levels
    
    def _evaluate_population(self):
        """Evaluate the fitness of all individuals in the population."""
        for individual in self.population:
            # Skip if already evaluated
            if individual['fitness'] is not None:
                continue
            
            # Decode the individual to get a full schedule
            makespan, security_utility = self._decode_and_evaluate(individual)
            
            # Update individual with results
            individual['makespan'] = makespan
            individual['security_utility'] = security_utility
            individual['fitness'] = self._calculate_fitness(makespan, security_utility)
    
    def _decode_and_evaluate(self, individual):
        """Decode the chromosome into a full schedule and evaluate its performance."""
        # Reset tasks and processors
        self._reset_tasks_and_processors()
        
        # Apply security levels to messages
        self._apply_security_levels(individual['security_levels'])
        
        # Apply task sequence and processor mapping
        task_seq = individual['task_seq']
        proc_map = individual['proc_map']
        
        # Check if we have a valid schedule
        if len(task_seq) != self.num_tasks or len(proc_map) != self.num_tasks:
            # Invalid schedule, return worst possible values
            return float('inf'), 0
        
        # Schedule tasks in the given sequence
        for task_id in task_seq:
            # Get the task and processor
            task = self._get_task_by_id(task_id)
            processor_id = proc_map[task_id]
            processor = self.processors[processor_id - 1]
            
            # Calculate earliest start time
            est = self._calculate_est(task, processor_id)
            
            # Calculate execution time
            exec_time = task.execution_times[processor_id - 1]
            
            # Calculate security overhead
            security_overhead = self._calculate_security_overhead(task, processor_id)
            
            # Set task attributes
            task.assigned_processor = processor_id
            task.start_time = est
            task.finish_time = est + exec_time + security_overhead
            task.is_scheduled = True
            
            # Update processor availability
            processor.available_time = task.finish_time
        
        # Calculate makespan
        makespan = max(task.finish_time for task in self.tasks if task.is_scheduled)
        
        # Check if makespan exceeds deadline
        if makespan > self.deadline:
            # Apply penalty
            return makespan, 0
        
        # Calculate security utility
        security_utility = self._calculate_security_utility()
        
        return makespan, security_utility
    
    def _reset_tasks_and_processors(self):
        """Reset tasks and processors to their initial state."""
        for task in self.tasks:
            task.assigned_processor = None
            task.start_time = None
            task.finish_time = None
            task.is_scheduled = False
        
        for processor in self.processors:
            processor.available_time = 0
    
    def _apply_security_levels(self, security_levels):
        """Apply security levels to messages."""
        for message in self.messages:
            if message.id in security_levels:
                levels = security_levels[message.id]
                message.assigned_security['confidentiality'] = levels['confidentiality']
                message.assigned_security['integrity'] = levels['integrity']
                message.assigned_security['authentication'] = levels['authentication']
    
    def _calculate_est(self, task, processor_id):
        """Calculate the earliest start time for a task on a processor."""
        processor = self.processors[processor_id - 1]
        processor_ready_time = processor.available_time
        
        # Consider predecessors
        max_pred_finish_time = 0
        for pred_id in task.predecessors:
            pred_task = self._get_task_by_id(pred_id)
            
            if not pred_task.is_scheduled:
                return float('inf')  # Predecessor not scheduled yet
            
            message = self._get_message(pred_id, task.task_id)
            if not message:
                continue
            
            # Calculate communication time
            source_proc = self.processors[pred_task.assigned_processor - 1]
            dest_proc = self.processors[processor_id - 1]
            comm_time = source_proc.get_communication_time(message, source_proc, dest_proc) if pred_task.assigned_processor != processor_id else 0
            
            # Calculate security overhead
            security_overhead = self._calc_security_overhead(message, pred_task.assigned_processor, processor_id)
            
            # Calculate when data can arrive
            data_arrival_time = pred_task.finish_time + comm_time + security_overhead
            
            # Update max predecessor finish time
            max_pred_finish_time = max(max_pred_finish_time, data_arrival_time)
        
        # Return the earliest start time
        return max(processor_ready_time, max_pred_finish_time)
    
    def _calculate_security_overhead(self, task, processor_id):
        """Calculate the security overhead for a task on a processor."""
        total_overhead = 0
        
        # Incoming messages (from predecessors)
        for pred_id in task.predecessors:
            pred_task = self._get_task_by_id(pred_id)
            if not pred_task.is_scheduled:
                continue
            
            message = self._get_message(pred_id, task.task_id)
            if not message:
                continue
            
            # Calculate receiver-side security overhead
            for service in ['confidentiality', 'integrity']:
                protocol_idx = message.assigned_security[service] + 1  # 1-indexed
                overhead_factor = self.security.overheads[service][protocol_idx][processor_id - 1]
                overhead = message.size / overhead_factor
                total_overhead += overhead
            
            # Authentication verification overhead
            auth_idx = message.assigned_security['authentication'] + 1
            auth_overhead = self.security.overheads['authentication'][auth_idx][processor_id - 1]
            total_overhead += auth_overhead
        
        # Outgoing messages (to successors)
        for succ_id in task.successors:
            message = self._get_message(task.task_id, succ_id)
            if not message:
                continue
            
            # Calculate sender-side security overhead
            for service in ['confidentiality', 'integrity']:
                protocol_idx = message.assigned_security[service] + 1  # 1-indexed
                overhead_factor = self.security.overheads[service][protocol_idx][processor_id - 1]
                overhead = message.size / overhead_factor
                total_overhead += overhead
            
            # Authentication generation overhead
            auth_idx = message.assigned_security['authentication'] + 1
            auth_overhead = self.security.overheads['authentication'][auth_idx][processor_id - 1]
            total_overhead += auth_overhead
        
        return total_overhead
    
    def _calc_security_overhead(self, message, source_proc, dest_proc):
        """Calculate the security overhead for a message between processors."""
        total_overhead = 0
        
        # Add overhead for all three services at both sender and receiver
        for service in ['confidentiality', 'integrity', 'authentication']:
            protocol_idx = message.assigned_security[service] + 1  # 1-indexed in the overhead table
            
            # Sender overhead
            sender_overhead_factor = self.security.overheads[service][protocol_idx][source_proc - 1]
            if service in ['confidentiality', 'integrity']:
                # Data-dependent overhead for confidentiality and integrity
                sender_overhead = (message.size / sender_overhead_factor)
            else:
                # Fixed overhead for authentication
                sender_overhead = sender_overhead_factor
            total_overhead += sender_overhead
            
            # Receiver overhead 
            receiver_overhead_factor = self.security.overheads[service][protocol_idx][dest_proc - 1]
            if service in ['confidentiality', 'integrity']:
                # Data-dependent overhead for confidentiality and integrity
                receiver_overhead = (message.size / receiver_overhead_factor)
            else:
                # Fixed overhead for authentication
                receiver_overhead = receiver_overhead_factor
            total_overhead += receiver_overhead
        
        return total_overhead
    
    def _calculate_security_utility(self):
        """Calculate the security utility for all messages."""
        total_utility = 0
        
        for message in self.messages:
            message_utility = 0
            
            # Calculate utility for each security service
            for service in ['confidentiality', 'integrity', 'authentication']:
                level = message.assigned_security[service]
                strength = self.security.strengths[service][level]
                weight = message.weights[service]
                
                service_utility = weight * strength
                message_utility += service_utility
            
            total_utility += message_utility
        
        return total_utility
    
    def _calculate_fitness(self, makespan, security_utility):
        """Calculate fitness value combining makespan and security utility."""
        # Check if makespan exceeds deadline
        if makespan > self.deadline:
            # Strong penalty for exceeding deadline
            return -1 * (makespan / self.deadline)
        
        # Calculate normalized values
        normalize_makespan = (self.deadline - makespan) / self.deadline
        
        # Weight factors (can be adjusted)
        makespan_weight = 0.4
        security_weight = 0.6
        
        # Combined fitness (higher is better)
        fitness = makespan_weight * normalize_makespan + security_weight * security_utility
        
        return fitness
    
    def _get_task_by_id(self, task_id):
        """Get a task by its ID."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
    
    def _get_message(self, source_id, dest_id):
        """Get a message between two tasks."""
        for message in self.messages:
            if message.source_id == source_id and message.dest_id == dest_id:
                return message
        return None
    
    def _create_new_population(self):
        """Create a new population using selection, crossover, and mutation."""
        new_population = []
        
        # Sort population by fitness (descending)
        sorted_pop = sorted(self.population, key=lambda x: x['fitness'] if x['fitness'] is not None else float('-inf'), reverse=True)
        
        # Elitism: Keep the best individuals
        elites = sorted_pop[:self.elite_size]
        new_population.extend(copy.deepcopy(elites))
        
        # Create the rest of the population
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.crossover_rate:
                offspring1, offspring2 = self._crossover(parent1, parent2)
            else:
                offspring1, offspring2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
            
            # Mutation
            if random.random() < self.mutation_rate:
                self._mutate(offspring1)
            
            if random.random() < self.mutation_rate:
                self._mutate(offspring2)
            
            # Repair if needed
            self._repair_solution(offspring1)
            self._repair_solution(offspring2)
            
            # Reset evaluation metrics
            offspring1['makespan'] = None
            offspring1['security_utility'] = None
            offspring1['fitness'] = None
            
            offspring2['makespan'] = None
            offspring2['security_utility'] = None
            offspring2['fitness'] = None
            
            # Add to new population
            new_population.append(offspring1)
            if len(new_population) < self.population_size:
                new_population.append(offspring2)
        
        return new_population
    
    def _tournament_selection(self, tournament_size=3):
        """Select an individual using tournament selection."""
        # Randomly select tournament_size individuals
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        
        # Return the best individual from the tournament
        return copy.deepcopy(max(tournament, key=lambda x: x['fitness'] if x['fitness'] is not None else float('-inf')))
    
    def _crossover(self, parent1, parent2):
        """Perform order-preserving crossover between two parents."""
        # Create offspring
        offspring1 = copy.deepcopy(parent1)
        offspring2 = copy.deepcopy(parent2)
        
        # Crossover task sequence using order-preserving crossover
        task_seq1, task_seq2 = self._order_preserving_crossover(parent1['task_seq'], parent2['task_seq'])
        offspring1['task_seq'] = task_seq1
        offspring2['task_seq'] = task_seq2
        
        # Crossover processor map
        proc_map1, proc_map2 = self._processor_map_crossover(parent1['proc_map'], parent2['proc_map'])
        offspring1['proc_map'] = proc_map1
        offspring2['proc_map'] = proc_map2
        
        # Crossover security levels
        sec_levels1, sec_levels2 = self._security_levels_crossover(parent1['security_levels'], parent2['security_levels'])
        offspring1['security_levels'] = sec_levels1
        offspring2['security_levels'] = sec_levels2
        
        return offspring1, offspring2
    
    def _order_preserving_crossover(self, seq1, seq2):
        """Perform order-preserving crossover on task sequences."""
        if not seq1 or not seq2:
            return seq1, seq2
        
        # Choose crossover points
        length = len(seq1)
        point1 = random.randint(0, length - 1)
        point2 = random.randint(point1, length - 1)
        
        # Create offspring sequences
        offspring1 = [None] * length
        offspring2 = [None] * length
        
        # Copy the crossover segment
        for i in range(point1, point2 + 1):
            offspring1[i] = seq1[i]
            offspring2[i] = seq2[i]
        
        # Fill the remaining positions with tasks from the other parent
        # while preserving precedence constraints
        
        # First, create a topological ordering of tasks
        topo_order = self._topological_sort()
        
        # Use this order to fill in the remaining positions
        for i in range(length):
            if offspring1[i] is None:
                # Find the next valid task from parent2
                for task_id in seq2:
                    if task_id not in offspring1:
                        # Check if all predecessors are scheduled before this position
                        task = self._get_task_by_id(task_id)
                        pred_ids = task.predecessors
                        
                        # Check if all predecessors are already in the sequence before this position
                        can_schedule = True
                        for pred_id in pred_ids:
                            if pred_id not in offspring1[:i]:
                                can_schedule = False
                                break
                        
                        if can_schedule:
                            offspring1[i] = task_id
                            break
            
            if offspring2[i] is None:
                # Find the next valid task from parent1
                for task_id in seq1:
                    if task_id not in offspring2:
                        # Check if all predecessors are scheduled before this position
                        task = self._get_task_by_id(task_id)
                        pred_ids = task.predecessors
                        
                        # Check if all predecessors are already in the sequence before this position
                        can_schedule = True
                        for pred_id in pred_ids:
                            if pred_id not in offspring2[:i]:
                                can_schedule = False
                                break
                        
                        if can_schedule:
                            offspring2[i] = task_id
                            break
        
        # In case the crossover creates invalid sequences, repair them
        if None in offspring1 or None in offspring2:
            # Identify missing tasks
            missing1 = [t for t in seq1 if t not in offspring1]
            missing2 = [t for t in seq2 if t not in offspring2]
            
            # Fill in missing tasks according to topological order
            pos1 = 0
            pos2 = 0
            
            for task_id in topo_order:
                if task_id in missing1:
                    while offspring1[pos1] is not None:
                        pos1 += 1
                        if pos1 >= length:
                            break
                    if pos1 < length:
                        offspring1[pos1] = task_id
                
                if task_id in missing2:
                    while offspring2[pos2] is not None:
                        pos2 += 1
                        if pos2 >= length:
                            break
                    if pos2 < length:
                        offspring2[pos2] = task_id
        
        return offspring1, offspring2
    
    def _processor_map_crossover(self, map1, map2):
        """Perform crossover on processor maps."""
        # Create new maps
        new_map1 = {}
        new_map2 = {}
        
        # Get all task IDs
        task_ids = set(map1.keys()).union(set(map2.keys()))
        
        # Random crossover point
        crossover_point = random.randint(1, len(task_ids) - 1)
        
        # Sort task IDs for deterministic crossover
        sorted_ids = sorted(list(task_ids))
        
        # First part from parent1, second part from parent2
        for i, task_id in enumerate(sorted_ids):
            if i < crossover_point:
                new_map1[task_id] = map1.get(task_id, random.randint(1, self.num_processors))
                new_map2[task_id] = map2.get(task_id, random.randint(1, self.num_processors))
            else:
                new_map1[task_id] = map2.get(task_id, random.randint(1, self.num_processors))
                new_map2[task_id] = map1.get(task_id, random.randint(1, self.num_processors))
        
        return new_map1, new_map2
    
    def _security_levels_crossover(self, levels1, levels2):
        """Perform crossover on security levels."""
        # Create new security levels
        new_levels1 = {}
        new_levels2 = {}
        
        # Get all message IDs
        message_ids = set(levels1.keys()).union(set(levels2.keys()))
        
        # Random crossover point
        crossover_point = random.randint(1, len(message_ids) - 1)
        
        # Sort message IDs for deterministic crossover
        sorted_ids = sorted(list(message_ids))
        
        # For each security service, cross over half of the values
        for i, msg_id in enumerate(sorted_ids):
            if i < crossover_point:
                # First half from parent 1
                new_levels1[msg_id] = levels1.get(msg_id, self._generate_random_security_level(msg_id))
                new_levels2[msg_id] = levels2.get(msg_id, self._generate_random_security_level(msg_id))
            else:
                # Second half from parent 2
                new_levels1[msg_id] = levels2.get(msg_id, self._generate_random_security_level(msg_id))
                new_levels2[msg_id] = levels1.get(msg_id, self._generate_random_security_level(msg_id))
        
        return new_levels1, new_levels2

    def _generate_random_security_level(self, msg_id):
        """Generate random security level for a specific message."""
        for message in self.messages:
            if message.id == msg_id:
                # Get the max levels for each service
                max_confidentiality = len(self.security.strengths['confidentiality']) - 1
                max_integrity = len(self.security.strengths['integrity']) - 1
                max_authentication = len(self.security.strengths['authentication']) - 1
                
                # Find minimum required levels
                min_conf_level = 0
                for i, strength in enumerate(self.security.strengths['confidentiality']):
                    if strength >= message.min_security['confidentiality']:
                        min_conf_level = i
                        break
                
                min_int_level = 0
                for i, strength in enumerate(self.security.strengths['integrity']):
                    if strength >= message.min_security['integrity']:
                        min_int_level = i
                        break
                
                min_auth_level = 0
                for i, strength in enumerate(self.security.strengths['authentication']):
                    if strength >= message.min_security['authentication']:
                        min_auth_level = i
                        break
                
                # Generate random levels at or above the minimum
                return {
                    'confidentiality': random.randint(min_conf_level, max_confidentiality),
                    'integrity': random.randint(min_int_level, max_integrity),
                    'authentication': random.randint(min_auth_level, max_authentication)
                }
        
        # Default if message not found (shouldn't happen)
        return {
            'confidentiality': 0,
            'integrity': 0,
            'authentication': 0
        }

    def _mutate(self, individual):
        """Perform mutation on an individual."""
        # Choose what to mutate
        mutation_type = random.choice(['task_seq', 'proc_map', 'security_levels'])
        
        if mutation_type == 'task_seq':
            self._mutate_task_sequence(individual)
        elif mutation_type == 'proc_map':
            self._mutate_processor_map(individual)
        else:
            self._mutate_security_levels(individual)
        
        return individual

    def _mutate_task_sequence(self, individual):
        """Mutate the task sequence while preserving precedence constraints."""
        task_seq = individual['task_seq']
        
        # Pick two random positions to swap
        length = len(task_seq)
        if length <= 1:
            return
        
        # Try up to 10 times to find valid mutation points
        for _ in range(10):
            pos1 = random.randint(0, length - 1)
            pos2 = random.randint(0, length - 1)
            
            # Ensure pos1 < pos2
            if pos1 > pos2:
                pos1, pos2 = pos2, pos1
                
            # Skip if same position
            if pos1 == pos2:
                continue
            
            # Check if swap would violate precedence constraints
            task1 = self._get_task_by_id(task_seq[pos1])
            task2 = self._get_task_by_id(task_seq[pos2])
            
            # Check if task2 depends on task1
            if task1.task_id in self._get_all_predecessors(task2):
                continue
                
            # Check if task1 depends on task2
            if task2.task_id in self._get_all_predecessors(task1):
                continue
            
            # If swapping adjacent positions, just do it
            if pos2 == pos1 + 1:
                task_seq[pos1], task_seq[pos2] = task_seq[pos2], task_seq[pos1]
                return
            
            # For non-adjacent positions, perform order-preserving "shift" mutation
            # Remove task at pos2 and insert it at pos1
            task_to_move = task_seq[pos2]
            task_seq.pop(pos2)
            task_seq.insert(pos1, task_to_move)
            return
        
        # If no valid mutation found, do nothing

    def _get_all_predecessors(self, task):
        """Get all predecessors recursively for a task."""
        all_preds = set()
        
        def collect_predecessors(task_id):
            for pred_id in self._get_task_by_id(task_id).predecessors:
                all_preds.add(pred_id)
                collect_predecessors(pred_id)
        
        collect_predecessors(task.task_id)
        return all_preds

    def _mutate_processor_map(self, individual):
        """Mutate the processor map."""
        proc_map = individual['proc_map']
        
        # Pick a random task
        task_id = random.choice(list(proc_map.keys()))
        
        # Assign a new processor (different from current)
        current_proc = proc_map[task_id]
        available_procs = list(range(1, self.num_processors + 1))
        available_procs.remove(current_proc)
        
        if available_procs:  # Only if there are other processors available
            proc_map[task_id] = random.choice(available_procs)

    def _mutate_security_levels(self, individual):
        """Mutate the security levels of messages."""
        security_levels = individual['security_levels']
        
        # Pick a random message
        if not security_levels:
            return
            
        msg_id = random.choice(list(security_levels.keys()))
        
        # Pick a random security service
        service = random.choice(['confidentiality', 'integrity', 'authentication'])
        
        # Find the message
        message = None
        for msg in self.messages:
            if msg.id == msg_id:
                message = msg
                break
        
        if not message:
            return
        
        # Get max level for this service
        max_level = len(self.security.strengths[service]) - 1
        
        # Find minimum required level
        min_level = 0
        for i, strength in enumerate(self.security.strengths[service]):
            if strength >= message.min_security[service]:
                min_level = i
                break
        
        # Get current level
        current_level = security_levels[msg_id][service]
        
        # Generate a new level (different from current)
        available_levels = list(range(min_level, max_level + 1))
        if current_level in available_levels and len(available_levels) > 1:
            available_levels.remove(current_level)
        
        if available_levels:  # Only if there are other levels available
            security_levels[msg_id][service] = random.choice(available_levels)

    def _repair_solution(self, individual):
        """Repair an individual if it violates any constraints."""
        # Repair task sequence if there are precedence violations
        self._repair_task_sequence(individual)
        
        # Repair processor map if there are missing tasks
        self._repair_processor_map(individual)
        
        # Repair security levels if they don't meet minimum requirements
        self._repair_security_levels(individual)

    def _repair_task_sequence(self, individual):
        """Repair task sequence to ensure precedence constraints are met."""
        task_seq = individual['task_seq']
        
        # Check if we have all tasks
        if len(task_seq) != self.num_tasks:
            # Regenerate task sequence using CTS approach
            individual['task_seq'] = self._generate_individual_with_cts()['task_seq']
            return
        
        # Check for precedence violations
        scheduled = set()
        valid_sequence = []
        violated = False
        
        for task_id in task_seq:
            task = self._get_task_by_id(task_id)
            
            # Check if all predecessors are scheduled
            if not all(pred in scheduled for pred in task.predecessors):
                violated = True
                continue
            
            # Add to valid sequence and mark as scheduled
            valid_sequence.append(task_id)
            scheduled.add(task_id)
        
        # If violations occurred, fix the sequence
        if violated:
            # Add missing tasks in topological order
            missing_tasks = [t.task_id for t in self.tasks if t.task_id not in scheduled]
            
            # Use topological sort to determine the order
            topo_order = self._topological_sort()
            
            # Add missing tasks in topological order
            for task_id in topo_order:
                if task_id in missing_tasks:
                    # Check if predecessors are all scheduled
                    task = self._get_task_by_id(task_id)
                    if all(pred in scheduled for pred in task.predecessors):
                        valid_sequence.append(task_id)
                        scheduled.add(task_id)
            
            # Update the task sequence
            individual['task_seq'] = valid_sequence

    def _repair_processor_map(self, individual):
        """Repair processor map to ensure all tasks are assigned."""
        proc_map = individual['proc_map']
        task_seq = individual['task_seq']
        
        # Check if all tasks have a processor assignment
        for task_id in task_seq:
            if task_id not in proc_map:
                # Assign a random processor
                proc_map[task_id] = random.randint(1, self.num_processors)

    def _repair_security_levels(self, individual):
        """Repair security levels to ensure minimum requirements are met."""
        security_levels = individual['security_levels']
        
        for message in self.messages:
            # Check if this message has security levels assigned
            if message.id not in security_levels:
                security_levels[message.id] = self._generate_random_security_level(message.id)
                continue
                
            levels = security_levels[message.id]
            
            # Check each security service
            for service in ['confidentiality', 'integrity', 'authentication']:
                # Find minimum required level
                min_level = 0
                for i, strength in enumerate(self.security.strengths[service]):
                    if strength >= message.min_security[service]:
                        min_level = i
                        break
                
                # Ensure the level meets minimum requirement
                if levels[service] < min_level:
                    levels[service] = min_level

    def _topological_sort(self):
        """Perform a topological sort on the task graph."""
        # Create a copy of the task graph
        in_degree = {task.task_id: len(task.predecessors) for task in self.tasks}
        
        # Queue of tasks with no dependencies
        queue = [task.task_id for task in self.tasks if len(task.predecessors) == 0]
        
        # Result list
        topo_order = []
        
        # Process queue
        while queue:
            current_id = queue.pop(0)
            topo_order.append(current_id)
            
            # Update in-degrees of successors
            for succ_id in self.successors.get(current_id, []):
                in_degree[succ_id] -= 1
                
                # If in-degree is 0, add to queue
                if in_degree[succ_id] == 0:
                    queue.append(succ_id)
        
        # Check if we have all tasks
        if len(topo_order) != self.num_tasks:
            # There is a cycle, use a different approach
            return [task.task_id for task in self.tasks]
        
        return topo_order

    def _adapt_parameters(self, generation):
        """Adapt GA parameters based on current generation and diversity."""
        # Calculate current diversity
        diversity = self._calculate_diversity()
        
        # Calculate progress ratio
        progress = generation / self.generations
        
        # Adapt crossover rate
        if diversity < 0.2:  # Low diversity
            self.crossover_rate = min(0.95, self.initial_crossover_rate * 1.1)  # Increase crossover
        elif diversity > 0.6:  # High diversity
            self.crossover_rate = max(0.5, self.initial_crossover_rate * 0.9)  # Decrease crossover
        else:
            self.crossover_rate = self.initial_crossover_rate
        
        # Adapt mutation rate
        if progress < 0.3:  # Early exploration
            self.mutation_rate = self.initial_mutation_rate * 1.2  # Higher exploration
        elif progress > 0.7:  # Late exploitation
            self.mutation_rate = self.initial_mutation_rate * 0.5  # Lower exploration
        else:
            # Linear decay from high to low
            self.mutation_rate = self.initial_mutation_rate * (1.0 - (progress - 0.3) / 0.4)

    def _calculate_diversity(self):
        """Calculate population diversity."""
        if not self.population:
            return 0
        
        # Sample a subset of the population for efficiency
        sample_size = min(len(self.population), 10)
        sample = random.sample(self.population, sample_size)
        
        total_distance = 0
        comparisons = 0
        
        # Calculate average pairwise distance
        for i in range(sample_size):
            for j in range(i+1, sample_size):
                ind1 = sample[i]
                ind2 = sample[j]
                
                # Calculate distance between individuals
                distance = self._calculate_distance(ind1, ind2)
                total_distance += distance
                comparisons += 1
        
        if comparisons == 0:
            return 0
        
        avg_distance = total_distance / comparisons
        
        # Normalize diversity to [0, 1]
        norm_diversity = avg_distance / (self.num_tasks + self.num_processors + len(self.messages) * 3)
        
        return norm_diversity

    def _calculate_distance(self, ind1, ind2):
        """Calculate distance between two individuals."""
        distance = 0
        
        # Task sequence distance (count positions where tasks differ)
        for i in range(min(len(ind1['task_seq']), len(ind2['task_seq']))):
            if ind1['task_seq'][i] != ind2['task_seq'][i]:
                distance += 1
        
        # Processor map distance
        common_tasks = set(ind1['proc_map'].keys()).intersection(set(ind2['proc_map'].keys()))
        for task_id in common_tasks:
            if ind1['proc_map'][task_id] != ind2['proc_map'][task_id]:
                distance += 1
        
        # Security levels distance
        common_msgs = set(ind1['security_levels'].keys()).intersection(set(ind2['security_levels'].keys()))
        for msg_id in common_msgs:
            for service in ['confidentiality', 'integrity', 'authentication']:
                if ind1['security_levels'][msg_id][service] != ind2['security_levels'][msg_id][service]:
                    distance += 1
        
        return distance

    def _apply_cataclysmic_mutation(self):
        """Apply cataclysmic mutation to escape local optima."""
        # Sort population by fitness
        sorted_pop = sorted(self.population, key=lambda x: x['fitness'] if x['fitness'] is not None else float('-inf'), reverse=True)
        
        # Keep only the best individual
        best_individual = copy.deepcopy(sorted_pop[0])
        
        # Reset the population
        self.population = [best_individual]
        
        # Fill with new individuals
        while len(self.population) < self.population_size:
            if random.random() < 0.5 and self.hsms_solution:
                # Generate individual by mutating HSMS solution
                new_ind = copy.deepcopy(self.hsms_solution)
                self._heavy_mutation(new_ind)
                self.population.append(new_ind)
            elif random.random() < 0.7 and self.shield_solution:
                # Generate individual by mutating SHIELD solution
                new_ind = copy.deepcopy(self.shield_solution)
                self._heavy_mutation(new_ind)
                self.population.append(new_ind)
            else:
                # Generate completely new individual
                new_ind = self._generate_individual_with_cts()
                self.population.append(new_ind)

    def _heavy_mutation(self, individual):
        """Apply heavy mutation to an individual."""
        # Mutate processor map - reassign 30% of tasks
        proc_map = individual['proc_map']
        tasks_to_mutate = random.sample(list(proc_map.keys()), int(len(proc_map) * 0.3))
        
        for task_id in tasks_to_mutate:
            proc_map[task_id] = random.randint(1, self.num_processors)
        
        # Mutate security levels - change 30% of messages
        security_levels = individual['security_levels']
        if security_levels:
            msgs_to_mutate = random.sample(list(security_levels.keys()), int(len(security_levels) * 0.3))
            
            for msg_id in msgs_to_mutate:
                # Find the message
                message = None
                for msg in self.messages:
                    if msg.id == msg_id:
                        message = msg
                        break
                
                if message:
                    # Generate new random security levels that meet minimum requirements
                    security_levels[msg_id] = self._generate_random_security_level(msg_id)
        
        # Task sequence mutation is more delicate due to precedence constraints
        # Use CTS approach to generate a valid sequence
        if random.random() < 0.3:
            new_seq = self._generate_individual_with_cts()['task_seq']
            individual['task_seq'] = new_seq
        
        # Reset evaluation metrics
        individual['makespan'] = None
        individual['security_utility'] = None
        individual['fitness'] = None

    def _get_best_solution(self):
        """Get the best solution from the current population."""
        best_solution = None
        best_fitness = float('-inf')
        
        for ind in self.population:
            if ind['fitness'] is not None and ind['fitness'] > best_fitness:
                best_fitness = ind['fitness']
                best_solution = ind
        
        return best_solution if best_solution else self.population[0]

    def _apply_lamarckian_learning(self, individual):
        """Apply Lamarckian learning to improve individual."""
        # Store original values
        original_fitness = individual['fitness']
        original_makespan = individual['makespan']
        original_security = individual['security_utility']
        
        # Try to improve processor assignment with local search
        self._local_search_processors(individual)
        
        # Try to improve security levels with local search
        self._local_search_security(individual)
        
        # Evaluate the improved individual
        makespan, security_utility = self._decode_and_evaluate(individual)
        
        # Update individual with results
        individual['makespan'] = makespan
        individual['security_utility'] = security_utility
        individual['fitness'] = self._calculate_fitness(makespan, security_utility)
        
        # If no improvement, revert to original
        if individual['fitness'] <= original_fitness:
            individual['fitness'] = original_fitness
            individual['makespan'] = original_makespan
            individual['security_utility'] = original_security

    def _local_search_processors(self, individual):
        """Local search to improve processor assignment."""
        proc_map = individual['proc_map']
        
        # Try reassigning each task to a different processor
        for task_id in proc_map:
            current_proc = proc_map[task_id]
            best_proc = current_proc
            best_fitness = individual['fitness']
            
            # Try each processor
            for proc_id in range(1, self.num_processors + 1):
                if proc_id == current_proc:
                    continue
                    
                # Temporarily assign task to this processor
                proc_map[task_id] = proc_id
                
                # Evaluate the new assignment
                makespan, security_utility = self._decode_and_evaluate(individual)
                fitness = self._calculate_fitness(makespan, security_utility)
                
                # If better, keep track of it
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_proc = proc_id
            
            # Revert to best processor
            proc_map[task_id] = best_proc

    def _local_search_security(self, individual):
        """Local search to improve security levels."""
        security_levels = individual['security_levels']
        
        # For each message, try increasing each security service level
        for msg_id in security_levels:
            # Find the message
            message = None
            for msg in self.messages:
                if msg.id == msg_id:
                    message = msg
                    break
                    
            if not message:
                continue
                
            # Try optimizing each security service
            for service in ['confidentiality', 'integrity', 'authentication']:
                current_level = security_levels[msg_id][service]
                best_level = current_level
                best_fitness = individual['fitness']
                
                # Get max level for this service
                max_level = len(self.security.strengths[service]) - 1
                
                # Try each level
                for level in range(current_level, max_level + 1):
                    # Temporarily set this level
                    security_levels[msg_id][service] = level
                    
                    # Evaluate the new assignment
                    makespan, security_utility = self._decode_and_evaluate(individual)
                    fitness = self._calculate_fitness(makespan, security_utility)
                    
                    # If better, keep track of it
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_level = level
                
                # Revert to best level
                security_levels[msg_id][service] = best_level

    def apply_baldwin_learning(self, individual):
        """Apply Baldwinian learning without changing the genotype."""
        # Create a copy of the individual
        copy_ind = copy.deepcopy(individual)
        
        # Apply Lamarckian learning to the copy
        self._apply_lamarckian_learning(copy_ind)
        
        # If the copy has better fitness, use its fitness but keep original genotype
        if copy_ind['fitness'] > individual['fitness']:
            individual['fitness'] = copy_ind['fitness']
            # Note: we don't update the genotype, but only the fitness
            # This is the key difference between Baldwinian and Lamarckian learning
            
    def visualize_gantt_chart(self):
        """
        Visualize the task scheduling as a Gantt chart.
        Shows which tasks are scheduled on which processors, with timing information.
        Also indicates security levels using color intensity.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
        
        if not self.best_solution:
            print("No solution available to visualize.")
            return
        
        # Apply the best solution to tasks for visualization
        self._reset_tasks_and_processors()
        self._apply_security_levels(self.best_solution['security_levels'])
        
        # Apply task sequence and processor mapping
        task_seq = self.best_solution['task_seq']
        proc_map = self.best_solution['proc_map']
        
        # Schedule tasks in the given sequence to calculate start/finish times
        for task_id in task_seq:
            task = self._get_task_by_id(task_id)
            processor_id = proc_map[task_id]
            processor = self.processors[processor_id - 1]
            
            # Calculate earliest start time
            est = self._calculate_est(task, processor_id)
            
            # Calculate execution time
            exec_time = task.execution_times[processor_id - 1]
            
            # Calculate security overhead
            security_overhead = self._calculate_security_overhead(task, processor_id)
            
            # Set task attributes
            task.assigned_processor = processor_id
            task.start_time = est
            task.finish_time = est + exec_time + security_overhead
            task.security_overhead = security_overhead
            task.computation_time = exec_time
            task.is_scheduled = True
            
            # Update processor availability
            processor.available_time = task.finish_time
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Colors for different processors
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_processors))
        
        # Process tasks for visualization
        y_ticks = []
        y_labels = []
        
        # Group tasks by processor
        processor_tasks = {i+1: [] for i in range(self.num_processors)}
        for task in self.tasks:
            if task.is_scheduled:
                processor_tasks[task.assigned_processor].append(task)
        
        y_pos = 0
        for proc_id in range(1, self.num_processors + 1):
            tasks = processor_tasks[proc_id]
            if not tasks:
                continue
                
            # Add processor label
            y_ticks.append(y_pos + 0.5)
            y_labels.append(f"Processor {proc_id}")
            
            # Draw processor timeline
            max_time = max([task.finish_time for task in tasks] + [0])
            ax.add_patch(
                patches.Rectangle(
                    (0, y_pos),
                    max_time,
                    1,
                    linewidth=1,
                    edgecolor='black',
                    facecolor='whitesmoke',
                    alpha=0.3
                )
            )
            
            # Draw tasks on this processor
            for task in tasks:
                # Calculate security intensity (normalized)
                incoming_msgs = [m for m in self.messages if m.dest_id == task.task_id]
                outgoing_msgs = [m for m in self.messages if m.source_id == task.task_id]
                
                all_msgs = incoming_msgs + outgoing_msgs
                if all_msgs:
                    # Average security level across all connected messages (normalized to 0-1)
                    security_levels = []
                    for msg in all_msgs:
                        levels = self.best_solution['security_levels'].get(msg.id, {'confidentiality': 0, 'integrity': 0, 'authentication': 0})
                        # Normalize by maximum levels
                        conf_norm = levels['confidentiality'] / (len(self.security.strengths['confidentiality']) - 1)
                        int_norm = levels['integrity'] / (len(self.security.strengths['integrity']) - 1)
                        auth_norm = levels['authentication'] / (len(self.security.strengths['authentication']) - 1)
                        security_levels.append((conf_norm + int_norm + auth_norm) / 3)
                    
                    security_intensity = sum(security_levels) / len(security_levels)
                else:
                    security_intensity = 0
                
                # Assign color based on processor with security intensity
                base_color = colors[proc_id - 1]
                task_color = tuple(list(base_color[:3]) + [0.3 + 0.7 * security_intensity])
                
                # Draw computation time
                ax.add_patch(
                    patches.Rectangle(
                        (task.start_time, y_pos),
                        task.computation_time,
                        1,
                        linewidth=1,
                        edgecolor='black',
                        facecolor=task_color
                    )
                )
                
                # Draw security overhead (if any)
                if task.security_overhead > 0:
                    ax.add_patch(
                        patches.Rectangle(
                            (task.start_time + task.computation_time, y_pos),
                            task.security_overhead,
                            1,
                            linewidth=1,
                            edgecolor='black',
                            facecolor='red',
                            alpha=0.5,
                            hatch='///'
                        )
                    )
                
                # Add task label
                ax.text(
                    task.start_time + (task.finish_time - task.start_time) / 2,
                    y_pos + 0.5,
                    f"T{task.task_id}",
                    ha='center',
                    va='center',
                    fontweight='bold'
                )
            
            y_pos += 1
        
        # Add communication arrows between tasks
        for message in self.messages:
            source_task = self._get_task_by_id(message.source_id)
            dest_task = self._get_task_by_id(message.dest_id)
            
            # Skip if tasks are not scheduled
            if not source_task.is_scheduled or not dest_task.is_scheduled:
                continue
                
            # Skip if tasks are on the same processor (no communication needed)
            if source_task.assigned_processor == dest_task.assigned_processor:
                continue
            
            # Find y positions
            source_proc_idx = list(processor_tasks.keys()).index(source_task.assigned_processor)
            dest_proc_idx = list(processor_tasks.keys()).index(dest_task.assigned_processor)
            
            # Get security levels
            levels = self.best_solution['security_levels'].get(message.id, {'confidentiality': 0, 'integrity': 0, 'authentication': 0})
            # Sum of security levels normalized
            sec_sum = (levels['confidentiality'] + levels['integrity'] + levels['authentication']) / 3
            security_intensity = sec_sum / ((len(self.security.strengths['confidentiality']) - 1 + 
                                            len(self.security.strengths['integrity']) - 1 + 
                                            len(self.security.strengths['authentication']) - 1) / 3)
            
            # Arrow color based on security level
            arrow_color = plt.cm.plasma(security_intensity)
            
            # Draw arrow from source task end to dest task start
            ax.annotate(
                "",
                xy=(dest_task.start_time, dest_proc_idx + 0.5),
                xytext=(source_task.finish_time, source_proc_idx + 0.5),
                arrowprops=dict(
                    arrowstyle="->",
                    color=arrow_color,
                    linewidth=1,
                    alpha=0.7
                )
            )
        
        # Configure plot
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel('Time')
        ax.set_title('Task Scheduling Gantt Chart')
        
        # Add makespan line
        makespan = self.best_solution['makespan']
        ax.axvline(x=makespan, color='red', linestyle='--', linewidth=2)
        ax.text(makespan + 0.1, 0, f"Makespan: {makespan:.2f}", color='red', va='bottom')
        
        # Add deadline line if available
        if self.deadline:
            ax.axvline(x=self.deadline, color='green', linestyle='--', linewidth=2)
            ax.text(self.deadline + 0.1, y_pos - 1, f"Deadline: {self.deadline:.2f}", color='green', va='top')
        
        # Add security level legend
        security_legend = fig.add_axes([0.92, 0.15, 0.03, 0.7])
        norm = plt.Normalize(0, 1)
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.plasma), cax=security_legend)
        cb.set_label('Message Security Level')
        
        # Add security overhead legend
        overhead_patch = patches.Patch(facecolor='red', alpha=0.5, hatch='///', label='Security Overhead')
        ax.legend(handles=[overhead_patch], loc='upper right')
        
        plt.tight_layout()
        plt.show()

    def visualize_convergence(self):
        """
        Visualize the convergence of the genetic algorithm.
        Shows makespan and security utility over generations.
        """
        import matplotlib.pyplot as plt
        
        if not self.best_fitness_history:
            print("No convergence data available to visualize.")
            return
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
        
        generations = range(len(self.best_fitness_history))
        
        # Plot fitness
        ax1.plot(generations, self.best_fitness_history, 'b-', label='Best Fitness')
        ax1.plot(generations, self.avg_fitness_history, 'g--', label='Average Fitness')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Fitness Convergence')
        ax1.legend()
        ax1.grid(True)
        
        # Plot makespan
        ax2.plot(generations, self.best_makespan_history, 'r-')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Makespan')
        ax2.set_title('Makespan Convergence')
        if self.deadline:
            ax2.axhline(y=self.deadline, color='green', linestyle='--', label='Deadline')
            ax2.legend()
        ax2.grid(True)
        
        # Plot security utility
        ax3.plot(generations, self.best_security_history, 'purple')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Security Utility')
        ax3.set_title('Security Utility Convergence')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()

    def visualize_3d_convergence(self):
        """
        Create a 3D visualization showing the relationship between 
        makespan, security utility, and generations.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        if not self.best_fitness_history:
            print("No convergence data available to visualize.")
            return
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Data
        generations = list(range(len(self.best_makespan_history)))
        makespans = self.best_makespan_history
        security = self.best_security_history
        
        # Create the scatter plot
        scatter = ax.scatter(
            generations, 
            makespans, 
            security, 
            c=self.best_fitness_history, 
            cmap='viridis',
            s=50,
            alpha=0.7
        )
        
        # Add line to show progression
        ax.plot(generations, makespans, security, 'r-', alpha=0.5)
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Fitness Value')
        
        # Add labels
        ax.set_xlabel('Generation')
        ax.set_ylabel('Makespan')
        ax.set_zlabel('Security Utility')
        ax.set_title('3D Convergence of Makespan vs Security Utility')
        
        # Add initial and final points with annotations
        ax.scatter([0], [makespans[0]], [security[0]], color='red', s=100, label='Initial Solution')
        ax.scatter(
            [len(generations)-1], 
            [makespans[-1]], 
            [security[-1]], 
            color='green', 
            s=100, 
            label='Final Solution'
        )
        
        # Show deadline plane if available
        if self.deadline:
            x_grid, z_grid = np.meshgrid(
                np.linspace(0, len(generations), 10), 
                np.linspace(min(security), max(security), 10)
            )
            y_grid = np.ones_like(x_grid) * self.deadline
            ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.2, color='green')
            
        ax.legend()
        plt.tight_layout()
        plt.show()

    def visualize_all_results(self):
        """
        Visualize all the results from the genetic algorithm in one go.
        """
        print("Visualizing Gantt Chart...")
        self.visualize_gantt_chart()
        
        print("\nVisualizing Convergence Plots...")
        self.visualize_convergence()
        
        print("\nVisualizing 3D Convergence...")
        self.visualize_3d_convergence()
        
        print(f"\nBest solution found:")
        print(f"Makespan: {self.best_solution['makespan']:.2f}")
        print(f"Security Utility: {self.best_solution['security_utility']:.4f}")
        print(f"Fitness: {self.best_solution['fitness']:.4f}")
        
        # Print statistics
        if self.best_fitness_history:
            initial_makespan = self.best_makespan_history[0]
            initial_security = self.best_security_history[0]
            final_makespan = self.best_makespan_history[-1]
            final_security = self.best_security_history[-1]
            
            print(f"\nImprovement statistics:")
            makespan_improvement = ((initial_makespan - final_makespan) / initial_makespan) * 100
            security_improvement = ((final_security - initial_security) / initial_security) * 100 if initial_security > 0 else float('inf')
            
            print(f"Makespan improvement: {makespan_improvement:.2f}%")
            print(f"Security utility improvement: {security_improvement:.2f}%")
            
            print(f"\nConvergence information:")
            generations_run = len(self.best_fitness_history)
            print(f"Total generations run: {generations_run}")
            
            # Find generation where convergence stabilized
            # (when improvement is less than 0.1% over 10 generations)
            convergence_gen = generations_run
            for i in range(10, generations_run):
                if (self.best_fitness_history[i] - self.best_fitness_history[i-10]) / self.best_fitness_history[i-10] < 0.001:
                    convergence_gen = i
                    break
                    
            print(f"Convergence detected at generation: {convergence_gen}")
            
        print("\nVisualization complete!")
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

    processors = [Processor(1,[0,500]), Processor(2,[500,0])]
    network = CommunicationNetwork(2)
    security_service = SecurityService()
    deadline = 2200  # As specified in the paper
    
    return tasks, messages, processors, network, security_service, deadline

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
    """Create a Gantt chart for visualizing the task schedule."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle
    
    # Setup the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Colors for different tasks
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    # Draw each task as a rectangle
    for entry in schedule:
        task_id = entry['task_id']
        processor = entry['processor']
        start = entry['start_time']
        finish = entry['finish_time']
        
        # Create a rectangle for the task
        rect = Rectangle((start, processor-0.4), finish-start, 0.8, 
                         color=colors[task_id % len(colors)], alpha=0.8)
        ax.add_patch(rect)
        
        # Add task label
        ax.text(start + (finish-start)/2, processor, f"T{task_id}", 
                ha='center', va='center', color='black')
    
    # Set up the axes
    ax.set_ylim(0.5, num_processors + 0.5)
    ax.set_xlim(0, makespan * 1.05)  # Add some padding
    ax.set_yticks(range(1, num_processors + 1))
    ax.set_yticklabels([f'Processor {i}' for i in range(1, num_processors + 1)])
    ax.set_xlabel('Time (ms)')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return fig, ax
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

def plot_modified_gantt_chart(title, schedule, tasks, processors, makespan):
    """
    Create a Gantt chart for the schedule showing:
    - Base execution time
    - Security overhead
    - Different colors for each component
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    fig, ax = plt.subplots(figsize=(14, 7))
    task_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Define colors for different components
    base_color = 'royalblue'
    security_color = 'tomato'
    
    # Create a mapping of task_id to task object
    task_map = {task.task_id: task for task in tasks if task.is_scheduled}
    
    # Create legend elements
    legend_elements = [
        Patch(facecolor=base_color, edgecolor='black', label='Base Execution'),
        Patch(facecolor=security_color, edgecolor='black', label='Security Overhead')
    ]
    
    for entry in schedule:
        task_id = entry['task_id']
        name = entry['name']
        proc = entry['processor']
        start = entry['start_time']
        finish = entry['finish_time']
        
        # Get task object and calculate base execution time
        task = task_map.get(task_id)
        if not task:
            continue
            
        base_exec_time = task.execution_times[task.assigned_processor - 1]
        security_overhead = (finish - start) - base_exec_time
        
        # Draw base execution time bar
        ax.barh(proc - 1, base_exec_time, left=start, color=base_color, edgecolor='black')
        
        # Draw security overhead bar (if any)
        if security_overhead > 0:
            ax.barh(proc - 1, security_overhead, left=start + base_exec_time, 
                   color=security_color, edgecolor='black')
        
        # Add task label in the middle of the base execution part
        ax.text(start + base_exec_time/2, proc - 1, f"T{task_id}", 
                va='center', ha='center', color='white', fontsize=10, weight='bold')
    
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Processor")
    ax.set_title(f"{title} - Makespan: {makespan:.2f} ms")
    ax.set_yticks(range(len(processors)))
    ax.set_yticklabels([f"P{p.proc_id}" for p in processors])
    ax.set_xlim(0, makespan * 1.1)  # Add some padding
    
    # Add legend
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    return fig


def main():
    """Run and compare HSMS and SHIELD algorithms."""
    
    print("Creating test case...")
    tasks, messages, processors, network, security_service, deadline = create_testcase()
    
    # Run HSMS first
    print("\nRunning HSMS scheduler...")
    hsms = HSMS(copy.deepcopy(tasks), copy.deepcopy(messages), 
                copy.deepcopy(processors), network, security_service, deadline)
    
    hsms_makespan, hsms_security_utility = hsms.run()
    
    if hsms_makespan:
        print(f"HSMS completed successfully: Makespan = {hsms_makespan}, Security Utility = {hsms_security_utility}")
        hsms.display_security_information("HSMS")
        hsms.print_task_schedule()
        plot_gantt_chart("HSMS Schedule", hsms.schedule, len(processors), hsms_makespan)
        plt.savefig("hsms_schedule.png")
        plt.show()
    else:
        print("HSMS failed to meet deadline requirements")
        return
    
    # Run SHIELD
    print("\nRunning SHIELD algorithm...")
    shield = SHIELD(copy.deepcopy(tasks), copy.deepcopy(messages), 
                    copy.deepcopy(processors), network, security_service, deadline)
    
    shield_makespan, shield_security_utility = shield.run()
    
    if shield_makespan:
        print(f"SHIELD completed successfully: Makespan = {shield_makespan}, Security Utility = {shield_security_utility}")
        shield.display_security_information("SHIELD")
        shield.print_task_schedule()
        shield.print_upgrade_history()
        shield.security_improvement_summary()
        plot_gantt_chart("SHIELD Schedule", shield.schedule, len(processors), shield_makespan)
        plt.savefig("shield_schedule.png")
        plt.show()
        
        # Compare results
        
        
        
        # Add more metrics if needed
        print("="*100)
    else:
        print("SHIELD failed to complete")
    print("Running GA Scheduler...")
    ga_shield = GeneticScheduler(copy.deepcopy(tasks), copy.deepcopy(messages), 
                    copy.deepcopy(processors), network, security_service, deadline)
    ga_makespan, ga_security_utility = ga_shield.run()
    if ga_makespan:
        print(f"GA completed successfully: Makespan = {ga_makespan}, Security Utility = {ga_security_utility}")
        ga_shield.visualize_gantt_chart()
        ga_shield.visualize_all_results()
        plt.show()
        
        # Compare results
        print("\n" + "="*100)
        print("COMPARISON: HSMS vs GA")
        print("="*100)
        print(f"{'Metric':<25} {'HSMS':<15} {'GA':<15} {'Difference':<15} {'% Change':<15}")
        print("-"*100)
        
        makespan_diff = ga_makespan - hsms_makespan
        makespan_pct = (makespan_diff / hsms_makespan * 100) if hsms_makespan > 0 else 0
        
        security_diff = ga_security_utility - hsms_security_utility
        security_pct = (security_diff / hsms_security_utility * 100) if hsms_security_utility > 0 else 0
        
        print(f"{'Makespan (ms)':<25} {hsms_makespan:<15.2f} {ga_makespan:<15.2f} "
             f"{makespan_diff:<15.2f} {makespan_pct:<15.2f}%")
        print(f"{'Security Utility':<25} {hsms_security_utility:<15.4f} {ga_security_utility:<15.4f} "
             f"{security_diff:<15.4f} {security_pct:<15.2f}%")
        print("\n")
        print("\n" + "="*100)
        print("COMPARISON: HSMS vs SHIELD")
        print("="*100)
        print(f"{'Metric':<25} {'HSMS':<15} {'SHIELD':<15} {'Difference':<15} {'% Change':<15}")
        print("-"*100)
        makespan_diff = shield_makespan - hsms_makespan
        makespan_pct = (makespan_diff / hsms_makespan * 100) if hsms_makespan > 0 else 0
        
        security_diff = shield_security_utility - hsms_security_utility
        security_pct = (security_diff / hsms_security_utility * 100) if hsms_security_utility > 0 else 0
        
        print(f"{'Makespan (ms)':<25} {hsms_makespan:<15.2f} {shield_makespan:<15.2f} "
             f"{makespan_diff:<15.2f} {makespan_pct:<15.2f}%")
        print(f"{'Security Utility':<25} {hsms_security_utility:<15.4f} {shield_security_utility:<15.4f} "
             f"{security_diff:<15.4f} {security_pct:<15.2f}%")
        
        
    else:
        print("GA failed to complete")
    ql_shield = QLSecurityScheduler(copy.deepcopy(tasks), copy.deepcopy(messages), 
                    copy.deepcopy(processors), network, security_service, deadline)
    ql_makespan, ql_security_utility = ql_shield.run()
    if ql_makespan:
        print(f"QL completed successfully: Makespan = {ql_makespan}, Security Utility = {ql_security_utility}")
        ql_shield.print_task_schedule()
        ql_shield.print_schedule_with_security()
        ql_shield.visualize_2d_plots()
        ql_shield.visualize_3d_plot()
        plt.show()
        
        # Compare results
        print("\n" + "="*100)
        print("COMPARISON: HSMS vs QL")
        print("="*100)
        print(f"{'Metric':<25} {'HSMS':<15} {'QL':<15} {'Difference':<15} {'% Change':<15}")
        print("-"*100)
        
        makespan_diff = ql_makespan - hsms_makespan
        makespan_pct = (makespan_diff / hsms_makespan * 100) if hsms_makespan > 0 else 0
        
        security_diff = ql_security_utility - hsms_security_utility
        security_pct = (security_diff / hsms_security_utility * 100) if hsms_security_utility > 0 else 0
        
        print(f"{'Makespan (ms)':<25} {hsms_makespan:<15.2f} {ql_makespan:<15.2f} "
             f"{makespan_diff:<15.2f} {makespan_pct:<15.2f}%")
        print(f"{'Security Utility':<25} {hsms_security_utility:<15.4f} {ql_security_utility:<15.4f} "
             f"{security_diff:<15.4f} {security_pct:<15.2f}%")
    else:
        print("QL failed to complete")
    print("Comparison complete.")
    # Show the plots
    plt.show()
if __name__ == "__main__":
    main()
