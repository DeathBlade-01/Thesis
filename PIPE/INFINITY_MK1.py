import math
import statistics
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as plt1
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import time
import random
import types
import heapq
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from collections import deque
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


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
            'confidentiality': [0.08, 0.14, 0.36, 0.40, 0.46, 0.64, 0.9, 1.0],
            'integrity': [0.18, 0.26, 0.36, 0.45, 0.63, 0.77, 1.0],
            'authentication': [0.55, 0.91, 1.0]
        }

        self.overheads = {
            'confidentiality': {
                1: [1012.5,1518.7,1181.2],
                2: [578.58, 867.87,675.01],
                3: [225.0, 337.5,262.5],
                4: [202.50, 303.75,236.25],
                5: [176.10, 264.15,205.45],
                6: [126.54,189.81,147.63],
                7: [90.0,135.0,105.0],
                8: [81.0,121.5,94.5]
            },
            'integrity': {
                1: [143.40, 215.10,167.30],
                2: [102.54 , 153.81,119.63],
                3: [72 , 108, 84.0],
                4: [58.38 , 87.57, 68.11],
                5: [41.28 , 61.92, 48.16],
                6: [34.14 , 51.21, 39.83],
                7: [26.16,39.24, 30.52]
            },
            'authentication': {
                1: [15, 10, 12.86],
                2: [24.67, 16.44, 21.14],
                3: [27.17, 18.11, 23.29]
            }
        }


class Scheduler:
    def __init__(self, tasks, messages, processors,  security_service, deadline):
        self.tasks = tasks
        self.messages = messages
        self.processors = processors
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
                    overhead = (message.size / overhead_factor)
                    
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
    
    def __init__(self, tasks, messages, processors,  security_service, deadline):
        super().__init__(tasks, messages, processors,  security_service, deadline)
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
            if isinstance(task, int):
                task = self.get_task_by_id(task)
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


class GASecurityScheduler(Scheduler):
    """Genetic Algorithm-based scheduler for security-aware task scheduling"""
    
    def __init__(self, tasks, messages, processors, security_service, deadline,
                 pop_size=50, generations=100, cataclysm_interval=20,
                 mutation_rate=0.1, crossover_rate=0.8, tournament_size=3,
                 hsms_seed=None, shield_seed=None, max_no_improvement=15):
        super().__init__(tasks, messages, processors, security_service, deadline)
        
        # GA parameters
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.cataclysm_interval = cataclysm_interval
        self.max_no_improvement = max_no_improvement
        
        # Initialize seed solutions if provided
        self.hsms_seed = hsms_seed
        self.shield_seed = shield_seed
        
        # Attributes to track best solution
        self.best_makespan = float('inf')
        self.best_utility = 0
        self.best_schedule = None
        self.best_chromosome = None
        
        # Set all messages to minimum security level
        self.assign_minimum_security()
        
        # Initialize population
        self.population = []
        self.initialize_population()
        
    def initialize_population(self):
        """Initialize the GA population with diverse chromosomes"""
        self.population = []
        
        # Add seeds if available
        if self.hsms_seed:
            hsms_chromosome = self.create_chromosome_from_seed(self.hsms_seed)
            self.population.append(hsms_chromosome)
            
        if self.shield_seed:
            shield_chromosome = self.create_chromosome_from_seed(self.shield_seed)
            self.population.append(shield_chromosome)
        
        # Generate remaining chromosomes
        while len(self.population) < self.pop_size:
            chromosome = self.generate_valid_chromosome_cts()
            self.population.append(chromosome)
    
    def create_chromosome_from_seed(self, seed):
        """Create a chromosome from a seed schedule"""
        # Extract task sequence from seed schedule
        task_seq = []
        for entry in sorted(seed.schedule, key=lambda x: x['start_time']):
            task_seq.append(entry['task_id'])
        
        # Extract processor assignments
        proc_assign = {}
        for entry in seed.schedule:
            proc_assign[entry['task_id']] = entry['processor']
        
        # Extract security levels
        security_levels = {}
        for message in seed.messages:
            if hasattr(message, 'assigned_security'):
                security_levels[message.id] = {
                    'confidentiality': message.assigned_security['confidentiality'],
                    'integrity': message.assigned_security['integrity'],
                    'authentication': message.assigned_security['authentication']
                }
        
        return {
            'task_seq': task_seq,
            'proc_assign': proc_assign,
            'security_levels': security_levels,
            'fitness': {'makespan': None, 'utility': None}
        }
    
    def generate_valid_chromosome_cts(self):
        """Generate a valid chromosome using the Candidate Task Set approach"""
        task_seq = []
        proc_assign = {}
        security_levels = {}
        
        # Reset scheduled status for all tasks
        for task in self.tasks:
            task.is_scheduled = False
        
        # Initialize candidate task set with entry tasks (no predecessors)
        cts = [task.task_id for task in self.tasks if not task.predecessors]
        
        # Generate task sequence based on CTS
        while cts:
            # Select a random task from CTS
            task_id = random.choice(cts)
            task_seq.append(task_id)
            cts.remove(task_id)
            
            # Mark task as scheduled
            task = self.get_task_by_id(task_id)
            task.is_scheduled = True
            
            # Assign a random processor
            proc_assign[task_id] = random.randint(1, len(self.processors))
            
            # Update CTS with newly eligible tasks
            for next_task in self.tasks:
                if not next_task.is_scheduled and all(self.get_task_by_id(pred_id).is_scheduled for pred_id in next_task.predecessors):
                    if next_task.task_id not in cts and next_task.task_id not in task_seq:
                        cts.append(next_task.task_id)
        
        # Assign minimum security levels for all messages
        for message in self.messages:
            security_levels[message.id] = {
                'confidentiality': message.assigned_security['confidentiality'],
                'integrity': message.assigned_security['integrity'],
                'authentication': message.assigned_security['authentication']
            }
        
        return {
            'task_seq': task_seq,
            'proc_assign': proc_assign,
            'security_levels': security_levels,
            'fitness': {'makespan': None, 'utility': None}
        }
    
    def evaluate_chromosome(self, chromosome):
        """Evaluate a chromosome and calculate fitness metrics (makespan and security utility)"""
        # Create a temporary copy of tasks and messages
        temp_tasks = copy.deepcopy(self.tasks)
        temp_messages = copy.deepcopy(self.messages)
        
        # Set processor assignments
        for task in temp_tasks:
            task.assigned_processor = chromosome['proc_assign'].get(task.task_id)
            task.is_scheduled = False
        
        # Set security levels
        for message in temp_messages:
            if message.id in chromosome['security_levels']:
                levels = chromosome['security_levels'][message.id]
                message.assigned_security = {
                    'confidentiality': levels['confidentiality'],
                    'integrity': levels['integrity'],
                    'authentication': levels['authentication']
                }
        
        # Schedule tasks according to chromosome's task sequence
        makespan = self.schedule_by_chromosome(temp_tasks, temp_messages, chromosome['task_seq'])
        
        # Calculate security utility
        security_utility = self.calculate_security_utility_for_chromosome(temp_tasks, temp_messages)
        
        # Update chromosome fitness
        chromosome['fitness'] = {
            'makespan': makespan,
            'utility': security_utility if makespan <= self.deadline else 0
        }
        
        return makespan, security_utility
    
    def schedule_by_chromosome(self, tasks, messages, task_seq):
        """Schedule tasks according to given task sequence"""
        # Reset processor available times
        for proc in self.processors:
            proc.available_time = 0
        
        # Schedule each task in order
        for task_id in task_seq:
            task = next((t for t in tasks if t.task_id == task_id), None)
            if not task:
                continue
                
            proc_id = task.assigned_processor
            processor = self.processors[proc_id - 1]
            
            # Calculate earliest start time based on predecessors
            est = 0
            pred_security_overhead = 0
            
            for pred_id in task.predecessors:
                pred_task = next((t for t in tasks if t.task_id == pred_id), None)
                if not pred_task or not pred_task.is_scheduled:
                    # Should not happen with valid CTS sequence
                    est = float('inf')
                    break
                
                # Get message between tasks
                message = next((m for m in messages if m.source_id == pred_id and m.dest_id == task_id), None)
                if not message:
                    continue
                
                # Calculate communication time
                source_proc = self.processors[pred_task.assigned_processor - 1]
                dest_proc = processor
                comm_time = source_proc.get_communication_time(message, source_proc, dest_proc) if source_proc != dest_proc else 0
                
                # Calculate security overhead for this message (receiver side)
                message_security_overhead = self.calculate_message_security_overhead(message, pred_task.assigned_processor, proc_id)
                pred_security_overhead += message_security_overhead
                
                # Update EST based on predecessor finish plus communication time
                pred_finish_with_comm = pred_task.finish_time + comm_time
                est = max(est, pred_finish_with_comm)
            
            # If any predecessor is not scheduled (should not happen with valid CTS)
            if est == float('inf'):
                continue
                
            # Account for processor availability
            est = max(est, processor.available_time)
            
            # Base execution time
            base_exec_time = task.execution_times[proc_id - 1]
            
            # Calculate outgoing security overhead (sender side)
            succ_security_overhead = 0
            for succ_id in task.successors:
                message = next((m for m in messages if m.source_id == task_id and m.dest_id == succ_id), None)
                if not message:
                    continue
                
                succ_security_overhead += self.calculate_message_security_overhead(message, proc_id, None)
            
            # Total security overhead
            total_security_overhead = pred_security_overhead + succ_security_overhead
            
            # Set task timing with security overhead
            task.start_time = est
            task.finish_time = est + base_exec_time + total_security_overhead
            task.is_scheduled = True
            
            # Update processor availability
            processor.available_time = task.finish_time
        
        # Calculate the makespan
        return max(task.finish_time for task in tasks if task.is_scheduled)
    
    def calculate_message_security_overhead(self, message, source_proc_id, dest_proc_id=None):
        """Calculate security overhead for a message"""
        total_overhead = 0
        
        # For outgoing security overhead calculation (sender only)
        if dest_proc_id is None:
            # Calculate security overhead for confidentiality and integrity (sender side)
            for service in ['confidentiality', 'integrity']:
                protocol_idx = message.assigned_security[service] + 1  # 1-indexed
                overhead_factor = self.security.overheads[service][protocol_idx][source_proc_id - 1]
                overhead = message.size / overhead_factor
                total_overhead += overhead
            
            # Add authentication generation overhead
            auth_idx = message.assigned_security['authentication'] + 1
            auth_overhead = self.security.overheads['authentication'][auth_idx][source_proc_id - 1]
            total_overhead += auth_overhead
            
        # For incoming security overhead calculation (receiver side)
        elif dest_proc_id is not None:
            # Calculate security overhead for confidentiality and integrity (receiver side)
            for service in ['confidentiality', 'integrity']:
                protocol_idx = message.assigned_security[service] + 1  # 1-indexed
                overhead_factor = self.security.overheads[service][protocol_idx][dest_proc_id - 1]
                overhead = message.size / overhead_factor
                total_overhead += overhead
            
            # Add authentication verification overhead
            auth_idx = message.assigned_security['authentication'] + 1
            auth_overhead = self.security.overheads['authentication'][auth_idx][dest_proc_id - 1]
            total_overhead += auth_overhead
        
        return total_overhead
    
    def calculate_security_utility_for_chromosome(self, tasks, messages):
        """Calculate total security utility for a chromosome"""
        total_utility = 0
        
        for message in messages:
            # Get source and destination tasks
            source_task = next((t for t in tasks if t.task_id == message.source_id), None)
            dest_task = next((t for t in tasks if t.task_id == message.dest_id), None)
            
            # Skip if either task is not scheduled
            if (not source_task or not dest_task or 
                not source_task.is_scheduled or not dest_task.is_scheduled):
                continue
            
            # Calculate utility for this message
            message_utility = 0
            for service in ['confidentiality', 'integrity', 'authentication']:
                protocol_idx = message.assigned_security[service]
                strength = self.security.strengths[service][protocol_idx]
                weight = message.weights[service]
                message_utility += weight * strength
            
            total_utility += message_utility
        
        return total_utility
    
    def select_parent(self):
        """Select a parent using tournament selection"""
        # Randomly select tournament_size chromosomes
        tournament = random.sample(self.population, self.tournament_size)
        
        # Select the best chromosome based on fitness
        best = None
        for chromosome in tournament:
            # Check if chromosome is valid (meets deadline)
            is_valid = chromosome['fitness']['makespan'] <= self.deadline
            
            if not best:
                if is_valid:
                    best = chromosome
            elif is_valid:
                # For valid chromosomes, prefer higher security utility
                if (not best['fitness']['makespan'] <= self.deadline or 
                    chromosome['fitness']['utility'] > best['fitness']['utility']):
                    best = chromosome
            elif not best['fitness']['makespan'] <= self.deadline:
                # If both invalid, prefer lower makespan
                if chromosome['fitness']['makespan'] < best['fitness']['makespan']:
                    best = chromosome
        
        # If no valid chromosome found, return any
        return best if best else random.choice(tournament)
    
    def crossover(self, parent1, parent2):
        """Perform crossover between two parent chromosomes"""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1)
        
        # Create empty child chromosome
        child = {
            'task_seq': [],
            'proc_assign': {},
            'security_levels': {},
            'fitness': {'makespan': None, 'utility': None}
        }
        
        # Crossover for task sequence using precedence-preserving crossover
        visited = set()
        task_count = len(parent1['task_seq'])
        
        # Start with a random task from either parent
        use_parent1 = random.choice([True, False])
        
        while len(child['task_seq']) < task_count:
            next_task = None
            
            # Try to get next task from current parent
            parent = parent1 if use_parent1 else parent2
            for task_id in parent['task_seq']:
                if task_id not in visited:
                    # Check if all predecessors are already in sequence
                    task = self.get_task_by_id(task_id)
                    if all(pred_id in visited for pred_id in task.predecessors):
                        next_task = task_id
                        break
            
            # If no task found, switch parent
            if next_task is None:
                use_parent1 = not use_parent1
                parent = parent1 if use_parent1 else parent2
                for task_id in parent['task_seq']:
                    if task_id not in visited:
                        task = self.get_task_by_id(task_id)
                        if all(pred_id in visited for pred_id in task.predecessors):
                            next_task = task_id
                            break
            
            # If still no task found, find any valid task
            if next_task is None:
                for task in self.tasks:
                    if task.task_id not in visited and all(pred_id in visited for pred_id in task.predecessors):
                        next_task = task.task_id
                        break
            
            if next_task:
                child['task_seq'].append(next_task)
                visited.add(next_task)
                
                # Randomly select processor assignment from either parent
                if random.random() < 0.5:
                    child['proc_assign'][next_task] = parent1['proc_assign'][next_task]
                else:
                    child['proc_assign'][next_task] = parent2['proc_assign'][next_task]
        
        # Crossover for security levels - select from either parent with 50% probability
        for message_id in parent1['security_levels']:
            child['security_levels'][message_id] = {}
            for service in ['confidentiality', 'integrity', 'authentication']:
                if random.random() < 0.5:
                    child['security_levels'][message_id][service] = parent1['security_levels'][message_id][service]
                else:
                    child['security_levels'][message_id][service] = parent2['security_levels'][message_id][service]
        
        return child
    
    def mutate(self, chromosome):
        """Apply Adjacency mutation to a chromosome"""
        # Deep copy to avoid modifying the original
        mutated = copy.deepcopy(chromosome)
        
        # Task sequence mutation - swap two adjacent tasks if precedence allows
        if random.random() < self.mutation_rate:
            # Try several times to find valid swap
            for _ in range(5):  # Limit attempts
                i = random.randint(0, len(mutated['task_seq']) - 2)
                task1_id = mutated['task_seq'][i]
                task2_id = mutated['task_seq'][i+1]
                
                task1 = self.get_task_by_id(task1_id)
                task2 = self.get_task_by_id(task2_id)
                
                # Check if swap would violate precedence
                if task1_id not in task2.predecessors and task2_id not in task1.predecessors:
                    # Check if any descendant of task1 is a predecessor of task2
                    descendants1 = self.get_all_descendants(task1)
                    if task2_id not in descendants1 and not any(desc in task2.predecessors for desc in descendants1):
                        # Safe to swap
                        mutated['task_seq'][i], mutated['task_seq'][i+1] = mutated['task_seq'][i+1], mutated['task_seq'][i]
                        break
        
        # Processor assignment mutation
        for task_id in mutated['proc_assign']:
            if random.random() < self.mutation_rate:
                # Assign a different processor
                current_proc = mutated['proc_assign'][task_id]
                available_procs = list(range(1, len(self.processors) + 1))
                available_procs.remove(current_proc)
                if available_procs:
                    mutated['proc_assign'][task_id] = random.choice(available_procs)
        
        # Security level mutation
        for message_id in mutated['security_levels']:
            for service in ['confidentiality', 'integrity', 'authentication']:
                if random.random() < self.mutation_rate:
                    # Change security level within valid range
                    max_level = len(self.security.strengths[service]) - 1
                    current_level = mutated['security_levels'][message_id][service]
                    
                    # Find message to check minimum requirements
                    message = next((m for m in self.messages if m.id == message_id), None)
                    if message:
                        min_strength = message.min_security[service]
                        valid_levels = [i for i, strength in enumerate(self.security.strengths[service]) 
                                      if strength >= min_strength]
                        
                        if valid_levels and valid_levels[0] != current_level:
                            # Select a different valid level
                            available_levels = valid_levels.copy()
                            if current_level in available_levels:
                                available_levels.remove(current_level)
                            if available_levels:
                                mutated['security_levels'][message_id][service] = random.choice(available_levels)
        
        return mutated
    
    def get_all_descendants(self, task):
        """Get all descendants of a task"""
        descendants = set()
        
        def collect_descendants(task_id):
            task = self.get_task_by_id(task_id)
            for succ_id in task.successors:
                if succ_id not in descendants:
                    descendants.add(succ_id)
                    collect_descendants(succ_id)
        
        collect_descendants(task.task_id)
        return descendants
    
    def run_phase1(self):
        """Run Phase 1 of the GA algorithm to find a good initial schedule with history tracking"""
        print("\nRunning GA Phase 1: Finding optimal task schedule...")
        
        # Initialize history tracking lists
        self.makespan_history = []
        self.utility_history = []
        
        # Evaluate initial population
        for chromosome in self.population:
            self.evaluate_chromosome(chromosome)
        
        # Find best solution in initial population
        best_chromosome = self.get_best_chromosome()
        if best_chromosome:
            self.best_chromosome = copy.deepcopy(best_chromosome)
            self.best_makespan = best_chromosome['fitness']['makespan']
            self.best_utility = best_chromosome['fitness']['utility']
            print(f"Initial best solution - Makespan: {self.best_makespan}, Utility: {self.best_utility}")
            
            # Record initial values in history
            self.makespan_history.append(self.best_makespan)
            self.utility_history.append(self.best_utility)
        
        # Main GA loop
        generations_no_improvement = 0
        for generation in range(self.generations):
            # Create new population
            new_population = []
            
            # Elitism: keep the best chromosome
            if self.best_chromosome:
                new_population.append(copy.deepcopy(self.best_chromosome))
            
            # Generate new chromosomes
            while len(new_population) < self.pop_size:
                # Select parents
                parent1 = self.select_parent()
                parent2 = self.select_parent()
                
                # Create offspring
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                
                # Evaluate child
                self.evaluate_chromosome(child)
                
                # Add to new population
                new_population.append(child)
            
            # Replace old population
            self.population = new_population
            
            # Find best solution in current population
            current_best = self.get_best_chromosome()
            
            if current_best:
                current_makespan = current_best['fitness']['makespan']
                current_utility = current_best['fitness']['utility']
                
                improved = False
                
                # Check if this solution beats our best so far
                if current_makespan <= self.deadline:  # Valid solution
                    if self.best_makespan > self.deadline or current_utility > self.best_utility:
                        # Better solution found
                        self.best_chromosome = copy.deepcopy(current_best)
                        self.best_makespan = current_makespan
                        self.best_utility = current_utility
                        improved = True
                        generations_no_improvement = 0
                        print(f"Generation {generation+1}: New best solution - Makespan: {current_makespan}, Utility: {current_utility}")
                
                # Record values in history (even if not improved - to show exploration)
                self.makespan_history.append(current_makespan)
                self.utility_history.append(current_utility)
                
                if not improved:
                    generations_no_improvement += 1
                    
                    # Apply cataclysm if stuck for too long
                    if generations_no_improvement >= self.cataclysm_interval:
                        print(f"Generation {generation+1}: Applying cataclysm due to {generations_no_improvement} generations without improvement")
                        self.apply_cataclysm()
                        generations_no_improvement = 0
                
                # Check termination condition
                if generations_no_improvement >= self.max_no_improvement:
                    print(f"Terminating after {generation+1} generations due to no improvement for {generations_no_improvement} generations")
                    break
            
            # Dynamically adjust mutation and crossover rates
            self.adjust_parameters(generation)
        
        # Check if we found a valid solution
        if self.best_makespan <= self.deadline:
            print(f"Phase 1 completed successfully - Best makespan: {self.best_makespan}, Utility: {self.best_utility}")
            return True
        else:
            print(f"Phase 1 failed to find valid solution - Best makespan: {self.best_makespan}, Deadline: {self.deadline}")
            return False
    
    def get_best_chromosome(self):
        """Get the best chromosome from the current population"""
        valid_chromosomes = [c for c in self.population if c['fitness']['makespan'] <= self.deadline]
        
        if valid_chromosomes:
            # Return chromosome with highest utility among valid ones
            return max(valid_chromosomes, key=lambda c: c['fitness']['utility'])
        else:
            # If no valid chromosomes, return one with lowest makespan
            return min(self.population, key=lambda c: c['fitness']['makespan'])
    
    def apply_cataclysm(self):
        """Apply cataclysm by keeping elite chromosomes and regenerating the rest"""
        # Keep top 10% of population
        self.population.sort(key=lambda c: (
            c['fitness']['makespan'] <= self.deadline,  # Valid solutions first
            -c['fitness']['makespan'],  # Lower makespan
            c['fitness']['utility']  # Higher utility
        ), reverse=True)
        
        elite_size = max(1, int(self.pop_size * 0.1))
        elite = self.population[:elite_size]
        
        # Regenerate the rest
        self.population = elite
        while len(self.population) < self.pop_size:
            chromosome = self.generate_valid_chromosome_cts()
            self.evaluate_chromosome(chromosome)
            self.population.append(chromosome)
    
    def adjust_parameters(self, generation):
        """Dynamically adjust mutation and crossover rates based on generation"""
        # Start with exploration (high mutation), move towards exploitation (low mutation)
        progress = min(1.0, generation / (self.generations * 0.7))
        self.mutation_rate = max(0.01, 0.2 - progress * 0.15)
        self.crossover_rate = min(0.95, 0.7 + progress * 0.2)
    
    def run_phase2(self):
        """Run Phase 2 of the GA algorithm to enhance security"""
        print("\nRunning GA Phase 2: Enhancing security utility...")
        
        if not self.best_chromosome or self.best_makespan > self.deadline:
            print("Cannot run Phase 2: No valid solution found in Phase 1")
            return False
        
        # Create temporary objects for Phase 2
        temp_tasks = copy.deepcopy(self.tasks)
        temp_messages = copy.deepcopy(self.messages)
        
        # Apply best chromosome from Phase 1
        for task in temp_tasks:
            task.assigned_processor = self.best_chromosome['proc_assign'].get(task.task_id)
            task.is_scheduled = False
        
        for message in temp_messages:
            if message.id in self.best_chromosome['security_levels']:
                levels = self.best_chromosome['security_levels'][message.id]
                message.assigned_security = {
                    'confidentiality': levels['confidentiality'],
                    'integrity': levels['integrity'],
                    'authentication': levels['authentication']
                }
        
        # Initialize max-heap with candidate security upgrades
        upgrade_heap = []
        
        # Schedule tasks according to best chromosome
        self.schedule_by_chromosome(temp_tasks, temp_messages, self.best_chromosome['task_seq'])
        
        # Add all possible security upgrades to heap
        for message in temp_messages:
            source_task = next((t for t in temp_tasks if t.task_id == message.source_id), None)
            dest_task = next((t for t in temp_tasks if t.task_id == message.dest_id), None)
            
            if not source_task or not dest_task or not source_task.is_scheduled or not dest_task.is_scheduled:
                continue
            
            for service in ['confidentiality', 'integrity', 'authentication']:
                current_level = message.assigned_security[service]
                max_level = len(self.security.strengths[service]) - 1
                
                # If not at maximum security level
                if current_level < max_level:
                    # Calculate benefit-to-cost ratio for upgrading
                    bcr = self.calculate_upgrade_bcr(message, service, current_level, source_task, dest_task)
                    
                    # Add to heap if upgrade is beneficial
                    if bcr > 0:
                        heapq.heappush(upgrade_heap, (-bcr, message.id, service, current_level))
        
        # Process upgrades in BCR order
        upgraded_count = 0
        while upgrade_heap:
            # Get next best upgrade
            bcr, message_id, service, current_level = heapq.heappop(upgrade_heap)
            bcr = -bcr  # Convert back to positive
            
            # Get message and related tasks
            message = next((m for m in temp_messages if m.id == message_id), None)
            if not message:
                continue
                
            source_task = next((t for t in temp_tasks if t.task_id == message.source_id), None)
            dest_task = next((t for t in temp_tasks if t.task_id == message.dest_id), None)
            
            if not source_task or not dest_task:
                continue
            
            # Try to upgrade security level
            if current_level == message.assigned_security[service]:  # Make sure it hasn't been upgraded already
                # Upgrade to next level
                next_level = current_level + 1
                
                # Try the upgrade
                old_level = message.assigned_security[service]
                message.assigned_security[service] = next_level
                
                # Recalculate schedule with upgraded security
                makespan = self.schedule_by_chromosome(temp_tasks, temp_messages, self.best_chromosome['task_seq'])
                
                # Check if deadline is still met
                if makespan <= self.deadline:
                    # Update best chromosome
                    self.best_chromosome['security_levels'][message_id][service] = next_level
                    upgraded_count += 1
                    print(f"Upgraded {message_id} {service} from level {old_level} to {next_level} (BCR: {bcr:.4f})")
                    
                    # Check if further upgrade is possible
                    if next_level < len(self.security.strengths[service]) - 1:
                        # Calculate BCR for next upgrade
                        bcr = self.calculate_upgrade_bcr(message, service, next_level, source_task, dest_task)
                        if bcr > 0:
                            heapq.heappush(upgrade_heap, (-bcr, message_id, service, next_level))
                else:
                    # Rollback upgrade
                    message.assigned_security[service] = old_level
                    # Restore schedule
                    self.schedule_by_chromosome(temp_tasks, temp_messages, self.best_chromosome['task_seq'])
        
        # Recalculate best solution with final security levels
        final_makespan = self.schedule_by_chromosome(temp_tasks, temp_messages, self.best_chromosome['task_seq'])
        final_utility = self.calculate_security_utility_for_chromosome(temp_tasks, temp_messages)
        
        print(f"Phase 2 completed: {upgraded_count} security upgrades performed")
        print(f"Final solution - Makespan: {final_makespan}, Security Utility: {final_utility}")
        
        # Update best fitness
        self.best_makespan = final_makespan
        self.best_utility = final_utility
        self.best_chromosome['fitness'] = {'makespan': final_makespan, 'utility': final_utility}
        
        return True
    def calculate_upgrade_bcr(self, message, service, current_level, source_task, dest_task):
        """Calculate the Benefit-to-Cost Ratio for upgrading a security service level"""
        # Get next level
        next_level = current_level + 1
        
        # Ensure next level is valid
        if next_level >= len(self.security.strengths[service]):
            return 0  # No upgrade possible
        
        # Calculate security utility gain
        current_strength = self.security.strengths[service][current_level]
        next_strength = self.security.strengths[service][next_level]
        weight = message.weights[service]
        
        # Utility gain formula: weight * (next_strength - current_strength)
        utility_gain = weight * (next_strength - current_strength)
        
        # Calculate overhead increase
        current_overhead = 0
        next_overhead = 0
        
        source_proc_id = source_task.assigned_processor
        dest_proc_id = dest_task.assigned_processor
        
        # Calculate sender-side overhead for both levels
        if service in ['confidentiality', 'integrity']:
            # For these services, overhead depends on message size and processor speed
            current_factor = self.security.overheads[service][current_level + 1][source_proc_id - 1]
            next_factor = self.security.overheads[service][next_level + 1][source_proc_id - 1]
            
            current_overhead += message.size / current_factor
            next_overhead += message.size / next_factor
        
        # For authentication, overhead is a fixed value
        if service == 'authentication':
            current_overhead += self.security.overheads[service][current_level + 1][source_proc_id - 1]
            next_overhead += self.security.overheads[service][next_level + 1][source_proc_id - 1]
        
        # Calculate receiver-side overhead for both levels (if tasks are on different processors)
        if source_proc_id != dest_proc_id:
            if service in ['confidentiality', 'integrity']:
                current_factor = self.security.overheads[service][current_level + 1][dest_proc_id - 1]
                next_factor = self.security.overheads[service][next_level + 1][dest_proc_id - 1]
                
                current_overhead += message.size / current_factor
                next_overhead += message.size / next_factor
            
            if service == 'authentication':
                current_overhead += self.security.overheads[service][current_level + 1][dest_proc_id - 1]
                next_overhead += self.security.overheads[service][next_level + 1][dest_proc_id - 1]
        
        # Calculate overhead increase
        overhead_increase = next_overhead - current_overhead
        
        # If there's no overhead increase, return a very large BCR
        if overhead_increase <= 0:
            return float('inf')
        
        # Calculate BCR
        bcr = utility_gain / overhead_increase
        
        return bcr
    
    def plot_ga_gantt_chart(scheduler):
        """
        Create a Gantt chart for visualizing the GA scheduler's task schedule with security information.
        
        Args:
            scheduler: The GASecurityScheduler instance
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Rectangle
        import copy
        
        # Create temporary tasks with final schedule
        temp_tasks = copy.deepcopy(scheduler.tasks)
        temp_messages = copy.deepcopy(scheduler.messages)

        # Apply best chromosome
        for task in temp_tasks:
            task.assigned_processor = scheduler.best_chromosome['proc_assign'].get(task.task_id)
            task.is_scheduled = False

        # Apply security levels
        for message in temp_messages:
            if message.id in scheduler.best_chromosome['security_levels']:
                levels = scheduler.best_chromosome['security_levels'][message.id]
                message.assigned_security = {
                    'confidentiality': levels['confidentiality'],
                    'integrity': levels['integrity'],
                    'authentication': levels['authentication']
                }

        # Schedule tasks according to best chromosome
        makespan = scheduler.schedule_by_chromosome(temp_tasks, temp_messages, scheduler.best_chromosome['task_seq'])

        # Convert to schedule format
        schedule = []
        for task in temp_tasks:
            if task.is_scheduled:
                schedule.append({
                    'task_id': task.task_id,
                    'processor': task.assigned_processor,
                    'start_time': task.start_time,
                    'finish_time': task.finish_time
                })
        
        # Setup the figure
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Colors for different tasks
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
        # Create title with metrics
        title = f'GA Security Schedule (Makespan: {makespan:.2f}, Utility: {scheduler.best_utility:.2f})'
        
        # Create task mapping for quick lookup
        task_map = {task.task_id: task for task in temp_tasks}
        
        # Create message mapping for quick lookup
        msg_map = {}
        for msg in temp_messages:
            if hasattr(msg, 'source_id') and hasattr(msg, 'dest_id'):
                key = f"{msg.source_id}_{msg.dest_id}"
                msg_map[key] = msg
        
        # Security protocol abbreviations for better readability
        security_protocols = {
            'confidentiality': ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'],
            'integrity': ['I0', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6'],
            'authentication': ['A0', 'A1', 'A2']
        }
        
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
            
            # Add basic task label
            task_label = f"T{task_id}"
            
            # Check if we have security information to show
            security_info = ""
            task = task_map.get(task_id)
            
            if task and hasattr(task, 'successors'):
                # Collect security levels for outgoing messages
                outgoing_security = []
                
                for succ_id in task.successors:
                    msg_key = f"{task_id}_{succ_id}"
                    msg = msg_map.get(msg_key)
                    
                    if msg and hasattr(msg, 'assigned_security'):
                        # Get security protocol names
                        conf_idx = msg.assigned_security.get('confidentiality', 0)
                        integ_idx = msg.assigned_security.get('integrity', 0)
                        auth_idx = msg.assigned_security.get('authentication', 0)
                        
                        conf = security_protocols['confidentiality'][conf_idx] if conf_idx < len(security_protocols['confidentiality']) else 'C?'
                        integ = security_protocols['integrity'][integ_idx] if integ_idx < len(security_protocols['integrity']) else 'I?'
                        auth = security_protocols['authentication'][auth_idx] if auth_idx < len(security_protocols['authentication']) else 'A?'
                        
                        outgoing_security.append(f"→T{succ_id}:{conf}/{integ}/{auth}")
                
                if outgoing_security:
                    security_info = "\n" + "\n".join(outgoing_security)
            
            # Add task label with security info
            ax.text(start + (finish-start)/2, processor, f"{task_label}{security_info}", 
                    ha='center', va='center', color='black', fontsize=10, weight='bold')
        
        # Draw deadline line
        ax.axvline(x=scheduler.deadline, color='red', linestyle='--', label='Deadline')
        
        # Set up the axes
        num_processors = len(scheduler.processors)
        ax.set_ylim(0.5, num_processors + 0.5)
        ax.set_xlim(0, makespan * 1.05)  # Add some padding
        ax.set_yticks(range(1, num_processors + 1))
        ax.set_yticklabels([f'Processor {i}' for i in range(1, num_processors + 1)])
        ax.set_xlabel('Time (ms)')
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add task sequence as text below the chart
        task_seq_text = "Task Sequence: " + " → ".join([f"T{tid}" for tid in scheduler.best_chromosome['task_seq']])
        fig.text(0.5, 0.01, task_seq_text, ha='center', fontsize=10)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Make room for task sequence text
        
        return fig, ax

    def plot_modified_ga_gantt_chart(scheduler):
        """
        Create a Gantt chart for GA schedule showing:
        - Base execution time
        - Security overhead
        - Different colors for each component
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Patch
        import copy
        
        # Create temporary tasks with final schedule
        temp_tasks = copy.deepcopy(scheduler.tasks)
        temp_messages = copy.deepcopy(scheduler.messages)

        # Apply best chromosome
        for task in temp_tasks:
            task.assigned_processor = scheduler.best_chromosome['proc_assign'].get(task.task_id)
            task.is_scheduled = False

        # Apply security levels
        for message in temp_messages:
            if message.id in scheduler.best_chromosome['security_levels']:
                levels = scheduler.best_chromosome['security_levels'][message.id]
                message.assigned_security = {
                    'confidentiality': levels['confidentiality'],
                    'integrity': levels['integrity'],
                    'authentication': levels['authentication']
                }

        # Schedule tasks according to best chromosome
        makespan = scheduler.schedule_by_chromosome(temp_tasks, temp_messages, scheduler.best_chromosome['task_seq'])
        
        # Create schedule entries with computed times
        schedule = []
        for task in temp_tasks:
            if task.is_scheduled:
                base_exec_time = task.execution_times[task.assigned_processor - 1]
                
                schedule.append({
                    'task_id': task.task_id,
                    'name': f"Task {task.task_id}",
                    'processor': task.assigned_processor,
                    'start_time': task.start_time,
                    'finish_time': task.finish_time,
                    'base_time': base_exec_time,
                    'security_overhead': task.finish_time - task.start_time - base_exec_time
                })
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Define colors for different components
        base_color = 'royalblue'
        security_color = 'tomato'
        
        # Create legend elements
        legend_elements = [
            Patch(facecolor=base_color, edgecolor='black', label='Base Execution'),
            Patch(facecolor=security_color, edgecolor='black', label='Security Overhead')
        ]
        
        for entry in schedule:
            task_id = entry['task_id']
            proc = entry['processor']
            start = entry['start_time']
            base_exec_time = entry['base_time']
            security_overhead = entry['security_overhead']
            
            # Draw base execution time bar
            ax.barh(proc-1, base_exec_time, left=start, color=base_color, edgecolor='black')
            
            # Draw security overhead bar (if any)
            if security_overhead > 0:
                ax.barh(proc-1, security_overhead, left=start + base_exec_time, 
                    color=security_color, edgecolor='black')
            
            # Add task label in the middle of the base execution part
            ax.text(start + base_exec_time/2, proc-1, f"T{task_id}", 
                    va='center', ha='center', color='white', fontsize=10, weight='bold')
        
        # Draw deadline line
        ax.axvline(x=scheduler.deadline, color='red', linestyle='--', label='Deadline')
        
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Processor")
        ax.set_title(f"GA Security Schedule - Base vs Overhead (Makespan: {makespan:.2f} ms)")
        ax.set_yticks(range(len(scheduler.processors)))
        ax.set_yticklabels([f"Processor {p+1}" for p in range(len(scheduler.processors))])
        ax.set_xlim(0, makespan * 1.1)  # Add some padding
        
        # Add legend
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        return fig, ax

    def plot_ga_performance(scheduler, makespan_history, utility_history):
        """
        Plot the makespan and security utility improvement over GA generations.
        Creates both 2D and 3D visualizations.
        
        Args:
            scheduler: The GASecurityScheduler instance
            makespan_history: List of makespan values by generation
            utility_history: List of security utility values by generation
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create figure with 3 subplots (2 for 2D plots and 1 for 3D)
        fig = plt.figure(figsize=(18, 12))
        
        # First subplot: Makespan vs Generation
        ax1 = fig.add_subplot(221)
        generations = range(1, len(makespan_history) + 1)
        ax1.plot(generations, makespan_history, 'b-', linewidth=2)
        ax1.axhline(y=scheduler.deadline, color='r', linestyle='--', label='Deadline')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Makespan (ms)')
        ax1.set_title('Makespan Improvement')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # Second subplot: Security Utility vs Generation
        ax2 = fig.add_subplot(222)
        ax2.plot(generations, utility_history, 'g-', linewidth=2)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Security Utility')
        ax2.set_title('Security Utility Improvement')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Third subplot: 3D visualization of Makespan vs Security Utility vs Generation
        ax3 = fig.add_subplot(212, projection='3d')
        
        # Create a colormap for the points
        colors = plt.cm.viridis(np.linspace(0, 1, len(generations)))
        
        # Plot the 3D trajectory
        ax3.plot3D(generations, makespan_history, utility_history, 'gray', alpha=0.5)
        
        # Scatter plot with color gradient by generation
        scatter = ax3.scatter(generations, makespan_history, utility_history, 
                            c=generations, cmap='viridis', s=40, alpha=0.8)
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax3, pad=0.1)
        cbar.set_label('Generation')
        
        # Add deadline plane
        x_grid, y_grid = np.meshgrid(
            np.linspace(min(generations), max(generations), 10),
            np.linspace(min(utility_history), max(utility_history), 10)
        )
        z_grid = np.full_like(x_grid, scheduler.deadline)
        ax3.plot_surface(x_grid, z_grid, y_grid, color='r', alpha=0.2)
        
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Makespan (ms)')
        ax3.set_zlabel('Security Utility')
        ax3.set_title('3D Visualization of GA Optimization')
        
        plt.tight_layout()
        return fig

    def show_security_assignments(scheduler):
        """Display security assignments for all messages in a tabular format."""
        import matplotlib.pyplot as plt
        
        # Apply best chromosome to get messages with assigned security
        temp_messages = copy.deepcopy(scheduler.messages)
        for message in temp_messages:
            if message.id in scheduler.best_chromosome['security_levels']:
                levels = scheduler.best_chromosome['security_levels'][message.id]
                message.assigned_security = {
                    'confidentiality': levels['confidentiality'],
                    'integrity': levels['integrity'],
                    'authentication': levels['authentication']
                }
        
        # Create data for the table
        message_ids = []
        confidentiality = []
        integrity = []
        authentication = []
        
        for message in temp_messages:
            if hasattr(message, 'assigned_security'):
                message_ids.append(f"M{message.id}")
                confidentiality.append(message.assigned_security['confidentiality'])
                integrity.append(message.assigned_security['integrity'])
                authentication.append(message.assigned_security['authentication'])
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, len(message_ids) * 0.4 + 1))
        ax.axis('tight')
        ax.axis('off')
        
        # Create the table
        table = ax.table(
            cellText=np.column_stack((confidentiality, integrity, authentication)),
            rowLabels=message_ids,
            colLabels=['Confidentiality', 'Integrity', 'Authentication'],
            loc='center',
            cellLoc='center'
        )
        
        # Adjust table appearance
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.title('Security Level Assignments')
        
        return fig, ax


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer for QL-SHIELD algorithm.
    Stores experiences with priorities for efficient learning.
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment
        self.epsilon = 1e-5  # Small constant to avoid zero priority

    def add(self, experience, priority=None):
        """Add experience to buffer with priority"""
        if priority is None:
            # Default to max priority for new experiences
            priority = max(self.priorities) if self.priorities else 1.0
        
        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size):
        """Sample batch_size experiences based on priorities"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
            
        # Convert priorities to probabilities
        probs = np.array(self.priorities) ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # Get samples
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        # Increase beta towards 1
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return samples, indices, weights

    def update_priorities(self, indices, new_priorities):
        """Update priorities for the experiences at the specified indices"""
        for idx, priority in zip(indices, new_priorities):
            self.priorities[idx] = priority + self.epsilon

    def __len__(self):
        return len(self.buffer)

class QLearningAgent:
    """Q-Learning Agent for optimal task scheduling"""
    
    def __init__(self, num_tasks, num_processors, alpha=0.1, gamma=0.9, epsilon=0.3, 
                 epsilon_decay=0.01, epsilon_min=0.05, replay_size=5000, batch_size=32):
        self.num_tasks = num_tasks
        self.num_processors = num_processors
        self.alpha = alpha          # Learning rate
        self.gamma = gamma          # Discount factor
        self.epsilon = epsilon      # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # State space: Combination of task ID and ready-tasks list
        # Action space: Processor assignment
        self.q_table = {}  # Will be a dictionary: {(task_id, tuple(ready_tasks)): [values for each processor]}
        
        # Initialize priority replay buffer
        self.replay_buffer = PriorityReplayBuffer(replay_size)
        self.batch_size = batch_size
        
    def get_q_value(self, state, action):
        """Get Q-value for a state-action pair"""
        # Convert ready tasks to tuple for hashable state
        state_key = self._get_state_key(state)
        
        # Initialize Q-values for new states
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.num_processors)
            
        return self.q_table[state_key][action]
    
    def _get_state_key(self, state):
        """Convert state to hashable key"""
        task_id, ready_tasks = state
        return (task_id, tuple(sorted(ready_tasks)))
    
    def choose_action(self, state, valid_processors):
        """Choose an action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            # Exploration: choose randomly from valid processors
            return np.random.choice(valid_processors)
        else:
            # Exploitation: choose best valid processor
            state_key = self._get_state_key(state)
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.num_processors)
            
            # Get Q-values only for valid processors
            valid_q_values = []
            for p in valid_processors:
                # Ensure processor index is 0-based for q_table access
                p_idx = p - 1
                if 0 <= p_idx < self.num_processors:
                    valid_q_values.append(self.q_table[state_key][p_idx])
                else:
                    valid_q_values.append(0)  # Default to 0 if index is out of bounds
            
            if not valid_q_values:
                return np.random.choice(valid_processors)  # Fallback if no valid values
                
            max_q_value = max(valid_q_values)
            
            # Find all actions with the max Q-value
            best_actions = [p for i, p in enumerate(valid_processors) 
                           if i < len(valid_q_values) and valid_q_values[i] == max_q_value]
            
            if not best_actions:
                return np.random.choice(valid_processors)  # Fallback if no best actions found
                
            # Randomly choose one of the best actions
            return np.random.choice(best_actions)
    
    def update_q_value(self, state, action, reward, next_state, next_valid_processors):
        """Update Q-value for a state-action pair"""
        state_key = self._get_state_key(state)
        
        # Initialize Q-values for new states
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.num_processors)
            
        # Ensure action index is within bounds
        if not (0 <= action < self.num_processors):
            action = min(max(0, action), self.num_processors - 1)
            
        # Get current Q-value
        current_q = self.q_table[state_key][action]
        
        # Get maximum Q-value for next state
        next_state_key = self._get_state_key(next_state)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.num_processors)
            
        # Get max Q-value only for valid processors in next state
        next_max_q = 0
        if next_valid_processors:
            next_q_values = []
            for p in next_valid_processors:
                # Convert 1-indexed processor to 0-indexed for q_table
                p_idx = p - 1
                if 0 <= p_idx < self.num_processors:
                    next_q_values.append(self.q_table[next_state_key][p_idx])
                else:
                    next_q_values.append(0)  # Default value for out of bounds
                    
            if next_q_values:  # Check if we have any valid q values
                next_max_q = max(next_q_values)
        
        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state_key][action] = new_q
        
        # Calculate TD error for priority replay
        td_error = abs(reward + self.gamma * next_max_q - current_q)
        
        # Store experience in replay buffer
        self.replay_buffer.add(
            (state, action, reward, next_state, next_valid_processors), 
            td_error
        )
        
        return td_error
    
    def replay(self):
        """Learn from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # Sample from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Update Q-values for each experience
        for experience, _ in batch:
            state, action, reward, next_state, next_valid_processors = experience
            
            # Get current Q-value
            state_key = self._get_state_key(state)
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.num_processors)
                
            # Ensure action index is within bounds
            if not (0 <= action < self.num_processors):
                action = min(max(0, action), self.num_processors - 1)
                
            current_q = self.q_table[state_key][action]
            
            # Get maximum Q-value for next state
            next_state_key = self._get_state_key(next_state)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.zeros(self.num_processors)
                
            # Get max Q-value only for valid processors in next state
            next_max_q = 0
            if next_valid_processors:
                next_q_values = []
                for p in next_valid_processors:
                    # Convert 1-indexed processor to 0-indexed for q_table
                    p_idx = p - 1
                    if 0 <= p_idx < self.num_processors:
                        next_q_values.append(self.q_table[next_state_key][p_idx])
                    else:
                        next_q_values.append(0)  # Default value for out of bounds
                        
                if next_q_values:  # Check if we have any valid q values
                    next_max_q = max(next_q_values)
            
            # Q-learning update
            new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
            self.q_table[state_key][action] = new_q
            
            # Update priority in replay buffer
            td_error = abs(reward + self.gamma * next_max_q - current_q)
            self.replay_buffer.update_priority(experience, td_error)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class PriorityReplayBuffer:
    """Priority Replay Buffer for more efficient learning"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.position = 0
        
    def __len__(self):
        return len(self.buffer)
        
    def add(self, experience, priority):
        """Add experience to buffer with priority"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
            
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """Sample batch_size experiences based on priorities"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
            
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Return experiences and their indices
        batch = [(self.buffer[i], self.priorities[i]) for i in indices]
        return batch
        
    def update_priority(self, experience, priority):
        """Update priority of an experience"""
        try:
            idx = self.buffer.index(experience)
            self.priorities[idx] = priority
        except ValueError:
            # Experience not in buffer
            pass

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import heapq

class DQNetwork(nn.Module):
    """Deep Q-Network for task scheduling decisions"""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer for storing and sampling experiences"""
    
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.alpha = alpha  # determines how much prioritization is used
        self.beta = beta    # importance-sampling weight
        self.beta_increment = beta_increment
        self.Experience = namedtuple('Experience', 
                                    field_names=['state', 'action', 'reward', 'next_state', 'done'])
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        max_priority = np.max(self.priorities) if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            
        e = self.Experience(state, action, reward, next_state, done)
        self.memory[self.position] = e
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample a batch of experiences based on priorities"""
        if len(self.memory) < batch_size:
            return None
            
        # Increase beta over time for more accurate bias correction
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get sampling probabilities from priorities
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
            
        # Calculate sampling probabilities and importance sampling weights
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        # Get experiences from memory
        experiences = [self.memory[idx] for idx in indices]
        
        # Extract components and convert to tensors
        states = torch.tensor(np.vstack([e.state for e in experiences]), dtype=torch.float32)
        actions = torch.tensor(np.vstack([e.action for e in experiences]), dtype=torch.long)
        rewards = torch.tensor(np.vstack([e.reward for e in experiences]), dtype=torch.float32)
        next_states = torch.tensor(np.vstack([e.next_state for e in experiences]), dtype=torch.float32)
        dones = torch.tensor(np.vstack([e.done for e in experiences]).astype(np.uint8), dtype=torch.float32)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, errors):
        """Update priorities based on TD errors"""
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error + 1e-5  # Small constant to ensure non-zero probability
            
    def __len__(self):
        """Return the current size of memory"""
        return len(self.memory)

class QLShield(HSMS):
    """
    Q-Learning-based SHIELD: Security-aware scheduling with RL-based task allocation
    and security utility maximization under real-time constraints.
    """
    
    def __init__(self, tasks, messages, processors, security_service, deadline):
        super().__init__(tasks, messages, processors, security_service, deadline)
        
        # Q-Learning parameters
        self.state_size = len(tasks) * 3  # Task status, processor assignment, sequence position
        self.action_size = len(processors) * len(tasks)  # Each action is assigning a task to a processor
        
        # DQN Network parameters
        self.learning_rate = 0.001
        self.batch_size = 64
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.update_target_every = 10  # Update target network every n episodes
        self.tau = 0.001  # Soft update parameter
        
        # Initialize networks
        self.qnetwork_local = DQNetwork(self.state_size, self.action_size)
        self.qnetwork_target = DQNetwork(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        
        # Initialize replay memory
        self.memory = PrioritizedReplayBuffer(10000)
        
        # Security enhancement related variables
        self.original_makespan = None
        self.bcr_heap = []  # Benefit-to-Cost Ratio heap
        self.upgrade_history = []  # Track security upgrades for reporting
        
        # Best schedule tracking
        self.best_makespan = float('inf')
        self.best_schedule = None
        self.best_task_assignments = None
    def seed_with_hsms_solution(self, hsms_scheduler):
        """Seed the QL replay buffer with experiences from HSMS solution."""
        print("Seeding QL-SHIELD with HSMS solution...")
        
        # Copy the task assignments from HSMS to our scheduler
        hsms_task_assignments = {}
        
        # First, get all task assignments from the HSMS solution
        for task in hsms_scheduler.tasks:
            if task.is_scheduled:
                hsms_task_assignments[task.task_id] = task.assigned_processor
        
        # Generate replay buffer experiences from the HSMS solution
        # First reset environment to clean state
        state = self.reset_environment()
        
        # Now step through the tasks according to HSMS schedule
        # We'll use the task ordering from HSMS
        sorted_tasks = sorted([t for t in hsms_scheduler.tasks if t.is_scheduled], 
                            key=lambda t: t.start_time)
        
        for task in sorted_tasks:
            # Find the corresponding task in our scheduler
            our_task = self.get_task_by_id(task.task_id)
            
            # Get the action (task-processor assignment)
            proc_id = task.assigned_processor
            action_idx = (task.task_id - 1) * len(self.processors) + (proc_id - 1)
            action = np.array([[action_idx]])
            
            # Take the action and observe result
            next_state, reward, done = self.take_action(state, action)
            
            # Store this experience in replay buffer with a higher priority
            # (higher priority will make the agent sample this more often early in training)
            self.memory.add(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            
            if done:
                break
        
        # Initialize best schedule with HSMS solution
        self.best_makespan = hsms_scheduler.makespan
        self.best_schedule = hsms_scheduler.schedule
        self.best_task_assignments = hsms_task_assignments
        
        print(f"Successfully seeded replay buffer with HSMS solution (makespan: {self.best_makespan})")
    def run(self):
        """Execute the QL-SHIELD algorithm in phases."""
        # Phase 1: Assign minimum security levels to all messages
        hsms_scheduler = HSMS(self.tasks, self.messages, self.processors, self.security, self.deadline)
        hsms_makespan, hsms_security_utility = hsms_scheduler.run()
        
        print(f"\nHSMS Baseline Solution - Makespan: {hsms_makespan}, Security Utility: {hsms_security_utility}")
        
        # Seed the learning with HSMS solution
        self.seed_with_hsms_solution(hsms_scheduler)
        self.assign_minimum_security()
        
        # Phase 2: Use QL to find the optimal schedule
        self.train_ql_scheduler(num_episodes=1000)
        
        # Apply the best schedule found
        self.apply_best_schedule()
        
        # Record original makespan
        self.original_makespan = self.makespan
        print(f"\nQL-SHIELD Phase 1-2 complete: Initial makespan={self.makespan}, Security utility={self.calculate_security_utility()}")
        
        # Phase 3: Security Enhancement
        print("\nQL-SHIELD Phase 3: Starting Security Enhancement...")
        self.enhance_security()
        
        # Calculate final results
        final_makespan = self.finalize_schedule()
        final_security_utility = self.calculate_security_utility()
        
        print(f"QL-SHIELD complete: Final makespan={final_makespan}, Final security utility={final_security_utility}")
        print(f"Original security utility: {self.calculate_security_utility_for_schedule(self.best_schedule)}")
        print(f"Security utility improvement: {(final_security_utility - self.calculate_security_utility_for_schedule(self.best_schedule)) / self.calculate_security_utility_for_schedule(self.best_schedule) * 100:.2f}%")
        
        return final_makespan, final_security_utility
    
    def train_ql_scheduler(self, num_episodes=1000):
        """Train the Q-Learning scheduler to find optimal task-processor mappings."""
        print(f"\nTraining Q-Learning Scheduler for {num_episodes} episodes...")
        
        for episode in range(1, num_episodes+1):
            # Reset environment and get initial state
            state = self.reset_environment()
            episode_reward = 0
            
            # Process each task in the DAG
            for step in range(len(self.tasks)):
                # Choose an action (task-processor assignment)
                action = self.get_action(state)
                
                # Take action and observe next state and reward
                next_state, reward, done = self.take_action(state, action)
                
                # Store experience in replay memory
                self.memory.add(state, action, reward, next_state, done)
                
                # Learn from experience
                if len(self.memory) > self.batch_size:
                    self.learn()
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Update exploration rate
            self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
            
            # Update target network periodically
            if episode % self.update_target_every == 0:
                self.soft_update()
            
            # Print progress and update best schedule
            if episode % 50 == 0:
                print(f"Episode {episode}/{num_episodes}, Reward: {episode_reward:.2f}, Epsilon: {self.epsilon:.4f}")
                print(f"Best makespan so far: {self.best_makespan:.2f}")
                
            # After certain episodes, print detailed task-processor mappings
            if episode % 200 == 0 or episode == num_episodes:
                if self.best_schedule:
                    print("\nCurrent best task-processor mappings:")
                    for task_id, proc_id in self.best_task_assignments.items():
                        task = self.get_task_by_id(task_id)
                        print(f"Task {task_id} ({task.name}) -> Processor {proc_id}")
        
        print(f"\nTraining complete. Best makespan: {self.best_makespan:.2f}")
    
    def reset_environment(self):
        """Reset the scheduling environment for a new episode."""
        # Reset task assignments and scheduling
        for task in self.tasks:
            task.is_scheduled = False
            task.assigned_processor = None
            task.start_time = None
            task.finish_time = None
            
        # Reset processors
        for proc in self.processors:
            proc.available_time = 0
            
        # Reset schedule
        self.schedule = []
        
        # Create initial state representation
        state = self.create_state_representation()
        
        return state
    
    def create_state_representation(self):
        """Create state representation for the current scheduling state."""
        state = np.zeros(self.state_size)
        
        for i, task in enumerate(self.tasks):
            # State features for each task:
            # 1. Is task scheduled (0 or 1)
            # 2. Which processor is it assigned to (normalized)
            # 3. Relative position in the schedule (normalized)
            
            base_idx = i * 3
            
            state[base_idx] = 1 if task.is_scheduled else 0
            
            if task.assigned_processor is not None:
                state[base_idx + 1] = (task.assigned_processor - 1) / (len(self.processors) - 1) if len(self.processors) > 1 else 0
            else:
                state[base_idx + 1] = 0
                
            if task.start_time is not None:
                # Normalize position relative to makespan
                current_makespan = max((t.finish_time for t in self.tasks if t.is_scheduled), default=1)
                state[base_idx + 2] = task.start_time / current_makespan if current_makespan > 0 else 0
            else:
                state[base_idx + 2] = 0
                
        return state
    
    def get_action(self, state):
        """Choose an action based on current state using epsilon-greedy policy."""
        # Find schedulable tasks (all predecessors scheduled)
        schedulable_tasks = self.get_schedulable_tasks()
        
        if not schedulable_tasks:
            # If no tasks can be scheduled, return a dummy action
            return np.array([[0]])
            
        # With probability epsilon, choose a random valid action
        if random.random() < self.epsilon:
            task = random.choice(schedulable_tasks)
            processor = random.randint(1, len(self.processors))
            action_idx = (task.task_id - 1) * len(self.processors) + (processor - 1)
            return np.array([[action_idx]])
        
        # Otherwise choose the best action according to the policy network
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()
        
        # Create a mask for valid actions
        valid_actions = np.full(self.action_size, -np.inf)
        for task in schedulable_tasks:
            for p in range(1, len(self.processors) + 1):
                action_idx = (task.task_id - 1) * len(self.processors) + (p - 1)
                valid_actions[action_idx] = action_values[0][action_idx].item()
        
        # Choose best valid action
        best_action = np.argmax(valid_actions)
        return np.array([[best_action]])
    
    def get_schedulable_tasks(self):
        """Get list of tasks that can be scheduled (all predecessors are scheduled)."""
        schedulable = []
        
        for task in self.tasks:
            if task.is_scheduled:
                continue
                
            # Check if all predecessors are scheduled
            all_preds_scheduled = all(
                self.get_task_by_id(pred_id).is_scheduled 
                for pred_id in task.predecessors
            )
            
            if all_preds_scheduled:
                schedulable.append(task)
                
        return schedulable
    
    def take_action(self, state, action):
        """
        Take the chosen action and update the environment state.
        Return the next state, reward, and done flag.
        """
        action_value = action[0][0]
        
        # If dummy action, all tasks are scheduled
        if action_value == 0 and all(task.is_scheduled for task in self.tasks):
            # Complete the schedule
            makespan = self.finalize_schedule()
            
            # Save if this is the best schedule so far
            if makespan <= self.deadline and makespan < self.best_makespan:
                self.best_makespan = makespan
                self.best_schedule = self.save_schedule_state()
                self.best_task_assignments = {task.task_id: task.assigned_processor for task in self.tasks}
                
            # Calculate reward based on makespan and deadline satisfaction
            reward = self.calculate_reward(makespan)
            
            # Next state is the same as current state
            next_state = state
            done = True
            return next_state, reward, done
            
        # Convert action value to task and processor ids
        num_processors = len(self.processors)
        task_idx = action_value // num_processors
        proc_idx = action_value % num_processors
        
        task_id = task_idx + 1
        processor_id = proc_idx + 1
        
        # Get the task to schedule
        task = self.get_task_by_id(task_id)
        
        # If task is already scheduled or not valid, give negative reward
        if not task or task.is_scheduled:
            return state, -10, False
            
        # Schedule the task on the chosen processor
        self.schedule_task(task, processor_id)
        
        # Check if all tasks are scheduled
        done = all(task.is_scheduled for task in self.tasks)
        
        # If all tasks are scheduled, calculate makespan and final reward
        if done:
            makespan = self.finalize_schedule()
            
            # Save if this is the best schedule so far
            if makespan <= self.deadline and makespan < self.best_makespan:
                self.best_makespan = makespan
                self.best_schedule = self.save_schedule_state()
                self.best_task_assignments = {task.task_id: task.assigned_processor for task in self.tasks}
                
            reward = self.calculate_reward(makespan)
        else:
            # Intermediate reward based on immediate impact of decision
            reward = self.calculate_immediate_reward(task, processor_id)
            
        # Create next state representation
        next_state = self.create_state_representation()
        
        return next_state, reward, done
    
    def schedule_task(self, task, processor_id):
        """Schedule a single task on a processor with security overhead calculation."""
        processor = self.processors[processor_id - 1]
        
        # Calculate earliest start time based on predecessors
        est = 0
        predecessor_overhead = 0  # Security overhead from incoming messages
        
        for pred_id in task.predecessors:
            pred_task = self.get_task_by_id(pred_id)
            if not pred_task or not pred_task.is_scheduled:
                continue
                
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
                overhead_factor = self.security.overheads[service][protocol_idx][processor_id - 1]
                overhead = message.size / overhead_factor
                message_security_overhead += overhead
                
            # Add authentication verification overhead
            auth_idx = message.assigned_security['authentication'] + 1
            auth_overhead = self.security.overheads['authentication'][auth_idx][processor_id - 1]
            message_security_overhead += auth_overhead
            
            # Update predecessor overhead
            predecessor_overhead += message_security_overhead
            
            # Update EST based on predecessor finish plus communication time
            pred_finish_with_comm = pred_task.finish_time + comm_time
            est = max(est, pred_finish_with_comm)
        
        # Account for processor availability
        est = max(est, processor.available_time)
        
        # Base execution time without security overhead
        base_exec_time = task.execution_times[processor_id - 1]
        
        # Calculate outgoing security overhead (sender side)
        successor_overhead = 0
        for succ_id in task.successors:
            message = self.get_message(task.task_id, succ_id)
            if not message:
                continue
                
            # Calculate overhead for each security service (sender side)
            for service in ['confidentiality', 'integrity']:
                protocol_idx = message.assigned_security[service] + 1  # 1-indexed
                overhead_factor = self.security.overheads[service][protocol_idx][processor_id - 1]
                overhead = message.size / overhead_factor
                successor_overhead += overhead
                
            # Include authentication generation overhead
            auth_idx = message.assigned_security['authentication'] + 1
            auth_overhead = self.security.overheads['authentication'][auth_idx][processor_id - 1]
            successor_overhead += auth_overhead
        
        # Total security overhead for this task on this processor
        total_security_overhead = predecessor_overhead + successor_overhead
        
        # Update task attributes
        task.assigned_processor = processor_id
        task.start_time = est
        task.finish_time = est + base_exec_time + total_security_overhead
        task.is_scheduled = True
        
        # Update processor availability time
        processor.available_time = task.finish_time
        
        # Add to schedule
        self.schedule.append({
            'task_id': task.task_id,
            'name': task.name,
            'processor': task.assigned_processor,
            'start_time': task.start_time,
            'finish_time': task.finish_time,
            'security_overhead': total_security_overhead
        })
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
    def calculate_reward(self, makespan):
        """Calculate final reward based on makespan and deadline satisfaction."""
        if makespan > self.deadline:
            # Penalize schedules that miss the deadline
            return -100
        else:
            # Reward is inversely proportional to makespan, encouraging shorter schedules
            # Normalized by deadline to keep rewards in reasonable range
            slack_time = self.deadline - makespan
            normalized_slack = slack_time / self.deadline
            return 50 + 50 * normalized_slack
    
    def calculate_immediate_reward(self, task, processor_id):
        """Calculate intermediate reward for a single task assignment."""
        # Base reward for successfully scheduling a task
        reward = 1
        
        # Get the task's finish time
        finish_time = task.finish_time
        
        # Calculate local slack for this task
        # Local slack is the time between this task's finish and its successors' earliest start
        local_slack = float('inf')
        has_successors = False
        
        for succ_id in task.successors:
            succ_task = self.get_task_by_id(succ_id)
            if not succ_task:
                continue
                
            has_successors = True
            
            # Calculate earliest possible start for successor
            earliest_start = finish_time
            for pred_id in succ_task.predecessors:
                pred_task = self.get_task_by_id(pred_id)
                if pred_task and pred_task.is_scheduled and pred_task.task_id != task.task_id:
                    earliest_start = max(earliest_start, pred_task.finish_time)
            
            task_slack = earliest_start - finish_time
            local_slack = min(local_slack, task_slack)
        
        if not has_successors:
            # End tasks should finish as early as possible
            local_slack = self.deadline - finish_time
        
        # Add reward component based on local slack
        # Normalize by deadline to keep values reasonable
        normalized_slack = local_slack / self.deadline if local_slack != float('inf') else 0
        reward += 2 * normalized_slack
        
        # Penalize for using slower processors when unnecessary
        processor_efficiency = task.execution_times[processor_id - 1] / min(task.execution_times)
        if processor_efficiency > 1.5:  # If more than 50% slower than optimal
            reward -= 0.5
        
        return reward
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
    def learn(self):
        """Update value parameters using a batch of experiences."""
        # Sample experiences from memory
        experiences = self.memory.sample(self.batch_size)
        if experiences is None:
            return
            
        states, actions, rewards, next_states, dones, indices, weights = experiences
        
        # Convert weights to tensor with matching device
        weights = torch.from_numpy(weights).float().to(next_states.device)
        
        # Get max predicted Q values for next states from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Calculate TD errors for updating priorities
        td_errors = torch.abs(Q_targets - Q_expected).cpu().data.numpy()
        self.memory.update_priorities(indices, td_errors)
        
        # Compute loss with importance sampling weights
        loss = (weights * F.mse_loss(Q_expected, Q_targets, reduction='none')).mean()
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def soft_update(self):
        """Soft update target network parameters."""
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def save_schedule_state(self):
        """Save the current schedule state for later restoration."""
        schedule_state = []
        for entry in self.schedule:
            schedule_state.append(entry.copy())
        return schedule_state
    
    def apply_best_schedule(self):
        """Apply the best schedule found during training."""
        if not self.best_schedule:
            print("No valid schedule found!")
            return
            
        print("\nApplying best schedule found during training...")
        
        # Reset task and processor states
        for task in self.tasks:
            task.is_scheduled = False
            task.assigned_processor = None
            task.start_time = None
            task.finish_time = None
            
        for proc in self.processors:
            proc.available_time = 0
            
        # Apply the best schedule
        self.schedule = self.best_schedule.copy()
        
        # Set task attributes based on schedule
        for entry in self.schedule:
            task = self.get_task_by_id(entry['task_id'])
            if task:
                task.assigned_processor = entry['processor']
                task.start_time = entry['start_time']
                task.finish_time = entry['finish_time']
                task.is_scheduled = True
                
                # Update processor availability
                proc_id = task.assigned_processor
                proc = self.processors[proc_id - 1]
                proc.available_time = max(proc.available_time, task.finish_time)
        
        # Calculate makespan
        self.makespan = max(task.finish_time for task in self.tasks if task.is_scheduled)
        
        print(f"Best schedule applied. Makespan: {self.makespan:.2f}")
    
    def calculate_security_utility_for_schedule(self, schedule):
        """Calculate security utility for a specific schedule."""
        if not schedule:
            return 0
            
        # Temporarily save current state
        current_schedule = self.schedule
        
        # Apply the given schedule
        self.schedule = schedule
        
        # Calculate security utility
        utility = self.calculate_security_utility()
        
        # Restore original state
        self.schedule = current_schedule
        
        return utility
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
    
    
    # Phase 3: Reuse the security enhancement methods from SHIELD
    def enhance_security(self):
        """Phase 3 of QL-SHIELD: Enhance security levels using available slack time."""
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
    security_service = SecurityService()
    deadline = 1600  # As specified in the paper
    
    return tasks, messages, processors,  security_service, deadline

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
class RFSHIELD(SHIELD):
    """
    RF-SHIELD: Random Forest enhanced Security-aware scHedulIng with sEcurity utiLity maximization unDer real-time constraints
    
    This algorithm extends SHIELD by using a Random Forest model to predict which security upgrades
    are likely to succeed, saving time by avoiding costly trial-and-error scheduling recalculations.
    """
    
    def __init__(self, tasks, messages, processors, security_service, deadline):
        super().__init__(tasks, messages, processors, security_service, deadline)
        self.rf_classifier = None
        self.scaler = StandardScaler()
        self.training_data = []
        self.training_labels = []
        self.upgrade_trials = 0
        self.upgrade_successes = 0
        self.prediction_time = 0
        self.trial_time = 0
        self.use_rf_predictions = False  # Will be enabled after training
        self.utility_weight = 0.8  # Added: Higher weight for security utility (vs. makespan)
        self.min_training_samples = 30  # Added: Increase minimum training samples
    
    def run(self):
        """Execute the RF-SHIELD algorithm in phases."""
        import time
        start_time = time.time()
        
        # Phase 1 & 2: Run HSMS first to get initial schedule (from SHIELD parent)
        makespan, security_utility = super(SHIELD, self).run()
        
        if makespan is None or makespan > self.deadline:
            print("RF-SHIELD cannot proceed: HSMS failed to meet deadline requirements")
            return None, None
        
        self.original_makespan = makespan
        self.original_security_utility = security_utility  # Added: Track original security utility
        print(f"\nRF-SHIELD Phase 1-2 complete: Initial makespan={makespan}, Security utility={security_utility}")
        
        # Phase 3: Training Data Collection - Try more upgrades and collect results for training
        print("\nRF-SHIELD Phase 3: Collecting training data...")
        self.collect_training_data()
        
        # Phase 4: Train Random Forest model
        print("\nRF-SHIELD Phase 4: Training Random Forest model...")
        self.train_rf_model()
        
        # Phase 5: Security Enhancement using RF predictions
        print("\nRF-SHIELD Phase 5: Starting ML-guided Security Enhancement...")
        self.use_rf_predictions = True
        self.enhance_security_with_rf()
        
        # Calculate final results
        final_makespan = self.finalize_schedule()
        final_security_utility = self.calculate_security_utility()
        
        # If security utility decreased, revert to the original configuration
        # Added: This safety check ensures we never produce worse security
        if final_security_utility < self.original_security_utility:
            print("WARNING: Security utility decreased. Reverting to original configuration.")
            # Revert to original schedule
            self.reset_security_levels()
            self.reapply_original_schedule()
            final_makespan = self.original_makespan
            final_security_utility = self.original_security_utility
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"RF-SHIELD complete: Final makespan={final_makespan}, Final security utility={final_security_utility}")
        print(f"Security utility improvement: {(final_security_utility - security_utility) / security_utility * 100:.2f}%")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"ML Prediction statistics: {self.upgrade_successes} successful upgrades out of {self.upgrade_trials} trials")
        
        return final_makespan, final_security_utility
    
    def collect_training_data(self):
        """Collect training data for the Random Forest model by trying a subset of upgrades."""
        # Use a larger sample of potential upgrades for training
        self.initialize_bcr_heap()
        original_heap = self.bcr_heap.copy()
        
        # Get total number of potential upgrades
        total_upgrades = len(original_heap)
        # Changed: Use at least 50% of possible upgrades or 100, whichever is greater
        training_samples = max(min(total_upgrades, 100), total_upgrades // 2)
        
        print(f"Collecting data on {training_samples} upgrade attempts for training...")
        
        # Create a more diverse training set by stratified sampling
        # Select upgrades across different services, levels, and message sizes
        selected_upgrades = self.stratified_sample_upgrades(original_heap, training_samples)
        
        for bcr, (message_id, service, current_level) in selected_upgrades:
            bcr = -bcr  # Negate back to positive value (heap is min-heap)
            
            # Find the message
            message = None
            for msg in self.messages:
                if msg.id == message_id:
                    message = msg
                    break
            
            if not message:
                continue
                
            # Create feature vector for this potential upgrade
            next_level = current_level + 1
            features = self.extract_features(message, service, current_level, next_level)
            
            # Try the upgrade to see if it works
            success = self.try_security_upgrade(message, service, next_level)
            
            # Store the result
            self.training_data.append(features)
            self.training_labels.append(1 if success else 0)
            
            if success:
                print(f"✅ Training sample: Successfully upgraded {message_id} {service} to level {next_level}")
            else:
                print(f"❌ Training sample: Failed to upgrade {message_id} {service} to level {next_level}")
        
        # Reset the BCR heap to its original state
        self.bcr_heap = original_heap
        
        print(f"Collected {len(self.training_data)} training samples: {sum(self.training_labels)} successful, {len(self.training_labels) - sum(self.training_labels)} failed")
    def stratified_sample_upgrades(self, upgrade_heap, n_samples):
        """
        Take a stratified sample from the upgrade heap to ensure diversity in training data.
        This helps the model learn patterns across different security services and levels.
        """
        # Group upgrades by service and level
        service_groups = {}
        for item in upgrade_heap:
            bcr, (message_id, service, current_level) = item
            key = (service, current_level)
            if key not in service_groups:
                service_groups[key] = []
            service_groups[key].append(item)
        
        # Calculate how many samples to take from each group
        n_groups = len(service_groups)
        samples_per_group = n_samples // n_groups
        
        # Ensure at least one sample per group
        samples_per_group = max(1, samples_per_group)
        
        # Take samples from each group
        selected_samples = []
        for group_key, group_items in service_groups.items():
            # Randomly select from each group
            import random
            # Take either all items in the group or the calculated number, whichever is smaller
            n_to_select = min(len(group_items), samples_per_group)
            selected = random.sample(group_items, n_to_select)
            selected_samples.extend(selected)
        
        # If we still need more samples, take randomly from the remaining items
        if len(selected_samples) < n_samples:
            remaining = [item for item in upgrade_heap if item not in selected_samples]
            n_remaining = n_samples - len(selected_samples)
            if remaining and n_remaining > 0:
                n_to_select = min(len(remaining), n_remaining)
                import random
                selected_samples.extend(random.sample(remaining, n_to_select))
        
        return selected_samples
    def train_rf_model(self):
        """Train a Random Forest model on collected data."""
        if not self.training_data or len(self.training_data) < self.min_training_samples:
            print(f"Insufficient training data available (need {self.min_training_samples}, have {len(self.training_data)}). Will fall back to standard SHIELD approach.")
            return False
        
        # Convert lists to numpy arrays
        X = np.array(self.training_data)
        y = np.array(self.training_labels)
        
        # Check class balance
        positive_ratio = np.mean(y)
        print(f"Class balance: {positive_ratio:.2f} positive examples ({sum(y)} out of {len(y)})")
        
        # If severe class imbalance, use class weights
        class_weights = None
        if positive_ratio < 0.2 or positive_ratio > 0.8:
            # Adjust weights to balance classes
            class_weights = {
                0: 1 / (2 * (1 - positive_ratio)) if positive_ratio < 1 else 1,
                1: 1 / (2 * positive_ratio) if positive_ratio > 0 else 1
            }
            print(f"Using class weights due to imbalance: {class_weights}")
        
        # Scale the features for better performance
        X_scaled = self.scaler.fit_transform(X)
        
        # Train the Random Forest classifier with improved hyperparameters
        self.rf_classifier = RandomForestClassifier(
            n_estimators=200,  # More trees
            max_depth=10,      # Deeper trees
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight=class_weights,
            random_state=42
        )
        
        self.rf_classifier.fit(X_scaled, y)
        
        # Calculate feature importances
        importances = self.rf_classifier.feature_importances_
        feature_names = [
            'message_size', 'service_weight', 'current_level', 'next_level',
            'utility_gain', 'overhead_increase', 'task_slack', 'global_slack',
            'deadline', 'avg_bandwidth', 'source_priority', 'dest_priority'
        ]
        
        # Print feature importances
        print("\nFeature importances:")
        for feature, importance in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
            print(f"  {feature}: {importance:.4f}")
        
        # Get training accuracy
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        y_pred = self.rf_classifier.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        
        print(f"Training metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        return True
    def calculate_bcr(self, message, service, current_level):
        """
        Calculate Benefit-Cost Ratio with higher emphasis on security utility.
        Modified to prioritize security utility gains over makespan implications.
        """
        next_level = current_level + 1
        if next_level >= len(self.security.strengths[service]):
            return 0  # Already at max level
        
        # Calculate benefit (security utility gain)
        service_weight = message.weights[service]
        current_strength = self.security.strengths[service][current_level]
        next_strength = self.security.strengths[service][next_level]
        utility_gain = service_weight * (next_strength - current_strength)
        
        # Calculate cost (additional overhead)
        source_task = self.get_task_by_id(message.source_id)
        dest_task = self.get_task_by_id(message.dest_id)
        source_proc = source_task.assigned_processor
        dest_proc = dest_task.assigned_processor
        
        current_overhead = self.calculate_service_overhead(message, service, current_level, source_proc, dest_proc)
        next_overhead = self.calculate_service_overhead(message, service, next_level, source_proc, dest_proc)
        overhead_increase = next_overhead - current_overhead
        
        # Avoid division by zero
        if overhead_increase <= 0:
            return float('inf') if utility_gain > 0 else 0
        
        # Calculate BCR with higher weight for utility
        # Changed: Emphasize utility gain more than overhead cost
        bcr = (utility_gain * self.utility_weight) / (overhead_increase * (1 - self.utility_weight))
        
        return bcr
    def calculate_security_utility_gain(self, message, service, current_level, next_level):
        """Calculate the security utility gain from upgrading a service."""
        service_weight = message.weights[service]
        current_strength = self.security.strengths[service][current_level]
        next_strength = self.security.strengths[service][next_level]
        return service_weight * (next_strength - current_strength)
    def save_current_security_configuration(self):
        """Save the current security configuration."""
        configuration = {}
        for message in self.messages:
            configuration[message.id] = {
                'confidentiality': message.assigned_security['confidentiality'],
                'integrity': message.assigned_security['integrity'],
                'authentication': message.assigned_security['authentication']
            }
        return configuration
    def restore_security_configuration(self, configuration):
        """Restore a saved security configuration."""
        for message in self.messages:
            if message.id in configuration:
                message.assigned_security['confidentiality'] = configuration[message.id]['confidentiality']
                message.assigned_security['integrity'] = configuration[message.id]['integrity']
                message.assigned_security['authentication'] = configuration[message.id]['authentication']
    
    def reset_security_levels(self):
        """Reset all security levels to minimum required levels."""
        for message in self.messages:
            message.assigned_security['confidentiality'] = message.min_security_levels['confidentiality']
            message.assigned_security['integrity'] = message.min_security_levels['integrity']
            message.assigned_security['authentication'] = message.min_security_levels['authentication']
    
    def reapply_original_schedule(self):
        """Reapply the original schedule."""
        # Reset all task scheduling information
        for task in self.tasks:
            task.is_scheduled = False
            task.start_time = None
            task.finish_time = None
            task.assigned_processor = None
        
        # Re-run the basic scheduling algorithm
        super(SHIELD, self).run()
    def enhance_security_with_rf(self):
        """Phase 5 of RF-SHIELD: Enhance security levels using RF predictions."""
        # Initialize BCR heap with all possible upgrades
        if not self.bcr_heap:
            self.initialize_bcr_heap()
        
        upgrade_count = 0
        prediction_count = 0
        true_positive = 0
        false_positive = 0
        

        slack_available = self.deadline - self.original_makespan
        
        print(f"Original makespan: {self.original_makespan}, Available slack: {slack_available}")
        
        # Keep track of best security utility achieved
        best_security_utility = self.calculate_security_utility()
        best_configuration = self.save_current_security_configuration()
        
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
            
            # Extract features for this potential upgrade
            features = self.extract_features(message, service, current_level, next_level)
            
            # Predict success using the trained model if available
            will_succeed = True  # Default to trying if no model
            prediction_confidence = 1.0
            prediction_time_start = time.time()
            
            if self.rf_classifier is not None and self.use_rf_predictions:
                # Scale features using the same scaler used during training
                features_scaled = self.scaler.transform([features])
                
                # Get prediction probabilities
                pred_proba = self.rf_classifier.predict_proba(features_scaled)[0]
                success_probability = pred_proba[1] if len(pred_proba) > 1 else 0.5
                
                # Only skip if the model is very confident it will fail
                # Changed: More conservative threshold for skipping
                will_succeed = success_probability >= 0.3
                prediction_confidence = success_probability
                prediction_count += 1
            
            prediction_time_end = time.time()
            self.prediction_time += prediction_time_end - prediction_time_start
            
            # Skip if RF predicts failure with high confidence
            if not will_succeed:
                print(f"⏩ Skipping upgrade {message_id} {service} from {current_level} to {next_level} - ML predicts failure (confidence: {prediction_confidence:.2f})")
                
                # ADDED: Double-check high utility gains that might be worth trying anyway
                # This helps prevent false negatives from causing us to miss valuable upgrades
                service_weight = message.weights[service]
                utility_gain = self.calculate_security_utility_gain(message, service, current_level, next_level)
                
                # If the potential gain is very high, try it anyway
                if utility_gain > 0.1 * self.original_security_utility:
                    print(f"🔍 High utility gain detected! Trying anyway despite prediction.")
                    will_succeed = True
                else:
                    continue
            
            print(f"\nTrying to upgrade {message_id} {service} from level {current_level} to {next_level} (BCR: {bcr:.4f}) - ML confidence: {prediction_confidence:.2f}")
            
            # Try to apply the upgrade
            trial_time_start = time.time()
            self.upgrade_trials += 1
            success = self.try_security_upgrade(message, service, next_level)
            trial_time_end = time.time()
            self.trial_time += trial_time_end - trial_time_start
            
            if success:
                self.upgrade_successes += 1
                true_positive += 1
                upgrade_count += 1
                
                # Record this upgrade
                self.upgrade_history.append({
                    'message_id': message_id,
                    'service': service,
                    'from_level': current_level,
                    'to_level': next_level,
                    'bcr': bcr
                })
                
                # Check if this improved the security utility
                current_security_utility = self.calculate_security_utility()
                if current_security_utility > best_security_utility:
                    best_security_utility = current_security_utility
                    best_configuration = self.save_current_security_configuration()
                    print(f"📈 New best security utility: {best_security_utility:.4f}")
                
                # If we can upgrade further, calculate new BCR and add back to heap
                if next_level < max_level:
                    new_bcr = self.calculate_bcr(message, service, next_level)
                    if new_bcr > 0:
                        heapq.heappush(self.bcr_heap, (-new_bcr, (message_id, service, next_level)))
                
                print(f"✅ Upgrade successful! New {service} level: {next_level}")
            else:
                false_positive += 1
                print(f"❌ Upgrade rejected: insufficient slack (ML prediction was incorrect)")
        
        # Restore the best configuration found
        print(f"\nRestoring best security configuration with utility {best_security_utility:.4f}")
        self.restore_security_configuration(best_configuration)
        
        # Print ML performance metrics
        if prediction_count > 0:
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            print(f"\nML Performance - Precision: {precision:.4f} ({true_positive} true positives, {false_positive} false positives)")
            print(f"Total time saved: {self.prediction_time:.2f} seconds in predictions vs {self.trial_time:.2f} seconds in trials")
        
        print(f"\nSecurity enhancement complete: {upgrade_count} upgrades applied")
        
        
    def extract_features(self, message, service, current_level, next_level):
        """Extract features for the Random Forest model with improved feature engineering."""
        source_task = self.get_task_by_id(message.source_id)
        dest_task = self.get_task_by_id(message.dest_id)
        
        # Calculate utility gain
        service_weight = message.weights[service]
        current_strength = self.security.strengths[service][current_level]
        next_strength = self.security.strengths[service][next_level]
        utility_gain = service_weight * (next_strength - current_strength)
        
        # Calculate additional overhead
        source_proc = source_task.assigned_processor
        dest_proc = dest_task.assigned_processor
        
        current_overhead = self.calculate_service_overhead(message, service, current_level, source_proc, dest_proc)
        next_overhead = self.calculate_service_overhead(message, service, next_level, source_proc, dest_proc)
        overhead_increase = next_overhead - current_overhead
        
        # Calculate task slack - time between this task finishes and its successors start
        task_slack = float('inf')
        for succ_id in source_task.successors:
            succ_task = self.get_task_by_id(succ_id)
            if succ_task and succ_task.is_scheduled:
                slack = succ_task.start_time - source_task.finish_time
                task_slack = min(task_slack, slack)
        
        if task_slack == float('inf'):
            task_slack = self.deadline - source_task.finish_time
        
        # Global slack is the difference between deadline and makespan
        global_slack = self.deadline - max(task.finish_time for task in self.tasks if task.is_scheduled)
        
        # Calculate average bandwidth between source and destination processors
        avg_bandwidth = 0
        if source_proc != dest_proc:
            source_processor = self.processors[source_proc - 1]
            
            # Assuming there's a bandwidth attribute or method in the Processor class
            if hasattr(source_processor, 'bandwidth'):
                avg_bandwidth = source_processor.bandwidth[dest_proc - 1]
            else:
                # Fallback if bandwidth isn't directly available
                avg_bandwidth = 1000  # Default value
        
        # ADDED: Calculate normalized utility gain (relative to current total utility)
        current_total_utility = self.calculate_security_utility()
        normalized_utility_gain = utility_gain / max(0.01, current_total_utility)
        
        # ADDED: Calculate deadline margin ratio
        deadline_margin_ratio = global_slack / self.deadline
        
        # Feature vector
        features = [
            message.size,                      # Message size
            service_weight,                    # Weight of security service 
            current_level,                     # Current security level
            next_level,                        # Next security level
            utility_gain,                      # Utility gain from upgrade
            overhead_increase,                 # Increase in overhead
            task_slack,                        # Available slack time
            global_slack,                      # Global slack until deadline
            self.deadline,                     # Deadline
            avg_bandwidth,                     # Average bandwidth between processors
            source_task.priority,              # Source task priority
            dest_task.priority if dest_task else 0  # Destination task priority
        ]
        
        return features
    def print_rf_analysis(self):
        """Print analysis of the Random Forest model performance."""
        if self.rf_classifier is None:
            print("No Random Forest model was trained.")
            return
            
        print("\n" + "="*120)
        print("RANDOM FOREST MODEL ANALYSIS")
        print("="*120)
        
        # Feature importances
        feature_names = [
            'message_size', 'service_weight', 'current_level', 'next_level',
            'utility_gain', 'overhead_increase', 'task_slack', 'global_slack',
            'deadline', 'avg_bandwidth', 'source_priority', 'dest_priority'
        ]
        
        importances = self.rf_classifier.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("Feature ranking:")
        for f in range(len(feature_names)):
            print(f"{f+1}. {feature_names[indices[f]]} ({importances[indices[f]]:.4f})")
        
        # Performance statistics
        print("\nPerformance Statistics:")
        print(f"Total upgrade trials: {self.upgrade_trials}")
        print(f"Successful upgrades: {self.upgrade_successes}")
        print(f"Success rate: {self.upgrade_successes / self.upgrade_trials * 100:.2f}% if trials > 0 else 'N/A'")
        
        # Time savings analysis
        if self.prediction_time > 0 and self.trial_time > 0:
            avg_prediction_time = self.prediction_time / max(1, self.upgrade_trials)
            avg_trial_time = self.trial_time / max(1, self.upgrade_trials)
            time_saving_ratio = avg_trial_time / avg_prediction_time if avg_prediction_time > 0 else 0
            
            print("\nTime Efficiency Analysis:")
            print(f"Average prediction time: {avg_prediction_time * 1000:.2f} ms")
            print(f"Average trial time: {avg_trial_time * 1000:.2f} ms")
            print(f"Speed improvement: {time_saving_ratio:.1f}x faster with ML predictions")
            
            # Estimated time savings
            total_potential_trials = len(self.training_data) + self.upgrade_trials
            actual_trials = self.upgrade_trials
            estimated_trials_saved = total_potential_trials - actual_trials
            estimated_time_saved = estimated_trials_saved * avg_trial_time
            
            print(f"Estimated time saved: {estimated_time_saved:.2f} seconds by avoiding {estimated_trials_saved} unnecessary trials")
        
        print("="*120)
        
class ImprovedRFSHIELD(RFSHIELD):
    """
    Enhanced version of RF-SHIELD with improved ML-based decision making
    and better training data collection strategy.
    """
    
    def __init__(self, tasks, messages, processors, security_service, deadline):
        super().__init__(tasks, messages, processors, security_service, deadline)
        self.adaptive_utility_weight = 0.7  # Starting weight for security utility
        self.confidence_thresholds = {
            'confidentiality': 0.5, 
            'integrity': 0.5, 
            'authentication': 0.5
        }
        self.service_success_rates = {
            'confidentiality': [], 
            'integrity': [], 
            'authentication': []
        }
        # Track rollbacks for analysis
        self.rollbacks = 0
        # Track upgrades by service
        self.upgrades_by_service = {
            'confidentiality': 0,
            'integrity': 0,
            'authentication': 0
        }
        # Critical path tracking
        self.critical_path = []
        self.critical_path_tasks = set()
        self.identify_critical_path()
    
    def identify_critical_path(self):
        """Identify the critical path in the task graph"""
        # Initialize earliest finish times for all tasks
        earliest_finish = {}
        
        # Topologically sort tasks
        task_order = self.topological_sort()
        
        # Forward pass to calculate earliest finish times
        for task_id in task_order:
            task = self.get_task_by_id(task_id)
            
            # Get the earliest start time (max of predecessors' finish times)
            earliest_start = 0
            for pred_id in self.get_predecessors(task_id):
                if pred_id in earliest_finish:
                    earliest_start = max(earliest_start, earliest_finish[pred_id])
            
            # Calculate earliest finish time
            earliest_finish[task_id] = earliest_start + self.get_task_execution_time(task)
        
        # Find the sink task(s) with the latest finish time
        sink_tasks = [t.task_id for t in self.tasks if not t.successors]
        critical_sink = max(sink_tasks, key=lambda t: earliest_finish[t])
        
        # Backtrack to find the critical path
        current = critical_sink
        self.critical_path = [current]
        self.critical_path_tasks.add(current)
        
        while True:
            predecessors = self.get_predecessors(current)
            if not predecessors:
                break
                
            # Find the critical predecessor
            critical_pred = max(predecessors, 
                               key=lambda p: earliest_finish[p])
            
            self.critical_path.insert(0, critical_pred)
            self.critical_path_tasks.add(critical_pred)
            current = critical_pred
        
        print(f"Identified critical path: {self.critical_path}")
        return self.critical_path
    
    def topological_sort(self):
        """Perform topological sort of tasks"""
        # Count incoming edges for each task
        in_degree = {task.task_id: 0 for task in self.tasks}
        for task in self.tasks:
            for succ_id in task.successors:
                in_degree[succ_id] = in_degree.get(succ_id, 0) + 1
        
        # Start with nodes that have no incoming edges
        queue = [task.task_id for task in self.tasks if in_degree[task.task_id] == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # Get the current task
            task = self.get_task_by_id(current)
            
            # Reduce in-degree of all successors
            for succ_id in task.successors:
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    queue.append(succ_id)
        
        return result
    
    def get_predecessors(self, task_id):
        """Get predecessors of a task"""
        predecessors = []
        for task in self.tasks:
            if task_id in task.successors:
                predecessors.append(task.task_id)
        return predecessors
    
    def get_task_execution_time(self, task):
        """Get execution time of a task"""
        # Use the first processor's execution time as an estimate
        return task.execution_times[0]
    
    def collect_training_data(self):
        """Enhanced training data collection strategy"""
        # Use a larger sample of potential upgrades for training
        self.initialize_bcr_heap()
        original_heap = self.bcr_heap.copy()
        
        # Get total number of potential upgrades
        total_upgrades = len(original_heap)
        # Use at least min_training_samples or 30% of total upgrades, whichever is greater
        training_samples = max(20, int(total_upgrades * 0.3))
        training_samples = min(training_samples, total_upgrades)
        
        print(f"Collecting data on {training_samples} upgrade attempts for training...")
        
        # Create a more diverse training set through intelligent sampling
        selected_upgrades = self.intelligent_sample_upgrades(original_heap, training_samples)
        
        successful_samples = 0
        failed_samples = 0
        
        for bcr, (message_id, service, current_level) in selected_upgrades:
            bcr = -bcr  # Negate back to positive value (heap is min-heap)
            
            # Find the message
            message = None
            for msg in self.messages:
                if msg.id == message_id:
                    message = msg
                    break
            
            if not message:
                continue
                
            # Create feature vector for this potential upgrade
            next_level = current_level + 1
            features = self.extract_enhanced_features(message, service, current_level, next_level)
            
            # Try the upgrade to see if it works
            success = self.try_security_upgrade(message, service, next_level)
            
            # Store the result
            self.training_data.append(features)
            self.training_labels.append(1 if success else 0)
            
            # Update service-specific success rates
            self.service_success_rates[service].append(1 if success else 0)
            
            if success:
                successful_samples += 1
                print(f"✅ Training sample: Successfully upgraded {message_id} {service} to level {next_level}")
            else:
                failed_samples += 1
                print(f"❌ Training sample: Failed to upgrade {message_id} {service} to level {next_level}")
        
        # Reset the BCR heap to its original state
        self.bcr_heap = original_heap
        
        # Adjust confidence thresholds based on observed success rates
        self._update_confidence_thresholds()
        
        print(f"Collected {len(self.training_data)} training samples: {successful_samples} successful, {failed_samples} failed")
        
        # If we have too few successful samples, add synthetic positive examples
        if successful_samples < 5 and successful_samples > 0:
            print("Adding synthetic positive examples to balance training data...")
            self._add_synthetic_samples(successful_samples)
    
    def _add_synthetic_samples(self, successful_samples):
        """Add synthetic samples to improve class balance"""
        # Find indices of successful upgrades
        success_indices = [i for i, label in enumerate(self.training_labels) if label == 1]
        
        if not success_indices:
            return
            
        # Calculate how many synthetic samples to add
        n_synthetic = min(5, 10 - successful_samples)
        
        import numpy as np
        from sklearn.neighbors import KernelDensity
        
        # Get features of successful samples
        X_success = np.array([self.training_data[i] for i in success_indices])
        
        # Fit KDE to successful samples
        kde = KernelDensity(bandwidth=0.5)
        kde.fit(X_success)
        
        # Generate synthetic samples
        for _ in range(n_synthetic):
            # Sample from KDE and add small noise
            x_new = kde.sample(1)[0] + np.random.normal(0, 0.1, size=X_success.shape[1])
            self.training_data.append(x_new.tolist())
            self.training_labels.append(1)
    
    def intelligent_sample_upgrades(self, upgrade_heap, n_samples):
        """
        Take a carefully balanced sample of upgrades to maximize learning potential:
        - Ensures coverage of all services and levels
        - Prioritizes upgrades on critical path tasks
        - Includes a mix of high and low BCR upgrades
        """
        # Group upgrades by service, level, and whether they're on the critical path
        upgrade_groups = {}
        critical_path_upgrades = []
        high_bcr_upgrades = []
        low_bcr_upgrades = []
        
        # Sort upgrades by BCR
        sorted_upgrades = sorted(upgrade_heap, key=lambda x: -x[0])
        
        # Mark top 25% as high BCR and bottom 25% as low BCR
        quarter = len(sorted_upgrades) // 4
        high_bcr_threshold = sorted_upgrades[quarter][0] if quarter > 0 else float('-inf')
        low_bcr_threshold = sorted_upgrades[-quarter][0] if quarter > 0 else float('inf')
        
        for item in upgrade_heap:
            bcr, (message_id, service, current_level) = item
            
            # Find message and related tasks
            message = None
            for msg in self.messages:
                if msg.id == message_id:
                    message = msg
                    break
            
            if not message:
                continue
                
            # Check if message connects critical path tasks
            source_task = self.get_task_by_id(message.source_id)
            dest_task = self.get_task_by_id(message.dest_id)
            on_critical_path = (message.source_id in self.critical_path_tasks and 
                               message.dest_id in self.critical_path_tasks)
            
            # Categorize by service, level
            key = (service, current_level)
            if key not in upgrade_groups:
                upgrade_groups[key] = []
            upgrade_groups[key].append(item)
            
            # Add to special categories
            if on_critical_path:
                critical_path_upgrades.append(item)
            
            if bcr >= high_bcr_threshold:
                high_bcr_upgrades.append(item)
            elif bcr <= low_bcr_threshold:
                low_bcr_upgrades.append(item)
        
        # Algorithm for selecting samples:
        # 1. 40% from critical path (or as many as available)
        # 2. 30% from high BCR
        # 3. 10% from low BCR
        # 4. 20% evenly distributed across service/level groups
        
        critical_path_quota = int(n_samples * 0.4)
        high_bcr_quota = int(n_samples * 0.3)
        low_bcr_quota = int(n_samples * 0.1)
        group_quota = n_samples - (critical_path_quota + high_bcr_quota + low_bcr_quota)
        
        selected_samples = []
        
        # Select critical path samples
        import random
        if critical_path_upgrades:
            to_select = min(len(critical_path_upgrades), critical_path_quota)
            selected_samples.extend(random.sample(critical_path_upgrades, to_select))
        
        # High BCR samples
        if high_bcr_upgrades:
            remaining_high_bcr = [x for x in high_bcr_upgrades if x not in selected_samples]
            to_select = min(len(remaining_high_bcr), high_bcr_quota)
            if to_select > 0:
                selected_samples.extend(random.sample(remaining_high_bcr, to_select))
        
        # Low BCR samples
        if low_bcr_upgrades:
            remaining_low_bcr = [x for x in low_bcr_upgrades if x not in selected_samples]
            to_select = min(len(remaining_low_bcr), low_bcr_quota)
            if to_select > 0:
                selected_samples.extend(random.sample(remaining_low_bcr, to_select))
        
        # Distribute remaining quota across groups
        if upgrade_groups and group_quota > 0:
            samples_per_group = max(1, group_quota // len(upgrade_groups))
            
            for group_key, group_items in upgrade_groups.items():
                remaining_in_group = [x for x in group_items if x not in selected_samples]
                to_select = min(len(remaining_in_group), samples_per_group)
                if to_select > 0:
                    selected_samples.extend(random.sample(remaining_in_group, to_select))
                
                # Stop if we've reached our quota
                if len(selected_samples) >= n_samples:
                    break
        
        # If we still need more samples, take randomly from the remaining
        if len(selected_samples) < n_samples:
            remaining = [x for x in upgrade_heap if x not in selected_samples]
            remaining_needed = n_samples - len(selected_samples)
            if remaining and remaining_needed > 0:
                to_select = min(len(remaining), remaining_needed)
                selected_samples.extend(random.sample(remaining, to_select))
        
        return selected_samples[:n_samples]  # Ensure we don't exceed n_samples
    
    def _update_confidence_thresholds(self):
        """Update confidence thresholds based on observed success rates"""
        for service, results in self.service_success_rates.items():
            if results:
                success_rate = sum(results) / len(results)
                # Adjust threshold based on success rate
                if success_rate > 0.7:
                    # High success rate: be more permissive
                    self.confidence_thresholds[service] = 0.3
                elif success_rate > 0.5:
                    # Medium success rate: be moderate
                    self.confidence_thresholds[service] = 0.4
                else:
                    # Low success rate: be more conservative
                    self.confidence_thresholds[service] = 0.6
                
                print(f"Service {service} success rate: {success_rate:.2f}, " +
                     f"adjusted threshold to {self.confidence_thresholds[service]:.2f}")
    
    def train_rf_model(self):
        """Enhanced training for the Random Forest model"""
        if not self.training_data or len(self.training_data) < 20:  # Keep min at 20
            print(f"Insufficient training data available (need 20, have {len(self.training_data)}). Will fall back to standard SHIELD approach.")
            return False
        
        # Convert lists to numpy arrays
        X = np.array(self.training_data)
        y = np.array(self.training_labels)
        
        # Check class balance
        positive_ratio = np.mean(y)
        print(f"Class balance: {positive_ratio:.2f} positive examples ({sum(y)} out of {len(y)})")
        
        # If severe class imbalance, use class weights and SMOTE
        class_weights = None
        if positive_ratio < 0.2 or positive_ratio > 0.8:
            # Adjust weights to balance classes
            class_weights = {
                0: 1 / (2 * (1 - positive_ratio)) if positive_ratio < 1 else 1,
                1: 1 / (2 * positive_ratio) if positive_ratio > 0 else 1
            }
            print(f"Using class weights due to imbalance: {class_weights}")
            
            # Try to apply SMOTE if we have at least 5 samples of minority class
            min_class_count = min(sum(y), len(y) - sum(y))
            if min_class_count >= 5:
                try:
                    from imblearn.over_sampling import SMOTE
                    smote = SMOTE(random_state=42)
                    X, y = smote.fit_resample(X, y)
                    print(f"Applied SMOTE: resampled to {len(y)} samples with {sum(y)} positive examples")
                except ImportError:
                    print("SMOTE not available, proceeding with class weights only")
        
        # Scale the features for better performance
        X_scaled = self.scaler.fit_transform(X)
        
        # Use RandomizedSearchCV if we have enough samples
        if len(X) >= 30:
            try:
                from sklearn.model_selection import RandomizedSearchCV
                # Define parameter space
                param_dist = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 4, 6],
                    'min_samples_leaf': [1, 2, 4],
                    'bootstrap': [True, False]
                }
                
                # Base RF model
                rf = RandomForestClassifier(random_state=42, class_weight=class_weights)
                
                # RandomizedSearchCV
                rf_random = RandomizedSearchCV(
                    estimator=rf, param_distributions=param_dist, 
                    n_iter=10, cv=3, random_state=42, n_jobs=-1
                )
                
                # Fit RandomizedSearchCV
                rf_random.fit(X_scaled, y)
                
                # Get best model
                self.rf_classifier = rf_random.best_estimator_
                print(f"Best hyperparameters found: {rf_random.best_params_}")
                
            except (ImportError, ValueError) as e:
                print(f"RandomizedSearchCV failed: {e}. Using default RF.")
                # Fall back to default RF model
                self.rf_classifier = RandomForestClassifier(
                    n_estimators=200, max_depth=10, min_samples_split=4,
                    min_samples_leaf=2, class_weight=class_weights, random_state=42
                )
                self.rf_classifier.fit(X_scaled, y)
        else:
            # Use default RF with tuned parameters
            self.rf_classifier = RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=4,
                min_samples_leaf=2, class_weight=class_weights, random_state=42
            )
            self.rf_classifier.fit(X_scaled, y)
        
        # Calculate feature importances
        importances = self.rf_classifier.feature_importances_
        feature_names = [
            'message_size', 'service_weight', 'current_level', 'next_level',
            'utility_gain', 'overhead_increase', 'task_slack', 'global_slack',
            'deadline', 'avg_bandwidth', 'source_priority', 'dest_priority',
            'on_critical_path', 'processor_utilization', 'is_bottleneck',
            'relative_msg_size', 'slack_ratio', 'bcr'
        ][:len(importances)]  # Ensure we don't exceed actual features used
        
        # Print feature importances
        print("\nFeature importances:")
        for feature, importance in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
            print(f"  {feature}: {importance:.4f}")
        
        # Get training accuracy
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        y_pred = self.rf_classifier.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        
        print(f"Training metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        return True
    
    def extract_enhanced_features(self, message, service, current_level, next_level):
        """Extract richer features for the Random Forest model"""
        source_task = self.get_task_by_id(message.source_id)
        dest_task = self.get_task_by_id(message.dest_id)
        
        # Calculate utility gain
        service_weight = message.weights[service]
        current_strength = self.security.strengths[service][current_level]
        next_strength = self.security.strengths[service][next_level]
        utility_gain = service_weight * (next_strength - current_strength)
        
        # Calculate additional overhead
        source_proc = source_task.assigned_processor
        dest_proc = dest_task.assigned_processor
        
        current_overhead = self.calculate_service_overhead(message, service, current_level, source_proc, dest_proc)
        next_overhead = self.calculate_service_overhead(message, service, next_level, source_proc, dest_proc)
        overhead_increase = next_overhead - current_overhead
        
        # Calculate task slack - time between this task finishes and its successors start
        task_slack = float('inf')
        for succ_id in source_task.successors:
            succ_task = self.get_task_by_id(succ_id)
            if succ_task and succ_task.is_scheduled:
                slack = succ_task.start_time - source_task.finish_time
                task_slack = min(task_slack, slack)
        
        if task_slack == float('inf'):
            task_slack = self.deadline - source_task.finish_time
        
        # Global slack is the difference between deadline and makespan
        global_slack = self.deadline - max(task.finish_time for task in self.tasks if task.is_scheduled)
        
        # Calculate average bandwidth between source and destination processors
        avg_bandwidth = 0
        if source_proc != dest_proc:
            source_processor = self.processors[source_proc - 1]
            
            # Assuming there's a bandwidth attribute or method in the Processor class
            if hasattr(source_processor, 'bandwidth'):
                avg_bandwidth = source_processor.bandwidth[dest_proc - 1]
            else:
                # Fallback if bandwidth isn't directly available
                avg_bandwidth = 1000  # Default value
        
        # New features
        
        # Is this message on the critical path?
        on_critical_path = (message.source_id in self.critical_path_tasks and 
                           message.dest_id in self.critical_path_tasks)
        on_critical_path = 1.0 if on_critical_path else 0.0
        
        # Calculate processor utilization
        processor_utilization = self.calculate_processor_utilization(source_proc)
        
        # Is this a bottleneck task?
        is_bottleneck = self.is_bottleneck_task(source_task.task_id)
        is_bottleneck = 1.0 if is_bottleneck else 0.0
        
        # Message size relative to average
        avg_message_size = sum(m.size for m in self.messages) / len(self.messages)
        relative_msg_size = message.size / avg_message_size
        
        # Slack ratio (task slack to global slack)
        slack_ratio = task_slack / global_slack if global_slack > 0 else 0
        
        # BCR value
        bcr = self.calculate_bcr(message, service, current_level)
        
        # Feature vector - original and new features
        features = [
            message.size,                      # Message size
            service_weight,                    # Weight of security service 
            current_level,                     # Current security level
            next_level,                        # Next security level
            utility_gain,                      # Utility gain from upgrade
            overhead_increase,                 # Increase in overhead
            task_slack,                        # Available slack time
            global_slack,                      # Global slack until deadline
            self.deadline,                     # Deadline
            avg_bandwidth,                     # Average bandwidth between processors
            source_task.priority if hasattr(source_task, 'priority') else 0,  # Source task priority
            dest_task.priority if dest_task and hasattr(dest_task, 'priority') else 0,  # Destination task priority
            on_critical_path,                  # On critical path (0 or 1)
            processor_utilization,             # Processor utilization
            is_bottleneck,                     # Is bottleneck task (0 or 1)
            relative_msg_size,                 # Message size relative to average
            slack_ratio,                       # Task slack to global slack ratio
            bcr                                # Benefit-cost ratio
        ]
        
        return features
    
    def calculate_processor_utilization(self, processor_id):
        """Calculate the utilization rate of a processor"""
        if not processor_id or processor_id < 1 or processor_id > len(self.processors):
            return 0
            
        # Get tasks assigned to this processor
        tasks_on_proc = [t for t in self.tasks if t.is_scheduled and t.assigned_processor == processor_id]
        
        if not tasks_on_proc:
            return 0
            
        # Calculate total execution time
        total_exec_time = sum(t.finish_time - t.start_time for t in tasks_on_proc)
        
        # Calculate makespan for this processor
        proc_start = min(t.start_time for t in tasks_on_proc)
        proc_end = max(t.finish_time for t in tasks_on_proc)
        proc_makespan = proc_end - proc_start
        
        # Calculate utilization
        if proc_makespan > 0:
            return total_exec_time / proc_makespan
        else:
            return 0
    
    def is_bottleneck_task(self, task_id):
        """Check if a task is a bottleneck (has many predecessors and successors)"""
        task = self.get_task_by_id(task_id)
        if not task:
            return False
            
        # Count predecessors
        pred_count = 0
        for t in self.tasks:
            if task_id in t.successors:
                pred_count += 1
                
        # Count successors
        succ_count = len(task.successors)
        
        # Consider bottleneck if connected to at least 1/4 of all tasks
        threshold = len(self.tasks) / 4
        return (pred_count + succ_count) >= threshold
    def calculate_processor_utilization(self, processor_id):
        """Calculate the utilization rate of a processor"""
        if not processor_id or processor_id < 1 or processor_id > len(self.processors):
            return 0
            
        # Get tasks assigned to this processor
        tasks_on_proc = [t for t in self.tasks if t.is_scheduled and t.assigned_processor == processor_id]
        
        if not tasks_on_proc:
            return 0
            
        # Calculate total execution time
        total_exec_time = sum(t.finish_time - t.start_time for t in tasks_on_proc)
        
        # Calculate makespan for this processor
        proc_start = min(t.start_time for t in tasks_on_proc)
        proc_end = max(t.finish_time for t in tasks_on_proc)
        proc_makespan = proc_end - proc_start
        
        # Calculate utilization
        if proc_makespan > 0:
            return total_exec_time / proc_makespan
        else:
            return 0

    def is_bottleneck_task(self, task_id):
        """Check if a task is a bottleneck (has many predecessors and successors)"""
        task = self.get_task_by_id(task_id)
        if not task:
            return False
            
        # Count predecessors
        pred_count = 0
        for t in self.tasks:
            if task_id in t.successors:
                pred_count += 1
                
        # Count successors
        succ_count = len(task.successors)
        
        # Consider bottleneck if connected to at least 1/4 of all tasks
        threshold = len(self.tasks) / 4
        return (pred_count + succ_count) >= threshold

    def calculate_adaptive_bcr(self, message, service, current_level):
        """
        Calculate an adaptive Benefit-Cost Ratio that considers:
        - How close we are to the deadline (affects utility weight)
        - Whether the message is on the critical path
        - Current security level vs max level
        """
        next_level = current_level + 1
        if next_level >= len(self.security.strengths[service]):
            return 0  # Already at max level
        
        # Calculate benefit (security utility gain)
        service_weight = message.weights[service]
        current_strength = self.security.strengths[service][current_level]
        next_strength = self.security.strengths[service][next_level]
        utility_gain = service_weight * (next_strength - current_strength)
        
        # Calculate cost (additional overhead)
        source_task = self.get_task_by_id(message.source_id)
        dest_task = self.get_task_by_id(message.dest_id)
        source_proc = source_task.assigned_processor
        dest_proc = dest_task.assigned_processor
        
        current_overhead = self.calculate_service_overhead(message, service, current_level, source_proc, dest_proc)
        next_overhead = self.calculate_service_overhead(message, service, next_level, source_proc, dest_proc)
        overhead_increase = next_overhead - current_overhead
        
        # Avoid division by zero
        if overhead_increase <= 0:
            return float('inf') if utility_gain > 0 else 0
        
        # Critical path adjustment
        on_critical_path = (message.source_id in self.critical_path_tasks and 
                        message.dest_id in self.critical_path_tasks)
        critical_path_factor = 0.9 if on_critical_path else 1.0
        

        # Level proximity factor - higher payoff for completing a service
        max_level = len(self.security.strengths[service]) - 1
        level_proximity = next_level / max_level  # 0.0 to 1.0
        # CHANGED: Increased bonus for completing security services
        level_bonus = 1.0 + (1.0 * level_proximity)  # 1.0 to 2.0
        
        # CHANGED: Simplified BCR with focus on utility
        # Using fixed weight of 1.0 to prioritize security utility
        bcr = (utility_gain * level_bonus) / (overhead_increase * critical_path_factor)
        
        return bcr


    def run(self):
        """Execute the improved RF-SHIELD algorithm with enhanced ML-guided security upgrade."""
        import time
        start_time = time.time()
        
        # Phase 1 & 2: Run HSMS first to get initial schedule (from parent class)
        makespan, security_utility = super(SHIELD, self).run()
        
        if makespan is None or makespan > self.deadline:
            print("Improved RF-SHIELD cannot proceed: HSMS failed to meet deadline requirements")
            return None, None
        
        self.original_makespan = makespan
        self.original_security_utility = security_utility
        print(f"\nImproved RF-SHIELD Phase 1-2 complete: Initial makespan={makespan}, Security utility={security_utility}")
        
        # Phase 3: Intelligent Training Data Collection
        print("\nImproved RF-SHIELD Phase 3: Collecting optimized training data...")
        self.collect_training_data()
        
        # Phase 4: Enhanced RF model training with feature importance analysis
        print("\nImproved RF-SHIELD Phase 4: Training enhanced Random Forest model...")
        model_trained = self.train_rf_model()
        
        # Phase 5: Adaptive Security Enhancement using RF predictions
        print("\nImproved RF-SHIELD Phase 5: Starting adaptive ML-guided Security Enhancement...")
        self.use_rf_predictions = model_trained
        
        # Save initial configuration for rollback
        best_configuration = self.save_current_security_configuration()
        best_security_utility = security_utility
        best_makespan = makespan
        
        # Track progress for early stopping
        improvement_window = []
        stagnation_counter = 0
        
        # Use the enhanced security upgrade algorithm
        if model_trained:
            self.enhance_adaptive_security()
        else:
            # Fallback to traditional approach if model training failed
            print("No ML model available. Falling back to standard BCR-based approach.")
            self.enhance_security_with_bcr()
        
        # Calculate final results
        final_makespan = self.finalize_schedule()
        final_security_utility = self.calculate_security_utility()
        
        # Safety check: if security utility decreased or deadline violated, revert
        if final_security_utility < best_security_utility or final_makespan > self.deadline:
            print("WARNING: Solution quality decreased or deadline violated. Reverting to best configuration.")
            self.restore_security_configuration(best_configuration)
            self.reapply_schedule()  # Re-schedule with the best configuration
            final_makespan = best_makespan
            final_security_utility = best_security_utility
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance statistics
        print(f"Improved RF-SHIELD complete in {total_time:.2f} seconds")
        print(f"Final makespan={final_makespan}, Final security utility={final_security_utility}")
        print(f"Security utility improvement: {(final_security_utility - security_utility) / security_utility * 100:.2f}%")
        
        # Print ML model statistics
        if model_trained:
            self.print_improved_rf_analysis()
        
        return final_makespan, final_security_utility

    def enhance_adaptive_security(self):
        """
        Enhanced security upgrade algorithm with focus on maximizing security
        utility while just meeting the deadline constraint.
        """
        # Initialize BCR heap with all possible upgrades
        self.initialize_bcr_heap_improved()
        
        upgrade_count = 0
        
        # CHANGED: Keep track of best security utility achieved and its makespan
        best_security_utility = self.calculate_security_utility()
        best_configuration = self.save_current_security_configuration()
        best_makespan = self.original_makespan
        
        # CHANGED: Start with aggressive low thresholds to try more upgrades
        for service in self.confidence_thresholds:
            self.confidence_thresholds[service] = 0.1
        
        # CHANGED: Try individual upgrades instead of batching
        # Process upgrades in order of highest adaptive BCR
        while self.bcr_heap:
            # Get the upgrade with the highest BCR
            bcr, (message_id, service, current_level) = heapq.heappop(self.bcr_heap)
            bcr = -bcr  # Convert back to positive value
            
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
            
            # Extract features for this potential upgrade
            features = self.extract_enhanced_features(message, service, current_level, next_level)
            
            # Get service-specific confidence threshold
            confidence_threshold = self.confidence_thresholds[service]
            
            # Predict success using the trained model
            will_succeed = True  # CHANGED: Default to trying almost always
            prediction_confidence = 0.5  # Default confidence
            
            if self.rf_classifier is not None:
                # Scale features
                features_scaled = self.scaler.transform([features])
                
                # Get prediction probabilities
                pred_proba = self.rf_classifier.predict_proba(features_scaled)[0]
                success_probability = pred_proba[1] if len(pred_proba) > 1 else 0.5
                
                # CHANGED: More permissive threshold
                will_succeed = success_probability >= confidence_threshold
                prediction_confidence = success_probability
            
            # CHANGED: Almost always try upgrades
            # Skip only if ML is very confident it will fail
            if not will_succeed and prediction_confidence < 0.05:
                continue
            
            # Try the upgrade
            print(f"\nTrying to upgrade {message.id} {service} from level {current_level} to {next_level} (BCR: {bcr:.4f}, Confidence: {prediction_confidence:.2f})")
            
            # Save current state before upgrade
            pre_upgrade_configuration = self.save_current_security_configuration()
            
            self.upgrade_trials += 1
            success = self.try_security_upgrade(message, service, next_level)
            
            if success:
                self.upgrade_successes += 1
                
                # Check if we're still meeting deadline
                current_makespan = self.finalize_schedule()
                current_security_utility = self.calculate_security_utility()
                
                if current_makespan <= self.deadline:
                    # We're good - keep the upgrade
                    upgrade_count += 1
                    
                    # Track upgrade by service
                    self.upgrades_by_service[service] += 1
                    
                    # CHANGED: Update best solution if security utility improved
                    if current_security_utility > best_security_utility:
                        best_security_utility = current_security_utility
                        best_configuration = self.save_current_security_configuration()
                        best_makespan = current_makespan
                        print(f"📈 New best security utility: {best_security_utility:.4f}")
                    
                    # If can upgrade further, add back to heap
                    if next_level < max_level:
                        new_bcr = self.calculate_adaptive_bcr(message, service, next_level)
                        if new_bcr > 0:
                            heapq.heappush(self.bcr_heap, (-new_bcr, (message.id, service, next_level)))
                    
                    print(f"✅ Upgrade successful! New {service} level: {next_level}")
                else:
                    # Violated deadline - rollback this upgrade
                    print(f"⚠️ Deadline violated. Rolling back.")
                    self.restore_security_configuration(pre_upgrade_configuration)
                    self.reapply_schedule()
                    self.rollbacks += 1
                    
                    # CHANGED: Don't adjust confidence threshold here
            else:
                print(f"❌ Upgrade rejected: insufficient slack")
        
        # Restore the best configuration found
        print(f"\nRestoring best security configuration with utility {best_security_utility:.4f}")
        self.restore_security_configuration(best_configuration)
        
        print(f"\nSecurity enhancement complete: {upgrade_count} upgrades applied, {self.rollbacks} rollbacks performed")
        print(f"Final makespan: {best_makespan}, Deadline: {self.deadline}")
        print(f"Upgrades by service: Confidentiality: {self.upgrades_by_service['confidentiality']}, " +
            f"Integrity: {self.upgrades_by_service['integrity']}, " +
            f"Authentication: {self.upgrades_by_service['authentication']}")

    def initialize_bcr_heap_improved(self):
        """Initialize the BCR heap using the adaptive BCR calculation."""
        self.bcr_heap = []
        
        # For each message, calculate BCR for each security service
        for message in self.messages:
            for service in ['confidentiality', 'integrity', 'authentication']:
                current_level = message.assigned_security[service]
                
                # Check if we can upgrade further
                if current_level < len(self.security.strengths[service]) - 1:
                    # Calculate adaptive BCR 
                    bcr = self.calculate_adaptive_bcr(message, service, current_level)
                    
                    if bcr > 0:
                        # Use negative BCR because heapq is a min-heap
                        heapq.heappush(self.bcr_heap, (-bcr, (message.id, service, current_level)))
        
        print(f"Initialized BCR heap with {len(self.bcr_heap)} potential upgrades")

    def reapply_schedule(self):
        """Reapply schedule with current security settings."""
        # Reset all task scheduling information
        for task in self.tasks:
            task.is_scheduled = False
            task.start_time = None
            task.finish_time = None
            task.assigned_processor = None
        
        # Re-run the scheduling algorithm to get the new makespan
        super(SHIELD, self).run()

    def enhance_security_with_bcr(self):
        """Fallback method to enhance security using standard BCR approach."""
        self.initialize_bcr_heap()
        
        upgrade_count = 0
        
        # Keep track of best security utility achieved
        best_security_utility = self.calculate_security_utility()
        best_configuration = self.save_current_security_configuration()
        
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
            
            # Try the upgrade
            success = self.try_security_upgrade(message, service, next_level)
            
            if success:
                upgrade_count += 1
                
                # Track this upgrade
                self.upgrade_history.append({
                    'message_id': message_id,
                    'service': service,
                    'from_level': current_level,
                    'to_level': next_level,
                    'bcr': bcr
                })
                
                # Update best configuration if utility improved
                current_security_utility = self.calculate_security_utility()
                if current_security_utility > best_security_utility:
                    best_security_utility = current_security_utility
                    best_configuration = self.save_current_security_configuration()
                
                # If we can upgrade further, calculate new BCR and add back to heap
                if next_level < max_level:
                    new_bcr = self.calculate_bcr(message, service, next_level)
                    if new_bcr > 0:
                        heapq.heappush(self.bcr_heap, (-new_bcr, (message_id, service, next_level)))
                
                print(f"✅ Upgrade successful! New {service} level: {next_level}")
            else:
                print(f"❌ Upgrade rejected: insufficient slack")
        
        # Restore the best configuration found
        print(f"\nRestoring best security configuration with utility {best_security_utility:.4f}")
        self.restore_security_configuration(best_configuration)
        
        print(f"\nFallback security enhancement complete: {upgrade_count} upgrades applied")

    def print_improved_rf_analysis(self):
        """Print enhanced analysis of the Random Forest model performance."""
        if self.rf_classifier is None:
            print("No Random Forest model was trained.")
            return
            
        print("\n" + "="*120)
        print("IMPROVED RANDOM FOREST MODEL ANALYSIS")
        print("="*120)
        
        # Feature importances
        feature_names = [
            'message_size', 'service_weight', 'current_level', 'next_level',
            'utility_gain', 'overhead_increase', 'task_slack', 'global_slack',
            'deadline', 'avg_bandwidth', 'source_priority', 'dest_priority',
            'on_critical_path', 'processor_utilization', 'is_bottleneck',
            'relative_msg_size', 'slack_ratio', 'bcr'
        ][:len(self.rf_classifier.feature_importances_)]
        
        importances = self.rf_classifier.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("Feature ranking:")
        for f in range(len(feature_names)):
            print(f"{f+1}. {feature_names[indices[f]]} ({importances[indices[f]]:.4f})")
        
        # Performance statistics
        print("\nPerformance Statistics:")
        print(f"Total upgrade trials: {self.upgrade_trials}")
        print(f"Successful upgrades: {self.upgrade_successes}")
        if self.upgrade_trials > 0:
            print(f"Success rate: {self.upgrade_successes / self.upgrade_trials * 100:.2f}%")
        
        # Additional statistics about rollbacks and service-specific upgrades
        print("\nRollback Statistics:")
        print(f"Total rollbacks: {self.rollbacks}")
        
        print("\nService-specific upgrades:")
        for service, count in self.upgrades_by_service.items():
            print(f"  {service}: {count}")
        
        # Confidence threshold evolution
        print("\nFinal confidence thresholds:")
        for service, threshold in self.confidence_thresholds.items():
            print(f"  {service}: {threshold:.2f}")
        
        print("\nCritical Path Analysis:")
        print(f"Critical path length: {len(self.critical_path)}")
        critical_path_msgs = 0
        for message in self.messages:
            if message.source_id in self.critical_path_tasks and message.dest_id in self.critical_path_tasks:
                critical_path_msgs += 1
        print(f"Messages on critical path: {critical_path_msgs} out of {len(self.messages)}")
        
        print("="*120)

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
    security_service = SecurityService()
    deadline = 500
    
    return tasks, messages, processors, security_service, deadline
        
        
def create_testcase_1():
    processors = [
        Processor(1, bandwidth=[0,1000, 300]),  # Proc 1 to 2 = 1000 KB/s, to 3 = 300 KB/s
        Processor(2, bandwidth=[1000,0, 500]),  # Proc 2 to 1 = 1000 KB/s, to 3 = 500 KB/s
        Processor(3, bandwidth=[300, 500,0])    # Proc 3 to 1 = 300 KB/s, to 2 = 500 KB/s
    ]
    tasks = [
        Task(1, "T1", [20, 25, 30]),
        Task(2, "T2", [30, 28, 35], predecessors=[1]),
        Task(3, "T3", [25, 22, 40], predecessors=[1]),
        Task(4, "T4", [18, 26, 20], predecessors=[2]),
        Task(5, "T5", [30, 34, 27], predecessors=[2]),
        Task(6, "T6", [35, 30, 40], predecessors=[3]),
        Task(7, "T7", [22, 28, 24], predecessors=[4, 5]),
        Task(8, "T8", [20, 18, 22], predecessors=[5]),
        Task(9, "T9", [33, 29, 25], predecessors=[6]),
        Task(10, "T10", [25, 22, 30], predecessors=[7, 8, 9])
    ]
    messages = [
        Message(1, 2, 500),
        Message(1, 3, 700),
        Message(2, 4, 300),
        Message(2, 5, 400),
        Message(3, 6, 600),
        Message(4, 7, 250),
        Message(5, 7, 350),
        Message(5, 8, 200),
        Message(6, 9, 550),
        Message(7, 10, 400),
        Message(8, 10, 300),
        Message(9, 10, 450)
    ]
    for msg in messages:
        msg.set_security_requirements(
            conf_min=0.6, integ_min=0.7, auth_min=0.8,
            conf_w=0.4, integ_w=0.3, auth_w=0.3
        )
    security_service = SecurityService()
    deadline = 550
    
    return tasks, messages, processors, security_service, deadline
        



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

# def plot_gantt_chart(title, schedule, num_processors, makespan,security_utility):

    
#     # Setup the figure
#     fig, ax = plt.subplots(figsize=(12, 6))
    
#     # Colors for different tasks
#     colors = plt.cm.tab20(np.linspace(0, 1, 20))
#     title = f'{title} (Makespan: {makespan:.2f}, Utility: {security_utility:.2f})'
    
#     # Draw each task as a rectangle
#     for entry in schedule:
#         task_id = entry['task_id']
#         processor = entry['processor']
#         start = entry['start_time']
#         finish = entry['finish_time']
#         # Create a rectangle for the task
#         rect = plt1.Rectangle((start, processor-0.4), finish-start, 0.8, 
#                          color=colors[task_id % len(colors)], alpha=0.8)
#         ax.add_patch(rect)
        
#         # Add task label
#         ax.text(start + (finish-start)/2, processor, f"T{task_id}", 
#                 ha='center', va='center', color='black')
    
#     # Set up the axes
#     ax.set_ylim(0.5, num_processors + 0.5)
#     ax.set_xlim(0, makespan * 1.05)  # Add some padding
#     ax.set_yticks(range(1, num_processors + 1))
#     ax.set_yticklabels([f'Processor {i}' for i in range(1, num_processors + 1)])
#     ax.set_xlabel('Time (ms)')
#     ax.set_title(title)
#     ax.grid(True, linestyle='--', alpha=0.7)
    
#     return fig, ax
def plot_gantt_chart(title, schedule, num_processors, makespan, security_utility=None, messages=None, tasks=None):
    """
    Create a Gantt chart showing task schedule with security information.
    
    Args:
        title: Chart title
        schedule: List of task schedule entries
        num_processors: Number of processors
        makespan: Total schedule length
        security_utility: Optional security utility value to display
        messages: List of message objects with security information
        tasks: List of task objects
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle
    
    # Setup the figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Colors for different tasks
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    # Update title with metrics if provided
    if security_utility is not None:
        display_title = f'{title} (Makespan: {makespan:.2f}, Utility: {security_utility:.2f})'
    else:
        display_title = f'{title} (Makespan: {makespan:.2f})'
    
    # Create task mapping for quick lookup
    task_map = {}
    if tasks:
        task_map = {task.task_id: task for task in tasks}
    
    # Create message mapping for quick lookup
    msg_map = {}
    if messages:
        for msg in messages:
            if hasattr(msg, 'source_id') and hasattr(msg, 'dest_id'):
                key = f"{msg.source_id}_{msg.dest_id}"
                msg_map[key] = msg
    
    # Security protocol abbreviations for better readability
    security_protocols = {
        'confidentiality': ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'],
        'integrity': ['I0', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6'],
        'authentication': ['A0', 'A1', 'A2']
    }
    
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
        
        # Add basic task label
        task_label = f"T{task_id}"
        
        # Check if we have security information to show
        security_info = ""
        task = task_map.get(task_id)
        
        if task and hasattr(task, 'successors') and messages:
            # Collect security levels for outgoing messages
            outgoing_security = []
            
            for succ_id in task.successors:
                msg_key = f"{task_id}_{succ_id}"
                msg = msg_map.get(msg_key)
                
                if msg and hasattr(msg, 'assigned_security'):
                    # Get security protocol names
                    conf_idx = msg.assigned_security.get('confidentiality', 0)
                    integ_idx = msg.assigned_security.get('integrity', 0)
                    auth_idx = msg.assigned_security.get('authentication', 0)
                    
                    conf = security_protocols['confidentiality'][conf_idx] if conf_idx < len(security_protocols['confidentiality']) else 'C?'
                    integ = security_protocols['integrity'][integ_idx] if integ_idx < len(security_protocols['integrity']) else 'I?'
                    auth = security_protocols['authentication'][auth_idx] if auth_idx < len(security_protocols['authentication']) else 'A?'
                    
                    outgoing_security.append(f"→T{succ_id}:{conf}/{integ}/{auth}")
            
            if outgoing_security:
                security_info = "\n" + "\n".join(outgoing_security)
        
        # Add task label with security info
        ax.text(start + (finish-start)/2, processor, f"{task_label}{security_info}", 
                ha='center', va='center', color='black', fontsize=13,weight='bold')
    
    # Set up the axes
    ax.set_ylim(0.5, num_processors + 0.5)
    ax.set_xlim(0, makespan * 1.05)  # Add some padding
    ax.set_yticks(range(1, num_processors + 1))
    ax.set_yticklabels([f'Processor {i}' for i in range(1, num_processors + 1)])
    ax.set_xlabel('Time (ms)')
    ax.set_title(display_title)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return fig, ax
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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import copy
import random
from collections import deque, namedtuple

# Define Checkpoint class to store state for rollbacks
class Checkpoint:
    def __init__(self, scheduler, security_utility, makespan, critical_path=None):
        """Store a complete state checkpoint for possible rollback"""
        self.scheduler = copy.deepcopy(scheduler)
        self.security_utility = security_utility
        self.makespan = makespan
        self.critical_path = critical_path
        
    def __str__(self):
        return f"Checkpoint[Makespan: {self.makespan:.2f}, Utility: {self.security_utility:.2f}]"

# Define the A3C Network
class A3CNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(A3CNetwork, self).__init__()
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=1)
        )
        
        # Value head (critic)
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        shared_output = self.shared(x)
        policy_output = self.policy(shared_output)
        value_output = self.value(shared_output)
        return policy_output, value_output

# Experience memory for A3C training
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class A3CAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create network
        self.network = A3CNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Memory for experiences
        self.memory = deque(maxlen=2000)
        
        # Hyperparameters
        self.entropy_beta = 0.01  # Entropy regularization coefficient
    
    def act(self, state, explore=True):
        """Choose an action based on current state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy, _ = self.network(state_tensor)
        
        if explore:
            # Sample action from the probability distribution
            m = Categorical(policy)
            action = m.sample().item()
        else:
            # Choose best action (greedy)
            action = torch.argmax(policy, dim=1).item()
            
        return action, policy.cpu().numpy()[0]
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def train(self, batch_size=32):
        """Train the network on a batch of experiences"""
        if len(self.memory) < batch_size:
            return 0.0  # Not enough samples
            
        # Sample random batch
        batch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.FloatTensor([float(e.done) for e in batch]).to(self.device)
        
        # Get current Q values and V values
        policies, values = self.network(states)
        
        # Get next V values for target
        _, next_values = self.network(next_states)
        next_values = next_values.detach().squeeze()
        
        # Target = reward + gamma * next_value * (1 - done)
        targets = rewards + self.gamma * next_values * (1 - dones)
        
        # Value loss (critic)
        value_loss = F.mse_loss(values.squeeze(), targets)
        
        # Policy loss (actor)
        log_probs = torch.log(policies + 1e-10)  # Add small constant for numerical stability
        selected_log_probs = log_probs.gather(1, actions).squeeze()
        advantages = (targets - values.squeeze()).detach()  # Detach to not backpropagate through critic
        
        policy_loss = -(selected_log_probs * advantages).mean()
        
        # Entropy regularization (helps exploration)
        entropy = -(policies * log_probs).sum(dim=1).mean()
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss - self.entropy_beta * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save(self, filepath):
        """Save model parameters"""
        torch.save(self.network.state_dict(), filepath)
    
    def load(self, filepath):
        """Load model parameters"""
        self.network.load_state_dict(torch.load(filepath, map_location=self.device))


class CriticalPathUpgrader:
    """Class to handle security upgrades on the non-critical paths"""
    
    def __init__(self, scheduler, max_iterations=100, checkpoint_interval=5, deadline_buffer=0.05):
        """
        Initialize the upgrader with a completed HSMS schedule.
        
        Args:
            scheduler: An HSMS scheduler with an initial feasible schedule
            max_iterations: Maximum number of upgrade attempts
            checkpoint_interval: How often to create checkpoints
            deadline_buffer: Buffer percentage to keep below deadline
        """
        self.scheduler = scheduler
        self.deadline = scheduler.deadline
        self.buffer_deadline = self.deadline * (1 - deadline_buffer)  # Apply buffer
        self.max_iterations = max_iterations
        self.checkpoint_interval = checkpoint_interval
        
        # Security protocols and their indices
        self.service_types = ['confidentiality', 'integrity', 'authentication']
        self.max_levels = {
            'confidentiality': 7,  # 0-7
            'integrity': 6,        # 0-6
            'authentication': 2    # 0-2
        }
        
        # For tracking progress
        self.best_utility = self.scheduler.calculate_security_utility()
        self.best_makespan = self.scheduler.makespan
        
        # Initialize A3C agent
        # State: [task_slack, message_security_levels, criticality_flags]
        num_messages = len(self.scheduler.messages)
        num_services = len(self.service_types)
        
        # State size: slack for each task + security level for each message-service pair + critical flags
        state_size = len(self.scheduler.tasks) + (num_messages * num_services) + num_messages
        
        # Action size: message_id * service_type combinations (each can be upgraded)
        action_size = num_messages * num_services
        
        self.agent = A3CAgent(state_size, action_size)
        
        # Checkpoint history
        self.checkpoints = []
        self.current_checkpoint_index = -1
        
        # Initial state
        self.create_checkpoint()
    
    def get_critical_path(self):
        """
        Identify the critical path in the current schedule using backtracking from the exit node.
        Returns a list of task IDs representing the critical path.
        """
        # Find the task with the latest finish time (exit node of critical path)
        latest_task = max(self.scheduler.tasks, key=lambda t: t.finish_time if t.is_scheduled else float('-inf'))
        
        critical_path = [latest_task.task_id]
        current_task = latest_task
        
        # Backtrack to find the critical path
        while current_task.predecessors:
            # Find the predecessor with the latest impact on current task's start time
            max_impact_pred = None
            max_impact_time = float('-inf')
            
            for pred_id in current_task.predecessors:
                pred_task = self.scheduler.get_task_by_id(pred_id)
                if not pred_task or not pred_task.is_scheduled:
                    continue
                
                # Calculate impact time (pred finish + comm + security overhead)
                message = self.scheduler.get_message(pred_id, current_task.task_id)
                if not message:
                    continue
                
                # Estimate communication time
                source_proc = self.scheduler.processors[pred_task.assigned_processor - 1]
                dest_proc = self.scheduler.processors[current_task.assigned_processor - 1]
                comm_time = source_proc.get_communication_time(message, source_proc, dest_proc) if source_proc != dest_proc else 0
                
                # Total impact is predecessor finish time + communication
                # (Note: security overhead is already included in the finish times)
                impact_time = pred_task.finish_time + comm_time
                
                if impact_time > max_impact_time:
                    max_impact_time = impact_time
                    max_impact_pred = pred_task
            
            # If we found a critical predecessor, add it to the path
            if max_impact_pred:
                critical_path.append(max_impact_pred.task_id)
                current_task = max_impact_pred
            else:
                # No more critical predecessors
                break
        
        # Return the critical path in topological order (entry to exit)
        return list(reversed(critical_path))

    def get_non_critical_messages(self, critical_path):
        """
        Identify messages that are not on the critical path and are candidates for security upgrades.
        """
        # Create pairs of tasks on the critical path
        critical_pairs = set()
        for i in range(len(critical_path) - 1):
            for j in range(i + 1, len(critical_path)):
                critical_pairs.add((critical_path[i], critical_path[j]))
        
        # Find messages that are not on the critical path
        non_critical_messages = []
        for message in self.scheduler.messages:
            if (message.source_id, message.dest_id) not in critical_pairs:
                # Check if both source and destination tasks are scheduled
                source_task = self.scheduler.get_task_by_id(message.source_id)
                dest_task = self.scheduler.get_task_by_id(message.dest_id)
                
                if (source_task and dest_task and 
                    source_task.is_scheduled and dest_task.is_scheduled):
                    non_critical_messages.append(message)
        
        return non_critical_messages

    def calculate_task_slack(self):
        """
        Calculate slack time for each task in the schedule.
        Slack is the amount of time a task can be delayed without affecting the makespan.
        """
        # Initialize list to store slack for each task
        task_slack = {task.task_id: 0 for task in self.scheduler.tasks if task.is_scheduled}
        
        # Find the critical path
        critical_path = self.get_critical_path()
        critical_tasks = set(critical_path)
        
        # Tasks on the critical path have zero slack
        for task_id in critical_tasks:
            task_slack[task_id] = 0
        
        # Calculate slack for non-critical tasks
        for task in self.scheduler.tasks:
            if not task.is_scheduled or task.task_id in critical_tasks:
                continue
            
            # Calculate latest allowable finish time based on successors
            latest_finish = float('inf')
            
            for succ_id in task.successors:
                succ_task = self.scheduler.get_task_by_id(succ_id)
                if not succ_task or not succ_task.is_scheduled:
                    continue
                
                # Get the message between task and successor
                message = self.scheduler.get_message(task.task_id, succ_id)
                if not message:
                    continue
                
                # Calculate communication time
                source_proc = self.scheduler.processors[task.assigned_processor - 1]
                dest_proc = self.scheduler.processors[succ_task.assigned_processor - 1]
                comm_time = source_proc.get_communication_time(message, source_proc, dest_proc) if source_proc != dest_proc else 0
                
                # Latest finish time = successor start time - communication time
                succ_start = succ_task.start_time
                this_latest = succ_start - comm_time
                latest_finish = min(latest_finish, this_latest)
            
            # If the task has no successors, use the deadline
            if latest_finish == float('inf'):
                latest_finish = self.scheduler.deadline
            
            # Slack = latest finish time - actual finish time
            slack = latest_finish - task.finish_time
            task_slack[task.task_id] = max(0, slack)  # Slack can't be negative
        
        return task_slack, critical_path

    def build_state(self, task_slack, critical_path):
        """
        Build the state representation for the A3C agent.
        State includes task slack, message security levels, and criticality flags.
        """
        # Get messages on the critical path
        critical_pairs = set()
        for i in range(len(critical_path) - 1):
            for j in range(i + 1, len(critical_path)):
                critical_pairs.add((critical_path[i], critical_path[j]))
        
        # Build state vector components
        # 1. Task slack (normalized)
        max_slack = max(task_slack.values()) if task_slack else 1.0
        normalized_slack = [task_slack.get(task.task_id, 0) / (max_slack if max_slack > 0 else 1) 
                          for task in self.scheduler.tasks]
        
        # 2. Message security levels (normalized)
        security_levels = []
        criticality_flags = []
        
        for message in self.scheduler.messages:
            # Check if message is on critical path
            is_critical = (message.source_id, message.dest_id) in critical_pairs
            criticality_flags.append(1 if is_critical else 0)
            
            # Add normalized security levels for each service
            for service in self.service_types:
                level = message.assigned_security[service]
                max_level = self.max_levels[service]
                security_levels.append(level / max_level)
        
        # Combine all state components
        state = normalized_slack + security_levels + criticality_flags
        return np.array(state)

    def decode_action(self, action_idx):
        """
        Decode action index into message ID and security service.
        
        Args:
            action_idx: Combined action index
            
        Returns:
            tuple: (message_index, service_type)
        """
        num_services = len(self.service_types)
        message_idx = action_idx // num_services
        service_idx = action_idx % num_services
        
        if message_idx >= len(self.scheduler.messages):
            return None, None
            
        return message_idx, self.service_types[service_idx]

    def apply_upgrade(self, message_idx, service_type):
        """
        Apply a security upgrade to a specific message and service.
        
        Args:
            message_idx: Index of the message to upgrade
            service_type: Type of security service to upgrade
            
        Returns:
            bool: True if upgrade was successful, False otherwise
        """
        # Get the message
        if message_idx >= len(self.scheduler.messages):
            return False
            
        message = self.scheduler.messages[message_idx]
        
        # Check if the message can be upgraded
        current_level = message.assigned_security[service_type]
        max_level = self.max_levels[service_type]
        
        if current_level >= max_level:
            return False  # Already at max level
        
        # Apply the upgrade
        message.assigned_security[service_type] += 1
        
        return True

    def calculate_reward(self, old_utility, new_utility, old_makespan, new_makespan):
        """
        Calculate reward for the A3C agent based on changes in utility and makespan.
        
        Args:
            old_utility: Security utility before upgrade
            new_utility: Security utility after upgrade
            old_makespan: Schedule makespan before upgrade
            new_makespan: Schedule makespan after upgrade
            
        Returns:
            float: Calculated reward
        """
        # Utility improvement reward
        utility_reward = (new_utility - old_utility) * 10  # Scale up to make more significant
        
        # Deadline violation penalty
        deadline_penalty = 0
        if new_makespan > self.buffer_deadline:
            # Stronger penalty the further we go over the buffer deadline
            deadline_penalty = -20 * ((new_makespan - self.buffer_deadline) / self.buffer_deadline)
        
        # Efficiency reward: reward for keeping makespan low
        efficiency_reward = (old_makespan - new_makespan) * 5
        
        # Total reward
        total_reward = utility_reward + deadline_penalty + efficiency_reward
        
        return total_reward

    def create_checkpoint(self):
        """Create a checkpoint of the current state"""
        # Calculate security utility for the current state
        security_utility = self.scheduler.calculate_security_utility()
        
        # Create and store checkpoint
        critical_path = self.get_critical_path()
        checkpoint = Checkpoint(self.scheduler, security_utility, self.scheduler.makespan, critical_path)
        
        # Add to checkpoint history
        self.checkpoints.append(checkpoint)
        self.current_checkpoint_index = len(self.checkpoints) - 1
        
        return checkpoint

    def rollback_to_checkpoint(self, checkpoint_index=None):
        """
        Roll back to a specific checkpoint.
        
        Args:
            checkpoint_index: Index of checkpoint to roll back to, or None for most recent
        """
        if checkpoint_index is None:
            checkpoint_index = self.current_checkpoint_index
        
        if 0 <= checkpoint_index < len(self.checkpoints):
            checkpoint = self.checkpoints[checkpoint_index]
            self.scheduler = copy.deepcopy(checkpoint.scheduler)
            self.current_checkpoint_index = checkpoint_index
            return True
            
        return False

    def upgrade_security(self, training_mode=True, episodes=100):
        """
        Main method to perform the security upgrade process using the A3C agent.
        
        Args:
            training_mode: Whether to train the agent or just use it for inference
            episodes: Number of episodes for training
            
        Returns:
            tuple: (makespan, security_utility) of the best solution found
        """
        print("\n" + "="*80)
        print("CRITICAL PATH-GUIDED SECURITY UPGRADE")
        print("="*80)
        
        best_checkpoint = self.create_checkpoint()  # Initial checkpoint
        improvement_streak = 0
        no_improvement_count = 0
        
        for episode in range(episodes):
            print(f"\nEpisode {episode+1}/{episodes}")
            print("-" * 40)
            
            # Reset to last checkpoint if training
            if training_mode and self.current_checkpoint_index >= 0:
                self.rollback_to_checkpoint()
            
            # Calculate task slack and critical path
            task_slack, critical_path = self.calculate_task_slack()
            
            # Print critical path information
            print(f"Critical Path: {' -> '.join(str(t) for t in critical_path)}")
            print(f"Current Makespan: {self.scheduler.makespan:.2f}, Deadline: {self.deadline:.2f}")
            
            # Get non-critical messages eligible for upgrade
            non_critical_messages = self.get_non_critical_messages(critical_path)
            print(f"Non-Critical Messages eligible for upgrade: {len(non_critical_messages)}/{len(self.scheduler.messages)}")
            
            # Start the upgrade loop
            iteration = 0
            upgrades_applied = 0
            
            while iteration < self.max_iterations:
                iteration += 1
                
                # Build state representation
                state = self.build_state(task_slack, critical_path)
                
                # Get action from agent
                action, action_probs = self.agent.act(state, explore=training_mode)
                message_idx, service_type = self.decode_action(action)
                
                if message_idx is None or service_type is None:
                    print(f"Invalid action: {action}, skipping")
                    continue
                
                # Check if message is eligible for upgrade (non-critical)
                message = self.scheduler.messages[message_idx]
                source_task = self.scheduler.get_task_by_id(message.source_id)
                dest_task = self.scheduler.get_task_by_id(message.dest_id)
                
                if not source_task or not dest_task or not source_task.is_scheduled or not dest_task.is_scheduled:
                    if training_mode:
                        # Negative reward for invalid action
                        reward = -1.0
                        self.agent.remember(state, action, reward, state, False)
                    continue
                
                # Store current state for potential rollback
                old_scheduler = copy.deepcopy(self.scheduler)
                old_utility = self.scheduler.calculate_security_utility()
                old_makespan = self.scheduler.makespan
                
                # Apply the upgrade
                success = self.apply_upgrade(message_idx, service_type)
                
                if not success:
                    if training_mode:
                        # Small negative reward for impossible upgrade
                        reward = -0.5
                        self.agent.remember(state, action, reward, state, False)
                    continue
                
                print(f"Iteration {iteration}: Upgraded message {message.id} ({message.source_id}->{message.dest_id}) {service_type}")
                
                # Recalculate schedule with new security levels
                self.scheduler.finalize_schedule()
                new_makespan = self.scheduler.makespan
                new_utility = self.scheduler.calculate_security_utility()
                
                print(f"  Security Utility: {old_utility:.2f} -> {new_utility:.2f}")
                print(f"  Makespan: {old_makespan:.2f} -> {new_makespan:.2f}")
                
                # Check if we've violated deadline
                if new_makespan > self.deadline:
                    print(f"  ❌ Deadline violated! Rolling back.")
                    # Roll back the upgrade
                    self.scheduler = old_scheduler
                    
                    if training_mode:
                        # Strong negative reward for deadline violation
                        reward = -5.0
                        # Build next state (same as current because we rolled back)
                        next_state = state
                        self.agent.remember(state, action, reward, next_state, True)
                        
                    continue
                
                # Get reward
                reward = self.calculate_reward(old_utility, new_utility, old_makespan, new_makespan)
                
                # Handle training
                if training_mode:
                    # Calculate new state after upgrade
                    new_task_slack, new_critical_path = self.calculate_task_slack()
                    next_state = self.build_state(new_task_slack, new_critical_path)
                    
                    # Store experience
                    done = (new_makespan > self.buffer_deadline)
                    self.agent.remember(state, action, reward, next_state, done)
                    
                    # Update state for next iteration
                    task_slack = new_task_slack
                    critical_path = new_critical_path
                
                # Track successful upgrades
                upgrades_applied += 1
                
                # Create checkpoint periodically
                if upgrades_applied % self.checkpoint_interval == 0:
                    checkpoint = self.create_checkpoint()
                    print(f"  📌 Created checkpoint {len(self.checkpoints)}: Makespan={checkpoint.makespan:.2f}, Utility={checkpoint.security_utility:.2f}")
                
                # Check if this is the best solution so far
                if new_utility > best_checkpoint.security_utility and new_makespan <= self.deadline:
                    best_checkpoint = self.create_checkpoint()
                    print(f"  ✅ New best solution! Utility: {new_utility:.2f}, Makespan: {new_makespan:.2f}")
                    improvement_streak += 1
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                # Early stopping if no improvements for a while
                if no_improvement_count >= 10:
                    print("No improvements for several iterations, stopping this episode")
                    break
            
            # Train the agent after each episode
            if training_mode and len(self.agent.memory) > 32:
                loss = self.agent.train(32)
                print(f"Trained agent. Loss: {loss:.4f}")
            
            print(f"Episode {episode+1} complete. Applied {upgrades_applied} upgrades.")
        
        # Restore the best solution found
        if best_checkpoint:
            self.scheduler = copy.deepcopy(best_checkpoint.scheduler)
            print("\nFinal Solution:")
            print(f"Makespan: {self.scheduler.makespan:.2f} (Deadline: {self.deadline:.2f})")
            print(f"Security Utility: {best_checkpoint.security_utility:.2f}")
        
        return self.scheduler.makespan, best_checkpoint.security_utility

    def print_security_upgrade_summary(self):
        """Print summary of security upgrades applied"""
        print("\n" + "="*80)
        print("SECURITY UPGRADE SUMMARY")
        print("="*80)
        
        # Define security protocols for better readability
        security_protocols = {
            'confidentiality': ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'],
            'integrity': ['I0', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6'],
            'authentication': ['A0', 'A1', 'A2']
        }
        
        # Print header
        print(f"{'Message ID':<12} {'Source→Dest':<15} {'Service':<15} {'Protocol':<10} {'Security Level':<15}")
        print("-"*80)
        
        for message in self.scheduler.messages:
            source_task = self.scheduler.get_task_by_id(message.source_id)
            dest_task = self.scheduler.get_task_by_id(message.dest_id)
            
            if not source_task or not dest_task or not source_task.is_scheduled or not dest_task.is_scheduled:
                continue
                
            source_dest = f"{source_task.name}→{dest_task.name}"
            
            # Print information for each security service
            for service in self.service_types:
                idx = message.assigned_security[service]
                protocol = security_protocols[service][idx]
                strength = self.scheduler.security.strengths[service][idx]
                
                print(f"{message.id:<12} {source_dest:<15} {service.capitalize():<15} {protocol:<10} {strength:<15.2f}")
        
        print("-"*80)
        print(f"Final Makespan: {self.scheduler.makespan:.2f}")
        print(f"Final Security Utility: {self.scheduler.calculate_security_utility():.2f}")
        print(f"Deadline: {self.deadline:.2f}")
        print("="*80)

def plot_critical_path_gantt(critical_path_upgrader, title=None, show_security=True, show_critical_path=True):
    """
    Create an enhanced Gantt chart for the CriticalPathUpgrader showing:
    - Task scheduling across processors
    - Critical path highlighting
    - Security levels on messages
    - Base execution vs security overhead
    
    Args:
        critical_path_upgrader: A CriticalPathUpgrader object with a completed schedule
        title: Custom title for the chart (default: auto-generated)
        show_security: Whether to show security protocol information on tasks
        show_critical_path: Whether to highlight the critical path
        
    Returns:
        fig, ax: The matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle, Patch
    
    # Get scheduler from the upgrader
    scheduler = critical_path_upgrader.scheduler
    
    # Calculate current critical path and task slack
    task_slack, critical_path = critical_path_upgrader.calculate_task_slack()
    critical_path_tasks = set(critical_path)
    
    # Setup the figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Colors for different tasks and components
    task_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    critical_color = 'firebrick'
    normal_color = 'royalblue'
    security_color = 'gold'
    
    # Get number of processors
    num_processors = len(scheduler.processors)
    makespan = scheduler.makespan
    security_utility = scheduler.calculate_security_utility()
    
    # Auto-generate title if not provided
    if title is None:
        title = f"Security-Enhanced Schedule (A3C-SHIELD)"
    
    display_title = f"{title} (Makespan: {makespan:.2f}, Utility: {security_utility:.2f})"
    
    # Security protocol abbreviations for better readability
    security_protocols = {
        'confidentiality': ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'],
        'integrity': ['I0', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6'],
        'authentication': ['A0', 'A1', 'A2']
    }
    
    # Create task mapping for quick lookup
    task_map = {task.task_id: task for task in scheduler.tasks if task.is_scheduled}
    
    # Create message mapping for quick lookup
    msg_map = {}
    for msg in scheduler.messages:
        if hasattr(msg, 'source_id') and hasattr(msg, 'dest_id'):
            key = f"{msg.source_id}_{msg.dest_id}"
            msg_map[key] = msg
    
    # Legend elements
    legend_elements = [
        Patch(facecolor=critical_color, edgecolor='black', label='Critical Path Task'),
        Patch(facecolor=normal_color, edgecolor='black', label='Normal Task'),
        Patch(facecolor=security_color, edgecolor='black', label='Security Overhead')
    ]
    
    # Get ordered schedule
    # Assuming there's a schedule list in the format similar to what's used in the plotting functions
    schedule = []
    for task in scheduler.tasks:
        if task.is_scheduled:
            # Create schedule entry similar to what's expected by plotting functions
            schedule.append({
                'task_id': task.task_id,
                'name': f"T{task.task_id}",
                'processor': task.assigned_processor,
                'start_time': task.start_time,
                'finish_time': task.finish_time
            })
    
    # Draw each task as a rectangle
    for entry in schedule:
        task_id = entry['task_id']
        processor = entry['processor']
        start = entry['start_time']
        finish = entry['finish_time']
        
        # Check if this task is on the critical path
        is_critical = task_id in critical_path_tasks if show_critical_path else False
        
        task = task_map.get(task_id)
        if not task:
            continue
            
        # Calculate base execution time and security overhead
        base_exec_time = task.execution_times[task.assigned_processor - 1] if hasattr(task, 'execution_times') else finish - start
        total_time = finish - start
        security_overhead = total_time - base_exec_time
        
        # Create a rectangle for the base execution time
        base_color = critical_color if is_critical else normal_color
        rect_base = Rectangle((start, processor-0.4), base_exec_time, 0.8, 
                    color=base_color, alpha=0.9)
        ax.add_patch(rect_base)
        
        # If there's security overhead, add another rectangle for it
        if security_overhead > 0:
            rect_security = Rectangle((start + base_exec_time, processor-0.4), security_overhead, 0.8, 
                        color=security_color, alpha=0.9)
            ax.add_patch(rect_security)
        
        # Add basic task label
        task_label = f"T{task_id}"
        
        # Add slack information if available
        if task_id in task_slack:
            slack_value = task_slack[task_id]
            if slack_value > 0:
                task_label += f" (S:{slack_value:.1f})"
        
        # Check if we have security information to show
        security_info = ""
        
        if show_security and hasattr(task, 'successors'):
            # Collect security levels for outgoing messages
            outgoing_security = []
            
            for succ_id in task.successors:
                msg_key = f"{task_id}_{succ_id}"
                msg = msg_map.get(msg_key)
                
                if msg and hasattr(msg, 'assigned_security'):
                    # Get security protocol names
                    conf_idx = msg.assigned_security.get('confidentiality', 0)
                    integ_idx = msg.assigned_security.get('integrity', 0)
                    auth_idx = msg.assigned_security.get('authentication', 0)
                    
                    conf = security_protocols['confidentiality'][conf_idx] if conf_idx < len(security_protocols['confidentiality']) else 'C?'
                    integ = security_protocols['integrity'][integ_idx] if integ_idx < len(security_protocols['integrity']) else 'I?'
                    auth = security_protocols['authentication'][auth_idx] if auth_idx < len(security_protocols['authentication']) else 'A?'
                    
                    outgoing_security.append(f"→T{succ_id}:{conf}/{integ}/{auth}")
            
            if outgoing_security:
                security_info = "\n" + "\n".join(outgoing_security)
        
        # Add task label with security info
        ax.text(start + (finish-start)/2, processor, f"{task_label}{security_info}", 
                ha='center', va='center', color='black', fontsize=11, weight='bold')
    
    # Highlight critical path with connections if enabled
    if show_critical_path and critical_path:
        cp_connections = []
        for i in range(len(critical_path) - 1):
            src_id = critical_path[i]
            dst_id = critical_path[i+1]
            
            if src_id in task_map and dst_id in task_map:
                src_task = task_map[src_id]
                dst_task = task_map[dst_id]
                
                # Add source and destination points to the connections list
                src_proc = src_task.assigned_processor
                dst_proc = dst_task.assigned_processor
                src_finish = src_task.finish_time
                dst_start = dst_task.start_time
                
                # Draw a line connecting the tasks
                ax.plot([src_finish, dst_start], [src_proc, dst_proc], 'r-', linewidth=2, alpha=0.6)
    
    # Set up the axes
    ax.set_ylim(0.5, num_processors + 0.5)
    ax.set_xlim(0, makespan * 1.05)  # Add some padding
    ax.set_yticks(range(1, num_processors + 1))
    ax.set_yticklabels([f'Processor {i}' for i in range(1, num_processors + 1)])
    ax.set_xlabel('Time (ms)')
    ax.set_title(display_title)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add critical path info in the plot
    if show_critical_path:
        cp_text = f"Critical Path: {' → '.join([f'T{t}' for t in critical_path])}"
        ax.text(0.5, 1.02, cp_text, transform=ax.transAxes, ha='center', fontsize=10)
        
    # Add deadline line if available
    if hasattr(critical_path_upgrader, 'deadline'):
        deadline = critical_path_upgrader.deadline
        ax.axvline(x=deadline, color='red', linestyle='--', linewidth=2)
        ax.text(deadline, num_processors + 0.3, f"Deadline: {deadline:.2f}", 
                color='red', ha='center', va='bottom', fontsize=10)
    
    # Add buffer deadline line if available
    if hasattr(critical_path_upgrader, 'buffer_deadline'):
        buffer_deadline = critical_path_upgrader.buffer_deadline
        ax.axvline(x=buffer_deadline, color='orange', linestyle=':', linewidth=1.5)
        ax.text(buffer_deadline, num_processors + 0.1, f"Buffer: {buffer_deadline:.2f}", 
                color='orange', ha='center', va='bottom', fontsize=9)
    
    # Add legend
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    return fig, ax
class CriticalPathUpgradeScheduler:
    """Class that builds on HSMS to upgrade security on non-critical paths"""
    
    def __init__(self, processors, tasks, messages, security, deadline):
        """
        Initialize the scheduler with the same parameters as HSMS.
        
        Args:
            processors: List of processors
            tasks: List of tasks
            messages: List of messages
            security: Security settings
            deadline: Deadline for the schedule
        """
        self.processors = processors
        self.tasks = tasks
        self.messages = messages
        self.security = security
        self.deadline = deadline
        
        # Initialize the schedule attribute - this was missing
        self.schedule = []
        
        # Other attributes that might be needed
        self.makespan = 0
        
        # Base HSMS scheduler to create initial schedule

        self.hsms_scheduler = SHIELD(
            tasks=copy.deepcopy(tasks),
            messages=copy.deepcopy(messages),
            processors=copy.deepcopy(processors),
            security_service=copy.deepcopy(security),
            deadline=deadline
        )


    def run(self):
        """Run the Critical Path Upgrade Scheduler."""
        # First, run the HSMS scheduler to get a feasible initial schedule
        makespan, security_utility = self.hsms_scheduler.run()
        
        if makespan is None:
            print("HSMS failed to create a feasible schedule within deadline.")
            return None, None
            
        # Copy the schedule from HSMS
        self.schedule = self.hsms_scheduler.schedule.copy()
        self.makespan = makespan
        
        # Now apply the critical path upgrader
        upgrader = CriticalPathUpgrader(self.hsms_scheduler)
        final_makespan, final_utility = upgrader.upgrade_security()
        fig, ax = plot_critical_path_gantt(upgrader, title="A3C-SHIELD Security-Enhanced Schedule")
        plt.show()

        plt.show()
        
        # Update our schedule from the upgraded schedule
        self.schedule = upgrader.scheduler.schedule
        self.makespan = final_makespan
        
        return final_makespan, final_utility
    
    def finalize_schedule(self):
        """Wrapper to call the HSMS finalize_schedule method."""
        return self.hsms_scheduler.finalize_schedule()
    
    def calculate_security_utility(self):
        """Wrapper to call the HSMS calculate_security_utility method."""
        return self.hsms_scheduler.calculate_security_utility()
    
    def print_task_schedule(self):
        """Wrapper to call the HSMS print_task_schedule method."""
        return self.hsms_scheduler.print_task_schedule()
    
    def display_security_information(self, algorithm_name="Critical Path Upgrade"):
        """Wrapper to call the HSMS display_security_information method."""
        return self.hsms_scheduler.display_security_information(algorithm_name)

def main():
    """Run and compare HSMS and SHIELD algorithms."""
    # Set the directory you want to save to
    save_dir = "/home/kannan/Thesis/PICS"

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save figures using the new path

    print("Creating test case...")
    tasks, messages, processors,  security_service, deadline = create_testcase_1()
    
    # Run HSMS first
    print("\nRunning HSMS scheduler...")
    hsms = HSMS(copy.deepcopy(tasks), copy.deepcopy(messages), 
                copy.deepcopy(processors),  security_service, deadline)
    
    hsms_makespan, hsms_security_utility = hsms.run()
    
    if hsms_makespan:
        print(f"HSMS completed successfully: Makespan = {hsms_makespan}, Security Utility = {hsms_security_utility}")
        hsms.display_security_information("HSMS")
        hsms.print_task_schedule()
        fig1,ax=plot_gantt_chart("HSMS Schedule", hsms.schedule, len(processors), hsms_makespan,hsms_security_utility,hsms.messages,hsms.tasks)
        fig2=plot_modified_gantt_chart("HSMS Schedule", hsms.schedule, hsms.tasks, hsms.processors, hsms_makespan)
        
        fig1.savefig(os.path.join(save_dir, "HSMS_GANTT_CHART(4).png"), dpi=300, bbox_inches='tight')
        fig2.savefig(os.path.join(save_dir, "HSMS_SECURITY_CHART(4).png"), dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("HSMS failed to meet deadline requirements")
        return
    
    # Run SHIELD
    print("\nRunning SHIELD algorithm...")
    shield = SHIELD(copy.deepcopy(tasks), copy.deepcopy(messages), 
                    copy.deepcopy(processors),  security_service, deadline)
    
    shield_makespan, shield_security_utility = shield.run()
    
    if shield_makespan:
        print(f"SHIELD completed successfully: Makespan = {shield_makespan}, Security Utility = {shield_security_utility}")
        shield.display_security_information("SHIELD")
        shield.print_task_schedule()
        shield.print_upgrade_history()
        shield.security_improvement_summary()
        fig1,ax=plot_gantt_chart("SHIELD Schedule", shield.schedule, len(processors), shield_makespan,shield_security_utility,shield.messages,shield.tasks)
        fig2=plot_modified_gantt_chart("SHIELD Schedule", shield.schedule, shield.tasks, shield.processors, shield_makespan)
        
        fig1.savefig(os.path.join(save_dir, "SHIELD_GANTT_CHART(4).png"), dpi=300, bbox_inches='tight')
        fig2.savefig(os.path.join(save_dir, "SHIELD_SECURITY_CHART(4).png"), dpi=300, bbox_inches='tight')
        
        plt.show()
        # Compare results
        
        
        
        # Add more metrics if needed
        print("="*100)
    else:
        print("SHIELD failed to complete")

    # if a3c_shield_makespan:
    #     print(f"A3C-SHIELD completed successfully: Makespan = {a3c_shield_makespan}, Security Utility = {a3c_shield_security_utility}")
    #     # a3c_shield.display_security_information("A3C-SHIELD")
    #     # a3c_shield.print_task_schedule()
    #     fig1, ax1 = plot_gantt_chart("A3C-SHIELD Schedule", a3c_shield.schedule, len(processors), a3c_shield_makespan,a3c_shield_security_utility,a3c_shield.messages,a3c_shield.tasks)
    #     fig2 = plot_modified_gantt_chart("A3C-SHIELD Schedule", a3c_shield.schedule, a3c_shield.tasks, a3c_shield.processors, a3c_shield_makespan)
        
    #     fig1.savefig(os.path.join(save_dir, "A3C_SHIELD_GANTT_CHART(4).png"), dpi=300, bbox_inches='tight')
    #     fig2.savefig(os.path.join(save_dir, "A3C_SHIELD_SECURITY_CHART(4).png"), dpi=300, bbox_inches='tight')
        
    #     plt.show()
    # print("Running GA Scheduler...")
    ga_shield = GASecurityScheduler(copy.deepcopy(tasks), copy.deepcopy(messages),
                    copy.deepcopy(processors),  security_service, deadline,50,100,20,0.1,0.8,3,hsms,shield,15)
    
    # Run phase 1 (task scheduling)
    if ga_shield.run_phase1():
        # Run phase 2 (security enhancement)
        ga_shield.run_phase2()
        
        # Create visualization plots
        fig1, ax1 = ga_shield.plot_ga_gantt_chart()
        fig2, ax2 = ga_shield.plot_modified_ga_gantt_chart()

        
        # Show plots
        plt.show()
        
        # Save plots if needed
        fig1.savefig(os.path.join(save_dir, "GA_GANTT_CHART(4).png"), dpi=300, bbox_inches='tight')
        fig2.savefig(os.path.join(save_dir, "GA_SECURITY_CHART(4).png"), dpi=300, bbox_inches='tight')

    else:
        print("GA scheduling failed")
    
    
    # Run QL-SHIELD
    output_dir = '/home/kannan/Thesis/PIPE/log.txt'  # or '/absolute/path/to/logs'
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    output_path = os.path.join(output_dir, 'output.txt')

    # with open(output_path, 'w') as f:
    #     sys.stdout = f
    #     sys.stderr = f
    ql_shield = QLShield(copy.deepcopy(tasks), copy.deepcopy(messages),
                    copy.deepcopy(processors), security_service, deadline)
    # Optional: Reset stdout/stderr
    # sys.stdout = sys.__stdout__
    # sys.stderr = sys.__stderr__
    ql_shield_makespan, ql_shield_security_utility = ql_shield.run()
    if ql_shield_makespan:
        print(f"QL-SHIELD completed successfully: Makespan = {ql_shield_makespan}, Security Utility = {ql_shield_security_utility}")
        ql_shield.display_security_information("QL-SHIELD")
        ql_shield.print_task_schedule()
        fig1, ax1 = plot_gantt_chart("QL-SHIELD Schedule", ql_shield.schedule, len(processors), ql_shield_makespan,ql_shield_security_utility,ql_shield.messages,ql_shield.tasks)
        fig2 = plot_modified_gantt_chart("QL-SHIELD Schedule", ql_shield.schedule, ql_shield.tasks, ql_shield.processors, ql_shield_makespan)
        
        fig1.savefig(os.path.join(save_dir, "QL_SHIELD_GANTT_CHART(4).png"), dpi=300, bbox_inches='tight')
        fig2.savefig(os.path.join(save_dir, "QL_SHIELD_SECURITY_CHART(4).png"), dpi=300, bbox_inches='tight')
        
        plt.show()
    else:
        print("QL-SHIELD failed to meet deadline requirements")
    
    a3c_shield = CriticalPathUpgradeScheduler(copy.deepcopy(processors), copy.deepcopy(tasks), copy.deepcopy(messages),
                    security_service, deadline)
    a3c_shield_makespan, a3c_shield_security_utility = a3c_shield.run()
    
    
    rf_shield = RFSHIELD(copy.deepcopy(tasks), copy.deepcopy(messages),
                    copy.deepcopy(processors), security_service, deadline)
    rf_shield_makespan, rf_shield_security_utility = rf_shield.run()
    
    if rf_shield_makespan:
        print(f"RF-SHIELD completed successfully: Makespan = {rf_shield_makespan}, Security Utility = {rf_shield_security_utility}")
        rf_shield.display_security_information("RF-SHIELD")
        rf_shield.print_task_schedule()
        fig1, ax1 = plot_gantt_chart("RF-SHIELD Schedule", rf_shield.schedule, len(processors), rf_shield_makespan,rf_shield_security_utility,rf_shield.messages,rf_shield.tasks)
        fig2 = plot_modified_gantt_chart("RF-SHIELD Schedule", rf_shield.schedule, rf_shield.tasks, rf_shield.processors, rf_shield_makespan)
        
        fig1.savefig(os.path.join(save_dir, "RF_SHIELD_GANTT_CHART(4).png"), dpi=300, bbox_inches='tight')
        fig2.savefig(os.path.join(save_dir, "RF_SHIELD_SECURITY_CHART(4).png"), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    rf_imp_shield = ImprovedRFSHIELD(copy.deepcopy(tasks), copy.deepcopy(messages),
                    copy.deepcopy(processors), security_service, deadline)
    rf_imp_shield_makespan, rf_imp_shield_security_utility = rf_imp_shield.run()
    if rf_imp_shield_makespan:
        print(f"RF-IMPROVED-SHIELD completed successfully: Makespan = {rf_imp_shield_makespan}, Security Utility = {rf_imp_shield_security_utility}")
        rf_imp_shield.display_security_information("RF-IMPROVED-SHIELD")
        rf_imp_shield.print_task_schedule()
        fig1, ax1 = plot_gantt_chart("RF-IMPROVED-SHIELD Schedule", rf_imp_shield.schedule, len(processors), rf_imp_shield_makespan,rf_imp_shield_security_utility,rf_imp_shield.messages,rf_imp_shield.tasks)
        fig2 = plot_modified_gantt_chart("RF-IMPROVED-SHIELD Schedule", rf_imp_shield.schedule, rf_imp_shield.tasks, rf_imp_shield.processors, rf_imp_shield_makespan)
        
        fig1.savefig(os.path.join(save_dir, "RF_IMPROVED_SHIELD_GANTT_CHART(4).png"), dpi=300, bbox_inches='tight')
        fig2.savefig(os.path.join(save_dir, "RF_IMPROVED_SHIELD_SECURITY_CHART(4).png"), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    # Run HSMS
    
    # Show improvement percentages
    if shield_makespan is not None and hsms_makespan is not None:
        print("\nImprovement Metrics:")
        
        # Security utility improvement
        shield_security_imp = ((ql_shield_security_utility - shield_security_utility) / shield_security_utility) * 100 if shield_security_utility > 0 else float('inf')
        hsms_security_imp = ((ql_shield_security_utility - hsms_security_utility) / hsms_security_utility) * 100 if hsms_security_utility > 0 else float('inf')
        
        # Makespan improvement (lower is better)
        shield_makespan_imp = ((shield_makespan - ql_shield_makespan) / shield_makespan) * 100
        hsms_makespan_imp = ((hsms_makespan - ql_shield_makespan) / hsms_makespan) * 100
        
        print(f"vs SHIELD - Security: +{shield_security_imp:.2f}%, Makespan: -{shield_makespan_imp:.2f}%")
        print(f"vs HSMS - Security: +{hsms_security_imp:.2f}%, Makespan: -{hsms_makespan_imp:.2f}%")
if __name__ == "__main__":
    main()
