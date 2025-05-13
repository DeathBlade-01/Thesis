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

import networkx as nx
import matplotlib.pyplot as plt

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
    
def create_random_dag(n_tasks, max_predecessors=3):
    """
    Generate a random DAG with specified number of tasks.
    
    Args:
    n_tasks (int): Number of tasks to generate
    max_predecessors (int): Maximum number of predecessors for a task
    
    Returns:
    tuple: (tasks, messages, processors, security_service, deadline)
    """
    # Generate tasks
    randomforgeneration = random.Random()
    tasks = []
    for i in range(1, n_tasks + 1):
        # Random execution times (3 different times)
        exec_times = [randomforgeneration.randint(30, 90) for _ in range(3)]
        
        # Determine predecessors
        possible_predecessors = list(range(1, i))
        n_preds = randomforgeneration.randint(0, min(max_predecessors, len(possible_predecessors)))
        predecessors = randomforgeneration.sample(possible_predecessors, n_preds) if n_preds > 0 else []
        
        # Create task
        task = Task(i, f'T{i}', exec_times, predecessors)
        tasks.append(task)
    
    # Generate messages between tasks with predecessors
    messages = []
    message_id = 1
    for task in tasks:
        for pred in task.predecessors:
            # randomforgeneration message size between 10 and 50 KB
            size = randomforgeneration.randint(10, 50)
            message = Message(pred, task.task_id, size)
            
            # Generate randomforgeneration security requirements
            conf_min = round(randomforgeneration.uniform(0.1, 0.5), 1)
            integ_min = round(randomforgeneration.uniform(0.1, 0.5), 1)
            auth_min = round(randomforgeneration.uniform(0.1, 0.5), 1)
            
            conf_w = round(randomforgeneration.uniform(0.1, 0.7), 1)
            integ_w = round(randomforgeneration.uniform(0.1, 0.7), 1)
            auth_w = round(randomforgeneration.uniform(0.1, 0.7), 1)
            
            message.set_security_requirements(conf_min, integ_min, auth_min, conf_w, integ_w, auth_w)
            messages.append(message)
            message_id += 1
    
    # Always 3 processors with randomforgeneration bandwidths
    processors = [
        Processor(1, [0, randomforgeneration.randint(1, 3), randomforgeneration.randint(1, 3)]),
        Processor(2, [randomforgeneration.randint(1, 3), 0, randomforgeneration.randint(1, 3)]),
        Processor(3, [randomforgeneration.randint(1, 3), randomforgeneration.randint(1, 3), 0])
    ]
    
    # Create security service
    security_service = SecurityService()
    
    # Calculate deadline (sum of max exec times + some buffer)
    max_exec_time = max(max(task.execution_times) for task in tasks)
    deadline = max_exec_time * n_tasks * 2
    
    return tasks, messages, processors, security_service, deadline

def visualize_dag(tasks, messages=None):
    """
    Visualize the Directed Acyclic Graph (DAG) of tasks with a hierarchical layout.
    
    Args:
    tasks (list): List of Task objects
    messages (list, optional): List of Message objects to show message sizes
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes (tasks)
    for task in tasks:
        G.add_node(task.task_id)
    
    # Add edges (dependencies)
    edge_labels = {}
    for task in tasks:
        for pred in task.predecessors:
            G.add_edge(pred, task.task_id)
            
            # Find message size if messages are provided
            if messages:
                matching_messages = [msg for msg in messages 
                                     if msg.source_id == pred and msg.dest_id == task.task_id]
                if matching_messages:
                    edge_labels[(pred, task.task_id)] = str(matching_messages[0].size)
    
    # Create a hierarchical layout
    def assign_layer(graph):
        # Topological sort to determine layers
        layers = {}
        for node in nx.topological_sort(graph):
            # Find the maximum layer of predecessors and add 1
            pred_layers = [layers.get(pred, -1) for pred in graph.predecessors(node)]
            layers[node] = max(pred_layers) + 1 if pred_layers else 0
        return layers
    
    # Compute layers
    layers = assign_layer(G)
    
    # Create position dictionary
    pos = {}
    layer_counts = {}
    
    # Assign positions based on layers
    for node, layer in layers.items():
        if layer not in layer_counts:
            layer_counts[layer] = 0
        
        # Spread nodes horizontally within their layer
        pos[node] = (layer_counts[layer], -layer)
        layer_counts[layer] += 1
    
    # Normalize x-coordinates
    max_nodes_in_layer = max(layer_counts.values())
    for node in pos:
        pos[node] = (pos[node][0] / (max_nodes_in_layer - 1) * 10, pos[node][1])
    
    # Draw the graph
    plt.figure(figsize=(10, 6))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                            node_size=700, 
                            edgecolors='black')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, 
                            edge_color='gray', 
                            arrows=True, 
                            arrowsize=20)
    
    # Node labels (task names)
    nx.draw_networkx_labels(G, pos, 
                             labels={node: f'T{node}' for node in G.nodes()}, 
                             font_weight='bold')
    
    # Edge labels (message sizes)
    if messages:
        nx.draw_networkx_edge_labels(G, pos, 
                                      edge_labels=edge_labels, 
                                      font_size=8)
    
    plt.title("Task Dependency Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_dag_with_times(tasks, messages=None):
    """
    Visualize the Directed Acyclic Graph (DAG) of tasks with execution times and hierarchical layout.
    
    Args:
    tasks (list): List of Task objects
    messages (list, optional): List of Message objects to show message sizes
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes (tasks)
    for task in tasks:
        G.add_node(task.task_id, label=f'T{task.task_id}\n{task.execution_times}')
    
    # Add edges (dependencies)
    edge_labels = {}
    for task in tasks:
        for pred in task.predecessors:
            G.add_edge(pred, task.task_id)
            
            # Find message size if messages are provided
            if messages:
                matching_messages = [msg for msg in messages 
                                     if msg.source_id == pred and msg.dest_id == task.task_id]
                if matching_messages:
                    edge_labels[(pred, task.task_id)] = str(matching_messages[0].size)
    
    # Create a hierarchical layout
    def assign_layer(graph):
        # Topological sort to determine layers
        layers = {}
        for node in nx.topological_sort(graph):
            # Find the maximum layer of predecessors and add 1
            pred_layers = [layers.get(pred, -1) for pred in graph.predecessors(node)]
            layers[node] = max(pred_layers) + 1 if pred_layers else 0
        return layers
    
    # Compute layers
    layers = assign_layer(G)
    
    # Create position dictionary
    pos = {}
    layer_counts = {}
    
    # Assign positions based on layers
    for node, layer in layers.items():
        if layer not in layer_counts:
            layer_counts[layer] = 0
        
        # Spread nodes horizontally within their layer
        pos[node] = (layer_counts[layer], -layer)
        layer_counts[layer] += 1
    
    # Normalize x-coordinates
    max_nodes_in_layer = max(layer_counts.values())
    for node in pos:
        pos[node] = (pos[node][0] / (max_nodes_in_layer - 1) * 10, pos[node][1])
    
    # Draw the graph
    plt.figure(figsize=(10, 6))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                            node_size=700, 
                            edgecolors='black')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, 
                            edge_color='gray', 
                            arrows=True, 
                            arrowsize=20)
    
    # Node labels (task names and execution times)
    nx.draw_networkx_labels(G, pos, 
                             labels={node: G.nodes[node]['label'] for node in G.nodes()}, 
                             font_weight='bold', 
                             font_size=8)
    
    # Edge labels (message sizes)
    if messages:
        nx.draw_networkx_edge_labels(G, pos, 
                                      edge_labels=edge_labels, 
                                      font_size=8)
    
    plt.title("Task Dependency Graph with Execution Times")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    
def main():
    """Run and compare HSMS and SHIELD algorithms."""
    # Set the directory you want to save to
    save_dir = "/home/kannan/Thesis/PICS"

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save figures using the new path

    print("Creating test case...")
    tasks, messages, processors,  security_service, deadline = create_testcase()
    visualize_dag(tasks,messages)
    print(f"Test case created with deadline: {deadline}")
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
        
    tasks, messages, processors, security_service, deadline = create_random_dag(5)
    
    # Print task details
    print("Tasks:")
    for task in tasks:
        print(f"Task {task.task_id}: Exec Times {task.execution_times}, Predecessors {task.predecessors}")
    
    print("\nMessages:")
    for msg in messages:
        print(f"From {msg.source_id} to {msg.dest_id}, Size: {msg.size} KB")
    
    print(f"\nDeadline: {deadline}")
    
    # Visualize the DAG
    visualize_dag(tasks,messages)
    
    
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

if __name__ == "__main__":
    main()