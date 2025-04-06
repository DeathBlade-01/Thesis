import numpy as np
from collections import defaultdict, deque
import heapq
import math


NUM_PROCESSES = 12  # Number of processes in the DAG
NUM_CORES = 1      # Modified to use only one core
CHIP_TDP = 9    # TDP limit for the entire chip in watts
CLOCK = 1        # Clock cycle
DEADLINE = 9     # Deadline in seconds


ADJACENCY_MATRIX = np.array([
    # Process 0  1  2  3  4  5  6  7  8  9  10  11  12
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Process 0 depends on...
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Process 1 depends on...
    [0, 1, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0],  # Process 2 depends on...
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Process 3 depends on...
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Process 4 depends on...
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#5
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],#6
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],#7
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],#8
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],#9
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],#10
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],#11
])

# Format: [core][process] = [power_consumption, execution_time]
POWER_EXEC_MATRIX = np.array([
    # Only one core now
    [
        [0 , 0],
        [3, 5],   # Process 0: 8W, 18 time units
        [2.5, 6],  # Process 1: 10W, 12 time units
        [3, 5],  # Process 2: 12W, 25 time units
        [3.5, 3],   # Process 3: 7W, 20 time units
        [4.5, 1],
        [4, 2],
        [3.5, 3],
        [4.5, 1],
        [4, 2],
        [3.5, 3],
        [0, 0],
    ],
])

CORE_TDP_LIMITS = np.array([9])  # Updated TDP limit for single core

def find_predecessors(process, adj_matrix):
    predecessors = []
    for i in range(adj_matrix.shape[0]):
        if adj_matrix[process, i] == 1:
            predecessors.append(i)
    return predecessors

# Calculate the earliest start time based on predecessors
def calculate_earliest_start_time(process, adj_matrix, process_finish_times):
    predecessors = find_predecessors(process, adj_matrix)
    if not predecessors:
        return 0
    
    max_finish_time = 0
    for pred in predecessors:
        if pred in process_finish_times and process_finish_times[pred] > max_finish_time:
            max_finish_time = process_finish_times[pred]
    
    return max_finish_time

# Check if all predecessors of a process are completed
def are_predecessors_completed(process, adj_matrix, completed_processes):
    predecessors = find_predecessors(process, adj_matrix)
    return all(pred in completed_processes for pred in predecessors)

# Calculate the priority of a process (critical path length)
def calculate_process_priority(process, adj_matrix, power_exec_matrix):
    # Start a DFS from this process to find the critical path
    visited = set()
    priority = {}
    
    def count_successors(node):
        # Count number of direct successors
        successors = sum(1 for i in range(adj_matrix.shape[0]) if adj_matrix[i, node] == 1)
        return max(1, successors)
    
    def dfs(node):
        if node in visited:
            return priority[node]
        
        visited.add(node)
        
        # Find all processes that depend on this one (successors)
        successors = []
        for i in range(adj_matrix.shape[0]):
            if adj_matrix[i, node] == 1:
                successors.append(i)
        
        if not successors:
            # Use the average execution time across all cores as the base priority
            priority[node] = np.mean(power_exec_matrix[0, node, 1])
        else:
            # Find the maximum priority path
            max_path = 0
            for succ in successors:
                path_length = dfs(succ)
                if path_length > max_path:
                    max_path = path_length
            
            # Priority is its own execution time plus the maximum path of its successors
            priority[node] = np.mean(power_exec_matrix[0, node, 1]) + max_path
            
            # Multiply by number of successors to ensure multiple of children
            priority[node] *= count_successors(node)
        
        return priority[node]
    
    return dfs(process)

# Main scheduling function
def schedule_processes_with_power_constraints(adj_matrix, power_exec_matrix, core_tdp_limits, chip_tdp, clock, deadline):
    num_processes = adj_matrix.shape[0]
    
    # Ready queue of processes (those with all predecessors completed)
    ready_queue = []
    
    # Track which processes are completed
    completed_processes = set()
    
    # Track finish time of each process
    process_finish_times = {}
    
    # Track power usage over time for chip
    max_time = deadline
    chip_power_usage = np.zeros(max_time)
    
    # Track which process is assigned to which core
    process_core_assignment = {}
    
    # Track the start and end times of each process
    process_schedule = {}
    
    # Intervals for process addition
    process_intervals = [i * (deadline // num_processes) for i in range(num_processes + 1)]
    
    # Initialize the ready queue with processes that have no predecessors
    for process in range(num_processes):
        if not find_predecessors(process, adj_matrix):
            # Calculate process priority for sorting
            priority = calculate_process_priority(process, adj_matrix, power_exec_matrix)
            # Use negative priority because heapq is a min-heap and we want highest priority first
            heapq.heappush(ready_queue, (-priority, process, 0))  # Added initial time
    
    current_time = 0
    last_scheduled_time = 0
    
    # Main scheduling loop
    while len(completed_processes) < num_processes:
        # Add new processes at specified intervals
        for interval_time in process_intervals:
            if current_time >= interval_time and current_time < interval_time + clock:
                for process in range(num_processes):
                    if process not in completed_processes and process not in [p for _, p, _ in ready_queue]:
                        if are_predecessors_completed(process, adj_matrix, completed_processes):
                            priority = calculate_process_priority(process, adj_matrix, power_exec_matrix)
                            heapq.heappush(ready_queue, (-priority, process, current_time))
        
        # Check if any process can be scheduled now
        if ready_queue:
            # Get the highest priority process
            _, process, arrival_time = heapq.heappop(ready_queue)
            
            # Calculate earliest possible start time based on predecessors
            earliest_start = max(calculate_earliest_start_time(process, adj_matrix, process_finish_times), current_time)
            
            # Use the single core (index 0)
            core = 0
            
            # Get execution time and power
            exec_time = int(power_exec_matrix[core, process, 1])
            process_power = power_exec_matrix[core, process, 0]
            
            # Ensure we don't exceed the deadline
            if earliest_start + exec_time > deadline:
                continue
            
            # Check power constraints
            valid_slot_found = True
            for t in range(earliest_start, earliest_start + exec_time):
                # Check chip TDP constraint
                if chip_power_usage[t] + process_power > chip_tdp:
                    valid_slot_found = False
                    break
            
            # If we found a valid schedule
            if valid_slot_found:
                # Update power usage
                for t in range(earliest_start, earliest_start + exec_time):
                    chip_power_usage[t] += process_power
                
                # Update process information
                process_finish_times[process] = earliest_start + exec_time
                process_core_assignment[process] = core
                process_schedule[process] = (earliest_start, earliest_start + exec_time)
                completed_processes.add(process)
                last_scheduled_time = max(last_scheduled_time, earliest_start + exec_time)
                
                # Add new processes to the ready queue
                for next_process in range(num_processes):
                    # Check if this process depends on the one we just completed
                    if adj_matrix[next_process, process] == 1 and next_process not in completed_processes:
                        # Check if all its predecessors are now completed
                        if are_predecessors_completed(next_process, adj_matrix, completed_processes):
                            # Calculate priority
                            priority = calculate_process_priority(next_process, adj_matrix, power_exec_matrix)
                            heapq.heappush(ready_queue, (-priority, next_process, current_time))
            else:
                # Couldn't schedule this process now, try again later
                priority = calculate_process_priority(process, adj_matrix, power_exec_matrix)
                heapq.heappush(ready_queue, (-priority, process, current_time))
        
        # Increment time
        current_time += clock
        
        # Break if we've exceeded the deadline
        if current_time > deadline:
            break
    
    # Makespan is either the last scheduled time or the deadline
    makespan = min(last_scheduled_time, deadline)
    
    return process_schedule, process_core_assignment, makespan

def main():
    print("Power-Aware DAG Scheduling for Single Core")
    print(f"Number of processes: {NUM_PROCESSES}")
    print(f"Number of cores: {NUM_CORES}")
    print(f"Chip TDP: {CHIP_TDP}W")
    print(f"Deadline: {DEADLINE} seconds")
    print(f"Clock: {CLOCK} second")
    
    # Print the process intervals
    process_intervals = [i * (DEADLINE // NUM_PROCESSES) for i in range(NUM_PROCESSES + 1)]
    print("\nProcess Addition Intervals:")
    for i, interval in enumerate(process_intervals):
        print(f"  Interval {i}: {interval} seconds")
    
    # Schedule the processes
    process_schedule, process_core_assignment, makespan = schedule_processes_with_power_constraints(
        ADJACENCY_MATRIX, POWER_EXEC_MATRIX, CORE_TDP_LIMITS, CHIP_TDP, CLOCK, DEADLINE)
    
    # Print the schedule
    print("\nSchedule:")
    sorted_processes = sorted(range(NUM_PROCESSES), 
                             key=lambda p: process_schedule[p][0] if p in process_schedule else float('inf'))
    
    for process in sorted_processes:
        if process in process_schedule:
            start_time, finish_time = process_schedule[process]
            core = process_core_assignment[process]
            actual_time = finish_time - start_time
            print(f"Process {process}: Core {core}, Start={start_time}, Finish={finish_time}, Duration={actual_time}")
    
    print(f"\nTotal execution time (makespan): {makespan}")

if __name__ == "__main__":
    main()