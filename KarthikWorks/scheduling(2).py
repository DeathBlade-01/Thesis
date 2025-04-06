import numpy as np
from collections import defaultdict, deque
import heapq
import math


NUM_PROCESSES = 12  # Number of processes in the DAG
NUM_CORES = 3      # Number of cores in the system
CHIP_TDP = 9    # TDP limit for the entire chip in watts


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
    # Core 0
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
    # Core 1
    [
        [0 , 0],
        [2, 7],   # Process 0: 8W, 18 time units
        [4, 3],  # Process 1: 10W, 12 time units
        [4, 3],  # Process 2: 12W, 25 time units
        [5, 2.5],   # Process 3: 7W, 20 time units
        [3.5, 2],
        [3.5, 2],
        [3, 4],
        [3.5, 2],
        [3.5, 2],
        [3, 4],
        [0, 0],
    ],
    # Core 2
    [
        [0 , 0],
        [1.5, 8],   # Process 0: 8W, 18 time units
        [5, 2.75],  # Process 1: 10W, 12 time units
        [2, 3],  # Process 2: 12W, 25 time units
        [2, 6],   # Process 3: 7W, 20 time units
        [2.5, 4],
        [2, 6],
        [2.75, 3],
        [2.5, 4],
        [2, 6],
        [2.75, 3],
        [0, 0],
    ],
])


CORE_TDP_LIMITS = np.array([4.5,3.5,3])  # TDP limits for each core in watts


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
            # Use the average execution time across all cores as the priority
            priority[node] = np.mean([power_exec_matrix[core, node, 1] for core in range(power_exec_matrix.shape[0])])
        else:
            # Find the maximum priority path
            max_path = 0
            for succ in successors:
                path_length = dfs(succ)
                if path_length > max_path:
                    max_path = path_length
            
            # Priority is its own execution time plus the maximum path of its successors
            priority[node] = np.mean([power_exec_matrix[core, node, 1] for core in range(power_exec_matrix.shape[0])]) + max_path
            priority[node] *= max(1, len(successors))
        
        return priority[node]
    
    return dfs(process)

# Main scheduling function
def schedule_processes_with_power_constraints(adj_matrix, power_exec_matrix, core_tdp_limits, chip_tdp):
    num_processes = adj_matrix.shape[0]
    num_cores = power_exec_matrix.shape[0]
    
    # Ready queue of processes (those with all predecessors completed)
    ready_queue = []
    
    # Track which processes are completed
    completed_processes = set()
    
    # Track finish time of each process
    process_finish_times = {}
    
    # Track power usage over time for each core and the entire chip
    max_time = int(np.sum(power_exec_matrix[:, :, 1].max(axis=0)) * 2)  # Conservative estimate
    core_power_usage = np.zeros((num_cores, max_time))
    chip_power_usage = np.zeros(max_time)
    
    # Track which process is assigned to which core
    process_core_assignment = {}
    
    # Track the start and end times of each process
    process_schedule = {}
    
    # Initialize the ready queue with processes that have no predecessors
    for process in range(num_processes):
        if not find_predecessors(process, adj_matrix):
            # Calculate process priority for sorting
            priority = calculate_process_priority(process, adj_matrix, power_exec_matrix)
            # Use negative priority because heapq is a min-heap and we want highest priority first
            heapq.heappush(ready_queue, (-priority, process))
    
    current_time = 0
    
    # Main scheduling loop
    while len(completed_processes) < num_processes:
        # Check if any process can be scheduled now
        if ready_queue:
            # Get the highest priority process
            _, process = heapq.heappop(ready_queue)
            
            # Find the best core for this process (minimum execution time)
            best_core = None
            best_start_time = float('inf')
            best_finish_time = float('inf')
            
            # Sort cores by execution time for this specific process
            cores_sorted = sorted(range(num_cores), 
                                 key=lambda c: power_exec_matrix[c, process, 1])
            
            for core in cores_sorted:
                # Skip this core if the process's power requirement exceeds core TDP limit
                process_power = power_exec_matrix[core, process, 0]
                if process_power > core_tdp_limits[core]:
                    continue
                
                # Calculate earliest possible start time based on predecessors
                earliest_start = calculate_earliest_start_time(process, adj_matrix, process_finish_times)
                
                # Find a valid time slot that satisfies power constraints
                start_time = earliest_start
                exec_time = int(power_exec_matrix[core, process, 1])
                
                # Check if power requirements are met
                valid_slot_found = False
                while start_time + exec_time < max_time and not valid_slot_found:
                    valid_slot_found = True
                    
                    # Check power constraints for each time unit
                    for t in range(start_time, start_time + exec_time):
                        # Check chip TDP constraint
                        if chip_power_usage[t] + process_power > chip_tdp:
                            valid_slot_found = False
                            start_time += 1
                            break
                    
                    # If we found a valid slot and it's better than our current best
                    if valid_slot_found and start_time < best_start_time:
                        best_core = core
                        best_start_time = start_time
                        best_finish_time = start_time + exec_time
                        # We found a good slot, but let's continue checking other cores
                        # to see if there's an even better option
                        break
                
            # If we found a valid schedule
            if best_core is not None:
                # Update power usage
                process_power = power_exec_matrix[best_core, process, 0]
                for t in range(best_start_time, best_finish_time):
                    core_power_usage[best_core, t] += process_power
                    chip_power_usage[t] += process_power
                
                # Update process information
                process_finish_times[process] = best_finish_time
                process_core_assignment[process] = best_core
                process_schedule[process] = (best_start_time, best_finish_time)
                completed_processes.add(process)
                
                # Add new processes to the ready queue
                for next_process in range(num_processes):
                    # Check if this process depends on the one we just completed
                    if adj_matrix[next_process, process] == 1 and next_process not in completed_processes:
                        # Check if all its predecessors are now completed
                        if are_predecessors_completed(next_process, adj_matrix, completed_processes):
                            # Calculate priority
                            priority = calculate_process_priority(next_process, adj_matrix, power_exec_matrix)
                            heapq.heappush(ready_queue, (-priority, next_process))
            else:
                # Couldn't schedule this process now, try again later
                current_time += 1
                priority = calculate_process_priority(process, adj_matrix, power_exec_matrix)
                heapq.heappush(ready_queue, (-priority, process))
        else:
            # No processes in ready queue, increment time and check if any new processes can be added
            current_time += 1
            
            # Check if any processes can be added to ready queue
            for process in range(num_processes):
                if process not in completed_processes and process not in [p for _, p in ready_queue]:
                    if are_predecessors_completed(process, adj_matrix, completed_processes):
                        priority = calculate_process_priority(process, adj_matrix, power_exec_matrix)
                        heapq.heappush(ready_queue, (-priority, process))
    
    makespan = max(finish_time for _, finish_time in process_schedule.values())
    
    return process_schedule, process_core_assignment, makespan

def main():
    print("Power-Aware DAG Scheduling for Heterogeneous Cores")
    print(f"Number of processes: {NUM_PROCESSES}")
    print(f"Number of cores: {NUM_CORES}")
    print(f"Chip TDP: {CHIP_TDP}W")
    
    if ADJACENCY_MATRIX.shape[0] != NUM_PROCESSES or ADJACENCY_MATRIX.shape[1] != NUM_PROCESSES:
        print("Error: Adjacency matrix dimensions do not match number of processes")
        return
    
    if POWER_EXEC_MATRIX.shape[0] != NUM_CORES or POWER_EXEC_MATRIX.shape[1] != NUM_PROCESSES:
        print("Error: Power execution matrix dimensions do not match number of cores/processes")
        return
    
    if CORE_TDP_LIMITS.shape[0] != NUM_CORES:
        print("Error: Core TDP limits array length does not match number of cores")
        return
    
    # Print the DAG adjacency matrix
    print("\nDAG Adjacency Matrix (row i depends on column j):")
    print(ADJACENCY_MATRIX)
    
    # Print the power and execution time matrix
    print("\nPower and Execution Time Matrix:")
    for core in range(NUM_CORES):
        print(f"Core {core}:")
        for process in range(NUM_PROCESSES):
            power = POWER_EXEC_MATRIX[core, process, 0]
            time = POWER_EXEC_MATRIX[core, process, 1]
            print(f"  Process {process}: Power={power:.2f}W, Time={time:.2f}")
    
    # Print core TDP limits
    print("\nCore TDP Limits:")
    for core in range(NUM_CORES):
        print(f"  Core {core}: {CORE_TDP_LIMITS[core]:.2f}W")
    
    # Schedule the processes
    process_schedule, process_core_assignment, makespan = schedule_processes_with_power_constraints(
        ADJACENCY_MATRIX, POWER_EXEC_MATRIX, CORE_TDP_LIMITS, CHIP_TDP)
    
    # Print the schedule
    print("\nSchedule:")
    sorted_processes = sorted(range(NUM_PROCESSES), 
                             key=lambda p: process_schedule[p][0])
    
    for process in sorted_processes:
        start_time, finish_time = process_schedule[process]
        core = process_core_assignment[process]
        actual_time = finish_time - start_time
        print(f"Process {process}: Core {core}, Start={start_time}, Finish={finish_time}, Duration={actual_time}")
    
    print(f"\nTotal execution time (makespan): {makespan}")

if __name__ == "__main__":
    main()