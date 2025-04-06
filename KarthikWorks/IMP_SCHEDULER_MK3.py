import numpy as np
from collections import defaultdict
import heapq
import math

class DAGScheduler:
    def __init__(self, num_processes, num_cores, chip_tdp, core_tdp_limits, 
                 adjacency_matrix, power_exec_matrix, deadline=9, clock=1):
        """
        Initialize the scheduler with system parameters
        
        Additional Parameters:
        - deadline (int): Maximum time allowed for scheduling all tasks
        - clock (int): Time increment for scheduling steps
        """
        self.num_processes = num_processes
        self.num_cores = num_cores
        self.chip_tdp = chip_tdp
        self.core_tdp_limits = core_tdp_limits
        self.adjacency_matrix = adjacency_matrix
        self.power_exec_matrix = power_exec_matrix
        self.deadline = deadline
        self.clock = clock
        
        # Track hyperperiod and repetitions
        self.hyperperiod = min(deadline, self._calculate_hyperperiod())
        
        # Initialize task tracking
        self.all_tasks = self._generate_task_instances()
        
        # Initialize scheduling data structures
        self.schedule = [[] for _ in range(num_cores)]
        self.free_slots = {c: [(0, self.deadline)] for c in range(num_cores)}
        self.power_profile_list = np.zeros(self.deadline + 1)  # Expanded to match deadline
    
    def _calculate_hyperperiod(self, base_deadline=100):
        """
        Calculate the hyperperiod for task scheduling
        
        Args:
            base_deadline (int): Base deadline for task repetitions
        
        Returns:
            int: Least Common Multiple serving as hyperperiod
        """
        return min(base_deadline, self.deadline)
    
    def _generate_task_instances(self, repeat_count=1):
        """
        Generate task instances considering the deadline
        
        Args:
            repeat_count (int): Number of times to repeat tasks
        
        Returns:
            list: Task instances with metadata
        """
        all_tasks = []
        interval = max(1, self.deadline // self.num_processes)
        
        for process in range(self.num_processes):
            for k in range(repeat_count):
                task_start = min(k * interval * process, self.deadline)
                task_deadline = min(task_start + interval, self.deadline)
                
                all_tasks.append({
                    'process': process,
                    'instance': k,
                    'start_time': task_start,
                    'deadline': task_deadline
                })
        
        return sorted(all_tasks, key=lambda x: x['start_time'])
    
    def _find_predecessors(self, process):
        """Find predecessors for a given process"""
        return [i for i in range(self.adjacency_matrix.shape[0]) 
                if self.adjacency_matrix[process, i] == 1]
    
    def _calculate_process_priority(self, process):
        """
        Calculate process priority using critical path length
        
        Args:
            process (int): Process index
        
        Returns:
            float: Priority score
        """
        visited = set()
        priority_cache = {}
        
        def dfs(node):
            if node in visited:
                return priority_cache[node]
            
            visited.add(node)
            
            # Find successors
            successors = [i for i in range(self.adjacency_matrix.shape[0]) 
                          if self.adjacency_matrix[i, node] == 1]
            
            if not successors:
                # Base case: use average execution time
                priority = np.mean([self.power_exec_matrix[core, node, 1] 
                                    for core in range(self.num_cores)])
            else:
                # Find maximum path length
                max_path = max([dfs(succ) for succ in successors], default=0)
                
                # Priority includes own execution time and max successor path
                priority = (np.mean([self.power_exec_matrix[core, node, 1] 
                                     for core in range(self.num_cores)]) 
                           + max_path)
                priority *= max(1, len(successors))
            
            priority_cache[node] = priority
            return priority
        
        return dfs(process)
    
    def _check_power_constraints(self, core, process, start_time, execution_time):
        """
        Check if scheduling a process violates power constraints
        
        Args:
            core (int): Target core
            process (int): Process to schedule
            start_time (int): Proposed start time
            execution_time (int): Process execution time
        
        Returns:
            bool: Whether scheduling is power-constraint compliant
        """
        process_power = self.power_exec_matrix[core, process, 0]
        
        # Check individual core TDP
        if process_power > self.core_tdp_limits[core]:
            return False
        
        # Ensure we don't go out of bounds
        end_time = min(start_time + int(execution_time), len(self.power_profile_list))
        
        # Check chip-level TDP
        for t in range(start_time, end_time):
            if (self.power_profile_list[t] + process_power > self.chip_tdp):
                return False
        
        return True
    
    def schedule_dag(self):
        """
        Main scheduling algorithm implementing power-aware DAG scheduling
        
        Returns:
            dict: Scheduling results with process mappings and makespan
        """
        completed_tasks = set()
        task_finish_times = {}
        current_time = 0
        
        # Sort tasks by priority and arrival time
        sorted_tasks = sorted(
            self.all_tasks, 
            key=lambda t: (-self._calculate_process_priority(t['process']), t['start_time'])
        )
        
        while current_time < self.deadline and sorted_tasks:
            for task in sorted_tasks[:]:
                process = task['process']
                
                # Skip if already completed or missed deadline
                if (process, task['instance']) in completed_tasks or task['deadline'] < current_time:
                    sorted_tasks.remove(task)
                    continue
                
                # Find possible cores, sorted by execution efficiency
                cores_sorted = sorted(
                    range(self.num_cores), 
                    key=lambda c: self.power_exec_matrix[c, process, 1]
                )
                
                # Find suitable core and time slot
                scheduled = False
                for core in cores_sorted:
                    # Find valid start time considering predecessors
                    start_time = max(
                        current_time, 
                        max([task_finish_times.get(pred, 0) for pred in self._find_predecessors(process)], default=current_time)
                    )
                    
                    execution_time = int(self.power_exec_matrix[core, process, 1])
                    
                    # Check power and deadline constraints
                    if (start_time + execution_time <= task['deadline'] and 
                        start_time + execution_time <= self.deadline and
                        self._check_power_constraints(core, process, start_time, execution_time)):
                        
                        # Update power profile
                        for t in range(start_time, start_time + execution_time):
                            self.power_profile_list[t] += self.power_exec_matrix[core, process, 0]
                        
                        # Record scheduling information
                        self.schedule[core].append({
                            'process': process,
                            'instance': task['instance'],
                            'start_time': start_time,
                            'finish_time': start_time + execution_time
                        })
                        
                        task_finish_times[(process, task['instance'])] = start_time + execution_time
                        completed_tasks.add((process, task['instance']))
                        sorted_tasks.remove(task)
                        
                        scheduled = True
                        break
                
                # If task couldn't be scheduled
                if not scheduled:
                    print(f"Warning: Could not schedule process {process}, instance {task['instance']}")
            
            # Increment time by clock
            current_time += self.clock
        
        # Calculate makespan
        makespan = max(
            max([task['finish_time'] for task in core_schedule] + [0]) 
            for core_schedule in self.schedule
        )
        
        return {
            'schedule': self.schedule,
            'makespan': makespan,
            'power_profile': self.power_profile_list
        }
    
    def print_results(self, results):
            """
            Print scheduling results in a readable format
            
            Args:
                results (dict): Scheduling results
            """
            print("\nDAG Scheduling Results:")
            print(f"Total Makespan: {results['makespan']} time units")
            
            for core, core_schedule in enumerate(results['schedule']):
                print(f"\nCore {core} Schedule:")
                sorted_tasks = sorted(core_schedule, key=lambda x: x['start_time'])
                for task in sorted_tasks:
                    print(f"  Process {task['process']} (Instance {task['instance']}): "
                        f"Start={task['start_time']}, Finish={task['finish_time']}")

def main():
    # Configuration matching original implementation
    NUM_PROCESSES = 7
    NUM_CORES = 3
    CHIP_TDP = 20
    DEADLINE = 15
    CLOCK = 1
    
    ADJACENCY_MATRIX = np.array([
        # Process 0  1  2  3  4  5  6  7  8  9  10  11  12
        [0, 0, 0, 0, 0, 0, 0, 0],  # Process 0 (no dependencies)
        [0, 0, 1, 0, 0, 0, 0, 0],  # Process 1 depends on Process 2
        [0, 0, 0, 1, 0, 0, 0, 0],  # Process 2 depends on Process 3
        [0, 0, 0, 0, 1, 0, 0, 0],  # Process 3 depends on Process 4
        [0, 0, 0, 0, 0, 1, 0, 0],  # Process 4 depends on Process 5
        [0, 0, 0, 0, 0, 0, 1, 0],  # Process 5 depends on Process 6
        [0, 0, 0, 0, 0, 0, 0, 1],  # Process 6 depends on Process 7
        [0, 0, 0, 0, 0, 0, 0, 0]  # Process 7 (no further dependencies)
    ])

    # Format: [core][process] = [power_consumption, execution_time]
    POWER_EXEC_MATRIX = np.array([
        # Core 0
        [
            [1.5,3],
            [2.0,2],
            [2.5,3],
            [3.0,4],
            [4.0,2],
            [2.5,3],
            [2.0,2],
        ],
        # Core 1
        [
            [0.5,3.5],
            [1.5,2.5],
            [1.5,5],
            [2.0,4.5],
            [2.0,3.5],
            [2.5,4],
            [1.5,2],
        ],
        # Core 2
        [
            [1.5,2],
            [2.0,2],
            [2.0,1.5],
            [3.5,2],
            [4.0,2.5],
            [2.5,2.5],
            [2.0,2],
        ],
    ])

    CORE_TDP_LIMITS = np.array([7,8,7])
    # Initialize and run scheduler with deadline and clock
    scheduler = DAGScheduler(
        num_processes=NUM_PROCESSES,
        num_cores=NUM_CORES,
        chip_tdp=CHIP_TDP,
        core_tdp_limits=CORE_TDP_LIMITS,
        adjacency_matrix=ADJACENCY_MATRIX,
        power_exec_matrix=POWER_EXEC_MATRIX,
        deadline=DEADLINE,  # Added deadline parameter
        clock=CLOCK        # Added clock parameter
    )
    
    results = scheduler.schedule_dag()
    scheduler.print_results(results)

if __name__ == "__main__":
    main()