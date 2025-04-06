import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import heapq
import math

class ImprovedDAGScheduler:
    def __init__(self, num_processes, num_cores, chip_tdp, core_tdp_limits, 
                 adjacency_matrix, power_exec_matrix, deadline=9, clock=1):
        """
        Enhanced DAG Scheduler with improved scheduling strategies
        
        Args:
            num_processes (int): Number of processes to schedule
            num_cores (int): Number of available cores
            chip_tdp (float): Total chip thermal design power
            core_tdp_limits (np.array): TDP limits for each core
            adjacency_matrix (np.array): Dependency graph
            power_exec_matrix (np.array): Power and execution time for each process on each core
            deadline (int): Maximum scheduling time
            clock (int): Scheduling time increment
        """
        self.num_processes = num_processes
        self.num_cores = num_cores
        self.chip_tdp = chip_tdp
        self.core_tdp_limits = core_tdp_limits
        self.adjacency_matrix = adjacency_matrix
        self.power_exec_matrix = power_exec_matrix
        self.deadline = deadline
        self.clock = clock
        
        # Enhanced tracking mechanisms
        self.core_load = np.zeros(num_cores)
        self.core_power_consumption = np.zeros(num_cores)
        self.process_dependencies = self._build_dependency_graph()
        
        # Initialize scheduling data structures
        self.schedule = [[] for _ in range(num_cores)]
        self.power_profile = np.zeros(self.deadline + 1)
        self.task_earliest_start = np.zeros(num_processes, dtype=int)
    
    def _build_dependency_graph(self):
        """
        Build a comprehensive dependency graph with more detailed tracking
        
        Returns:
            dict: Detailed dependency information for each process
        """
        dependencies = {}
        for process in range(self.num_processes):
            # Direct predecessors
            predecessors = np.where(self.adjacency_matrix[:, process] == 1)[0].tolist()
            
            # All upstream dependencies (transitive closure)
            upstream = set()
            stack = predecessors.copy()
            while stack:
                current = stack.pop()
                if current not in upstream:
                    upstream.add(current)
                    stack.extend([p for p in np.where(self.adjacency_matrix[:, current] == 1)[0] 
                                  if p not in upstream])
            
            # Downstream dependencies
            downstream = set()
            stack = np.where(self.adjacency_matrix[process, :] == 1)[0].tolist()
            while stack:
                current = stack.pop()
                if current not in downstream:
                    downstream.add(current)
                    stack.extend([d for d in np.where(self.adjacency_matrix[current, :] == 1)[0] 
                                  if d not in downstream])
            
            dependencies[process] = {
                'direct_predecessors': predecessors,
                'upstream_dependencies': list(upstream),
                'downstream_dependencies': list(downstream)
            }
        
        return dependencies
    
    def _calculate_advanced_priority(self, process):
        """
        Advanced priority calculation considering multiple factors
        
        Args:
            process (int): Process to calculate priority for
        
        Returns:
            float: Comprehensive priority score
        """
        # Dependency complexity factor
        dependency_factor = (
            len(self.process_dependencies[process]['upstream_dependencies']) * 1.5 +
            len(self.process_dependencies[process]['downstream_dependencies']) * 1.2
        )
        
        # Power efficiency factor
        power_efficiency = np.mean([
            1 / (self.power_exec_matrix[core, process, 0] / 
                 self.power_exec_matrix[core, process, 1]) 
            for core in range(self.num_cores)
        ])
        
        # Execution time variability
        exec_time_variance = np.std([
            self.power_exec_matrix[core, process, 1] 
            for core in range(self.num_cores)
        ])
        
        # Combine factors with weighted scoring
        priority = (
            dependency_factor * 2.0 +  # Dependency impact
            power_efficiency * 1.5 +   # Power efficiency
            (1 / (exec_time_variance + 1)) * 1.0  # Consistency bonus
        )
        
        return priority
    
    def _find_optimal_core(self, process, current_time):
        """
        Find the most suitable core for a process
        
        Args:
            process (int): Process to schedule
            current_time (int): Current scheduling time
        
        Returns:
            tuple: (optimal_core, start_time, is_schedulable)
        """
        best_core = -1
        best_start_time = current_time
        best_score = float('-inf')
        
        # Consider minimum start time based on dependencies
        min_start_time = max([
            self.task_earliest_start[dep] 
            for dep in self.process_dependencies[process]['direct_predecessors']
        ], default=current_time)
        
        for core in range(self.num_cores):
            # Check power and TDP constraints
            process_power = self.power_exec_matrix[core, process, 0]
            process_time = int(self.power_exec_matrix[core, process, 1])
            
            if (process_power <= self.core_tdp_limits[core] and 
                min_start_time + process_time <= self.deadline):
                
                # Calculate core suitability score
                load_balance_score = -self.core_load[core]
                power_score = -process_power
                
                # Combined scoring mechanism
                core_score = (
                    load_balance_score * 2.0 +  # Prioritize less loaded cores
                    power_score * 1.5           # Consider power efficiency
                )
                
                if core_score > best_score:
                    best_core = core
                    best_start_time = min_start_time
                    best_score = core_score
        
        # Verify schedulability
        is_schedulable = best_core != -1
        
        return best_core, best_start_time, is_schedulable
    
    def _update_power_and_load(self, core, process, start_time):
        """
        Update core load and power consumption tracking
        
        Args:
            core (int): Target core
            process (int): Scheduled process
            start_time (int): Process start time
        """
        process_time = int(self.power_exec_matrix[core, process, 1])
        process_power = self.power_exec_matrix[core, process, 0]
        
        # Update core-specific tracking
        self.core_load[core] += process_time
        self.core_power_consumption[core] += process_power * process_time
        
        # Global power profile update
        end_time = min(start_time + process_time, len(self.power_profile))
        for t in range(start_time, end_time):
            self.power_profile[t] += process_power
    
    def schedule_dag(self):
        """
        Enhanced DAG scheduling algorithm
        
        Returns:
            dict: Comprehensive scheduling results
        """
        # Sort processes by advanced priority
        processes = sorted(
            range(self.num_processes), 
            key=self._calculate_advanced_priority, 
            reverse=True
        )
        
        current_time = 0
        scheduled_processes = set()
        
        while current_time < self.deadline and len(scheduled_processes) < self.num_processes:
            for process in processes:
                if process in scheduled_processes:
                    continue
                
                # Check if all dependencies are met
                dependencies_met = all(
                    dep in scheduled_processes 
                    for dep in self.process_dependencies[process]['direct_predecessors']
                )
                
                if dependencies_met:
                    # Find optimal core and scheduling time
                    core, start_time, is_schedulable = self._find_optimal_core(process, current_time)
                    
                    if is_schedulable:
                        process_time = int(self.power_exec_matrix[core, process, 1])
                        
                        # Record scheduling information
                        self.schedule[core].append({
                            'process': process,
                            'start_time': start_time,
                            'finish_time': start_time + process_time
                        })
                        
                        # Update tracking
                        self._update_power_and_load(core, process, start_time)
                        scheduled_processes.add(process)
                        self.task_earliest_start[process] = start_time + process_time
            
            current_time += self.clock
        
        # Calculate makespan and performance metrics
        makespan = max(
            max([task['finish_time'] for task in core_schedule] + [0]) 
            for core_schedule in self.schedule
        )
        
        return {
            'schedule': self.schedule,
            'makespan': makespan,
            'power_profile': self.power_profile,
            'core_load': self.core_load,
            'core_power_consumption': self.core_power_consumption
        }
    
    def print_results(self, results):
        """
        Enhanced results printing with additional insights
        
        Args:
            results (dict): Scheduling results
        """
        print("\nEnhanced DAG Scheduling Results:")
        print(f"Total Makespan: {results['makespan']} time units")
        print("\nCore Utilization:")
        for core, load in enumerate(results['core_load']):
            print(f"  Core {core}: Load = {load} | Power Consumption = {results['core_power_consumption'][core]:.2f}")
        
        print("\nDetailed Core Schedules:")
        for core, core_schedule in enumerate(results['schedule']):
            print(f"\nCore {core} Schedule:")
            sorted_tasks = sorted(core_schedule, key=lambda x: x['start_time'])
            for task in sorted_tasks:
                print(f"  Process {task['process']}: "
                      f"Start={task['start_time']}, Finish={task['finish_time']}")
    def visualize_schedule(self, results):
        """
        Create a comprehensive and accurate visualization of the scheduling results
        
        Args:
            results (dict): Scheduling results from schedule_dag()
        """
        plt.figure(figsize=(15, 10))
        
        # Schedule Gantt Chart
        plt.subplot(2, 1, 1)
        plt.title('Process Scheduling Gantt Chart', fontsize=15)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Core', fontsize=12)
        
        # Color palette for processes
        colors = plt.cm.Set3(np.linspace(0, 1, self.num_processes))
        
        # Track maximum makespan for x-axis
        max_makespan = results['makespan']
        
        for core_idx, core_schedule in enumerate(results['schedule']):
            # Sort tasks by start time to ensure correct ordering
            sorted_tasks = sorted(core_schedule, key=lambda x: x['start_time'])
            
            for task in sorted_tasks:
                process = task['process']
                start_time = task['start_time']
                duration = task['finish_time'] - task['start_time']
                
                # Determine power for this task
                power = self.power_exec_matrix[core_idx, process, 0]
                
                # Create rectangle with color indicating process
                rect = patches.Rectangle(
                    (start_time, core_idx), 
                    duration, 
                    0.8, 
                    facecolor=colors[process],
                    edgecolor='black',
                    linewidth=1,
                    alpha=0.7
                )
                plt.gca().add_patch(rect)
                
                # Add process label
                plt.text(
                    start_time + duration/2, 
                    core_idx + 0.4, 
                    f'P{process}\n{power:.1f}W', 
                    ha='center', 
                    va='center',
                    fontsize=8
                )
        
        plt.yticks(range(self.num_cores), [f'Core {i}' for i in range(self.num_cores)])
        plt.ylim(-0.5, self.num_cores - 0.5)
        plt.xlim(0, max_makespan)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Power Profile
        plt.subplot(2, 1, 2)
        plt.title('Power Consumption Profile', fontsize=15)
        plt.plot(results['power_profile'], label='Total Power', color='red', linewidth=2)
        plt.axhline(y=self.chip_tdp, color='black', linestyle='--', label='Chip TDP Limit')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Power (W)', fontsize=12)
        plt.xlim(0, max_makespan)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

def main():
    # Configuration parameters
    NUM_PROCESSES = 7
    NUM_CORES = 3
    CHIP_TDP = 20
    DEADLINE = 15
    CLOCK = 1
    
    # Dependency matrix (same as previous implementation)
    ADJACENCY_MATRIX = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],  # Process 0 (no dependencies)
        [0, 0, 1, 0, 0, 0, 0, 0],  # Process 1 depends on Process 2
        [0, 0, 0, 1, 0, 0, 0, 0],  # Process 2 depends on Process 3
        [0, 0, 0, 0, 1, 0, 0, 0],  # Process 3 depends on Process 4
        [0, 0, 0, 0, 0, 1, 0, 0],  # Process 4 depends on Process 5
        [0, 0, 0, 0, 0, 0, 1, 0],  # Process 5 depends on Process 6
        [0, 0, 0, 0, 0, 0, 0, 1],  # Process 6 depends on Process 7
        [0, 0, 0, 0, 0, 0, 0, 0]   # Process 7 (no further dependencies)
    ])

    # Power and execution matrix (same as previous implementation)
    POWER_EXEC_MATRIX = np.array([
        # Core 0: [power_consumption, execution_time]
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
    
    # Initialize and run scheduler
    scheduler = ImprovedDAGScheduler(
        num_processes=NUM_PROCESSES,
        num_cores=NUM_CORES,
        chip_tdp=CHIP_TDP,
        core_tdp_limits=CORE_TDP_LIMITS,
        adjacency_matrix=ADJACENCY_MATRIX,
        power_exec_matrix=POWER_EXEC_MATRIX,
        deadline=DEADLINE,
        clock=CLOCK
    )
    
    results = scheduler.schedule_dag()
    scheduler.print_results(results)

    scheduler.visualize_schedule(results)

if __name__ == "__main__":
    main()