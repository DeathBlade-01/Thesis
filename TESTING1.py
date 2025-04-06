import heapq
import numpy as np
import matplotlib.pyplot as plt
import itertools

class Task:
    def __init__(self, task_id, execution_times, predecessors, message_sizes):
        self.task_id = task_id
        self.execution_times = execution_times  # Execution time on each processor
        self.predecessors = predecessors  # List of dependent tasks
        self.message_sizes = message_sizes  # Dict of message sizes to successors
        self.security_strength = [
            [0.08, 0.14, 0.36, 0.40, 0.46, 0.64, 0.90, 1],
            [0.18, 0.26, 0.36, 0.45, 0.63, 0.77, 1],
            [0.55, 0.91, 1]
        ]  # Security overhead as per paper
        self.X = [
            [168.75, 96.43, 37.50, 33.75, 29.35, 21.09, 15.00, 13.50],
            [23.90, 17.09, 12.00, 9.73, 6.88, 5.69, 4.36],
            [90, 148, 163]
        ]
        self.priority = None
        self.assigned_processor = None
        self.successors = []
        self.start_time = None
        self.finish_time = None
        self.so = 0  # Security Overhead
        self.soN = 0  # Normalized Security Overhead
        self.rank = None  # Rank used for task sorting

class Processor:
    def __init__(self, proc_id, bandwidths):
        self.proc_id = proc_id
        self.bandwidths = bandwidths  # Dict of bandwidths to other processors
        self.available_time = 0  # Next available time slot

class HSMS_Scheduler:
    def __init__(self, tasks, processors):
        self.tasks = {t.task_id: t for t in tasks}
        self.processors = processors
        self.schedule = []
        self.find_successors()
        self.calculate_so()

    def find_successors(self):
        """ Identify successor tasks for each task """
        for task in self.tasks.values():
            for possible_succ in self.tasks.values():
                if task.task_id in possible_succ.predecessors:
                    task.successors.append(possible_succ.task_id)

    def calculate_so(self):
        """ Compute security overhead for each task """
        for task in self.tasks.values():
            task.so = 0
            # Compute Security Overhead from Predecessors
            for pred_task_id in task.predecessors:
                if pred_task_id in self.tasks:
                    pred_task = self.tasks[pred_task_id]
                    if task.task_id in pred_task.message_sizes:
                        msg_size = pred_task.message_sizes[task.task_id]
                        task.so += sum((msg_size / x) + (msg_size / y) + z
                                       for x, y, z in itertools.product(task.X[0], task.X[1], task.X[2]))

            # Compute Security Overhead from Successors
            for succ_task_id in task.successors:
                if succ_task_id in self.tasks:
                    succ_task = self.tasks[succ_task_id]
                    if succ_task_id in task.message_sizes:
                        msg_size = task.message_sizes[succ_task_id]
                        task.so += sum((msg_size / x) + (msg_size / y) + z
                                       for x, y, z in itertools.product(task.X[0], task.X[1], task.X[2]))

            task.soN = task.so  # Direct assignment for now, normalization can be added
            print(f"Task {task.task_id} - soN: {task.soN}")

    def compute_task_priorities(self):
        """ Compute task rank for prioritization in scheduling """
        def calculate_rank(task):
            if task.rank is not None:
                return task.rank
            
            max_successor_rank = max((self.tasks[succ].rank for succ in task.successors if succ in self.tasks), default=0)
            task.rank = np.mean(task.execution_times) + max_successor_rank + task.soN
            return task.rank

        for task in self.tasks.values():
            calculate_rank(task)

    def schedule_tasks(self):
        """ Schedule tasks based on EST, EFT, and Security Overhead considerations """
        sorted_tasks = sorted(self.tasks.values(), key=lambda t: -t.rank)
        for task in sorted_tasks:
            best_processor = None
            best_finish_time = float('inf')
            best_start_time = 0

            for processor in self.processors:
                # Calculate EST
                est = max(processor.available_time, max(
                    (self.tasks[pred].finish_time for pred in task.predecessors if pred in self.tasks), default=0
                ))
                
                # Calculate EFT (EST + execution time + security overhead)
                finish_time = est + task.execution_times[processor.proc_id - 1] + task.soN
                if finish_time < best_finish_time:
                    best_processor, best_finish_time, best_start_time = processor, finish_time, est

            # Assign best processor
            task.assigned_processor = best_processor.proc_id
            task.start_time = best_start_time
            task.finish_time = best_finish_time
            best_processor.available_time = best_finish_time
            self.schedule.append((task.task_id, task.assigned_processor, task.start_time, task.finish_time))

    def plot_gantt_chart(self):
        """ Visualize the task schedule as a Gantt chart """
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['blue', 'green']

        for task_id, proc, start, finish in self.schedule:
            ax.barh(proc - 1, finish - start, left=start, color=colors[proc - 1], edgecolor='black')
            ax.text(start + (finish - start) / 2, proc - 1, f"T{task_id}", va='center', ha='center', color='white', fontsize=10, weight='bold')

        ax.set_xlabel("Time")
        ax.set_ylabel("Processor")
        ax.set_title("HSMS Scheduling Gantt Chart")
        ax.set_yticks(range(len(self.processors)))
        ax.set_yticklabels([f"P{p.proc_id}" for p in self.processors])
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    def run(self):
        self.compute_task_priorities()
        self.schedule_tasks()
        self.plot_gantt_chart()
        return self.schedule
    
    
tasks = [
    Task(1, [205,200], [], {6: 512}),
    Task(2, [207,200], [], {6: 512}),
    Task(3, [190,210], [], {6: 512}),
    Task(4, [200,198], [], {6: 512}),

    Task(5, [150,155], [], {8: 128}),
    Task(6, [297,300], [1,2,3,4], {8: 128}),
    Task(7, [175,180], [], {8: 128}),

    Task(8, [250,260], [5,6,7], {9: 256, 10: 256}),
    Task(9, [146,154], [8], {}),
    Task(10, [199,201], [8], {}),
]
processors = [Processor(1, {2: 5}), Processor(2, {1:10})]

scheduler = HSMS_Scheduler(tasks, processors)
schedule = scheduler.run()
print("Final Schedule:", schedule)