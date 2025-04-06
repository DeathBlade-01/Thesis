import heapq
import numpy as np
import matplotlib.pyplot as plt

class Task:
    def __init__(self, task_id, execution_times, predecessors, message_sizes):
        self.task_id = task_id
        self.execution_times = execution_times  # Execution time on each processor
        self.predecessors = predecessors  # List of dependent tasks
        self.message_sizes = message_sizes  # Dict of message sizes to successors
        self.priority = None  # Task priority for scheduling
        self.assigned_processor = None
        self.start_time = None
        self.finish_time = None

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
    
    def compute_task_priorities(self):
        def calculate_priority(task):
            if task.priority is not None:
                return task.priority
            
            max_priority = 0
            for successor_id in task.message_sizes:
                if successor_id in self.tasks:
                    successor_task = self.tasks[successor_id]
                    comm_time = task.message_sizes[successor_id] / max(self.processors[0].bandwidths.values())
                    max_priority = max(max_priority, calculate_priority(successor_task) + comm_time)
            
            task.priority = np.mean(task.execution_times) + max_priority
            return task.priority

        for task in self.tasks.values():
            calculate_priority(task)
    
    def schedule_tasks(self):
        sorted_tasks = sorted(self.tasks.values(), key=lambda t: -t.priority)
        for task in sorted_tasks:
            best_processor = None
            best_finish_time = float('inf')
            
            for processor in self.processors:
                earliest_start_time = max(processor.available_time, max(
                    (self.tasks[pred].finish_time for pred in task.predecessors if pred in self.tasks), default=0
                ))
                finish_time = earliest_start_time + task.execution_times[processor.proc_id - 1]
                if finish_time < best_finish_time:
                    best_processor, best_finish_time = processor, finish_time
            
            task.assigned_processor = best_processor.proc_id
            task.start_time = max(best_processor.available_time, max(
                (self.tasks[pred].finish_time for pred in task.predecessors if pred in self.tasks), default=0
            ))
            task.finish_time = task.start_time + task.execution_times[task.assigned_processor - 1]
            best_processor.available_time = task.finish_time
            self.schedule.append((task.task_id, task.assigned_processor, task.start_time, task.finish_time))
    
    def plot_gantt_chart(self):
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

# Example Usage
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