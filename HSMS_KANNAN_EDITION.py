import heapq
import numpy as np
import matplotlib.pyplot as plt
import itertools

class Task:
    def __init__(self, task_id, execution_times, predecessors, message_sizes):
        self.task_id = task_id
        self.execution_times = execution_times  
        self.predecessors = predecessors  
        self.message_sizes = message_sizes  
        self.security_strength = [
            [0.08,0.14,0.36,0.40,0.46,0.64,0.90,1],
            [0.18,0.26,0.36,0.45,0.63,0.77,1],
            [0.55,0.91,1]
        ]  
        self.X = [
            [168.75,96.43,37.50,33.75,29.35,21.09,15.00,13.50],
            [23.90,17.09,12.00,9.73,6.88,5.69,4.36],
            [90,148,163]
        ]
        self.priority = None  
        self.assigned_processor = None
        self.successors = None
        self.start_time = None
        self.finish_time = None
        self.so = None
        self.soN = None

class Processor:
    def __init__(self, proc_id, bandwidths):
        self.proc_id = proc_id
        self.bandwidths = bandwidths  
        self.available_time = 0  

class HSMS_Scheduler:
    def __init__(self, tasks, processors):
        self.tasks = {t.task_id: t for t in tasks}
        self.processors = processors
        self.schedule = []
        self.find_successors()
        self.calculate_so()
    
    def find_successors(self):
        for task in self.tasks.values():
            task.successors = []

        for task in self.tasks.values():
            for possible_succ in self.tasks.values():
                if task.task_id in possible_succ.predecessors:
                    task.successors.append(possible_succ.task_id)
    
    def calculate_so(self):
        for task in self.tasks.values():
            task.so = []
            task.soN = 0  

        for task in self.tasks.values():
            for pred_task_id in task.predecessors:
                if pred_task_id in self.tasks:
                    pred_task = self.tasks[pred_task_id]
                    msg_size = pred_task.message_sizes.get(task.task_id, 0)  
                    for x, y, z in itertools.product(task.X[0], task.X[1], task.X[2]):
                        task.so.append((msg_size / x) + (msg_size / y) + z)

            if not task.predecessors:  
                for succ_task_id in task.successors:
                    if succ_task_id in self.tasks:
                        succ_task = self.tasks[succ_task_id]
                        msg_size = task.message_sizes.get(succ_task_id, 0)  
                        for x, y, z in itertools.product(task.X[0], task.X[1], task.X[2]):
                            task.so.append((msg_size / x) + (msg_size / y) + z)

            task.soN = sum(task.so) if task.so else 0  

    def compute_task_priorities(self):
        def calculate_priority(task):
            if task.priority is not None:
                return task.priority
            
            max_priority = 0
            for successor_id in task.message_sizes:
                if successor_id in self.tasks:
                    successor_task = self.tasks[successor_id]
                    comm_time = task.message_sizes.get(successor_id, 1) / max(self.processors[0].bandwidths.values())
                    max_priority = max(max_priority, calculate_priority(successor_task) + comm_time)
            
            security_overhead = task.soN / 3  
            task.priority = np.mean(task.execution_times) + security_overhead + max_priority
            return task.priority

        for task in self.tasks.values():
            calculate_priority(task)
    
    def schedule_tasks(self):
        sorted_tasks = sorted(self.tasks.values(), key=lambda t: -t.priority)

        for task in sorted_tasks:
            best_processor = None
            best_finish_time = float('inf')
            best_start_time = 0
            optimal_pred = None  

            for processor in self.processors:
                est = max(processor.available_time, max(
                    (self.tasks[pred].finish_time for pred in task.predecessors if pred in self.tasks), default=0
                ))

                comm_time = 0
                if task.predecessors:
                    pred_finish_times = {}
                    for pred_id in task.predecessors:
                        pred_task = self.tasks[pred_id]
                        if pred_task.assigned_processor is not None and pred_task.assigned_processor != processor.proc_id:
                            comm_time += task.message_sizes.get(pred_id, 1) / processor.bandwidths.get(pred_task.assigned_processor, 2)
                        pred_finish_times[pred_id] = pred_task.finish_time

                    optimal_pred_id = max(pred_finish_times, key=pred_finish_times.get, default=None)
                    if optimal_pred_id is not None:
                        optimal_pred = self.tasks[optimal_pred_id]
                        est = max(est, optimal_pred.finish_time + comm_time)

                security_overhead = task.soN / 3  

                finish_time = est + task.execution_times[processor.proc_id - 1] + security_overhead  

                if finish_time < best_finish_time:
                    best_processor, best_finish_time, best_start_time = processor, finish_time, est

            task.assigned_processor = best_processor.proc_id
            task.start_time = best_start_time
            task.finish_time = best_finish_time
            best_processor.available_time = best_finish_time  

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
