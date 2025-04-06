import heapq
import numpy as np
import matplotlib.pyplot as plt

class Task:
    def __init__(self, task_id, execution_times, dependencies, communication_cost, min_security):
        self.task_id = task_id
        self.execution_times = execution_times  # List of execution times on each processor
        self.dependencies = dependencies  # List of dependent tasks
        self.communication_cost = communication_cost  # Dict of message sizes to successors
        self.min_security = min_security  # Dict of minimum security required per message
        self.rank = 0  # Will be calculated
        self.processor = None  # Assigned processor
        self.start_time = None  # Assigned start time
        self.finish_time = None  # Assigned finish time

class Processor:
    def __init__(self, proc_id, bandwidths):
        self.proc_id = proc_id
        self.bandwidths = bandwidths  # Dict of bandwidths to other processors
        self.available_time = 0  # Next available time slot

class SHIELD_Scheduler:
    def __init__(self, tasks, processors, security_protocols):
        self.tasks = {t.task_id: t for t in tasks}
        self.processors = processors
        self.security_protocols = security_protocols  # Dict of protocol strengths and overheads
        self.schedule = []
    
    def compute_task_priorities(self):
        def compute_rank(task):
            if task.rank > 0:
                return task.rank
            
            max_rank = 0
            for succ_id in task.communication_cost:
                succ_task = self.tasks[succ_id]
                comm_time = task.communication_cost[succ_id] / max(self.processors[0].bandwidths.values())
                max_rank = max(max_rank, compute_rank(succ_task) + comm_time)
            
            task.rank = np.mean(task.execution_times) + max_rank
            return task.rank

        for task in self.tasks.values():
            compute_rank(task)
    
    def allocate_processors(self):
        sorted_tasks = sorted(self.tasks.values(), key=lambda t: -t.rank)
        for task in sorted_tasks:
            best_proc = None
            best_finish = float('inf')
            
            for proc in self.processors:
                start_time = max(proc.available_time, max(
                    (self.tasks[dep].finish_time for dep in task.dependencies), default=0
                ))
                finish_time = start_time + task.execution_times[proc.proc_id]
                if finish_time < best_finish:
                    best_proc, best_finish = proc, finish_time
            
            task.processor = best_proc.proc_id
            task.start_time = max(best_proc.available_time, max(
                (self.tasks[dep].finish_time for dep in task.dependencies), default=0
            ))
            task.finish_time = task.start_time + task.execution_times[task.processor]
            best_proc.available_time = task.finish_time
            self.schedule.append((task.task_id, task.processor, task.start_time, task.finish_time))
    
    def enhance_security(self):
        security_heap = []
        for task in self.tasks.values():
            for succ_id, min_sec in task.min_security.items():
                message_size = task.communication_cost[succ_id]
                best_upgrade = None
                max_benefit = 0
                
                for protocol, (strength, overhead) in self.security_protocols.items():
                    if strength > min_sec:
                        benefit = strength - min_sec
                        cost = overhead * message_size
                        if benefit / cost > max_benefit:
                            max_benefit = benefit / cost
                            best_upgrade = (task.task_id, succ_id, protocol)
                
                if best_upgrade:
                    heapq.heappush(security_heap, (-max_benefit, best_upgrade))
        
        while security_heap:
            _, (src, dst, protocol) = heapq.heappop(security_heap)
            print(f"Upgrading security of message from Task {src} to Task {dst} using {protocol}")
    
    def visualize_schedule(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for task_id, proc, start, finish in self.schedule:
            ax.barh(proc, finish - start, left=start, color=colors[proc % len(colors)], edgecolor='black')
            ax.text(start + (finish - start) / 2, proc, f"T{task_id}", va='center', ha='center', color='white', fontsize=10, weight='bold')
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Processor")
        ax.set_title("SHIELD Scheduling Gantt Chart")
        ax.set_yticks(range(len(self.processors)))
        ax.set_yticklabels([f"P{p.proc_id}" for p in self.processors])
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()
    
    def run(self):
        self.compute_task_priorities()
        self.allocate_processors()
        self.enhance_security()
        self.visualize_schedule()
        return self.schedule

# Example Usage
tasks = [
    Task(0, [10, 12, 8], [], {1: 5}, {1: 0.2}),
    Task(1, [15, 18, 12], [0], {2: 8}, {2: 0.3}),
    Task(2, [20, 22, 18], [1], {}, {}),
]
processors = [Processor(0, {1: 10, 2: 15}), Processor(1, {0: 10, 2: 12}), Processor(2, {0: 15, 1: 12})]
security_protocols = {"AES": (0.5, 0.02), "RSA": (0.7, 0.05), "SHA-256": (0.6, 0.03)}

scheduler = SHIELD_Scheduler(tasks, processors, security_protocols)
schedule = scheduler.run()
print("Final Schedule:", schedule)