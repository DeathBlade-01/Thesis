import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import time
from LASTRESORT import Task, Message, Processor, SecurityService, CommunicationNetwork, Scheduler

class ImprovedQLearningScheduler(Scheduler):
    def __init__(self, tasks, messages, processors, network, security_service, deadline, seed=42):
        super().__init__(tasks, messages, processors, network, security_service, deadline)
        random.seed(seed)
        np.random.seed(seed)

        self.q_table = {}
        self.alpha = 0.2
        self.gamma = 0.9
        self.epsilon = 0.8
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.97
        self.episodes = 500

        self.best_makespan = float('inf')
        self.best_schedule = None
        self.best_security_utility = 0

        self.initialize_q_table()

    def initialize_q_table(self):
        """Initialize Q-table with default values for all state-action pairs."""
        for task in self.tasks:
            for proc_id in range(1, len(self.processors) + 1):
                self.q_table[(task.task_id, proc_id)] = 0.0

    def update_q_value(self, task_id, processor, reward, next_task_id=None):
        """Update Q-value using the Q-learning formula."""
        current_q = self.q_table.get((task_id, processor), 0)
        if next_task_id:
            next_max_q = max(self.q_table.get((next_task_id, p), 0) for p in range(1, len(self.processors) + 1))
            self.q_table[(task_id, processor)] = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * next_max_q)
        else:
            self.q_table[(task_id, processor)] = (1 - self.alpha) * current_q + self.alpha * reward

    def choose_processor(self, task):
        """Choose a processor using an epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            return random.choice(range(1, len(self.processors) + 1))
        else:
            return max(range(1, len(self.processors) + 1), key=lambda p: self.q_table.get((task.task_id, p), 0))

    def calculate_improved_reward(self, task, processor_id):
        """Calculate reward for scheduling a task on a processor."""
        processor = self.processors[processor_id - 1]
        slack = self.deadline - task.finish_time
        locality_bonus = sum(10 for pred_id in task.predecessors if self.get_task_by_id(pred_id).assigned_processor == processor_id)
        return max(0, slack) + locality_bonus - processor.available_time

    def schedule_tasks(self):
        """Schedule tasks using Q-learning."""
        for episode in range(self.episodes):
            for processor in self.processors:
                processor.available_time = 0
            for task in self.tasks:
                task.is_scheduled = False
                task.assigned_processor = None
                task.start_time = None
                task.finish_time = None

            for task in sorted(self.tasks, key=lambda t: -t.priority):
                processor_id = self.choose_processor(task)
                est = self.calculate_est(task, processor_id)
                eft = est + task.execution_times[processor_id - 1]

                task.assigned_processor = processor_id
                task.start_time = est
                task.finish_time = eft
                task.is_scheduled = True
                self.processors[processor_id - 1].available_time = eft

                reward = self.calculate_improved_reward(task, processor_id)
                self.update_q_value(task.task_id, processor_id, reward)

            makespan = max(task.finish_time for task in self.tasks)
            if makespan < self.best_makespan:
                self.best_makespan = makespan
                self.best_schedule = copy.deepcopy(self.tasks)
                self.best_security_utility = self.calculate_security_utility()

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return self.best_makespan

    def calculate_security_utility(self):
        """Calculate the total security utility."""
        total_utility = 0
        for message in self.messages:
            for service in ['confidentiality', 'integrity', 'authentication']:
                protocol_idx = message.assigned_security[service]
                strength = self.security.strengths[service][protocol_idx]
                weight = message.weights[service]
                total_utility += weight * strength
        return total_utility