import random
import numpy as np
import heapq
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class Task:
    """Task in a workflow graph"""
    def __init__(self, task_id, name, execution_times, predecessors=None, successors=None):
        self.task_id = task_id
        self.name = name
        self.execution_times = execution_times  # List of execution times on each processor
        self.predecessors = predecessors if predecessors else []
        self.successors = successors if successors else []
        self.assigned_processor = None
        self.start_time = None
        self.finish_time = None
        self.is_scheduled = False
        self.priority = 0  # For priority-based scheduling
        self.upward_rank = 0  # For HEFT-based scheduling
        self.average_security_overhead = 0  # For security-aware scheduling


class Message:
    """Message passed between tasks in a workflow"""
    def __init__(self, source_id, dest_id, size):
        self.id = f"{source_id}-{dest_id}"
        self.source_id = source_id
        self.dest_id = dest_id
        self.size = size  # Message size in KB
        
        # Security requirements and weights
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
        
        # Assigned security levels (indices, not actual security values)
        self.assigned_security = {
            'confidentiality': 0,
            'integrity': 0,
            'authentication': 0
        }
    
    def set_security_requirements(self, conf_min, integ_min, auth_min, conf_weight, integ_weight, auth_weight):
        """Set minimum security requirements and weights for this message"""
        self.min_security['confidentiality'] = conf_min
        self.min_security['integrity'] = integ_min
        self.min_security['authentication'] = auth_min
        
        self.weights['confidentiality'] = conf_weight
        self.weights['integrity'] = integ_weight
        self.weights['authentication'] = auth_weight


class Processor:
    """Processing unit in a heterogeneous system"""
    def __init__(self, proc_id, comm_costs=None):
        self.proc_id = proc_id
        self.available_time = 0
        self.comm_costs = comm_costs if comm_costs else []
    
    def get_communication_time(self, message, source_proc, dest_proc):
        """Calculate communication time based on message size and link speed"""
        if source_proc == dest_proc:
            return 0
        
        # Use comm_costs matrix for communication cost
        link_cost = self.comm_costs[dest_proc.proc_id - 1]
        return message.size * link_cost


class SecurityService:
    """Security service with available protocols and associated costs/strengths"""
    def __init__(self):
        # Security strengths for each protocol level
        self.strengths = {
            'confidentiality': [0.08, 0.14, 0.36, 0.40, 0.46, 0.64, 0.9, 1.0],
            'integrity': [0.18, 0.26, 0.36, 0.45, 0.63, 0.77, 1.0],
            'authentication': [0.55, 0.91, 1.0]
        }
        
        # Security overheads for each protocol level on each processor
        # Index is 1-based to match protocol level (0 = no security, 1 = level 1, etc.)
        self.overheads = {
            'confidentiality': {
                1: [1012.5, 1518.7, 1181.2],
                2: [578.58, 867.87, 675.01],
                3: [225.0, 337.5, 262.5],
                4: [202.50, 303.75, 236.25],
                5: [176.10, 264.15, 205.45],
                6: [126.54, 189.81, 147.63],
                7: [90.0, 135.0, 105.0],
                8: [81.0, 121.5, 94.5]
            },
            'integrity': {
                1: [143.40, 215.10, 167.30],
                2: [102.54, 153.81, 119.63],
                3: [72, 108, 84.0],
                4: [58.38, 87.57, 68.11],
                5: [41.28, 61.92, 48.16],
                6: [34.14, 51.21, 39.83],
                7: [26.16, 39.24, 30.52]
            },
            'authentication': {
                1: [15, 10, 12.86],
                2: [24.67, 16.44, 21.14],
                3: [27.17, 18.11, 23.29]
            }
        }


class RF_SHIELD:
    """Random Forest-based SHIELD implementation"""
    def __init__(self, tasks, messages, processors, security_service, deadline):
        self.tasks = tasks
        self.messages = messages
        self.processors = processors
        self.security = security_service
        self.deadline = deadline
        self.makespan = 0
        self.schedule = []
        self.original_makespan = None
        self.bcr_heap = []  # Benefit-to-Cost Ratio heap
        self.upgrade_history = []  # Track security upgrades for reporting
        
        # Initialize RF models
        self.priority_model = None
        self.processor_assignment_model = None
        self.security_upgrade_model = None
        
        # Prepare task successors (if not already set)
        self._prepare_task_successors()
        
        # Create message lookup by source and destination
        self.message_lookup = {}
        for message in self.messages:
            self.message_lookup[(message.source_id, message.dest_id)] = message
    
    def _prepare_task_successors(self):
        """Initialize task successors based on predecessors"""
        # Clear existing successors
        for task in self.tasks:
            task.successors = []
        
        # Build successor lists
        for task in self.tasks:
            for pred_id in task.predecessors:
                pred_task = self.get_task_by_id(pred_id)
                if pred_task:
                    if task.task_id not in pred_task.successors:
                        pred_task.successors.append(task.task_id)
    
    def get_task_by_id(self, task_id):
        """Get task by ID"""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
    
    def get_message(self, source_id, dest_id):
        """Get message by source and destination task IDs"""
        return self.message_lookup.get((source_id, dest_id), None)
    
    def assign_minimum_security(self):
        """Assign minimum required security levels to all messages"""
        for message in self.messages:
            for service in ['confidentiality', 'integrity', 'authentication']:
                min_level = message.min_security[service]
                strengths = self.security.strengths[service]
                
                # Find the smallest level that meets the minimum requirement
                for i, strength in enumerate(strengths):
                    if strength >= min_level:
                        message.assigned_security[service] = i
                        break
    
    def run(self):
        """Execute the RF-SHIELD algorithm"""
        # Train models with the given task data
        self.train_models()
        
        # Phase 1 & 2: Use RF models for task sequencing and processor mapping
        makespan = self.schedule_tasks_with_rf()
        
        if makespan is None or makespan > self.deadline:
            print("RF-SHIELD cannot proceed: Initial scheduling failed to meet deadline requirements")
            return None, None
        
        self.original_makespan = makespan
        security_utility = self.calculate_security_utility()
        print(f"\nRF-SHIELD Phase 1-2 complete: Initial makespan={makespan}, Security utility={security_utility}")
        
        # Phase 3: Security Enhancement using RF prediction
        print("\nRF-SHIELD Phase 3: Starting Security Enhancement...")
        self.enhance_security_with_rf()
        
        # Calculate final results
        final_makespan = self.finalize_schedule()
        final_security_utility = self.calculate_security_utility()
        
        print(f"RF-SHIELD complete: Final makespan={final_makespan}, Final security utility={final_security_utility}")
        print(f"Security utility improvement: {(final_security_utility - security_utility) / security_utility * 100:.2f}%")
        
        return final_makespan, final_security_utility
    
    def train_models(self):
        """Train the Random Forest models using the available task data"""
        self.train_priority_model()
        self.train_processor_assignment_model()
        self.train_security_upgrade_model()
    
    def train_priority_model(self):
        """Train RF model for task priority prediction"""
        # For task priority, we'll use a simple approach with synthetic data
        # since we only have one test case
        
        # First, compute some base priorities using HEFT upward rank
        self.compute_upward_ranks()
        
        # Calculate average security overhead for each task
        self.calculate_task_security_overheads()
        
        # Create synthetic training data
        X_train = []
        y_train = []
        
        # Use different random seeds to create variations
        for seed in range(10):
            random.seed(seed)
            
            for task in self.tasks:
                # Features for priority model
                features = [
                    task.upward_rank,  # HEFT upward rank
                    sum(task.execution_times) / len(task.execution_times),  # Avg execution time
                    len(task.predecessors),  # Number of predecessors
                    len(task.successors),  # Number of successors
                    task.average_security_overhead,  # Average security overhead
                    random.uniform(0.9, 1.1)  # Random variation for synthetic data
                ]
                
                # Priority target (perturbed upward rank + security factor)
                priority = (task.upward_rank * 
                           (1 + 0.3 * task.average_security_overhead) * 
                           random.uniform(0.95, 1.05))
                
                X_train.append(features)
                y_train.append(priority)
        
        # Train the model
        self.priority_model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.priority_model.fit(X_train, y_train)
        
        print("Priority prediction model trained")
    
    def train_processor_assignment_model(self):
        """Train RF model for processor assignment prediction"""
        # For processor assignment, create synthetic training data
        X_train = []
        y_train = []
        
        # Assign tasks to processors that minimize execution time as a baseline
        for seed in range(10):
            random.seed(seed)
            
            for task in self.tasks:
                for proc_idx, processor in enumerate(self.processors):
                    proc_id = processor.proc_id
                    
                    # Features for processor assignment
                    features = [
                        task.execution_times[proc_id - 1],  # Execution time on this processor
                        len(task.successors),  # Number of successors (parallel potential)
                        processor.available_time * random.uniform(0.9, 1.1),  # Current availability
                        len(self.tasks) / (task.task_id + 1),  # Position in workflow (normalized)
                        proc_id,  # Processor ID
                        random.uniform(0.9, 1.1)  # Random variation
                    ]
                    
                    # Target: processor suitability score (lower is better)
                    # Base score on execution time, add random variation
                    score = task.execution_times[proc_id - 1] * random.uniform(0.9, 1.1)
                    
                    # Adjust score based on processor availability
                    avail_factor = processor.available_time / 100.0  # Normalize
                    score *= (1 + avail_factor) * random.uniform(0.9, 1.1)
                    
                    X_train.append(features)
                    y_train.append(score)  # Lower score = better processor assignment
        
        # Train the model
        self.processor_assignment_model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.processor_assignment_model.fit(X_train, y_train)
        
        print("Processor assignment model trained")
    
    def train_security_upgrade_model(self):
        """Train RF model to predict successful security upgrades"""
        X_train = []
        y_train = []
        
        # Generate synthetic training data
        for seed in range(20):
            random.seed(seed)
            
            for message in self.messages:
                source_task = self.get_task_by_id(message.source_id)
                dest_task = self.get_task_by_id(message.dest_id)
                
                if not source_task or not dest_task:
                    continue
                
                for service in ['confidentiality', 'integrity', 'authentication']:
                    current_level = message.assigned_security[service]
                    max_level = len(self.security.strengths[service]) - 1
                    
                    if current_level >= max_level:
                        continue
                    
                    # Current and next level strengths
                    current_strength = self.security.strengths[service][current_level]
                    next_strength = self.security.strengths[service][current_level + 1]
                    
                    # Calculate security benefit
                    benefit = message.weights[service] * (next_strength - current_strength)
                    
                    # Estimate cost (overhead increase)
                    current_overhead = self.estimate_service_overhead(message, service, current_level)
                    next_overhead = self.estimate_service_overhead(message, service, current_level + 1)
                    cost = next_overhead - current_overhead
                    
                    # BCR (benefit-to-cost ratio)
                    bcr = benefit / max(cost, 0.01)  # Avoid division by zero
                    
                    # Simulate task and global slack
                    source_slack = random.uniform(0, 50)
                    dest_slack = random.uniform(0, 50)
                    global_slack = random.uniform(0, 100)
                    
                    # Features for upgrade success prediction
                    features = [
                        message.size,  # Message size
                        message.weights[service],  # Service weight
                        current_level,  # Current security level
                        current_level + 1,  # Next security level
                        cost,  # Estimated overhead increase
                        bcr,  # Benefit-to-cost ratio
                        source_slack,  # Source task slack
                        dest_slack,  # Destination task slack
                        global_slack,  # Global slack
                        len(source_task.successors),  # Source task successors
                        len(dest_task.successors)  # Destination task successors
                    ]
                    
                    # Success is more likely with high BCR and sufficient slack
                    success_prob = 0.8 * (bcr / 10.0) + 0.2 * (global_slack / 100.0)
                    success = 1 if random.random() < success_prob else 0
                    
                    X_train.append(features)
                    y_train.append(success)
        
        # Train the model
        self.security_upgrade_model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.security_upgrade_model.fit(X_train, y_train)
        
        print("Security upgrade prediction model trained")
    
    def compute_upward_ranks(self):
        """Compute HEFT upward ranks for all tasks"""
        # First, compute average execution time for each task
        for task in self.tasks:
            task.avg_exec_time = sum(task.execution_times) / len(task.execution_times)
        
        # Compute upward rank in reverse topological order
        self.compute_upward_rank_recursive(self.get_exit_tasks())
    
    def get_exit_tasks(self):
        """Get tasks with no successors (exit tasks)"""
        exit_tasks = []
        for task in self.tasks:
            if not task.successors:
                exit_tasks.append(task)
        return exit_tasks
    
    def compute_upward_rank_recursive(self, tasks):
        """Recursively compute upward rank for tasks"""
        for task in tasks:
            if task.upward_rank != 0:
                continue  # Already computed
            
            max_successor_rank = 0
            
            for succ_id in task.successors:
                succ_task = self.get_task_by_id(succ_id)
                if not succ_task:
                    continue
                
                # If successor rank not computed yet, compute it first
                if succ_task.upward_rank == 0:
                    self.compute_upward_rank_recursive([succ_task])
                
                # Get the message between tasks
                message = self.get_message(task.task_id, succ_id)
                comm_cost = 0
                if message:
                    # Average communication cost across all processor pairs
                    comm_cost = message.size * 2  # Simple approximation
                
                successor_rank = succ_task.upward_rank + comm_cost
                max_successor_rank = max(max_successor_rank, successor_rank)
            
            # Compute upward rank
            task.upward_rank = task.avg_exec_time + max_successor_rank
    
    def calculate_task_security_overheads(self):
        """Calculate average security overhead for each task based on minimum security requirements"""
        # Assign minimum security levels first
        self.assign_minimum_security()
        
        for task in self.tasks:
            incoming_overhead = 0
            incoming_count = 0
            outgoing_overhead = 0
            outgoing_count = 0
            
            # Calculate incoming security overhead
            for pred_id in task.predecessors:
                message = self.get_message(pred_id, task.task_id)
                if not message:
                    continue
                
                # Calculate overhead for each security service
                for service in ['confidentiality', 'integrity', 'authentication']:
                    level = message.assigned_security[service]
                    overhead = self.estimate_service_overhead(message, service, level)
                    incoming_overhead += overhead
                    incoming_count += 1
            
            # Calculate outgoing security overhead
            for succ_id in task.successors:
                message = self.get_message(task.task_id, succ_id)
                if not message:
                    continue
                
                # Calculate overhead for each security service
                for service in ['confidentiality', 'integrity', 'authentication']:
                    level = message.assigned_security[service]
                    overhead = self.estimate_service_overhead(message, service, level)
                    outgoing_overhead += overhead
                    outgoing_count += 1
            
            # Calculate average overhead
            total_count = max(1, incoming_count + outgoing_count)
            task.average_security_overhead = (incoming_overhead + outgoing_overhead) / total_count
    
    def estimate_service_overhead(self, message, service, level):
        """Estimate overhead for a security service at given level, averaged across processors"""
        if level == 0:  # No security
            return 0
        
        protocol_idx = level + 1  # Convert to 1-indexed
        
        if service in ['confidentiality', 'integrity']:
            # Data-dependent overhead (average across processors)
            avg_overhead = 0
            for proc_id in range(1, len(self.processors) + 1):
                overhead_factor = self.security.overheads[service][protocol_idx][proc_id - 1]
                avg_overhead += message.size / overhead_factor
            return avg_overhead / len(self.processors)
        else:  # Authentication
            # Fixed overhead (average across processors)
            avg_overhead = 0
            for proc_id in range(1, len(self.processors) + 1):
                avg_overhead += self.security.overheads[service][protocol_idx][proc_id - 1]
            return avg_overhead / len(self.processors)
    
    def schedule_tasks_with_rf(self):
        """Schedule tasks using RF models for priority and processor assignment"""
        # Assign minimum security levels first
        self.assign_minimum_security()
        
        # Predict task priorities using RF model
        self.predict_task_priorities()
        
        # Sort tasks by predicted priority (descending)
        sorted_tasks = sorted(self.tasks, key=lambda t: -t.priority)
        
        # Schedule each task
        for task in sorted_tasks:
            best_processor = None
            best_processor_score = float('inf')
            earliest_start_time = 0
            
            # Create a list to store processor scores
            processor_scores = []
            
            # Try each processor
            for proc_idx, processor in enumerate(self.processors):
                proc_id = processor.proc_id
                
                # Calculate earliest start time based on predecessors
                est = self.calculate_est(task, processor)
                
                # Skip if task cannot be scheduled yet
                if est == float('inf'):
                    continue
                
                # Prepare features for processor assignment prediction
                features = [
                    task.execution_times[proc_id - 1],  # Execution time on this processor
                    len(task.successors),  # Number of successors (parallel potential)
                    processor.available_time,  # Current processor availability
                    len(self.tasks) / (task.task_id + 1),  # Position in workflow (normalized)
                    proc_id,  # Processor ID
                    1.0  # No random variation in prediction phase
                ]
                
                # Predict processor suitability score (lower is better)
                score = self.processor_assignment_model.predict([features])[0]
                processor_scores.append((proc_id, score))
                
                # Keep track of the best processor assignment
                if score < best_processor_score:
                    best_processor_score = score
                    best_processor = processor
                    earliest_start_time = est
            
            # Assign task to the best processor
            if best_processor:
                self.assign_task_to_processor(task, best_processor, earliest_start_time)
                print(f"Scheduled Task {task.task_id} ({task.name}) on Processor {task.assigned_processor}")
                print(f"  Predicted processor scores: {processor_scores}")
                print(f"  Start time: {task.start_time:.2f}")
                print(f"  Finish time: {task.finish_time:.2f}")
            else:
                print(f"Failed to schedule Task {task.task_id} - predecessors not scheduled")
                return None
        
        # Calculate the makespan
        self.makespan = max(task.finish_time for task in self.tasks if task.is_scheduled)
        return self.makespan
    
    def predict_task_priorities(self):
        """Predict task priorities using the trained RF model"""
        for task in self.tasks:
            # Prepare features for priority prediction
            features = [
                task.upward_rank,  # HEFT upward rank
                sum(task.execution_times) / len(task.execution_times),  # Avg execution time
                len(task.predecessors),  # Number of predecessors
                len(task.successors),  # Number of successors
                task.average_security_overhead,  # Average security overhead
                1.0  # No random variation in prediction phase
            ]
            
            # Predict priority
            task.priority = self.priority_model.predict([features])[0]
            
            print(f"Task {task.task_id}: Upward rank = {task.upward_rank:.2f}, "
                 f"Security overhead = {task.average_security_overhead:.2f}, "
                 f"Predicted priority = {task.priority:.2f}")
    
    def calculate_est(self, task, processor):
        """Calculate earliest start time for a task on a processor"""
        proc_id = processor.proc_id
        est = 0
        
        # Calculate EST based on predecessors
        for pred_id in task.predecessors:
            pred_task = self.get_task_by_id(pred_id)
            if not pred_task or not pred_task.is_scheduled:
                # Cannot schedule this task yet
                return float('inf')
            
            # Get the message between predecessor and current task
            message = self.get_message(pred_id, task.task_id)
            if not message:
                continue
            
            # Calculate communication time
            source_proc = self.processors[pred_task.assigned_processor - 1]
            dest_proc = processor
            comm_time = source_proc.get_communication_time(message, source_proc, dest_proc) if source_proc != dest_proc else 0
            
            # Calculate security overhead
            security_overhead = self.calculate_security_overhead(message, pred_task.assigned_processor, proc_id)
            
            # Update EST based on predecessor finish time plus communication and security overhead
            pred_finish_with_comm = pred_task.finish_time + comm_time + security_overhead
            est = max(est, pred_finish_with_comm)
        
        # Also consider processor availability
        est = max(est, processor.available_time)
        
        return est
    
    def calculate_security_overhead(self, message, source_proc_id, dest_proc_id):
        """Calculate security overhead for a message between two processors"""
        total_overhead = 0
        
        # Calculate overhead for each security service
        for service in ['confidentiality', 'integrity']:
            level = message.assigned_security[service]
            if level > 0:  # Skip if no security
                protocol_idx = level + 1  # Convert to 1-indexed
                
                # Source processor overhead
                source_factor = self.security.overheads[service][protocol_idx][source_proc_id - 1]
                source_overhead = message.size / source_factor
                
                # Destination processor overhead
                dest_factor = self.security.overheads[service][protocol_idx][dest_proc_id - 1]
                dest_overhead = message.size / dest_factor
                
                total_overhead += source_overhead + dest_overhead
        
        # Authentication overhead
        auth_level = message.assigned_security['authentication']
        if auth_level > 0:
            protocol_idx = auth_level + 1
            
            # Source and destination authentication overhead
            source_overhead = self.security.overheads['authentication'][protocol_idx][source_proc_id - 1]
            dest_overhead = self.security.overheads['authentication'][protocol_idx][dest_proc_id - 1]
            
            total_overhead += source_overhead + dest_overhead
        
        return total_overhead
    
    def assign_task_to_processor(self, task, processor, start_time):
        """Assign a task to a processor with calculated times"""
        proc_id = processor.proc_id
        task.assigned_processor = proc_id
        task.start_time = start_time
        
        # Calculate base execution time
        base_exec_time = task.execution_times[proc_id - 1]
        
        # Calculate outgoing security overhead
        outgoing_security_overhead = 0
        for succ_id in task.successors:
            message = self.get_message(task.task_id, succ_id)
            if not message:
                continue
            
            # Only calculate source-side overhead
            outgoing_security_overhead += self.calculate_sender_security_overhead(message, proc_id)
        
        # Set finish time with security overhead included
        task.finish_time = start_time + base_exec_time + outgoing_security_overhead
        task.is_scheduled = True
        
        # Update processor availability time
        processor.available_time = task.finish_time
        
        # Add to schedule
        self.schedule.append({
            'task_id': task.task_id,
            'name': task.name,
            'processor': task.assigned_processor,
            'start_time': task.start_time,
            'finish_time': task.finish_time,
            'security_overhead': outgoing_security_overhead
        })
    
    def calculate_sender_security_overhead(self, message, processor_id):
        """Calculate only the sender-side security overhead for a message"""
        total_overhead = 0
        
        # Add overhead for confidentiality and integrity at the sender side
        for service in ['confidentiality', 'integrity']:
            level = message.assigned_security[service]
            if level > 0:
                protocol_idx = level + 1  # Convert to 1-indexed
                factor = self.security.overheads[service][protocol_idx][processor_id - 1]
                overhead = message.size / factor
                total_overhead += overhead
        
        # Add authentication overhead (sender side)
        auth_level = message.assigned_security['authentication']
        if auth_level > 0:
            protocol_idx = auth_level + 1
            total_overhead += self.security.overheads['authentication'][protocol_idx][processor_id - 1]
        
        return total_overhead
    
    def enhance_security_with_rf(self):
        """Enhance security levels using RF model to predict successful upgrades"""
        # Initialize all potential security upgrades
        upgrade_candidates = self.get_all_upgrade_candidates()
        print(f"Found {len(upgrade_candidates)} potential security upgrades")
        
        # Calculate BCR for each upgrade
        upgrades_with_bcr = []
        for message_id, service, current_level in upgrade_candidates:
            message = self.get_message_by_id(message_id)
            if not message:
                continue
            
            bcr = self.calculate_bcr(message, service, current_level)
            if bcr > 0:
                upgrades_with_bcr.append((bcr, message_id, service, current_level))
        
        # Sort by BCR (descending)
        upgrades_with_bcr.sort(reverse=True)
        print(f"Calculated BCR for {len(upgrades_with_bcr)} upgrades")
        
        upgrade_count = 0
        rejected_count = 0
        
        # Process upgrades in order of highest BCR
        for bcr, message_id, service, current_level in upgrades_with_bcr:
            message = self.get_message_by_id(message_id)
            # Skip if no message found
            if not message:
                continue
            
            # Skip if already at max security level
            max_level = len(self.security.strengths[service]) - 1
            if current_level >= max_level:
                continue
            
            # Get source and destination tasks
            source_task = self.get_task_by_id(message.source_id)
            dest_task = self.get_task_by_id(message.dest_id)
            
            if not source_task or not dest_task:
                continue
            
            # Calculate task slack
            source_slack = self.calculate_task_slack(source_task)
            dest_slack = self.calculate_task_slack(dest_task)
            
            # Calculate global slack
            global_slack = self.deadline - self.makespan
            
            # Prepare features for security upgrade prediction
            features = [
                message.size,  # Message size
                message.weights[service],  # Service weight
                current_level,  # Current security level
                current_level + 1,  # Next security level
                self.estimate_upgrade_cost(message, service, current_level),  # Estimated overhead increase
                bcr,  # Benefit-to-cost ratio
                source_slack,  # Source task slack
                dest_slack,  # Destination task slack
                global_slack,  # Global slack
                len(source_task.successors),  # Source task successors
                len(dest_task.successors)  # Destination task successors
            ]
            
            # Predict upgrade success probability
            success_prob = self.security_upgrade_model.predict_proba([features])[0][1]
            
            print(f"Upgrade candidate: Message {message_id} {service} from level {current_level} to {current_level+1}")
            print(f"  BCR: {bcr:.4f}, Success probability: {success_prob:.4f}")
            
            # Apply upgrade if success probability is high enough
            if success_prob > 0.5:  # Threshold can be adjusted
                # Apply the upgrade
                old_level = message.assigned_security[service]
                message.assigned_security[service] = current_level + 1
                
                # Recalculate the makespan with the new security level
                new_makespan = self.recalculate_makespan()
                
                if new_makespan <= self.deadline:
                    # Upgrade successful
                    upgrade_count += 1
                    print(f"  Upgrade ACCEPTED: New makespan {new_makespan:.2f} <= deadline {self.deadline}")
                    
                    # Record the upgrade for reporting
                    self.upgrade_history.append({
                        'message_id': message_id,
                        'service': service,
                        'old_level': old_level,
                        'new_level': current_level + 1,
                        'bcr': bcr,
                        'success_prob': success_prob,
                        'new_makespan': new_makespan
                    })
                else:
                    # Revert the upgrade if deadline violated
                    message.assigned_security[service] = old_level
                    rejected_count += 1
                    print(f"  Upgrade REJECTED: New makespan {new_makespan:.2f} > deadline {self.deadline}")
            else:
                rejected_count += 1
                print(f"  Upgrade SKIPPED: Low success probability")
        
        print(f"\nSecurity Enhancement Complete:")
        print(f"  {upgrade_count} upgrades applied")
        print(f"  {rejected_count} upgrades rejected")
    
    def get_all_upgrade_candidates(self):
        """Get all potential security upgrade candidates"""
        upgrade_candidates = []
        
        for message in self.messages:
            for service in ['confidentiality', 'integrity', 'authentication']:
                current_level = message.assigned_security[service]
                max_level = len(self.security.strengths[service]) - 1
                
                if current_level < max_level:
                    upgrade_candidates.append((message.id, service, current_level))
        
        return upgrade_candidates
    
    def get_message_by_id(self, message_id):
        """Get message by ID"""
        for message in self.messages:
            if message.id == message_id:
                return message
        return None
    
    def calculate_bcr(self, message, service, current_level):
        """Calculate Benefit-to-Cost Ratio for a security upgrade"""
        # Calculate benefit
        next_level = current_level + 1
        current_strength = self.security.strengths[service][current_level]
        next_strength = self.security.strengths[service][next_level]
        
        benefit = message.weights[service] * (next_strength - current_strength)
        
        # Calculate cost (overhead increase)
        source_task = self.get_task_by_id(message.source_id)
        dest_task = self.get_task_by_id(message.dest_id)
        
        if not source_task or not dest_task or not source_task.is_scheduled or not dest_task.is_scheduled:
            return 0
        
        cost = self.estimate_upgrade_cost(message, service, current_level)
        
        # Calculate BCR
        if cost <= 0:
            return float('inf')
        
        return benefit / cost
    
    def estimate_upgrade_cost(self, message, service, current_level):
        """Estimate the cost (overhead increase) for upgrading from current to next level"""
        next_level = current_level + 1
        source_proc = self.get_task_by_id(message.source_id).assigned_processor
        dest_proc = self.get_task_by_id(message.dest_id).assigned_processor
        
        # Calculate current overhead
        current_overhead = 0
        if current_level > 0:
            protocol_idx = current_level + 1  # Convert to 1-indexed
            
            if service in ['confidentiality', 'integrity']:
                # Source and destination overhead
                source_factor = self.security.overheads[service][protocol_idx][source_proc - 1]
                dest_factor = self.security.overheads[service][protocol_idx][dest_proc - 1]
                
                current_overhead = message.size / source_factor + message.size / dest_factor
            else:  # Authentication
                source_overhead = self.security.overheads[service][protocol_idx][source_proc - 1]
                dest_overhead = self.security.overheads[service][protocol_idx][dest_proc - 1]
                
                current_overhead = source_overhead + dest_overhead
        
        # Calculate next level overhead
        next_overhead = 0
        protocol_idx = next_level + 1  # Convert to 1-indexed
        
        if service in ['confidentiality', 'integrity']:
            # Source and destination overhead
            source_factor = self.security.overheads[service][protocol_idx][source_proc - 1]
            dest_factor = self.security.overheads[service][protocol_idx][dest_proc - 1]
            
            next_overhead = message.size / source_factor + message.size / dest_factor
        else:  # Authentication
            source_overhead = self.security.overheads[service][protocol_idx][source_proc - 1]
            dest_overhead = self.security.overheads[service][protocol_idx][dest_proc - 1]
            
            next_overhead = source_overhead + dest_overhead
        
        return next_overhead - current_overhead
    
    def calculate_task_slack(self, task):
        """Calculate slack time for a task"""
        if not task.is_scheduled:
            return 0
        
        # Calculate latest finish time (LFT)
        lft = self.calculate_lft(task)
        
        # Slack = LFT - Actual Finish Time
        return max(0, lft - task.finish_time)
    
    def calculate_lft(self, task):
        """Calculate latest finish time for a task based on deadline"""
        if not task.successors:
            # Exit task LFT is the deadline
            return self.deadline
        
        min_succ_lst = float('inf')
        
        for succ_id in task.successors:
            succ_task = self.get_task_by_id(succ_id)
            if not succ_task or not succ_task.is_scheduled:
                continue
            
            # Get the message between tasks
            message = self.get_message(task.task_id, succ_id)
            comm_time = 0
            security_overhead = 0
            
            if message:
                # Calculate communication time
                source_proc = self.processors[task.assigned_processor - 1]
                dest_proc = self.processors[succ_task.assigned_processor - 1]
                comm_time = source_proc.get_communication_time(message, source_proc, dest_proc) if source_proc != dest_proc else 0
                
                # Calculate security overhead
                security_overhead = self.calculate_security_overhead(message, task.assigned_processor, succ_task.assigned_processor)
            
            # Calculate successor LST (Latest Start Time)
            succ_lst = succ_task.start_time
            
            # LST of current task based on this successor
            lst = succ_lst - comm_time - security_overhead
            
            min_succ_lst = min(min_succ_lst, lst)
        
        # If no valid successors, use deadline
        if min_succ_lst == float('inf'):
            return self.deadline
        
        # LFT = LST
        return min_succ_lst
    
    def recalculate_makespan(self):
        """Recalculate the makespan after security level changes"""
        # Reset all task schedules
        for task in self.tasks:
            task.start_time = None
            task.finish_time = None
            task.is_scheduled = False
        
        for processor in self.processors:
            processor.available_time = 0
        
        # Reschedule all tasks based on the current assignments
        # Sort tasks in topological order
        scheduled_tasks = []
        remaining_tasks = list(self.tasks)
        
        while remaining_tasks:
            for task in list(remaining_tasks):
                # Check if all predecessors are scheduled
                all_preds_scheduled = True
                for pred_id in task.predecessors:
                    pred_scheduled = False
                    for scheduled_task in scheduled_tasks:
                        if scheduled_task.task_id == pred_id:
                            pred_scheduled = True
                            break
                    
                    if not pred_scheduled:
                        all_preds_scheduled = False
                        break
                
                if all_preds_scheduled:
                    # Schedule this task
                    processor = self.processors[task.assigned_processor - 1]
                    est = self.calculate_est(task, processor)
                    
                    if est != float('inf'):
                        task.start_time = est
                        
                        # Calculate base execution time
                        base_exec_time = task.execution_times[task.assigned_processor - 1]
                        
                        # Calculate outgoing security overhead
                        outgoing_security_overhead = 0
                        for succ_id in task.successors:
                            message = self.get_message(task.task_id, succ_id)
                            if not message:
                                continue
                            
                            # Only calculate source-side overhead
                            outgoing_security_overhead += self.calculate_sender_security_overhead(message, task.assigned_processor)
                        
                        # Set finish time with security overhead included
                        task.finish_time = task.start_time + base_exec_time + outgoing_security_overhead
                        task.is_scheduled = True
                        
                        # Update processor availability time
                        processor.available_time = max(processor.available_time, task.finish_time)
                        
                        scheduled_tasks.append(task)
                        remaining_tasks.remove(task)
        
        # Calculate the new makespan
        if all(task.is_scheduled for task in self.tasks):
            return max(task.finish_time for task in self.tasks)
        else:
            return float('inf')  # Could not schedule all tasks
    
    def finalize_schedule(self):
        """Finalize the schedule and return the makespan"""
        # Recalculate makespan with all security enhancements
        final_makespan = self.recalculate_makespan()
        
        # Update the schedule for reporting
        self.schedule = []
        for task in sorted(self.tasks, key=lambda t: t.task_id):
            # Calculate outgoing security overhead
            outgoing_security_overhead = 0
            for succ_id in task.successors:
                message = self.get_message(task.task_id, succ_id)
                if message:
                    outgoing_security_overhead += self.calculate_sender_security_overhead(message, task.assigned_processor)
            
            self.schedule.append({
                'task_id': task.task_id,
                'name': task.name,
                'processor': task.assigned_processor,
                'start_time': task.start_time,
                'finish_time': task.finish_time,
                'security_overhead': outgoing_security_overhead
            })
        
        return final_makespan
    
    def calculate_security_utility(self):
        """Calculate total security utility based on assigned security levels"""
        total_utility = 0
        
        for message in self.messages:
            message_utility = 0
            
            for service in ['confidentiality', 'integrity', 'authentication']:
                level = message.assigned_security[service]
                strength = self.security.strengths[service][level]
                weight = message.weights[service]
                
                message_utility += weight * strength
            
            total_utility += message_utility
        
        return total_utility
    
    def print_results(self):
        """Print detailed results of the RF-SHIELD algorithm"""
        print("\n=== RF-SHIELD Results ===")
        print(f"Makespan: {self.makespan:.2f} (Deadline: {self.deadline})")
        print(f"Original makespan: {self.original_makespan:.2f}")
        print(f"Security utility: {self.calculate_security_utility():.4f}")
        
        print("\nTask Schedule:")
        for task_info in sorted(self.schedule, key=lambda t: t['task_id']):
            print(f"Task {task_info['task_id']} ({task_info['name']}):")
            print(f"  Processor: {task_info['processor']}")
            print(f"  Start time: {task_info['start_time']:.2f}")
            print(f"  Finish time: {task_info['finish_time']:.2f}")
        
        print("\nSecurity Assignments:")
        for message in sorted(self.messages, key=lambda m: m.id):
            print(f"Message {message.id} (from T{message.source_id} to T{message.dest_id}, size {message.size}KB):")
            for service in ['confidentiality', 'integrity', 'authentication']:
                level = message.assigned_security[service]
                strength = self.security.strengths[service][level]
                print(f"  {service.capitalize()}: Level {level} (Strength: {strength:.4f})")
        
        print("\nSecurity Upgrades Applied:")
        for upgrade in self.upgrade_history:
            print(f"Message {upgrade['message_id']} {upgrade['service']}:")
            print(f"  Level {upgrade['old_level']} → {upgrade['new_level']}")
            print(f"  BCR: {upgrade['bcr']:.4f}, Success prob: {upgrade['success_prob']:.4f}")
            print(f"  New makespan: {upgrade['new_makespan']:.2f}")


def main():
    """Main function to test the RF-SHIELD algorithm"""
    print("Creating test case...")
    tasks, messages, processors, security_service, deadline = create_testcase()
    
    print("\nInitializing RF-SHIELD algorithm...")
    rf_shield = RF_SHIELD(tasks, messages, processors, security_service, deadline)
    
    print("\nRunning RF-SHIELD algorithm...")
    makespan, security_utility = rf_shield.run()
    
    if makespan is not None:
        print("\n=== Final Results ===")
        print(f"Makespan: {makespan:.2f}")
        print(f"Security utility: {security_utility:.4f}")
        
        # Print detailed results
        rf_shield.print_results()
    else:
        print("RF-SHIELD failed to find a feasible schedule")


def create_testcase():
    """Create a test case for the RF-SHIELD algorithm"""
    # Define tasks
    tasks = [
        Task(1, 'T1', [64, 56, 78], []),
        Task(2, 'T2', [46, 36, 58], [1]),
        Task(3, 'T3', [42, 63, 49], [1]),
        Task(4, 'T4', [36, 54, 42], [1]),
        Task(5, 'T5', [91, 45, 55], [2]),
        Task(6, 'T6', [58, 72, 84], [2, 3, 4]),
        Task(7, 'T7', [54, 81, 63], [3, 4]),
        Task(8, 'T8', [75, 42, 61], [5, 6, 7]),
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
        (0.3, 0.2, 0.4, 0.2, 0.3, 0.5),
        (0.1, 0.2, 0.2, 0.5, 0.3, 0.2),
        (0.1, 0.4, 0.3, 0.3, 0.6, 0.1),
        (0.2, 0.2, 0.4, 0.3, 0.5, 0.2),
        (0.4, 0.2, 0.3, 0.7, 0.1, 0.2),
        (0.3, 0.1, 0.1, 0.2, 0.4, 0.4),
        (0.3, 0.2, 0.1, 0.1, 0.3, 0.6),
        (0.2, 0.5, 0.3, 0.2, 0.6, 0.2),
        (0.3, 0.1, 0.4, 0.2, 0.4, 0.4),
        (0.3, 0.1, 0.4, 0.2, 0.4, 0.4),
        (0.3, 0.2, 0.3, 0.3, 0.6, 0.1),
        (0.4, 0.3, 0.4, 0.2, 0.3, 0.5),
    ]
    
    for msg, (conf_min, integ_min, auth_min, conf_w, integ_w, auth_w) in zip(messages, security_values):
        msg.set_security_requirements(conf_min, integ_min, auth_min, conf_w, integ_w, auth_w)
    
    processors = [
        Processor(1, [0, 1, 2]),
        Processor(2, [1, 0, 3]),
        Processor(3, [2, 3, 0])
    ]
    
    security_service = SecurityService()
    deadline = 500
    
    return tasks, messages, processors, security_service, deadline


if __name__ == "__main__":
    main()