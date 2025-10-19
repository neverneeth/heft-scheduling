"""
HEFT (Heterogeneous Earliest Finish Time) Algorithm.

This module implements the classic HEFT scheduling algorithm, which uses
upward ranking to prioritize tasks and earliest finish time (EFT) for
processor allocation.

Reference:
    Topcuoglu, H., Hariri, S., & Wu, M. Y. (2002). Performance-effective and
    low-complexity task scheduling for heterogeneous computing. IEEE
    transactions on parallel and distributed systems, 13(3), 260-274.
"""

from typing import Dict, List, Tuple
from collections import defaultdict
from src.algorithms.base import SchedulingAlgorithm
from src.core.workflow_dag import WorkflowDAG
from src.core.schedule_result import ScheduleResult


class HEFTAlgorithm(SchedulingAlgorithm):
    """
    HEFT (Heterogeneous Earliest Finish Time) scheduling algorithm.
    
    HEFT operates in two phases:
    1. Task Prioritization: Compute upward rank for each task
    2. Processor Selection: Assign tasks to processors based on EFT
    
    The upward rank represents the length of the critical path from a task
    to the exit tasks, considering both computation and communication costs.
    """
    
    def __init__(self):
        """Initialize the HEFT algorithm."""
        super().__init__(name="HEFT")
    
    def schedule(self, dag: WorkflowDAG) -> ScheduleResult:
        """
        Schedule the workflow using HEFT algorithm.
        
        Args:
            dag: The workflow DAG to schedule
            
        Returns:
            ScheduleResult with task assignments and timing information
        """
        # Phase 1: Compute upward ranks
        upward_ranks = self._compute_upward_rank(dag)
        
        # Phase 2: Sort tasks by upward rank (descending)
        task_order = sorted(upward_ranks.keys(), key=lambda t: upward_ranks[t], reverse=True)
        
        # Phase 3: Schedule tasks using EFT
        task_schedule, processor_schedules, makespan = self._eft_scheduler(dag, task_order)
        
        # Create result with metadata
        metadata = {
            'upward_ranks': upward_ranks,
            'task_order': task_order
        }
        
        return ScheduleResult(
            task_schedule=task_schedule,
            processor_schedules=processor_schedules,
            makespan=makespan,
            algorithm_name=self.name,
            metadata=metadata
        )
    
    def _compute_upward_rank(self, dag: WorkflowDAG) -> Dict[str, float]:
        """
        Compute the upward rank for each task.
        
        The upward rank of a task is recursively defined as:
        rank_u(t) = w_avg(t) + max{c(t,s) + rank_u(s)} for all successors s
        
        where:
        - w_avg(t) is the average computation cost of task t
        - c(t,s) is the communication cost from t to s
        
        Args:
            dag: The workflow DAG
            
        Returns:
            Dictionary mapping task IDs to their upward ranks
        """
        rank = {}
        
        # Process tasks in reverse topological order (from exit to entry tasks)
        topological_order = dag.get_topological_order()
        
        for task in reversed(topological_order):
            w_avg = dag.get_avg_computation_cost(task)
            successors = dag.get_successors(task)
            
            if not successors:
                # Exit task: rank is just the average computation cost
                rank[task] = w_avg
                print(f"Task {task} is an exit task with rank {rank[task]:.2f}")
            else:
                # Rank is avg computation cost + max{comm_cost + successor_rank}
                max_successor_cost = max(
                    dag.get_communication_cost(task, succ) + rank[succ]
                    for succ in successors
                )
                rank[task] = w_avg + max_successor_cost
                print(f"Task {task} has successors {successors} with rank {rank[task]:.2f}")
        return rank
    
    def _eft_scheduler(self, dag: WorkflowDAG, task_order: List[str]) -> Tuple[Dict, Dict, float]:
        """
        Schedule tasks using Earliest Finish Time (EFT) heuristic with insertion-based approach.
        
        For each task in the given order, find the processor and time slot that allows
        the earliest finish time considering:
        - Processor availability and existing schedule
        - Data arrival time from predecessors
        - Task execution time on that processor
        - Insertion into idle time slots between existing tasks
        
        Args:
            dag: The workflow DAG
            task_order: List of tasks in scheduling priority order
            
        Returns:
            Tuple of (task_schedule, processor_schedules, makespan)
        """
        num_processors = dag.num_processors
        task_schedule = {}
        processor_schedules = defaultdict(list)
        
        for task in task_order:
            best_processor = 0
            best_eft = float('inf')
            best_est = 0.0
            
            # Try each processor to find the one with earliest finish time
            for proc in range(num_processors):
                execution_time = dag.get_computation_cost(task, proc)
                
                # Calculate data ready time (when all predecessor data is available)
                data_ready_time = 0.0
                for pred in dag.get_predecessors(task):
                    if pred in task_schedule:
                        pred_finish_time = task_schedule[pred]['finish_time']
                        pred_processor = task_schedule[pred]['processor']
                        
                        # Add communication cost if predecessor is on different processor
                        if pred_processor != proc:
                            comm_delay = dag.get_communication_cost(pred, task)
                        else:
                            comm_delay = 0
                        
                        data_ready_time = max(data_ready_time, pred_finish_time + comm_delay)
                
                # Find earliest available time slot on this processor (insertion-based)
                est, eft = self._find_earliest_slot(
                    processor_schedules[proc], 
                    data_ready_time, 
                    execution_time
                )
                
                # Keep track of best processor
                if eft < best_eft:
                    best_eft = eft
                    best_processor = proc
                    best_est = est
            
            # Assign task to best processor
            task_schedule[task] = {
                'processor': best_processor,
                'start_time': best_est,
                'finish_time': best_eft,
                'execution_time': dag.get_computation_cost(task, best_processor)
            }
            
            # Insert task into processor schedule in sorted order
            processor_schedules[best_processor].append({
                'task': task,
                'start': best_est,
                'finish': best_eft,
                'duration': dag.get_computation_cost(task, best_processor)
            })
            # Keep processor schedule sorted by start time
            processor_schedules[best_processor].sort(key=lambda x: x['start'])
        
        # Calculate makespan
        makespan = max(info['finish_time'] for info in task_schedule.values())
        
        return task_schedule, dict(processor_schedules), makespan
    
    def _find_earliest_slot(self, processor_schedule: List[Dict], data_ready_time: float, 
                           execution_time: float) -> Tuple[float, float]:
        """
        Find the earliest available time slot on a processor using insertion-based approach.
        
        This method searches for idle time slots between already scheduled tasks
        and can insert the new task into such slots if they are large enough.
        
        Args:
            processor_schedule: List of already scheduled tasks on this processor
            data_ready_time: Earliest time when task can start (data dependencies met)
            execution_time: Required execution time for the task
            
        Returns:
            Tuple of (earliest_start_time, earliest_finish_time)
        """
        if not processor_schedule:
            # Empty processor - can start as soon as data is ready
            return data_ready_time, data_ready_time + execution_time
        
        # Sort schedule by start time (should already be sorted)
        schedule = sorted(processor_schedule, key=lambda x: x['start'])
        
        # Try to insert before the first task
        if data_ready_time + execution_time <= schedule[0]['start']:
            return data_ready_time, data_ready_time + execution_time
        
        # Try to insert between consecutive tasks
        for i in range(len(schedule) - 1):
            current_task = schedule[i]
            next_task = schedule[i + 1]
            
            # Available slot starts after current task finishes and data is ready
            slot_start = max(current_task['finish'], data_ready_time)
            slot_end = next_task['start']
            
            # Check if task fits in this slot
            if slot_start + execution_time <= slot_end:
                return slot_start, slot_start + execution_time
        
        # If no slot found, append after the last task
        last_task = schedule[-1]
        est = max(last_task['finish'], data_ready_time)
        return est, est + execution_time
