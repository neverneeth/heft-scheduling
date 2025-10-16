"""
ScheduleResult: Encapsulates the output of a scheduling algorithm.

This module defines the data structure for storing and analyzing
scheduling results, including task assignments, timing information,
and performance metrics.
"""

from typing import Dict, List, Any
from collections import defaultdict


class ScheduleResult:
    """
    Represents the result of scheduling a workflow onto processors.
    
    Contains complete information about task assignments, timing,
    and performance metrics for a scheduled workflow.
    
    Attributes:
        task_schedule (Dict[str, Dict[str, Any]]): Task-level scheduling details
        processor_schedules (Dict[int, List[Dict[str, Any]]]): Processor-level scheduling details
        makespan (float): Total completion time of the workflow
        algorithm_name (str): Name of the algorithm that produced this schedule
        metadata (Dict[str, Any]): Additional algorithm-specific information
    """
    
    def __init__(
        self,
        task_schedule: Dict[str, Dict[str, Any]],
        processor_schedules: Dict[int, List[Dict[str, Any]]],
        makespan: float,
        algorithm_name: str,
        metadata: Dict[str, Any] = None
    ):
        """
        Initialize a ScheduleResult.
        
        Args:
            task_schedule: Dictionary mapping task IDs to scheduling information
                          Format: {task_id: {'processor': int, 'start_time': float,
                                             'finish_time': float, 'execution_time': float}}
            processor_schedules: Dictionary mapping processor IDs to their task lists
                                Format: {proc_id: [{'task': str, 'start': float,
                                                    'finish': float, 'duration': float}]}
            makespan: Total workflow completion time
            algorithm_name: Name of the scheduling algorithm
            metadata: Optional dictionary for algorithm-specific data
        """
        self.task_schedule = task_schedule
        self.processor_schedules = processor_schedules
        self.makespan = makespan
        self.algorithm_name = algorithm_name
        self.metadata = metadata if metadata is not None else {}
    
    def get_task_processor(self, task: str) -> int:
        """
        Get the processor assigned to a task.
        
        Args:
            task: Task identifier
            
        Returns:
            Processor ID
        """
        return self.task_schedule[task]['processor']
    
    def get_task_start_time(self, task: str) -> float:
        """
        Get the start time of a task.
        
        Args:
            task: Task identifier
            
        Returns:
            Start time
        """
        return self.task_schedule[task]['start_time']
    
    def get_task_finish_time(self, task: str) -> float:
        """
        Get the finish time of a task.
        
        Args:
            task: Task identifier
            
        Returns:
            Finish time
        """
        return self.task_schedule[task]['finish_time']
    
    def get_processor_utilization(self) -> Dict[int, float]:
        """
        Calculate the utilization of each processor.
        
        Returns:
            Dictionary mapping processor ID to utilization percentage
        """
        utilization = {}
        for proc_id, tasks in self.processor_schedules.items():
            if not tasks:
                utilization[proc_id] = 0.0
            else:
                busy_time = sum(task['duration'] for task in tasks)
                utilization[proc_id] = (busy_time / self.makespan) * 100 if self.makespan > 0 else 0.0
        return utilization
    
    def get_average_utilization(self) -> float:
        """
        Calculate the average processor utilization across all processors.
        
        Returns:
            Average utilization percentage
        """
        utilizations = self.get_processor_utilization()
        return sum(utilizations.values()) / len(utilizations) if utilizations else 0.0
    
    def get_schedule_summary(self) -> str:
        """
        Generate a human-readable summary of the schedule.
        
        Returns:
            Multi-line string summary
        """
        lines = [
            f"Algorithm: {self.algorithm_name}",
            f"Makespan: {self.makespan:.2f}",
            f"Tasks scheduled: {len(self.task_schedule)}",
            f"Average processor utilization: {self.get_average_utilization():.2f}%",
            "\nTask Schedule:"
        ]
        
        for task in sorted(self.task_schedule.keys()):
            info = self.task_schedule[task]
            lines.append(
                f"  {task}: P{info['processor']} "
                f"[{info['start_time']:.1f} - {info['finish_time']:.1f}]"
            )
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """String representation of the ScheduleResult."""
        return (
            f"ScheduleResult(algorithm={self.algorithm_name}, "
            f"makespan={self.makespan:.2f}, tasks={len(self.task_schedule)})"
        )
