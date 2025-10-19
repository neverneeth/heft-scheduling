"""
Schedule Validator: Validates schedule results against workflow DAG constraints.

This module provides functionality to check if a schedule is valid with respect
to task dependencies, processor constraints, and execution timing.
"""

from typing import List, Set, Dict, Any
from src.core.workflow_dag import WorkflowDAG
from src.core.schedule_result import ScheduleResult


class ScheduleValidator:
    """
    Validates schedules against workflow DAG constraints.
    
    This class performs multiple validation checks on a schedule result
    to ensure it represents a feasible solution that respects the 
    precedence constraints defined in the workflow DAG.
    """
    
    def __init__(self, dag: WorkflowDAG, schedule: ScheduleResult):
        """
        Initialize a schedule validator.
        
        Args:
            dag: WorkflowDAG containing task dependencies and costs
            schedule: ScheduleResult to validate
        """
        self.dag = dag
        self.schedule = schedule
        self.errors = []
        self.warnings = []
    
    def validate(self, verbose: bool = True) -> bool:
        """
        Validate the schedule against the DAG constraints.
        
        Args:
            verbose: Whether to print the validation report
            
        Returns:
            True if the schedule is valid, False otherwise
        """
        self.errors = []
        self.warnings = []
        
        self._check_completeness()
        self._check_precedence_constraints()
        self._check_processor_conflicts()
        self._check_task_durations()
        self._check_makespan()
        
        is_valid = len(self.errors) == 0
        
        if verbose:
            print(self.get_validation_report())
        
        return is_valid
    
    def _check_completeness(self):
        """Check that all tasks in the DAG are scheduled."""
        scheduled_tasks = set(self.schedule.task_schedule.keys())
        dag_tasks = set(self.dag.task_list)
        
        if scheduled_tasks != dag_tasks:
            missing_tasks = dag_tasks - scheduled_tasks
            extra_tasks = scheduled_tasks - dag_tasks
            
            if missing_tasks:
                self.errors.append(
                    f"Missing tasks: {', '.join(sorted(missing_tasks))} are in the DAG but not scheduled"
                )
            
            if extra_tasks:
                self.errors.append(
                    f"Extra tasks: {', '.join(sorted(extra_tasks))} are scheduled but not in the DAG"
                )
    
    def _check_precedence_constraints(self):
        """Check that tasks respect their precedence constraints."""
        for task in self.schedule.task_schedule:
            task_info = self.schedule.task_schedule[task]
            task_start = task_info['start_time']
            task_processor = task_info['processor']
            
            # Check all predecessors
            for pred_task in self.dag.get_predecessors(task):
                if pred_task not in self.schedule.task_schedule:
                    self.errors.append(f"Predecessor {pred_task} of task {task} is not scheduled")
                    continue
                
                pred_info = self.schedule.task_schedule[pred_task]
                pred_finish = pred_info['finish_time']
                pred_processor = pred_info['processor']
                
                # Calculate communication time (0 if on same processor)
                comm_time = 0 if task_processor == pred_processor else self.dag.get_communication_cost(pred_task, task)
                
                # Check if the task starts after its predecessor finishes plus communication time
                if task_start < pred_finish + comm_time - 1e-6:  # Allow for floating point imprecision
                    self.errors.append(
                        f"Precedence violation: Task {task} starts at {task_start:.2f} before its "
                        f"predecessor {pred_task} finishes at {pred_finish:.2f} "
                        f"plus communication time {comm_time:.2f}"
                    )
    
    def _check_processor_conflicts(self):
        """Check that no two tasks overlap on the same processor."""
        for processor, tasks in self.schedule.processor_schedules.items():
            # Sort tasks by start time
            sorted_tasks = sorted(tasks, key=lambda x: x['start'])
            
            # Check for overlaps
            for i in range(1, len(sorted_tasks)):
                prev_task = sorted_tasks[i-1]
                curr_task = sorted_tasks[i]
                
                prev_end = prev_task['start'] + prev_task['duration']
                
                if prev_end > curr_task['start'] + 1e-6:  # Allow for floating point imprecision
                    self.errors.append(
                        f"Processor conflict: Tasks {prev_task['task']} and {curr_task['task']} "
                        f"overlap on processor {processor}. "
                        f"{prev_task['task']} ends at {prev_end:.2f}, but "
                        f"{curr_task['task']} starts at {curr_task['start']:.2f}"
                    )
    
    def _check_task_durations(self):
        """Check that task durations match their computation costs."""
        for task, info in self.schedule.task_schedule.items():
            processor = info['processor']
            expected_duration = self.dag.get_computation_cost(task, processor)
            actual_duration = info['finish_time'] - info['start_time']
            
            # Allow for small floating-point differences
            if abs(expected_duration - actual_duration) > 1e-6:
                self.errors.append(
                    f"Duration mismatch: Task {task} on processor {processor} has duration "
                    f"{actual_duration:.2f}, but expected {expected_duration:.2f}"
                )
    
    def _check_makespan(self):
        """Check that the reported makespan matches the actual completion time."""
        actual_makespan = max(
            info['finish_time'] 
            for info in self.schedule.task_schedule.values()
        )
        
        if abs(actual_makespan - self.schedule.makespan) > 1e-6:
            self.warnings.append(
                f"Makespan inconsistency: Reported makespan is {self.schedule.makespan:.2f}, "
                f"but the latest task finishes at {actual_makespan:.2f}"
            )
    
    def get_validation_report(self) -> str:
        """
        Generate a human-readable validation report.
        
        Returns:
            A multi-line string summarizing the validation results
        """
        if not self.errors and not self.warnings:
            return "✅ Schedule is valid. All constraints are satisfied."
        
        report = []
        
        if self.errors:
            report.append(f"❌ Schedule validation failed with {len(self.errors)} errors:")
            for i, error in enumerate(self.errors):
                report.append(f"   {i+1}. {error}")
            report.append("")
        
        if self.warnings:
            report.append(f"⚠️ Schedule has {len(self.warnings)} warnings:")
            for i, warning in enumerate(self.warnings):
                report.append(f"   {i+1}. {warning}")
        
        return "\n".join(report)


def validate_schedule(dag: WorkflowDAG, schedule: ScheduleResult, verbose: bool = True) -> bool:
    """
    Validate a schedule against a DAG.
    
    Args:
        dag: WorkflowDAG object containing task dependencies
        schedule: ScheduleResult object to validate
        verbose: Whether to print validation results
    
    Returns:
        True if the schedule is valid, False otherwise
    """
    validator = ScheduleValidator(dag, schedule)
    return validator.validate(verbose=verbose)