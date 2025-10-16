"""
Custom Algorithm Example

This script demonstrates how to create and test custom scheduling algorithms
using the framework's extensible design.
"""

import sys
sys.path.append('..')

from src.core import WorkflowDAG, ScheduleResult
from src.algorithms import SchedulingAlgorithm, HEFTAlgorithm
from src.utils import DAGGenerator, Visualizer, SanityChecker


class RandomSchedulingAlgorithm(SchedulingAlgorithm):
    """
    Example custom algorithm: Random task ordering with EFT allocation.
    
    This demonstrates how to create a new scheduling algorithm by
    extending the SchedulingAlgorithm base class.
    """
    
    def __init__(self):
        super().__init__(name="Random-EFT")
    
    def schedule(self, dag: WorkflowDAG) -> ScheduleResult:
        """Schedule tasks in random order using EFT allocation."""
        import random
        from collections import defaultdict
        
        # Create random task ordering
        task_order = list(dag.task_list)
        random.shuffle(task_order)
        
        # Use HEFT's EFT scheduler for processor allocation
        heft = HEFTAlgorithm()
        task_schedule, processor_schedules, makespan = heft._eft_scheduler(dag, task_order)
        
        return ScheduleResult(
            task_schedule=task_schedule,
            processor_schedules=processor_schedules,
            makespan=makespan,
            algorithm_name=self.name,
            metadata={'task_order': task_order}
        )


def main():
    """Demonstrate custom algorithm usage."""
    print("Custom Algorithm Example\n")
    print("=" * 80)
    
    # Create a custom algorithm list
    algorithms = [
        HEFTAlgorithm(),
        RandomSchedulingAlgorithm()
    ]
    
    # Run sanity check with custom algorithms
    checker = SanityChecker()
    results = checker.run_sanity_check(
        num_tasks=10,
        num_processors=3,
        algorithms=algorithms,
        random_seed=42
    )
    
    print("\n✓ Custom algorithm test complete!")


if __name__ == "__main__":
    main()
