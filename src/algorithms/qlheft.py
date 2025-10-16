"""
QL-HEFT: Q-Learning Enhanced HEFT Algorithm.

This module implements Q-learning based task ordering strategies combined
with HEFT's EFT processor allocation. Multiple variants are provided:
- Large State: Q(scheduled_set, next_task)
- Small State: Q(last_task, next_task)
"""

import random
import math
from typing import Dict, List, Tuple, Set, FrozenSet
from collections import defaultdict
from src.algorithms.base import SchedulingAlgorithm
from src.core.workflow_dag import WorkflowDAG
from src.core.schedule_result import ScheduleResult


class QLHEFTLargeState(SchedulingAlgorithm):
    """
    QL-HEFT with Large State Space Q-learning.
    
    State: Set of all scheduled tasks (frozenset)
    Action: Next task to schedule from viable tasks
    Reward: Upward rank of the selected task
    
    This variant maintains a Q-table with entries Q[(scheduled_tasks, next_task)]
    which can capture complex dependencies but requires more training episodes.
    """
    
    def __init__(
        self,
        num_episodes: int = 10000,
        epsilon: float = 0.1,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9
    ):
        """
        Initialize QL-HEFT Large State algorithm.
        
        Args:
            num_episodes: Number of Q-learning training episodes
            epsilon: Exploration rate for epsilon-greedy policy
            learning_rate: Learning rate (alpha) for Q-value updates
            discount_factor: Discount factor (gamma) for future rewards
        """
        super().__init__(name="QL-HEFT-LargeState")
        self.num_episodes = num_episodes
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
    
    def schedule(self, dag: WorkflowDAG) -> ScheduleResult:
        """
        Schedule the workflow using Q-learning with large state space.
        
        Args:
            dag: The workflow DAG to schedule
            
        Returns:
            ScheduleResult with task assignments and timing information
        """
        # Compute upward ranks for rewards
        upward_ranks = self._compute_upward_rank(dag)
        
        # Train Q-learning model
        Q = self._train_q_learning(dag, upward_ranks)
        
        # Extract optimal task order from learned Q-table
        task_order = self._extract_task_order(dag, Q)
        
        # Use HEFT's EFT scheduler for processor allocation
        task_schedule, processor_schedules, makespan = self._eft_scheduler(dag, task_order)
        
        metadata = {
            'upward_ranks': upward_ranks,
            'task_order': task_order,
            'num_episodes': self.num_episodes,
            'q_table_size': len(Q)
        }
        
        return ScheduleResult(
            task_schedule=task_schedule,
            processor_schedules=processor_schedules,
            makespan=makespan,
            algorithm_name=self.name,
            metadata=metadata
        )
    
    def _train_q_learning(
        self,
        dag: WorkflowDAG,
        upward_ranks: Dict[str, float]
    ) -> Dict[Tuple[FrozenSet[str], str], float]:
        """
        Train Q-learning model with large state representation.
        
        Args:
            dag: The workflow DAG
            upward_ranks: Pre-computed upward ranks for rewards
            
        Returns:
            Trained Q-table
        """
        Q = defaultdict(float)
        
        for episode in range(self.num_episodes):
            scheduled = frozenset()
            
            while len(scheduled) != len(dag.task_list):
                # Get viable tasks (actions)
                viable = dag.get_viable_tasks(scheduled)
                
                # Epsilon-greedy action selection
                if random.random() < self.epsilon:
                    action = random.choice(viable)
                else:
                    # Choose action with highest Q-value
                    q_values = {a: Q.get((scheduled, a), 0) for a in viable}
                    max_q = max(q_values.values())
                    best_actions = [a for a, v in q_values.items() if v == max_q]
                    action = random.choice(best_actions)
                
                # Get reward (upward rank of selected task)
                reward = upward_ranks[action]
                
                # Move to next state
                next_state = scheduled.union({action})
                next_viable = dag.get_viable_tasks(next_state)
                
                # Calculate max Q-value for next state
                if next_viable:
                    max_next_q = max(Q.get((next_state, a), 0) for a in next_viable)
                else:
                    max_next_q = 0
                
                # Q-learning update
                old_q = Q.get((scheduled, action), 0)
                Q[(scheduled, action)] = old_q + self.learning_rate * (
                    reward + self.discount_factor * max_next_q - old_q
                )
                
                scheduled = next_state
        
        return dict(Q)
    
    def _extract_task_order(
        self,
        dag: WorkflowDAG,
        Q: Dict[Tuple[FrozenSet[str], str], float]
    ) -> List[str]:
        """
        Extract optimal task ordering from learned Q-table.
        
        Args:
            dag: The workflow DAG
            Q: Trained Q-table
            
        Returns:
            List of tasks in optimal order
        """
        task_order = []
        state = frozenset()
        
        while len(task_order) != len(dag.task_list):
            viable = dag.get_viable_tasks(state)
            
            # Choose action with highest Q-value
            q_values = {a: Q.get((state, a), 0) for a in viable}
            best_action = max(q_values, key=q_values.get)
            
            task_order.append(best_action)
            state = state.union({best_action})
        
        return task_order
    
    def _compute_upward_rank(self, dag: WorkflowDAG) -> Dict[str, float]:
        """Compute upward rank (same as HEFT)."""
        rank = {}
        topological_order = dag.get_topological_order()
        
        for task in reversed(topological_order):
            w_avg = dag.get_avg_computation_cost(task)
            successors = dag.get_successors(task)
            
            if not successors:
                rank[task] = w_avg
            else:
                max_successor_cost = max(
                    dag.get_communication_cost(task, succ) + rank[succ]
                    for succ in successors
                )
                rank[task] = w_avg + max_successor_cost
        
        return rank
    
    def _eft_scheduler(self, dag: WorkflowDAG, task_order: List[str]) -> Tuple[Dict, Dict, float]:
        """EFT scheduler (same as HEFT)."""
        num_processors = dag.num_processors
        processor_avail = [0.0] * num_processors
        task_schedule = {}
        processor_schedules = defaultdict(list)
        
        for task in task_order:
            best_processor = 0
            best_eft = float('inf')
            best_est = 0.0
            
            for proc in range(num_processors):
                est = processor_avail[proc]
                
                for pred in dag.get_predecessors(task):
                    if pred in task_schedule:
                        pred_finish_time = task_schedule[pred]['finish_time']
                        pred_processor = task_schedule[pred]['processor']
                        comm_delay = dag.get_communication_cost(pred, task) if pred_processor != proc else 0
                        est = max(est, pred_finish_time + comm_delay)
                
                execution_time = dag.get_computation_cost(task, proc)
                eft = est + execution_time
                
                if eft < best_eft:
                    best_eft = eft
                    best_processor = proc
                    best_est = est
            
            task_schedule[task] = {
                'processor': best_processor,
                'start_time': best_est,
                'finish_time': best_eft,
                'execution_time': dag.get_computation_cost(task, best_processor)
            }
            
            processor_avail[best_processor] = best_eft
            
            processor_schedules[best_processor].append({
                'task': task,
                'start': best_est,
                'finish': best_eft,
                'duration': dag.get_computation_cost(task, best_processor)
            })
        
        makespan = max(info['finish_time'] for info in task_schedule.values())
        return task_schedule, dict(processor_schedules), makespan


class QLHEFTSmallState(SchedulingAlgorithm):
    """
    QL-HEFT with Small State Space Q-learning.
    
    State: Last scheduled task
    Action: Next task to schedule from viable tasks
    Reward: Upward rank of the selected task
    
    This variant uses Q[(last_task, next_task)] which is more compact
    and faster to train, though it captures less state information.
    """
    
    def __init__(
        self,
        num_episodes: int = 50000,
        epsilon: float = 0.2,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        convergence_window: int = 40,
        convergence_threshold: float = 0.2,
        learning_rate_decay: str = "none"
    ):
        """
        Initialize QL-HEFT Small State algorithm.
        
        Args:
            num_episodes: Maximum number of training episodes
            epsilon: Exploration rate for epsilon-greedy policy
            learning_rate: Initial learning rate (alpha)
            discount_factor: Discount factor (gamma)
            convergence_window: Window size for convergence detection
            convergence_threshold: Threshold for mean absolute Q-value change
            learning_rate_decay: Type of decay ("none", "harmonic", "exponential")
        """
        super().__init__(name="QL-HEFT-SmallState")
        self.num_episodes = num_episodes
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold
        self.learning_rate_decay = learning_rate_decay
    
    def schedule(self, dag: WorkflowDAG) -> ScheduleResult:
        """
        Schedule the workflow using Q-learning with small state space.
        
        Args:
            dag: The workflow DAG to schedule
            
        Returns:
            ScheduleResult with task assignments and timing information
        """
        upward_ranks = self._compute_upward_rank(dag)
        Q, mean_abs_diffs, episodes_run = self._train_q_learning(dag, upward_ranks)
        task_order = self._extract_task_order(dag, Q)
        task_schedule, processor_schedules, makespan = self._eft_scheduler(dag, task_order)
        
        metadata = {
            'upward_ranks': upward_ranks,
            'task_order': task_order,
            'episodes_run': episodes_run,
            'q_table_size': len(Q),
            'convergence_history': mean_abs_diffs
        }
        
        return ScheduleResult(
            task_schedule=task_schedule,
            processor_schedules=processor_schedules,
            makespan=makespan,
            algorithm_name=self.name,
            metadata=metadata
        )
    
    def _get_learning_rate(self, episode: int) -> float:
        """Calculate learning rate with optional decay."""
        if self.learning_rate_decay == "harmonic":
            return self.learning_rate / (1 + episode)
        elif self.learning_rate_decay == "exponential":
            return self.learning_rate * (0.99 ** math.sqrt(episode))
        else:
            return self.learning_rate
    
    def _train_q_learning(
        self,
        dag: WorkflowDAG,
        upward_ranks: Dict[str, float]
    ) -> Tuple[Dict[Tuple[str, str], float], List[float], int]:
        """
        Train Q-learning model with small state representation.
        
        Returns:
            Tuple of (Q-table, convergence history, episodes run)
        """
        Q = defaultdict(float)
        recent_diffs = []
        mean_abs_diffs = []
        converged = False
        episode = 0
        
        while not converged and episode < self.num_episodes:
            lr = self._get_learning_rate(episode)
            
            scheduled = set()
            entry_tasks = dag.get_entry_tasks()
            last_task = random.choice(entry_tasks)
            scheduled.add(last_task)
            
            abs_diffs = []
            
            while len(scheduled) != len(dag.task_list):
                viable = [t for t in dag.get_viable_tasks(scheduled) if t not in scheduled]
                
                if not viable:
                    break
                
                # Epsilon-greedy action selection
                if random.random() < self.epsilon:
                    action = random.choice(viable)
                else:
                    q_values = {a: Q.get((last_task, a), 0) for a in viable}
                    max_q = max(q_values.values())
                    best_actions = [a for a, v in q_values.items() if v == max_q]
                    action = random.choice(best_actions)
                
                reward = upward_ranks[action]
                old_q = Q.get((last_task, action), 0)
                
                scheduled.add(action)
                next_viable = [t for t in dag.get_viable_tasks(scheduled) if t not in scheduled]
                
                max_next_q = max(Q.get((action, a), 0) for a in next_viable) if next_viable else 0
                
                # Q-learning update
                Q[(last_task, action)] = old_q + lr * (reward + self.discount_factor * max_next_q - old_q)
                abs_diffs.append(abs(old_q - Q[(last_task, action)]))
                
                last_task = action
            
            # Track convergence
            mean_diff = sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0
            recent_diffs.append(mean_diff)
            mean_abs_diffs.append(mean_diff)
            
            if len(recent_diffs) > self.convergence_window:
                recent_diffs.pop(0)
            
            if len(recent_diffs) == self.convergence_window:
                avg_recent_diff = sum(recent_diffs) / len(recent_diffs)
                if avg_recent_diff < self.convergence_threshold:
                    converged = True
            
            episode += 1
        
        return dict(Q), mean_abs_diffs, episode
    
    def _extract_task_order(
        self,
        dag: WorkflowDAG,
        Q: Dict[Tuple[str, str], float]
    ) -> List[str]:
        """Extract optimal task ordering from learned Q-table."""
        task_order = []
        scheduled = set()
        
        entry_tasks = dag.get_entry_tasks()
        last_task = max(entry_tasks, key=lambda t: Q.get((None, t), 0))
        task_order.append(last_task)
        scheduled.add(last_task)
        
        while len(scheduled) != len(dag.task_list):
            viable = [t for t in dag.get_viable_tasks(scheduled) if t not in scheduled]
            
            if not viable:
                break
            
            q_values = {a: Q.get((last_task, a), 0) for a in viable}
            max_q = max(q_values.values())
            best_actions = [a for a, v in q_values.items() if v == max_q]
            next_task = random.choice(best_actions)
            
            task_order.append(next_task)
            scheduled.add(next_task)
            last_task = next_task
        
        return task_order
    
    def _compute_upward_rank(self, dag: WorkflowDAG) -> Dict[str, float]:
        """Compute upward rank (same as HEFT)."""
        rank = {}
        topological_order = dag.get_topological_order()
        
        for task in reversed(topological_order):
            w_avg = dag.get_avg_computation_cost(task)
            successors = dag.get_successors(task)
            
            if not successors:
                rank[task] = w_avg
            else:
                max_successor_cost = max(
                    dag.get_communication_cost(task, succ) + rank[succ]
                    for succ in successors
                )
                rank[task] = w_avg + max_successor_cost
        
        return rank
    
    def _eft_scheduler(self, dag: WorkflowDAG, task_order: List[str]) -> Tuple[Dict, Dict, float]:
        """EFT scheduler (same as HEFT)."""
        num_processors = dag.num_processors
        processor_avail = [0.0] * num_processors
        task_schedule = {}
        processor_schedules = defaultdict(list)
        
        for task in task_order:
            best_processor = 0
            best_eft = float('inf')
            best_est = 0.0
            
            for proc in range(num_processors):
                est = processor_avail[proc]
                
                for pred in dag.get_predecessors(task):
                    if pred in task_schedule:
                        pred_finish_time = task_schedule[pred]['finish_time']
                        pred_processor = task_schedule[pred]['processor']
                        comm_delay = dag.get_communication_cost(pred, task) if pred_processor != proc else 0
                        est = max(est, pred_finish_time + comm_delay)
                
                execution_time = dag.get_computation_cost(task, proc)
                eft = est + execution_time
                
                if eft < best_eft:
                    best_eft = eft
                    best_processor = proc
                    best_est = est
            
            task_schedule[task] = {
                'processor': best_processor,
                'start_time': best_est,
                'finish_time': best_eft,
                'execution_time': dag.get_computation_cost(task, best_processor)
            }
            
            processor_avail[best_processor] = best_eft
            
            processor_schedules[best_processor].append({
                'task': task,
                'start': best_est,
                'finish': best_eft,
                'duration': dag.get_computation_cost(task, best_processor)
            })
        
        makespan = max(info['finish_time'] for info in task_schedule.values())
        return task_schedule, dict(processor_schedules), makespan
