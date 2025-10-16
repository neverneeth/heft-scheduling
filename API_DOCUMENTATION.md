# API Documentation

Complete API reference for the HEFT Scheduling Framework.

## Table of Contents

1. [Core Module](#core-module)
2. [Algorithms Module](#algorithms-module)
3. [Utils Module](#utils-module)

---

## Core Module

### WorkflowDAG

Represents a workflow as a Directed Acyclic Graph.

#### Constructor

```python
WorkflowDAG(
    edges: List[Tuple[str, str]],
    computation_costs: List[List[float]],
    communication_costs: Dict[Tuple[str, str], float]
)
```

**Parameters:**
- `edges`: List of (source, destination) tuples representing dependencies
- `computation_costs`: 2D array where W[i][j] = cost of task i on processor j
- `communication_costs`: Dictionary mapping (source, dest) to communication cost

**Raises:**
- `ValueError`: If graph contains cycles or costs are inconsistent

#### Properties

- `graph` (nx.DiGraph): NetworkX directed graph
- `task_list` (List[str]): Ordered list of task identifiers
- `task_index` (Dict[str, int]): Mapping from task ID to index
- `computation_costs` (List[List[float]]): Computation cost matrix
- `communication_costs` (Dict): Communication cost dictionary
- `num_tasks` (int): Number of tasks
- `num_processors` (int): Number of processors

#### Methods

##### get_predecessors(task: str) -> List[str]
Returns immediate predecessors of a task.

##### get_successors(task: str) -> List[str]
Returns immediate successors of a task.

##### get_entry_tasks() -> List[str]
Returns tasks with no predecessors.

##### get_exit_tasks() -> List[str]
Returns tasks with no successors.

##### get_viable_tasks(scheduled: Set[str]) -> List[str]
Returns tasks ready to be scheduled (all predecessors scheduled).

##### get_computation_cost(task: str, processor: int) -> float
Returns execution time of task on processor.

##### get_communication_cost(source: str, dest: str) -> float
Returns communication cost between two tasks.

##### get_avg_computation_cost(task: str) -> float
Returns average computation cost across all processors.

##### get_topological_order() -> List[str]
Returns tasks in topological order.

##### copy() -> WorkflowDAG
Creates a deep copy of the DAG.

---

### SystemModel

Represents the computing environment.

#### Constructor

```python
SystemModel(
    num_processors: int,
    processor_names: Optional[List[str]] = None
)
```

**Parameters:**
- `num_processors`: Number of processors in the system
- `processor_names`: Optional custom names for processors (defaults to P0, P1, ...)

#### Properties

- `num_processors` (int): Number of processors
- `processor_names` (List[str]): Processor identifiers

#### Methods

##### get_processor_name(processor_id: int) -> str
Returns the name of a processor by its ID.

---

### ScheduleResult

Encapsulates scheduling algorithm output.

#### Constructor

```python
ScheduleResult(
    task_schedule: Dict[str, Dict[str, Any]],
    processor_schedules: Dict[int, List[Dict[str, Any]]],
    makespan: float,
    algorithm_name: str,
    metadata: Dict[str, Any] = None
)
```

**Parameters:**
- `task_schedule`: Task-level scheduling details
  - Format: `{task: {'processor': int, 'start_time': float, 'finish_time': float, 'execution_time': float}}`
- `processor_schedules`: Processor-level scheduling details
  - Format: `{proc: [{'task': str, 'start': float, 'finish': float, 'duration': float}]}`
- `makespan`: Total workflow completion time
- `algorithm_name`: Name of the algorithm
- `metadata`: Optional algorithm-specific data

#### Properties

- `task_schedule` (Dict): Per-task scheduling information
- `processor_schedules` (Dict): Per-processor task lists
- `makespan` (float): Total completion time
- `algorithm_name` (str): Algorithm identifier
- `metadata` (Dict): Additional information

#### Methods

##### get_task_processor(task: str) -> int
Returns the processor assigned to a task.

##### get_task_start_time(task: str) -> float
Returns the start time of a task.

##### get_task_finish_time(task: str) -> float
Returns the finish time of a task.

##### get_processor_utilization() -> Dict[int, float]
Returns utilization percentage for each processor.

##### get_average_utilization() -> float
Returns average processor utilization across all processors.

##### get_schedule_summary() -> str
Generates a human-readable summary of the schedule.

---

## Algorithms Module

### SchedulingAlgorithm (Abstract Base Class)

Base class for all scheduling algorithms.

#### Constructor

```python
SchedulingAlgorithm(
    name: str,
    config: Dict[str, Any] = None
)
```

**Parameters:**
- `name`: Algorithm identifier
- `config`: Optional configuration parameters

#### Abstract Methods

##### schedule(dag: WorkflowDAG) -> ScheduleResult
**Must be implemented by subclasses.**

Schedules the workflow DAG onto processors.

**Parameters:**
- `dag`: Workflow DAG to schedule

**Returns:**
- `ScheduleResult`: Complete scheduling result

#### Methods

##### get_name() -> str
Returns the algorithm name.

##### get_config() -> Dict[str, Any]
Returns a copy of the configuration.

##### set_config(key: str, value: Any)
Sets a configuration parameter.

---

### HEFTAlgorithm

HEFT (Heterogeneous Earliest Finish Time) implementation.

#### Constructor

```python
HEFTAlgorithm()
```

No parameters required.

#### Methods

##### schedule(dag: WorkflowDAG) -> ScheduleResult
Schedules using upward rank prioritization and EFT allocation.

**Returns:**
- `ScheduleResult` with metadata:
  - `upward_ranks`: Computed ranks for each task
  - `task_order`: Task scheduling order

---

### QLHEFTLargeState

Q-Learning HEFT with large state space representation.

#### Constructor

```python
QLHEFTLargeState(
    num_episodes: int = 10000,
    epsilon: float = 0.1,
    learning_rate: float = 0.1,
    discount_factor: float = 0.9
)
```

**Parameters:**
- `num_episodes`: Number of Q-learning training episodes
- `epsilon`: Exploration rate (0-1)
- `learning_rate`: Learning rate alpha (0-1)
- `discount_factor`: Discount factor gamma (0-1)

#### Methods

##### schedule(dag: WorkflowDAG) -> ScheduleResult
Schedules using Q-learning with state = set of scheduled tasks.

**Returns:**
- `ScheduleResult` with metadata:
  - `upward_ranks`: Computed ranks
  - `task_order`: Learned task order
  - `num_episodes`: Episodes trained
  - `q_table_size`: Size of Q-table

---

### QLHEFTSmallState

Q-Learning HEFT with small state space representation.

#### Constructor

```python
QLHEFTSmallState(
    num_episodes: int = 50000,
    epsilon: float = 0.2,
    learning_rate: float = 0.1,
    discount_factor: float = 0.9,
    convergence_window: int = 40,
    convergence_threshold: float = 0.2,
    learning_rate_decay: str = "none"
)
```

**Parameters:**
- `num_episodes`: Maximum training episodes
- `epsilon`: Exploration rate
- `learning_rate`: Initial learning rate
- `discount_factor`: Discount factor gamma
- `convergence_window`: Window size for convergence detection
- `convergence_threshold`: Threshold for mean absolute Q-value change
- `learning_rate_decay`: Type of decay ("none", "harmonic", "exponential")

#### Methods

##### schedule(dag: WorkflowDAG) -> ScheduleResult
Schedules using Q-learning with state = last scheduled task.

**Returns:**
- `ScheduleResult` with metadata:
  - `upward_ranks`: Computed ranks
  - `task_order`: Learned task order
  - `episodes_run`: Actual episodes run
  - `q_table_size`: Size of Q-table
  - `convergence_history`: List of mean absolute Q-value changes

---

## Utils Module

### DAGGenerator

Utilities for generating random workflow DAGs.

All methods are static and return a `WorkflowDAG` object.

#### generate_random_dag

```python
DAGGenerator.generate_random_dag(
    num_tasks: int,
    num_processors: int,
    edge_probability: float = 0.3,
    computation_cost_range: Tuple[float, float] = (5, 50),
    communication_cost_range: Tuple[float, float] = (1, 20),
    random_seed: Optional[int] = None
) -> WorkflowDAG
```

Generates a random DAG with specified parameters.

**Parameters:**
- `num_tasks`: Number of tasks
- `num_processors`: Number of processors
- `edge_probability`: Probability of edge between adjacent layers
- `computation_cost_range`: (min, max) for execution times
- `communication_cost_range`: (min, max) for transfer times
- `random_seed`: Seed for reproducibility

#### generate_layered_dag

```python
DAGGenerator.generate_layered_dag(
    num_layers: int,
    tasks_per_layer: int,
    num_processors: int,
    edge_density: float = 0.5,
    computation_cost_range: Tuple[float, float] = (5, 50),
    communication_cost_range: Tuple[float, float] = (1, 20),
    random_seed: Optional[int] = None
) -> WorkflowDAG
```

Generates a layered DAG with uniform structure.

**Parameters:**
- `num_layers`: Number of layers
- `tasks_per_layer`: Tasks in each layer
- `num_processors`: Number of processors
- `edge_density`: Proportion of inter-layer edges (0-1)
- Other parameters same as `generate_random_dag`

#### generate_fork_join_dag

```python
DAGGenerator.generate_fork_join_dag(
    num_initial_tasks: int,
    num_parallel_tasks: int,
    num_final_tasks: int,
    num_processors: int,
    computation_cost_range: Tuple[float, float] = (5, 50),
    communication_cost_range: Tuple[float, float] = (1, 20),
    random_seed: Optional[int] = None
) -> WorkflowDAG
```

Generates a fork-join style DAG.

**Parameters:**
- `num_initial_tasks`: Number of initial (fork) tasks
- `num_parallel_tasks`: Number of parallel middle tasks
- `num_final_tasks`: Number of final (join) tasks
- Other parameters same as `generate_random_dag`

---

### Visualizer

Visualization utilities for DAGs and schedules.

All methods are static and return a `matplotlib.Figure` object.

#### visualize_dag

```python
Visualizer.visualize_dag(
    dag: WorkflowDAG,
    title: str = "Workflow DAG",
    figsize: tuple = (10, 6),
    show: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure
```

Visualizes the structure of a workflow DAG.

**Parameters:**
- `dag`: Workflow DAG to visualize
- `title`: Plot title
- `figsize`: Figure size (width, height)
- `show`: Whether to display the plot
- `save_path`: Optional path to save figure

#### visualize_gantt_chart

```python
Visualizer.visualize_gantt_chart(
    result: ScheduleResult,
    title: Optional[str] = None,
    figsize: tuple = (12, 6),
    show: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure
```

Creates a Gantt chart visualization of a schedule.

**Parameters:**
- `result`: Scheduling result to visualize
- `title`: Plot title (defaults to algorithm name)
- Other parameters same as `visualize_dag`

#### visualize_convergence

```python
Visualizer.visualize_convergence(
    convergence_history: List[float],
    title: str = "Q-Learning Convergence",
    window_size: int = 100,
    figsize: tuple = (10, 5),
    show: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure
```

Visualizes Q-learning convergence history.

**Parameters:**
- `convergence_history`: List of mean absolute Q-value changes
- `window_size`: Size of moving average window
- Other parameters same as `visualize_dag`

#### compare_algorithms

```python
Visualizer.compare_algorithms(
    results: List[ScheduleResult],
    figsize: tuple = (10, 6),
    show: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure
```

Creates comparison charts for multiple algorithm results.

**Parameters:**
- `results`: List of ScheduleResult objects to compare
- Other parameters same as `visualize_dag`

---

### SanityChecker

Comprehensive testing and validation utility.

#### Constructor

```python
SanityChecker()
```

No parameters required.

#### Methods

##### run_sanity_check

```python
run_sanity_check(
    num_tasks: int = 9,
    num_processors: int = 3,
    algorithms: Optional[List[SchedulingAlgorithm]] = None,
    dag_type: str = "random",
    random_seed: Optional[int] = None,
    visualize: bool = True
) -> Dict[str, Any]
```

Runs a comprehensive sanity check.

**Parameters:**
- `num_tasks`: Number of tasks in workflow
- `num_processors`: Number of processors
- `algorithms`: List of algorithms to test (defaults to HEFT and QL-HEFT variants)
- `dag_type`: Type of DAG ("random", "layered", "fork_join")
- `random_seed`: Seed for reproducibility
- `visualize`: Whether to show visualizations

**Returns:**
- Dictionary with:
  - `text_summary`: Human-readable summary
  - `data`: Structured result data

##### get_results() -> List[ScheduleResult]
Returns the list of scheduling results.

##### get_dag() -> Optional[WorkflowDAG]
Returns the generated DAG.

---

### quick_sanity_check (Function)

```python
quick_sanity_check(
    num_tasks: int = 9,
    num_processors: int = 3,
    random_seed: Optional[int] = 42
) -> Dict[str, Any]
```

Convenience function for quick sanity checking.

**Parameters:**
- `num_tasks`: Number of tasks
- `num_processors`: Number of processors
- `random_seed`: Random seed

**Returns:**
- Summary dictionary (same as `SanityChecker.run_sanity_check`)

---

## Usage Examples

### Basic Workflow

```python
from src.core import WorkflowDAG
from src.algorithms import HEFTAlgorithm
from src.utils import Visualizer

# Create or generate DAG
dag = ...  # Your DAG

# Run algorithm
heft = HEFTAlgorithm()
result = heft.schedule(dag)

# Visualize
Visualizer.visualize_gantt_chart(result)

# Access results
print(f"Makespan: {result.makespan}")
print(f"Utilization: {result.get_average_utilization()}%")
```

### Custom Algorithm

```python
from src.algorithms import SchedulingAlgorithm
from src.core import ScheduleResult

class MyAlgorithm(SchedulingAlgorithm):
    def __init__(self):
        super().__init__(name="MyAlgorithm")
    
    def schedule(self, dag):
        # Your implementation
        return ScheduleResult(...)

# Use it
algo = MyAlgorithm()
result = algo.schedule(dag)
```

### Batch Comparison

```python
from src.utils import SanityChecker
from src.algorithms import HEFTAlgorithm, QLHEFTSmallState

algorithms = [
    HEFTAlgorithm(),
    QLHEFTSmallState(num_episodes=10000),
    # Your custom algorithms...
]

checker = SanityChecker()
summary = checker.run_sanity_check(
    num_tasks=15,
    num_processors=4,
    algorithms=algorithms
)
```

---

**Version**: 1.0.0  
**Last Updated**: Phase 1 Complete
