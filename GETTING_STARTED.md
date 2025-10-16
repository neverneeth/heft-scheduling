# Getting Started with HEFT Scheduling Framework

Welcome! This guide will get you up and running in 5 minutes.

## Quick Install

1. **Check Python version** (3.8+ required)
```powershell
python --version
```

2. **Install dependencies**
```powershell
pip install networkx matplotlib numpy pandas seaborn
```

That's it! No complex setup needed.

## Your First Sanity Check (30 seconds)

### Option 1: Using Python REPL

```powershell
python
```

```python
from src.utils import quick_sanity_check

# Run sanity check - will generate DAG, run algorithms, show charts
quick_sanity_check(num_tasks=9, num_processors=3, random_seed=42)
```

### Option 2: Run Example Script

```powershell
python examples/simple_sanity_check.py
```

## What Just Happened?

The sanity check:
1. ✓ Generated a random 9-task DAG
2. ✓ Visualized the DAG structure
3. ✓ Ran HEFT algorithm
4. ✓ Ran QL-HEFT algorithms (2 variants)
5. ✓ Showed Gantt charts for each
6. ✓ Compared algorithm performance

## Basic Usage (5 minutes)

### Step 1: Generate a DAG

```python
from src.utils import DAGGenerator

# Create a random workflow
dag = DAGGenerator.generate_random_dag(
    num_tasks=10,
    num_processors=3,
    random_seed=42
)

print(dag)  # WorkflowDAG(tasks=10, processors=3, edges=...)
```

### Step 2: Visualize the DAG

```python
from src.utils import Visualizer

Visualizer.visualize_dag(dag, title="My Workflow")
```

### Step 3: Run an Algorithm

```python
from src.algorithms import HEFTAlgorithm

# Create and run HEFT
heft = HEFTAlgorithm()
result = heft.schedule(dag)

print(f"Makespan: {result.makespan:.2f}")
print(f"Utilization: {result.get_average_utilization():.2f}%")
```

### Step 4: Show Gantt Chart

```python
Visualizer.visualize_gantt_chart(result)
```

### Step 5: Compare Multiple Algorithms

```python
from src.algorithms import QLHEFTSmallState

# Run another algorithm
ql_heft = QLHEFTSmallState(num_episodes=5000)
result2 = ql_heft.schedule(dag)

# Compare both
Visualizer.compare_algorithms([result, result2])
```

## Common Tasks

### Generate Different DAG Types

```python
# Layered DAG (pipeline style)
dag = DAGGenerator.generate_layered_dag(
    num_layers=4,
    tasks_per_layer=3,
    num_processors=3
)

# Fork-Join DAG (MapReduce style)
dag = DAGGenerator.generate_fork_join_dag(
    num_initial_tasks=2,
    num_parallel_tasks=6,
    num_final_tasks=2,
    num_processors=4
)
```

### Access Result Details

```python
result = heft.schedule(dag)

# Overall metrics
print(f"Makespan: {result.makespan}")
print(f"Avg Util: {result.get_average_utilization()}%")

# Per-processor utilization
for proc, util in result.get_processor_utilization().items():
    print(f"Processor {proc}: {util:.2f}%")

# Per-task details
for task, info in result.task_schedule.items():
    print(f"{task}: P{info['processor']} "
          f"[{info['start_time']:.1f} - {info['finish_time']:.1f}]")
```

### Save Visualizations

```python
# Save to file instead of showing
Visualizer.visualize_gantt_chart(
    result,
    save_path="my_schedule.png",
    show=False
)
```

## Run All Examples

```powershell
# Simple sanity check
python examples/simple_sanity_check.py

# Custom algorithm demo
python examples/custom_algorithm.py

# DAG generation demo
python examples/advanced_dag_generation.py
```

## Troubleshooting

### Import Error: "No module named 'src'"

**Solution:** Run Python from the framework root directory:
```powershell
cd c:\Users\navan\IDEA\heft_scheduling_framework
python
```

Or add to script:
```python
import sys
sys.path.append('c:/Users/navan/IDEA/heft_scheduling_framework')
```

### No plots showing

**Solution:** Matplotlib might be in non-interactive mode:
```python
import matplotlib.pyplot as plt
plt.ion()  # Turn on interactive mode
```

### QL-HEFT is slow

**Solution:** Reduce training episodes:
```python
ql_heft = QLHEFTSmallState(num_episodes=1000)  # Instead of 50000
```

## Next Steps

### Learn More
1. Read [FRAMEWORK_README.md](FRAMEWORK_README.md) for complete guide
2. Check [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for reference
3. See [PHASE1_SUMMARY.md](PHASE1_SUMMARY.md) for what's implemented

### Try These
- [ ] Generate different sized DAGs (5, 10, 20 tasks)
- [ ] Compare algorithm makespans
- [ ] Create your own custom algorithm
- [ ] Try different processor counts
- [ ] Experiment with cost ranges

### Customize
```python
# Custom computation costs
dag = DAGGenerator.generate_random_dag(
    num_tasks=10,
    num_processors=4,
    computation_cost_range=(20, 100),  # Slower tasks
    communication_cost_range=(5, 30)    # More communication
)

# Custom QL-HEFT parameters
ql_heft = QLHEFTSmallState(
    num_episodes=10000,
    epsilon=0.3,              # More exploration
    learning_rate=0.05,       # Slower learning
    learning_rate_decay="exponential"
)
```

## Quick Reference

### Import Everything
```python
from src.core import WorkflowDAG, ScheduleResult
from src.algorithms import HEFTAlgorithm, QLHEFTSmallState
from src.utils import DAGGenerator, Visualizer, SanityChecker
```

### One-Liner Workflow
```python
from src.utils import quick_sanity_check
quick_sanity_check(num_tasks=12, num_processors=4)
```

### Complete Example
```python
# Generate, schedule, visualize
from src.utils import DAGGenerator, Visualizer
from src.algorithms import HEFTAlgorithm

dag = DAGGenerator.generate_random_dag(10, 3, random_seed=42)
result = HEFTAlgorithm().schedule(dag)
Visualizer.visualize_gantt_chart(result)
```

## Getting Help

1. **Check documentation**: All classes and methods have detailed docstrings
2. **Run tests**: `python test_framework.py` to verify setup
3. **Try examples**: Working code in `examples/` directory
4. **Read inline docs**: Every function is documented

## You're Ready! 🚀

You now know how to:
- ✓ Generate DAGs
- ✓ Run scheduling algorithms
- ✓ Visualize results
- ✓ Compare performance

Start experimenting and have fun with workflow scheduling!

---

**Need more details?** Check out [FRAMEWORK_README.md](FRAMEWORK_README.md)  
**API reference?** See [API_DOCUMENTATION.md](API_DOCUMENTATION.md)  
**Want to contribute?** Extend the base classes and add your algorithms!
