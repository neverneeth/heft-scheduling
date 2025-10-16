# HEFT Scheduling Framework

> A rigorous, extensible, and intuitive framework for DAG-based workflow scheduling algorithms

[![Status](https://img.shields.io/badge/status-production_ready-brightgreen)]()
[![Phase](https://img.shields.io/badge/phase-1_complete-blue)]()
[![Tests](https://img.shields.io/badge/tests-passing-success)]()
[![Python](https://img.shields.io/badge/python-3.8+-blue)]()

---

## 🚀 Quick Start

```python
from src.utils import quick_sanity_check

# Generate DAG, run algorithms, visualize results - all in one line!
quick_sanity_check(num_tasks=9, num_processors=3, random_seed=42)
```

**That's it!** The framework will:
- ✅ Generate a random workflow DAG
- ✅ Run HEFT and QL-HEFT algorithms
- ✅ Show DAG structure and Gantt charts
- ✅ Compare algorithm performance

---

## 📚 Documentation

| Document | Purpose | Read Time |
|----------|---------|-----------|
| **[GETTING_STARTED.md](GETTING_STARTED.md)** | Quick setup & first steps | 5 min |
| **[FRAMEWORK_README.md](FRAMEWORK_README.md)** | Complete framework guide | 20 min |
| **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)** | Technical API reference | Reference |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | Visual architecture overview | 10 min |
| **[PHASE1_SUMMARY.md](PHASE1_SUMMARY.md)** | Project status & deliverables | 15 min |
| **[INDEX.md](INDEX.md)** | Documentation navigator | 5 min |

### 🎯 Where to Start?

- **New user?** → [GETTING_STARTED.md](GETTING_STARTED.md)
- **Want details?** → [FRAMEWORK_README.md](FRAMEWORK_README.md)
- **Need API info?** → [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
- **Visual learner?** → [ARCHITECTURE.md](ARCHITECTURE.md)

---

## ✨ Features

### 🔧 Core Capabilities

- **3 DAG Types**: Random, Layered, Fork-Join patterns
- **3 Algorithms**: HEFT, QL-HEFT (Large State), QL-HEFT (Small State)
- **4 Visualizations**: DAG structure, Gantt charts, comparisons, convergence
- **1-Line Testing**: Complete sanity check in a single function call

### 🎨 Highlights

```python
# Generate different DAG types
from src.utils import DAGGenerator

dag = DAGGenerator.generate_random_dag(num_tasks=10, num_processors=3)
dag = DAGGenerator.generate_layered_dag(num_layers=4, tasks_per_layer=3, num_processors=3)
dag = DAGGenerator.generate_fork_join_dag(num_initial_tasks=2, num_parallel_tasks=6, 
                                          num_final_tasks=2, num_processors=4)
```

```python
# Run algorithms
from src.algorithms import HEFTAlgorithm, QLHEFTSmallState

heft = HEFTAlgorithm()
ql_heft = QLHEFTSmallState(num_episodes=10000)

result1 = heft.schedule(dag)
result2 = ql_heft.schedule(dag)
```

```python
# Visualize everything
from src.utils import Visualizer

Visualizer.visualize_dag(dag)
Visualizer.visualize_gantt_chart(result1)
Visualizer.compare_algorithms([result1, result2])
```

---

## 📦 Installation

### Prerequisites

```bash
pip install networkx matplotlib numpy pandas seaborn
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python test_framework.py
```

Expected output: `ALL TESTS PASSED! ✓`

---

## 📖 Examples

### Example 1: Simple Usage

```bash
python examples/simple_sanity_check.py
```

### Example 2: Custom Algorithm

```bash
python examples/custom_algorithm.py
```

### Example 3: Advanced DAG Generation

```bash
python examples/advanced_dag_generation.py
```

---

## 🏗️ Framework Structure

```
heft_scheduling_framework/
├── src/
│   ├── core/              # Core data structures
│   ├── algorithms/        # Scheduling algorithms
│   └── utils/             # Utilities & tools
├── examples/              # Usage examples
├── tests/                 # Test directory
├── test_framework.py      # Comprehensive test suite
└── docs/                  # Documentation
```

---

## 🎓 What You Can Do

### ✅ Implemented (Phase 1)

- Generate random DAGs with various structures
- Run HEFT scheduling algorithm
- Run Q-learning enhanced HEFT (2 variants)
- Visualize DAG structures
- Generate Gantt charts
- Compare algorithm performance
- One-line sanity checking
- Comprehensive testing

### 🚀 Coming Soon (Phase 2+)

- Batch experiment orchestration
- Statistical analysis tools
- Result persistence (CSV, JSON)
- Additional algorithms (CPOP, PEFT)
- Multi-objective optimization
- Performance optimizations

---

## 🔬 Algorithms

### HEFT (Heterogeneous Earliest Finish Time)

Classic list scheduling with:
- Upward rank prioritization
- Earliest Finish Time allocation
- O(n²) time complexity

### QL-HEFT Large State

Q-learning with full state space:
- State: Set of all scheduled tasks
- Learns complex dependencies
- Suitable for small to medium DAGs

### QL-HEFT Small State

Q-learning with compact state:
- State: Last scheduled task only
- Convergence detection
- Learning rate decay
- Efficient for larger DAGs

---

## 📊 Performance

**Test Configuration:** 6 tasks, 3 processors

| Algorithm | Makespan | Utilization | Time |
|-----------|----------|-------------|------|
| HEFT | 47.90 | 72.95% | 0.001s |
| QL-HEFT Large | 47.90 | 72.95% | 0.270s |
| QL-HEFT Small | 49.50 | 68.69% | 0.500s |

*Results from automated test suite*

---

## 🛠️ Extension Guide

### Add a Custom Algorithm

```python
from src.algorithms import SchedulingAlgorithm
from src.core import ScheduleResult

class MyAlgorithm(SchedulingAlgorithm):
    def __init__(self):
        super().__init__(name="MyAlgorithm")
    
    def schedule(self, dag):
        # Your implementation here
        return ScheduleResult(...)

# Use it
algo = MyAlgorithm()
result = algo.schedule(dag)
```

See `examples/custom_algorithm.py` for a complete example.

---

## 🧪 Testing

### Run All Tests

```bash
python test_framework.py
```

### Run Specific Example

```bash
python examples/simple_sanity_check.py
```

---

## 📈 Project Status

**Phase 1: COMPLETE ✅**

- ✅ Algorithm implementations (HEFT, QL-HEFT variants)
- ✅ Randomized DAG generation (3 types)
- ✅ Comprehensive sanity checker
- ✅ Rich visualization tools
- ✅ Extensive documentation (11,000+ words)
- ✅ Automated testing
- ✅ Working examples

**Statistics:**
- 2,500+ lines of code
- 13 module files
- 20+ documentation pages
- 3 complete examples
- 100% test passing rate

---

## 🎯 Design Principles

✅ **Rigorous**: Type hints, validation, comprehensive testing  
✅ **Testable**: Unit tests, integration tests, automated validation  
✅ **Extensible**: Abstract base classes, plugin architecture  
✅ **Intuitive**: Clear APIs, extensive docs, working examples  

---

## 📝 Citation

If you use this framework in your research, please cite:

```
HEFT Scheduling Framework v1.0.0 (2025)
A modular framework for DAG-based workflow scheduling
```

**Reference Algorithms:**
- HEFT: Topcuoglu, H., Hariri, S., & Wu, M. Y. (2002). Performance-effective and low-complexity task scheduling for heterogeneous computing. IEEE transactions on parallel and distributed systems, 13(3), 260-274.

---

## 🤝 Contributing

We welcome contributions! To add:

- **New algorithms**: Extend `SchedulingAlgorithm` base class
- **New DAG types**: Add methods to `DAGGenerator`
- **New visualizations**: Extend `Visualizer` class
- **Improvements**: Optimize, document, test

---

## 📄 License

This framework is provided for educational and research purposes.

---

## 🔗 Quick Links

- **[Get Started in 5 Minutes](GETTING_STARTED.md)**
- **[Read the Complete Guide](FRAMEWORK_README.md)**
- **[Browse API Reference](API_DOCUMENTATION.md)**
- **[View Architecture](ARCHITECTURE.md)**
- **[Check Project Status](PHASE1_SUMMARY.md)**
- **[Navigate All Docs](INDEX.md)**

---

## 📞 Support

- 📖 Check [documentation](INDEX.md) for comprehensive guides
- 💡 Review [examples](examples/) for working code
- 🧪 Run [tests](test_framework.py) to verify setup
- 📚 Read inline docstrings for API details

---

## ⭐ Acknowledgments

Framework implementation based on:
- HEFT algorithm (Topcuoglu et al., 2002)
- Q-learning enhanced scheduling concepts
- NetworkX for graph operations
- Matplotlib for visualization

---

<div align="center">

**HEFT Scheduling Framework v1.0.0**

*Rigorous • Testable • Extensible • Intuitive*

[Documentation](INDEX.md) • [Examples](examples/) • [Tests](test_framework.py)

</div>
