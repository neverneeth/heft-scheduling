# HEFT Scheduling Framework - Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     HEFT SCHEDULING FRAMEWORK v1.0.0                        │
│                         Phase 1: Complete & Tested                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────┐          ┌──────────────────────┐               │
│  │  quick_sanity_check  │          │   SanityChecker      │               │
│  │  ────────────────────│          │  ────────────────────│               │
│  │  One-line testing    │          │  Full test suite     │               │
│  │  Auto visualization  │          │  Custom algorithms   │               │
│  │  Result comparison   │          │  Flexible config     │               │
│  └──────────────────────┘          └──────────────────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ALGORITHM LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │              SchedulingAlgorithm (Abstract Base)                    │   │
│  │              ────────────────────────────────                       │   │
│  │              + schedule(dag: WorkflowDAG) -> ScheduleResult         │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                        │                │                │                 │
│           ┌────────────┘                │                └────────────┐    │
│           │                             │                             │    │
│           ▼                             ▼                             ▼    │
│  ┌─────────────────┐        ┌────────────────────┐      ┌──────────────┐ │
│  │  HEFTAlgorithm  │        │ QLHEFTLargeState   │      │ QLHEFTSmall  │ │
│  │  ───────────────│        │ ──────────────────  │      │ ──────────── │ │
│  │  • Upward rank  │        │ • Full state Q(S,a)│      │ • Q(s,a)     │ │
│  │  • EFT schedule │        │ • 10K episodes     │      │ • Convergence│ │
│  │  • Fast O(n²)   │        │ • Exploration      │      │ • LR decay   │ │
│  └─────────────────┘        └────────────────────┘      └──────────────┘ │
│                                                                             │
│  [Extensible: Add your own algorithms by extending base class]             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CORE DATA LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────┐      ┌──────────────────┐      ┌───────────────┐ │
│  │   WorkflowDAG       │      │  ScheduleResult  │      │ SystemModel   │ │
│  │   ─────────────     │      │  ───────────────  │      │ ────────────  │ │
│  │  • Task graph       │      │  • Task schedule │      │ • Processors  │ │
│  │  • Computation costs│      │  • Makespan      │      │ • Resources   │ │
│  │  • Comm costs       │      │  • Utilization   │      │ • Config      │ │
│  │  • Dependencies     │      │  • Metadata      │      │               │ │
│  │                     │      │                  │      │               │ │
│  │  Methods:           │      │  Methods:        │      │               │ │
│  │  • get_viable()     │      │  • get_util()    │      │               │ │
│  │  • get_successors() │      │  • get_summary() │      │               │ │
│  │  • get_costs()      │      │                  │      │               │ │
│  └─────────────────────┘      └──────────────────┘      └───────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
            │                                                      ▲
            │ Uses                                                 │ Produces
            ▼                                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│                          UTILITIES LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                          DAGGenerator                                 │  │
│  │                          ────────────                                 │  │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │  │
│  │  │   Random     │    │   Layered    │    │    Fork-Join         │   │  │
│  │  │   ──────     │    │   ───────    │    │    ─────────         │   │  │
│  │  │ • Any size   │    │ • Pipeline   │    │ • MapReduce style    │   │  │
│  │  │ • Edge prob  │    │ • Uniform    │    │ • Initial-Parallel   │   │  │
│  │  │ • Cost range │    │ • Layers     │    │ • Final merge        │   │  │
│  │  └──────────────┘    └──────────────┘    └──────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                          Visualizer                                   │  │
│  │                          ──────────                                   │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────┐  │  │
│  │  │  DAG Plot    │  │ Gantt Chart  │  │ Comparison   │  │ Q-Learn │  │  │
│  │  │  ────────    │  │ ───────────  │  │ ──────────   │  │ ─────── │  │  │
│  │  │ • Structure  │  │ • Timeline   │  │ • Makespan   │  │ • Conv. │  │  │
│  │  │ • Layers     │  │ • Processors │  │ • Utilization│  │ • Smooth│  │  │
│  │  │ • Comm costs │  │ • Tasks      │  │ • Bar charts │  │ • Track │  │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └─────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1. Generate DAG        2. Run Algorithm      3. Visualize Results        │
│      │                       │                      │                       │
│      ▼                       ▼                      ▼                       │
│   DAGGenerator  ──────▶  Algorithm  ───────▶  Visualizer                   │
│      │                       │                      │                       │
│      │ WorkflowDAG           │ ScheduleResult       │ Charts/Plots         │
│      │                       │                      │                       │
│      └───────────────────────┴──────────────────────┘                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXTENSIBILITY POINTS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Add New Algorithm:    Extend SchedulingAlgorithm base class            │
│  2. Add New DAG Type:     Add method to DAGGenerator                       │
│  3. Add New Visualization: Add method to Visualizer                        │
│  4. Add New Metrics:      Extend ScheduleResult                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           KEY FEATURES                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ✓ Rigorous:      Type hints, validation, error handling                   │
│  ✓ Testable:      Unit tests, integration tests, sanity checker            │
│  ✓ Extensible:    Abstract base classes, plugin architecture               │
│  ✓ Intuitive:     Clear APIs, comprehensive docs, examples                 │
│  ✓ Visual:        DAG plots, Gantt charts, comparisons                     │
│  ✓ Fast:          HEFT in milliseconds, caching where needed               │
│  ✓ Documented:    Docstrings, README, API docs, tutorials                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          TYPICAL WORKFLOW                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  from src.utils import DAGGenerator, Visualizer                            │
│  from src.algorithms import HEFTAlgorithm                                  │
│                                                                             │
│  # Generate                                                                 │
│  dag = DAGGenerator.generate_random_dag(num_tasks=10, num_processors=3)    │
│                                                                             │
│  # Visualize DAG                                                            │
│  Visualizer.visualize_dag(dag)                                             │
│                                                                             │
│  # Schedule                                                                 │
│  result = HEFTAlgorithm().schedule(dag)                                    │
│                                                                             │
│  # Visualize Schedule                                                       │
│  Visualizer.visualize_gantt_chart(result)                                  │
│                                                                             │
│  # Analyze                                                                  │
│  print(f"Makespan: {result.makespan}")                                     │
│  print(f"Utilization: {result.get_average_utilization()}%")               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          QUICK REFERENCE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  File                         Purpose                    Lines              │
│  ────────────────────────────────────────────────────────────────          │
│  workflow_dag.py              DAG representation         230                │
│  heft.py                      HEFT algorithm             180                │
│  qlheft.py                    QL-HEFT algorithms         450                │
│  dag_generator.py             DAG generation             280                │
│  visualizer.py                Visualization              320                │
│  sanity_checker.py            Testing utilities          300                │
│                                                                             │
│  FRAMEWORK_README.md          Main documentation         4500+ words        │
│  API_DOCUMENTATION.md         API reference              3000+ words        │
│  GETTING_STARTED.md           Quick start guide          800+ words         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

                         Framework Status: ✅ READY FOR USE
                         Test Status: ✅ ALL TESTS PASSING
                         Phase 1: ✅ COMPLETE
```
