# HEFT

The Heterogeneous Earliest Finish Time (HEFT) algorithm is a classic and efficient static list scheduling algorithm designed for task scheduling in heterogeneous computing environments, with the objective of minimizing makespan. The HEFT algorithm is divided into two major phases: the **task prioritizing phase** and the **processor selection phase**.

The algorithm requires that the application model is represented as a Directed Acyclic Graph (DAG), $G = (T, E, C, W)$, where $T$ is the set of tasks, $E$ is the set of edges (dependencies), $C$ is the set of communication costs, and $W$ is the set of computation costs.

Here is the detailed algorithm for HEFT:

### Algorithm: HEFT (Heterogeneous Earliest Finish Time)

#### Phase 1: Task Prioritizing Phase

This phase determines the order in which tasks are considered for scheduling by calculating a priority value for each task called the **upward rank ($\text{rank}_{\text{u}}$)**. The $\text{rank}_{\text{u}}$ value incorporates the task's average computational cost, the average communication cost to its successor tasks, and the influence of the priorities of those successor tasks. A task with a higher $\text{rank}_{\text{u}}$ value is assigned a higher priority and is scheduled earlier.

1.  **Input Preparation:** Set the computation costs of tasks and communication costs of edges using mean/average values (calculated over all available processors/buses).
    *   **Average Computation Cost ($\bar{w}_i$):** This is the mean execution time of task $T_i$ across all $m$ processors ($P_k$):
        $$\bar{w}_i = \frac{1}{m} \sum_{k=1}^{m} w_{i,k}$$.
    *   **Average Communication Cost ($\bar{c}_{i,j}$):** This is the average cost of the edge $E_{i,j}$ between task $T_i$ and task $T_j$.

2.  **Compute Upward Rank ($\text{rank}_{\text{u}}$):** The $\text{rank}_{\text{u}}$ is computed recursively by traversing the DAG **upward**, starting from the exit task ($\text{T}_{\text{exit}}$).
    *   For the exit task $\text{T}_{\text{exit}}$:
        $$\text{rank}_{\text{u}}(\text{T}_{\text{exit}}) = \bar{w}_{\text{exit}}$$.
    *   For any other task $T_i$:
        $$\text{rank}_{\text{u}}(T_i) = \bar{w}_i + \max_{T_j \in \text{succ}(T_i)} \left( \bar{c}_{i,j} + \text{rank}_{\text{u}}(T_j) \right)$$.
        Where $\text{succ}(T_i)$ is the set of immediate successor tasks of $T_i$.

3.  **Sort Tasks:** Create a scheduling list by sorting all tasks in **non-increasing order** of their $\text{rank}_{\text{u}}$ values (highest rank first).

#### Phase 2: Processor Selection Phase

This phase iterates through the priority list generated in Phase 1 and assigns the currently selected task to the processor that ensures the **Earliest Finish Time (EFT)**. This process involves an insertion-based approach.

The main steps involve a `while` loop that continues as long as there are unscheduled tasks in the list:

1.  **Task Selection:** Select the first task ($T_i$) from the ordered scheduling list.

2.  **Processor Evaluation:** For each available processor $P_k$ in the processor set:
    *   **Compute Earliest Start Time ($\text{EST}(T_i, P_k)$):** The $\text{EST}$ is the time at which $T_i$ can begin execution on $P_k$. This is determined by the maximum of two values:
        a.  The earliest available time of processor $P_k$, denoted as $\text{AT}(P_k)$.
        b.  The time when all data required from all immediate predecessor tasks ($T_q \in \text{pred}(T_i)$) has arrived at $P_k$.
        $$\text{EST}(T_i, P_k) = \max \left\{ \max_{T_q \in \text{pred}(T_i)} \{\text{FT}(T_q) + c_{q,i}\}, \text{AT}(P_k) \right\}$$.
        (The $\text{EST}$ of an entry task is 0).
        *(Note: $c_{q,i}$ is the communication cost between $T_q$ and $T_i$. If $T_q$ and $T_i$ are scheduled on the same processor, $c_{q,i} = 0$).*
    *   **Compute Earliest Finish Time ($\text{EFT}(T_i, P_k)$):** The $\text{EFT}$ is calculated by adding the computation cost ($w_{i,k}$) to the earliest start time ($\text{EST}$):
        $$\text{EFT}(T_i, P_k) = w_{i,k} + \text{EST}(T_i, P_k)$$.

3.  **Task Assignment:** Assign task $T_i$ to the processor ($P_j$) that minimizes the $\text{EFT}(T_i, P_k)$ value. The status of $P_j$ (its availability time) is updated accordingly.

4.  **Repeat:** Remove $T_i$ from the scheduling list and repeat the process until all tasks are scheduled.

The overall makespan (schedule length) is the finish time of the exit task.

$$\text{Makespan} = \max_{i \in \{1, 2, \dots, n\}} (\text{EFT}(T_i))$$.


# QL-HEFT

The QL-HEFT (Q-Learning Heterogeneous Earliest Finish Time) algorithm is a novel static task scheduling scheme designed for cloud computing environments. It combines the reinforcement learning approach of Q-learning with the HEFT heuristic algorithm to minimize the overall makespan.

The algorithm is divided into two major phases: a task sorting phase based on Q-learning to determine the optimal task execution order, and a processor allocation phase using the Earliest Finish Time (EFT) strategy.

### I. Task Prioritizing Phase (Q-Learning)

The primary goal of this phase is to use Q-learning to obtain an optimal execution order of all tasks, thereby maximizing the reward. This phase leverages the $\text{rank}_{\text{u}}$ value from HEFT as the immediate reward to enhance the agent's learning performance.

**A. Initialization and Components**

1.  **Define the Model:** The process is based on a Markov Decision Process (MDP) framework, where the agent learns through trial-and-error interactions with the environment.
    *   **State ($s$):** The state represents the environment state set, typically encompassing the currently scheduled tasks.
    *   **Action ($a$):** An action is the selection of a task to be scheduled next. Actions must be **legal**, meaning the selected task follows the dependency relationship (i.e., all parent tasks must have been executed).
    *   **Q-Table:** A table storing the action value function $Q(s, a)$, representing the cumulative return value expected after executing action $a$ in state $s$.
    *   **Parameters:** Set the learning rate ($\alpha$) and discount factor ($\gamma$). For experimental optimization, the learning rate is often set to $1.0$ and the discount factor to $0.8$.

2.  **Calculate Immediate Reward ($r$):** The immediate reward $r$ is the **upward rank ($\text{rank}_{\text{u}}$)** value of the task chosen as the action.
    *   The $\text{rank}_{\text{u}}$ is computed recursively by traversing the task graph upward, starting from the exit task ($T_{\text{exit}}$).
    *   For the exit task: $\text{rank}_{\text{u}}(T_{\text{exit}}) = \bar{w}_{\text{exit}}$.
    *   For any other task $T_i$:
        $$\text{rank}_{\text{u}}(T_i) = \bar{w}_i + \max_{T_j \in \text{succ}(T_i)} \left( \bar{c}_{i,j} + \text{rank}_{\text{u}}(T_j) \right)$$.
        *Where $\bar{w}_i$ is the average computational cost of task $T_i$, $\bar{c}_{i,j}$ is the average communication cost between $T_i$ and $T_j$, and $\text{succ}(T_i)$ is the set of immediate successor tasks of $T_i$.

**B. Learning Process (Iterations)**

The learning process involves continuous episodes where the agent interacts with the environment to update the Q-table (self-learning).

1.  **Action Selection (Exploration):** In each state $s$, the agent uses a random selection strategy to choose an action $a$ (a legal/viable task) and transfer to the next state $s'$.
2.  **Q-Value Update:** The Q-value $Q(s, a)$ is updated using the temporal difference (TD(0)) algorithm, incorporating the immediate reward ($r = \text{rank}_{\text{u}}(a)$) and the maximum expected future reward ($\max_{a'} Q(s', a')$):
    $$Q_{t+1}(s, a) = Q_t(s, a) + \alpha \left[ r + \gamma \max_{a'} Q_t(s', a') - Q_t(s, a) \right]$$.
3.  **Convergence:** This process is repeated for a finite number of iterations. The algorithm converges when the Q-table stabilizes (no longer changes).

**C. Generating the Optimal Task Order**

1.  **Post-Convergence:** Once the Q-table has converged, the algorithm uses the **maximum Q-value strategy** (exploitation) to derive the optimal task execution order.
2.  **Task Selection:** At each step, the task (action) is selected that corresponds to the maximum Q-value in the current state, ensuring the task selection remains legal. The converged Q-table is used, where Q-values for illegal state-action pairs are effectively set to zero for selection purposes.
3.  **Output:** An optimal task execution order (list of tasks) is produced.

### II. Processor Allocation Phase

The second major phase assigns processors to the tasks based on the sequence obtained from the Q-learning phase.

1.  **Allocation Strategy:** The phase utilizes the **Earliest Finish Time (EFT) allocation strategy**.
2.  **Process:** Tasks are allocated sequentially according to the optimal order derived in Phase I.
3.  **Selection Criterion:** Each task $T_i$ is assigned to the processor $P_k$ that minimizes its $\text{EFT}(T_i, P_k)$.
4.  **EFT Calculation:** The EFT of task $T_i$ on processor $P_k$ is defined as:
    $$\text{EFT}(T_i, P_k) = w_{i,k} + \text{EST}(T_i, P_k)$$.
    *Where:*
    *   $w_{i,k}$ is the computation cost of $T_i$ on $P_k$.
    *   $\text{EST}(T_i, P_k)$ (Earliest Start Time) is the maximum of two values: the earliest available time of $P_k$ ($\text{AT}(P_k)$), and the time when all immediate predecessor tasks $T_q$ have finished execution and their required data has arrived:
        $$\text{EST}(T_i, P_k) = \max \left\{ \max_{T_q \in \text{pred}(T_i)} \{\text{FT}(T_q) + c_{q,i}\}, \text{AT}(P_k) \right\}$$. (The $\text{EST}$ of an entry task is 0).
5.  **Makespan Determination:** When the last working processor completes its task, the completion time of all tasks is obtained, which defines the overall **makespan** of the schedule.

### Summary Flowchart of QL-HEFT

The QL-HEFT process follows these steps until convergence and a final schedule are achieved:

1.  **Initialize** the Q-table and parameters ($\alpha$, $\gamma$).
2.  **Calculate** the immediate reward ($\text{rank}_{\text{u}}$ values).
3.  **Loop (Q-Learning Iterations):**
    a.  **Update Q-table** based on the current state, selected action, immediate reward, and maximum future reward.
    b.  If the Q-table is **not converged**, repeat step 3.
4.  **Get a Task Order** (using the maximum Q-value strategy on the converged Q-table).
5.  **Allocate Processor** to each task in the optimal order (using the EFT strategy).
6.  **Get the Makespan**.
7.  **Output** the best scheduling results.


# System Model

The Heterogeneous Earliest Finish Time (HEFT) and QL-HEFT (Q-Learning Heterogeneous Earliest Finish Time) algorithms both utilize a system model based on **task scheduling in heterogeneous computing environments**, typically represented by a workflow or application structure, and a model of the computing platform.

Since QL-HEFT is an extension of HEFT, their fundamental system models share many core components, though QL-HEFT specifically frames the scheduling problem within a cloud computing context using Q-learning.

Here is a detailed description of the system model used for HEFT and QL-HEFT, based on the provided sources:

### 1. Application Model: Directed Acyclic Graph (DAG)

The application (workflow or set of tasks) is fundamentally represented as a **Directed Acyclic Graph (DAG)**, $G = (T, E, C, W)$.

*   **Tasks ($T$ or $V$):** $T = \{T_i\}$ is the set of all tasks (or nodes in the graph) that need to be scheduled.
    *   In the CC-TMS formulation (which shares model elements with HEFT), the node set $V$ may contain both computation tasks ($T$) and message nodes ($M$) specifying communication demands.
    *   Tasks have **precedence constraints**, meaning a child task cannot begin execution until all of its parent tasks have finished and the necessary data has been transferred.
*   **Edges ($E$):** $E$ is the set of directed edges (dependencies) between tasks, where an edge $E_{i,j}$ from $T_i$ to $T_j$ represents that $T_i$ must be completed before $T_j$ can start.
*   **Computation Costs ($W$):** $W$ is the set of weights representing the **computation costs** (execution times) of the tasks.
    *   This is often represented by an $n \times p$ matrix $W$ (or $ET$), where $w_{i,k}$ (or $ET_{i,r}$) is the estimated execution time of task $T_i$ on processor $P_k$.
    *   The DAG model uses the **average computation cost ($\bar{w}_i$)** of a task across all processors for calculating task priorities ($\text{rank}_{\text{u}}$).
*   **Communication Costs ($C$):** $C$ is the set of communication costs between tasks connected by edges.
    *   $c_{i,j}$ represents the communication cost between $T_i$ and $T_j$. If $T_i$ and $T_j$ are scheduled on the **same processor**, the communication cost $c_{i,j}$ is considered **zero**.
    *   The DAG model uses the **average communication cost ($\bar{c}_{i,j}$) **of an edge for calculating task priorities.
*   **Entry/Exit Tasks:** The DAG usually has a single entry task ($T_{\text{entry}}$ or $T_{\text{source}}$) with zero in-degree and a single exit task ($T_{\text{exit}}$ or $T_{\text{sink}}$) with zero out-degree. If the original application has multiple entry/exit nodes, dummy tasks with zero computation cost are typically added to standardize the graph.

### 2. Computing Platform Model: Heterogeneous Systems

Both algorithms are designed for **heterogeneous computing environments**.

*   **Processors ($P$ or $U$):** The system consists of a set of computing units (processors) $P = \{P_1, P_2, \dots, P_p\}$ (or $U=\{u_1, u_2, \dots, u_p\}$). These processors are **heterogeneous**, meaning they have different processing capabilities, leading to varying execution times for the same task.
*   **Connectivity (HEFT/QL-HEFT Default):** The underlying assumption for the HEFT algorithm (and often QL-HEFT, unless explicitly extended) is a platform where computing units are connected by **dedicated communication channels** (a fully connected network), implying that communication **contention is not considered**.
    *   *Note:* The CC-TMS algorithm, which is closely related to HEFT in the sources, extends this model to include communication via **shared buses** (heterogeneous shared buses).

### 3. Scheduling and Cost Metrics

The models define specific time metrics necessary for executing the algorithms and evaluating the outcome:

*   **Earliest Start Time (EST):** The earliest time a task $T_i$ can begin execution on a processor $P_k$ is the maximum of two conditions:
    1.  The earliest available time of processor $P_k$ ($\text{AT}(P_k)$).
    2.  The time when the task's execution results and all necessary data from all predecessor tasks ($T_q$) have arrived at $P_k$, calculated as the maximum of ($\text{FT}(T_q) + c_{q,i}$) over all predecessors.
*   **Earliest Finish Time (EFT):** The time $T_i$ completes execution on $P_k$:
    $$ \text{EFT}(T_i, P_k) = \text{EST}(T_i, P_k) + w_{i,k} $$
*   **Makespan (Schedule Length):** The primary objective function is to **minimize the overall makespan** ($\text{SL}$), which is the earliest finish time of the exit task, $T_{\text{exit}}$.
    $$\text{Makespan} = \max_{i \in T} (\text{EFT}(T_i))$$

### 4. QL-HEFT Specific Reinforcement Learning Model

QL-HEFT superimposes a reinforcement learning structure onto this existing DAG model to determine the optimal task ordering:

*   **Framework:** The process is modeled as a Markov Decision Process (MDP) defined by a 4-tuple $\langle S, A, P, R \rangle$.
*   **State ($S$):** Represents the current environment state, typically encompassing the set of scheduled tasks.
*   **Action ($A$):** The selection of a legal/viable task to be scheduled next.
*   **Reward ($R$):** The **immediate reward ($r$)** for selecting a task (action) is the **upward rank ($\text{rank}_{\text{u}}$) value** of that task, borrowed directly from the HEFT prioritization scheme. This serves to guide the Q-learning agent toward better execution orders.
*   **Goal:** The agent learns the optimal policy ($\pi^*$) to maximize the cumulative expected return, leading to an **optimal task execution order**.