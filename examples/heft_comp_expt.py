"""
Comprehensive experiment to compare scheduling algorithms across different DAG types and parameters.

Algorithms compared:
- HEFT (classic algorithm)
- QL-HEFT (Large State)
- XGBoost-HEFT (regression-based)
- MLP-HEFT (neural network-based)

Experiment design:
- DAG Types = ['laplace', 'stencil', 'epigenomics']
- DAG Params = [laplace: [4, 6, 7], stencil: [3, 5, 8], epigenomics: [2, 4, 6]]
- Computation cost range = (10-30)
- Communication cost range = (5-15)
- Number of processors = [2, 3, 4]
- 50 iterations per parameter combination
"""

import os
import time
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.algorithms.heft import HEFTAlgorithm
from src.algorithms.qlheft import QLHEFTLargeState
from src.ml.regression_heft import RegressionHEFT
from src.ml.mlp_heft import MLPHEFT
from src.utils.dag_generator import DAGGenerator
from src.core.workflow_dag import WorkflowDAG
from src.utils.visualizer import Visualizer


class HEFTComparisonExperiment:
    """Class to manage the HEFT algorithm comparison experiments."""
    
    def __init__(
        self,
        output_dir: str = "experiment_results",
        xgboost_model_path: str = "ml_final_output/models/xgboost_model.joblib",
        mlp_model_path: str = "ml_final_output/models/mlp_heft_model.joblib",
        num_iterations: int = 50,
        seed: int = 42
    ):
        """
        Initialize the experiment.
        
        Args:
            output_dir: Directory to save results
            xgboost_model_path: Path to trained XGBoost model
            mlp_model_path: Path to trained MLP model
            num_iterations: Number of iterations per parameter combination
            seed: Random seed for reproducibility
        """
        self.output_dir = output_dir
        self.num_iterations = num_iterations
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize DAG generator
        self.dag_generator = DAGGenerator()
        
        # Initialize algorithms
        self.heft = HEFTAlgorithm()
        self.ql_heft = QLHEFTLargeState(num_episodes=10000, epsilon=0.1)
        
        # Load ML-based algorithms if models exist
        self.xgboost_model_path = xgboost_model_path
        self.mlp_model_path = mlp_model_path
        
        self.xgboost_heft = self._load_model(RegressionHEFT, xgboost_model_path, "XGBoost-HEFT")
        self.mlp_heft = self._load_model(MLPHEFT, mlp_model_path, "MLP-HEFT")
        
        # Define experiment parameters
        self.dag_types = ['laplace', 'stencil', 'epigenomics']
        self.dag_params = {
            'laplace': [4, 6, 7],
            'stencil': [3, 5, 8], 
            'epigenomics': [2, 4, 6]
        }
        self.comp_cost_range = (10, 30)
        self.comm_cost_range = (5, 15)
        self.processor_counts = [2, 3, 4]
        
        # Initialize results storage
        self.results = []
        self.iteration_results = []
        
        print(f"Experiment initialized with {num_iterations} iterations per parameter set")
        
    def _load_model(self, algorithm_class, model_path, fallback_name=None):
        """
        Load a trained model or return None if not available.
        
        Args:
            algorithm_class: Class of the algorithm to instantiate
            model_path: Path to the model file
            fallback_name: Name to use if model can't be loaded
            
        Returns:
            Instantiated algorithm or None
        """
        try:
            if os.path.exists(model_path):
                return algorithm_class(model_path=model_path)
            else:
                print(f"Warning: Model not found at {model_path}. {fallback_name} will be excluded.")
                return None
        except Exception as e:
            print(f"Error loading {fallback_name} model: {str(e)}")
            return None
    
    def generate_dag(self, dag_type: str, param: int, num_processors: int) -> WorkflowDAG:
        """
        Generate a DAG of the specified type and parameters.
        
        Args:
            dag_type: Type of DAG ('laplace', 'stencil', 'epigenomics')
            param: Size parameter for the DAG
            num_processors: Number of processors
            
        Returns:
            Generated WorkflowDAG
        """
        if dag_type == 'laplace':
            edges, comp_costs, comm_costs = self.dag_generator.generate_laplace_dag(
                phi=param,
                num_processors=num_processors,
                computation_cost_range=self.comp_cost_range,
                communication_cost_range=self.comm_cost_range,
                random_seed=random.randint(0, 10000)
            )
        elif dag_type == 'stencil':
            edges, comp_costs, comm_costs = self.dag_generator.generate_stencil_dag(
                xi=param,
                num_processors=num_processors,
                computation_cost_range=self.comp_cost_range,
                communication_cost_range=self.comm_cost_range,
                random_seed=random.randint(0, 10000)
            )
        elif dag_type == 'epigenomics':
            edges, comp_costs, comm_costs = self.dag_generator.generate_epigenomics_dag(
                gamma=param,
                num_processors=num_processors,
                computation_cost_range=self.comp_cost_range,
                communication_cost_range=self.comm_cost_range,
                random_seed=random.randint(0, 10000)
            )
        else:
            raise ValueError(f"Unknown DAG type: {dag_type}")
        
        # Return the constructed WorkflowDAG
        return WorkflowDAG(edges, comp_costs, comm_costs)
    
    def run_experiment(self) -> pd.DataFrame:
        """
        Run the full experiment across all parameter combinations.
        
        Returns:
            DataFrame with the results
        """
        print("Starting experiment...")
        start_time = time.time()
        
        # Define experiment runs
        total_combinations = len(self.dag_types) * sum(len(params) for params in self.dag_params.values()) * len(self.processor_counts)
        print(f"Total parameter combinations: {total_combinations}")
        print(f"Total experiment runs: {total_combinations * self.num_iterations}")
        
        # Use tqdm for progress tracking
        experiment_progress = tqdm(total=total_combinations, desc="Parameter combinations")
        
        # For each combination of parameters
        for dag_type in self.dag_types:
            for param in self.dag_params[dag_type]:
                for num_processors in self.processor_counts:
                    # Run multiple iterations with the same structure but different costs
                    self._run_parameter_combination(dag_type, param, num_processors)
                    experiment_progress.update(1)
        
        experiment_progress.close()
        
        # Convert results to DataFrames
        results_df = pd.DataFrame(self.results)
        iteration_df = pd.DataFrame(self.iteration_results)  # NEW
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_csv = os.path.join(self.output_dir, f"heft_comparison_results_{timestamp}.csv")
        iterations_csv = os.path.join(self.output_dir, f"heft_comparison_iterations_{timestamp}.csv")  # NEW
        
        results_df.to_csv(results_csv, index=False)
        iteration_df.to_csv(iterations_csv, index=False)  # NEW
        
        print(f"Aggregated results saved to {results_csv}")
        print(f"Per-iteration results saved to {iterations_csv}")  # NEW
        
        # Generate summary
        self._generate_summary(results_df, iteration_df)  # Update this method
        
        total_time = time.time() - start_time
        print(f"Experiment completed in {total_time:.2f} seconds")
        
        return results_df, iteration_df  # Return both DataFrames
        
    def _run_parameter_combination(self, dag_type: str, param: int, num_processors: int):
        """Run experiment for a specific parameter combination."""
        combination_results = {
            'heft': {'makespan': [], 'execution_time': []},
            'ql_heft': {'makespan': [], 'execution_time': [], 'episodes': []},
        }
        
        if self.xgboost_heft:
            combination_results['xgboost_heft'] = {'makespan': [], 'execution_time': []}
        
        if self.mlp_heft:
            combination_results['mlp_heft'] = {'makespan': [], 'execution_time': []}
        
        # Run multiple iterations
        iterations_progress = tqdm(
            total=self.num_iterations, 
            desc=f"{dag_type} (param={param}, procs={num_processors})",
            leave=False
        )
        
        for iteration in range(self.num_iterations):
            # Generate DAG with different costs but same structure
            dag = self.generate_dag(dag_type, param, num_processors)
            
            # Run each algorithm and record metrics
            algorithms = {
                'heft': self.heft,
                'ql_heft': self.ql_heft
            }
            
            if self.xgboost_heft:
                algorithms['xgboost_heft'] = self.xgboost_heft
            
            if self.mlp_heft:
                algorithms['mlp_heft'] = self.mlp_heft
            
            for algo_name, algo in algorithms.items():
                start_time = time.time()
                
                try:
                    result = algo.schedule(dag)
                    execution_time = time.time() - start_time
                    
                    # Store for aggregation
                    combination_results[algo_name]['makespan'].append(result.makespan)
                    combination_results[algo_name]['execution_time'].append(execution_time)
                    
                    # Record episodes for QL-HEFT
                    if algo_name == 'ql_heft' and 'episodes_run' in result.metadata:
                        combination_results[algo_name]['episodes'].append(result.metadata['episodes_run'])
                    
                    # NEW: Store per-iteration result
                    iteration_result = {
                        'algorithm': algo_name,
                        'dag_type': dag_type,
                        'dag_param': param,
                        'num_processors': num_processors,
                        'iteration': iteration,
                        'makespan': result.makespan,
                        'execution_time': execution_time,
                        'num_tasks': dag.num_tasks
                    }
                    
                    if algo_name == 'ql_heft' and 'episodes_run' in result.metadata:
                        iteration_result['episodes'] = result.metadata['episodes_run']
                        
                    self.iteration_results.append(iteration_result)
                    
                except Exception as e:
                    print(f"Error with {algo_name} on {dag_type} (param={param}): {str(e)}")
                    # Record as NaN for aggregation
                    combination_results[algo_name]['makespan'].append(np.nan)
                    combination_results[algo_name]['execution_time'].append(np.nan)
                    if algo_name == 'ql_heft':
                        combination_results[algo_name]['episodes'].append(np.nan)
                    
                    # Record error in per-iteration results
                    iteration_result = {
                        'algorithm': algo_name,
                        'dag_type': dag_type,
                        'dag_param': param,
                        'num_processors': num_processors,
                        'iteration': iteration,
                        'makespan': np.nan,
                        'execution_time': np.nan,
                        'num_tasks': dag.num_tasks,
                        'error': True
                    }
                    self.iteration_results.append(iteration_result)
                
            iterations_progress.update(1)
        
        iterations_progress.close()
        
        # Keep the original aggregation code for backward compatibility
        for algo_name in combination_results:
            # Calculate means, ignoring NaN values
            mean_makespan = np.nanmean(combination_results[algo_name]['makespan'])
            mean_execution_time = np.nanmean(combination_results[algo_name]['execution_time'])
            std_makespan = np.nanstd(combination_results[algo_name]['makespan'])
            
            result_entry = {
                'algorithm': algo_name,
                'dag_type': dag_type,
                'dag_param': param,
                'num_processors': num_processors,
                'avg_makespan': mean_makespan,
                'std_makespan': std_makespan,
                'avg_execution_time': mean_execution_time,
                'num_tasks': dag.num_tasks
            }
            
            if algo_name == 'ql_heft' and 'episodes' in combination_results[algo_name]:
                episodes = combination_results[algo_name]['episodes']
                if len(episodes) > 0 and not all(np.isnan(ep) for ep in episodes):
                    result_entry['avg_episodes'] = np.nanmean(episodes)
                else:
                    result_entry['avg_episodes'] = np.nan
            
            self.results.append(result_entry)
    
    def _generate_summary(self, results_df: pd.DataFrame, iteration_df: pd.DataFrame):
        """
        Generate summary visualizations of the results.
        
        Args:
            results_df: DataFrame with experiment results
        """
        # Create visualizations directory
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # 1. Makespan comparison across algorithms for each DAG type/param
        self._plot_makespan_comparison(results_df, vis_dir)
        
        # 2. Execution time comparison
        self._plot_execution_time_comparison(results_df, vis_dir)
        
        # 3. Efficiency comparison (makespan ratio to HEFT)
        self._plot_efficiency_comparison(results_df, vis_dir)
        
        # 4. Summary table
        self._create_summary_table(results_df, vis_dir)
        self._plot_iteration_details(iteration_df, vis_dir)
    
    def _plot_makespan_comparison(self, df: pd.DataFrame, vis_dir: str):
        """Plot makespan comparison across algorithms."""
        plt.figure(figsize=(15, 10))
        
        # Create a grid of subplots, one for each DAG type
        dag_types = df['dag_type'].unique()
        fig, axes = plt.subplots(1, len(dag_types), figsize=(5*len(dag_types), 6), sharey=False)
        
        for i, dag_type in enumerate(dag_types):
            df_subset = df[df['dag_type'] == dag_type]
            ax = axes[i]
            
            # Group by DAG param and processor count
            for param in df_subset['dag_param'].unique():
                df_param = df_subset[df_subset['dag_param'] == param]
                
                # Pivot to get algorithms as columns
                pivot_df = df_param.pivot(index='num_processors', columns='algorithm', values='avg_makespan')
                pivot_df.plot(marker='o', ax=ax)
                
                ax.set_title(f"{dag_type.capitalize()} DAG (param={param})")
                ax.set_xlabel("Number of Processors")
                ax.set_xticks(df_param['num_processors'].unique())
            
            if i == 0:
                ax.set_ylabel("Average Makespan")
            
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "makespan_comparison.png"), dpi=300)
        plt.close()

    def _plot_iteration_details(self, df: pd.DataFrame, vis_dir: str):
        """Plot detailed per-iteration results."""
        # Create a directory for iteration plots
        iter_dir = os.path.join(vis_dir, 'iterations')
        os.makedirs(iter_dir, exist_ok=True)
        
        # Group by DAG type, param, processors
        for dag_type in df['dag_type'].unique():
            for param in df[df['dag_type'] == dag_type]['dag_param'].unique():
                for procs in df[(df['dag_type'] == dag_type) & (df['dag_param'] == param)]['num_processors'].unique():
                    # Filter data for this combination
                    combo_df = df[(df['dag_type'] == dag_type) & 
                                (df['dag_param'] == param) & 
                                (df['num_processors'] == procs)]
                    
                    # Create makespan plot
                    plt.figure(figsize=(12, 6))
                    sns.lineplot(data=combo_df, x='iteration', y='makespan', hue='algorithm', marker='o')
                    plt.title(f"{dag_type.capitalize()} DAG (param={param}, procs={procs}): Makespan by Iteration")
                    plt.xlabel("Iteration")
                    plt.ylabel("Makespan")
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    plt.savefig(os.path.join(iter_dir, f"makespan_{dag_type}_p{param}_proc{procs}.png"), dpi=300)
                    plt.close()
                    
                    # Create execution time plot
                    plt.figure(figsize=(12, 6))
                    sns.lineplot(data=combo_df, x='iteration', y='execution_time', hue='algorithm', marker='o')
                    plt.title(f"{dag_type.capitalize()} DAG (param={param}, procs={procs}): Execution Time by Iteration")
                    plt.xlabel("Iteration")
                    plt.ylabel("Execution Time (seconds)")
                    plt.yscale('log')  # Use log scale for better visualization
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    plt.savefig(os.path.join(iter_dir, f"exectime_{dag_type}_p{param}_proc{procs}.png"), dpi=300)
                    plt.close()
    
    def _plot_execution_time_comparison(self, df: pd.DataFrame, vis_dir: str):
        """Plot execution time comparison across algorithms."""
        plt.figure(figsize=(10, 6))
        
        sns.barplot(x='algorithm', y='avg_execution_time', hue='dag_type', data=df)
        plt.title("Algorithm Execution Time Comparison")
        plt.xlabel("Algorithm")
        plt.ylabel("Average Execution Time (seconds)")
        plt.yscale('log')  # Use log scale for better visualization
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()
        
        plt.savefig(os.path.join(vis_dir, "execution_time_comparison.png"), dpi=300)
        plt.close()
    
    def _plot_efficiency_comparison(self, df: pd.DataFrame, vis_dir: str):
        """Plot efficiency comparison (makespan ratio to HEFT)."""
        # Create efficiency metric
        pivot_df = df.pivot_table(
            index=['dag_type', 'dag_param', 'num_processors'],
            columns='algorithm',
            values='avg_makespan'
        )
        
        # Calculate ratio to HEFT (>1 means worse than HEFT, <1 means better)
        for algo in pivot_df.columns:
            if algo != 'heft':
                pivot_df[f"{algo}_ratio"] = pivot_df[algo] / pivot_df['heft']
        
        # Extract ratio columns
        ratio_cols = [col for col in pivot_df.columns if col.endswith('_ratio')]
        ratio_df = pivot_df[ratio_cols].reset_index()
        
        # Melt for easier plotting
        melted_df = pd.melt(
            ratio_df,
            id_vars=['dag_type', 'dag_param', 'num_processors'],
            var_name='algorithm_ratio',
            value_name='makespan_ratio'
        )
        melted_df['algorithm'] = melted_df['algorithm_ratio'].str.replace('_ratio', '')
        
        plt.figure(figsize=(12, 8))
        
        # Box plot of ratios
        sns.boxplot(x='algorithm', y='makespan_ratio', hue='dag_type', data=melted_df)
        
        plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.7, label="HEFT Baseline")
        plt.title("Algorithm Efficiency Relative to HEFT")
        plt.xlabel("Algorithm")
        plt.ylabel("Makespan Ratio (algorithm/HEFT)")
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.legend(title="DAG Type")
        plt.tight_layout()
        
        plt.savefig(os.path.join(vis_dir, "efficiency_comparison.png"), dpi=300)
        plt.close()
    
    def _create_summary_table(self, df: pd.DataFrame, vis_dir: str):
        """Create summary table of results."""
        # Group by algorithm and calculate average metrics
        summary = df.groupby('algorithm').agg({
            'avg_makespan': ['mean', 'std'],
            'avg_execution_time': ['mean', 'std']
        }).reset_index()
        
        # Flatten MultiIndex columns
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
        
        # Save summary to CSV
        summary.to_csv(os.path.join(vis_dir, "summary_table.csv"), index=False)
        
        # Calculate percentage improvement over HEFT
        pivot_df = df.pivot_table(
            index=['dag_type', 'dag_param', 'num_processors'],
            columns='algorithm',
            values='avg_makespan'
        )
        
        heft_baseline = pivot_df['heft']
        improvement_data = {}
        
        for algo in pivot_df.columns:
            if algo != 'heft':
                # Calculate percentage improvement: (heft - algo) / heft * 100
                improvement = (heft_baseline - pivot_df[algo]) / heft_baseline * 100
                improvement_data[f"{algo}_improvement"] = improvement.mean()
        
        # Create improvement summary
        improvement_df = pd.DataFrame([improvement_data])
        improvement_df.to_csv(os.path.join(vis_dir, "improvement_summary.csv"), index=False)


def main():
    """Run the HEFT comparison experiment."""
    
    # Define paths
    output_dir = "heft_comparison_results"
    xgboost_model_path = "ml_final_output/models/xgboost_model.joblib"
    mlp_model_path = "ml_final_output/models/mlp_heft_model.joblib"

    # Check if ML models exist
    print("Checking for trained models...")
    if not os.path.exists(xgboost_model_path):
        print(f"Warning: XGBoost model not found at {xgboost_model_path}")
    else:
        print(f"Found XGBoost model at {xgboost_model_path}")
    
    if not os.path.exists(mlp_model_path):
        print(f"Warning: MLP model not found at {mlp_model_path}")
    else:
        print(f"Found MLP model at {mlp_model_path}")
    
    # Create experiment instance with reduced iterations for testing
    experiment = HEFTComparisonExperiment(
        output_dir=output_dir,
        xgboost_model_path=xgboost_model_path,
        mlp_model_path=mlp_model_path,
        num_iterations=50,  # Reduced from 50 for testing; change to 50 for full experiment
        seed=42
    )
    
    # Run experiment
    results, iterations = experiment.run_experiment()
    
    # Display summary
    print("\nExperiment Summary:")
    print("===================")
    
    # Group by algorithm and show average improvement
    algo_summary = results.groupby('algorithm')['avg_makespan'].mean()
    heft_baseline = algo_summary['heft']
    
    print(f"HEFT average makespan: {heft_baseline:.2f}")
    
    for algo, avg_makespan in algo_summary.items():
        if algo != 'heft':
            improvement = (heft_baseline - avg_makespan) / heft_baseline * 100
            print(f"{algo} average makespan: {avg_makespan:.2f} ({improvement:.2f}% improvement)")


if __name__ == "__main__":
    main()