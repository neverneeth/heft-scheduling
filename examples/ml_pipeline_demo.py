"""
ML Pipeline Demo: Complete demonstration of Dataset V1 and Model V1.

This example shows the full machine learning pipeline:
1. Generate expert dataset from HEFT/QL-HEFT runs
2. Train regression models to predict upward ranks
3. Use trained model for scheduling new DAGs
4. Compare performance against original HEFT
"""

import os
import time
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.ml import ExpertDatasetGenerator, RegressionHEFT
from src.algorithms.heft import HEFTAlgorithm
from src.utils.dag_generator import DAGGenerator
from src.core.workflow_dag import WorkflowDAG
from src.utils.visualizer import Visualizer
from src.utils.schedule_validator import validate_schedule

class MLPipelineDemo:
    """Demonstrates the complete ML pipeline for HEFT scheduling."""
    
    def __init__(self, output_dir: str = "ml_demo_output"):
        """Initialize the demo."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize algorithms
        self.heft_original = HEFTAlgorithm()
        self.dataset_generator = ExpertDatasetGenerator(
            output_dir=os.path.join(output_dir, "datasets")
        )
        
    def run_complete_pipeline(self) -> Dict:
        """Run the complete ML pipeline demonstration."""
        print("Starting ML Pipeline Demo for HEFT Scheduling")
        print("="*60)
        
        results = {}
        
        # Step 1: Generate expert dataset
        print("\nStep 1: Generating Expert Dataset (Dataset V1)")
        dataset_start = time.time()
        
        dataset_df = self.dataset_generator.generate_dataset_v1(
            num_dags=5,  # Smaller dataset for demo
            ql_runs_per_dag=3,  # Fewer runs for speed
            save_intermediate=False
        )
        
        dataset_time = time.time() - dataset_start
        dataset_path = os.path.join(self.output_dir, "datasets", "dataset_v1.csv")
        
        results['dataset'] = {
            'samples': len(dataset_df),
            'features': len([c for c in dataset_df.columns if c not in ['dag_id', 'task_id', 'upward_rank']]),
            'generation_time': dataset_time,
            'path': dataset_path
        }
        
        print(f"Dataset generated: {len(dataset_df)} samples in {dataset_time:.2f}s")
        
        # Step 2: Train multiple models
        print("\nStep 2: Training Regression Models (Model V1)")
        model_results = {}
        
        model_types = ["xgboost", "random_forest", "gradient_boosting", "ridge"]
        
        for model_type in model_types:
            print(f"\n  Training {model_type} model...")
            
            # Initialize and train model
            reg_heft = RegressionHEFT(model_type=model_type)
            
            model_start = time.time()
            train_results = reg_heft.train(
                dataset_data=dataset_path,
                test_size=0.2,
                tune_hyperparameters=True,
                save_model_path=os.path.join(
                    self.output_dir, "models", f"{model_type}_model.joblib"
                )
            )
            model_time = time.time() - model_start
            
            model_results[model_type] = {
                **train_results,
                'training_time': model_time
            }
            
            print(f" {model_type}: R² = {train_results['test_metrics']['r2']:.3f}")
        
        results['models'] = model_results
        
        # Step 3: Evaluate on test DAGs
        print("\nStep 3: Evaluating on Test DAGs")
        
        # Load best model based on test R²
        best_model_type = max(
            model_results.keys(), 
            key=lambda k: model_results[k]['test_metrics']['r2']
        )
        
        print(f"  Best model: {best_model_type} (R² = {model_results[best_model_type]['test_metrics']['r2']:.3f})")
        
        # Load the best trained model
        reg_heft_best = RegressionHEFT(
            model_path=os.path.join(self.output_dir, "models", f"{best_model_type}_model.joblib")
        )
        
        # Generate test DAGs
        test_dags = self._generate_test_dags(num_dags=20)
        
        # Compare algorithms
        comparison_results = self._compare_algorithms(
            test_dags, 
            {
                'Original HEFT': self.heft_original,
                f'Regression HEFT ({best_model_type})': reg_heft_best
            }
        )
        
        results['evaluation'] = comparison_results
        
        # Step 4: Generate analysis and visualizations
        print("\nStep 4: Generating Analysis")
        self._generate_analysis_report(results, best_model_type, reg_heft_best)

        print("\nML Pipeline Demo Complete!")
        print(f"Results saved to: {self.output_dir}")

        return results
    
    def _generate_test_dags(self, num_dags: int = 20) -> List[WorkflowDAG]:
        """Generate a diverse set of test DAGs."""
        test_dags = []
        
        # Random DAGs
        for i in range(num_dags // 4):
            edges, costs, comm = DAGGenerator.generate_random_dag(
                num_tasks=15 + i * 2,
                num_processors=4,
                edge_probability=0.3,
                random_seed=1000 + i
            )
            test_dags.append(WorkflowDAG(edges, costs, comm))
        
        # Layered DAGs
        for i in range(num_dags // 4):
            dag = DAGGenerator.generate_layered_dag(
                num_layers=4,
                tasks_per_layer=4,
                num_processors=4,
                random_seed=2000 + i
            )
            test_dags.append(dag)
        
        # Fork-Join DAGs
        for i in range(num_dags // 4):
            dag = DAGGenerator.generate_fork_join_dag(
                num_initial_tasks=2,
                num_parallel_tasks=8,
                num_final_tasks=2,
                num_processors=4,
                random_seed=3000 + i
            )
            test_dags.append(dag)
        
        # Benchmark DAGs
        remaining = num_dags - len(test_dags)
        for i in range(remaining):
            edges, costs, comm = DAGGenerator.generate_gaussian_elimination_dag(
                chi=4 + i % 3,
                num_processors=4,
                random_seed=4000 + i
            )
            test_dags.append(WorkflowDAG(edges, costs, comm))
        
        return test_dags
    
    def _compare_algorithms(self, test_dags: List[WorkflowDAG], algorithms: Dict) -> Dict:
        """Compare different scheduling algorithms on test DAGs."""
        results = {alg_name: {'makespans': [], 'times': []} 
                  for alg_name in algorithms.keys()}
        
        print(f"  Testing on {len(test_dags)} DAGs...")
        
        for i, dag in enumerate(test_dags):
            print(f"    DAG {i+1}/{len(test_dags)}: {dag.num_tasks} tasks")
            
            for alg_name, algorithm in algorithms.items():
                start_time = time.time()
                result = algorithm.schedule(dag)
                # Visualizer.visualize_gantt_chart(result=result, title=f"{alg_name} Schedule for DAG {i+1}")
                execution_time = time.time() - start_time
                is_valid = validate_schedule(dag, result, verbose=True)
                if is_valid:
                    results[alg_name]['makespans'].append(result.makespan)
                    results[alg_name]['times'].append(execution_time)
                else:
                    print(f"Invalid schedule produced by {alg_name} for DAG {i+1}, skipping.")

        # Calculate summary statistics
        for alg_name in results:
            makespans = results[alg_name]['makespans']
            times = results[alg_name]['times']
            
            results[alg_name]['stats'] = {
                'avg_makespan': sum(makespans) / len(makespans),
                'min_makespan': min(makespans),
                'max_makespan': max(makespans),
                'avg_time': sum(times) / len(times),
                'total_time': sum(times)
            }
        
        # Calculate relative performance
        original_avg = results['Original HEFT']['stats']['avg_makespan']
        for alg_name in results:
            if alg_name != 'Original HEFT':
                ml_avg = results[alg_name]['stats']['avg_makespan']
                improvement = (original_avg - ml_avg) / original_avg * 100
                results[alg_name]['stats']['improvement_percent'] = improvement
        
        return results
    
    def _generate_analysis_report(self, results: Dict, best_model_type: str, reg_heft: RegressionHEFT):
        """Generate comprehensive analysis report with visualizations."""
        
        # 1. Model Performance Comparison Plot
        plt.figure(figsize=(12, 8))
        
        # Model comparison
        plt.subplot(2, 2, 1)
        model_names = list(results['models'].keys())
        r2_scores = [results['models'][model]['test_metrics']['r2'] for model in model_names]
        
        bars = plt.bar(model_names, r2_scores, color=['skyblue', 'lightgreen', 'orange', 'pink'])
        plt.title('Model Performance Comparison (Test R²)')
        plt.ylabel('R² Score')
        plt.xticks(rotation=45)
        
        # Highlight best model
        best_idx = model_names.index(best_model_type)
        bars[best_idx].set_color('gold')
        
        # Scheduling performance comparison
        plt.subplot(2, 2, 2)
        alg_names = list(results['evaluation'].keys())
        avg_makespans = [results['evaluation'][alg]['stats']['avg_makespan'] 
                        for alg in alg_names]
        
        bars = plt.bar(alg_names, avg_makespans, color=['lightcoral', 'lightblue'])
        plt.title('Average Makespan Comparison')
        plt.ylabel('Makespan')
        plt.xticks(rotation=45)
        
        # Makespan distribution
        plt.subplot(2, 2, 3)
        for alg_name in alg_names:
            makespans = results['evaluation'][alg_name]['makespans']
            plt.hist(makespans, alpha=0.6, label=alg_name, bins=10)
        
        plt.title('Makespan Distribution')
        plt.xlabel('Makespan')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Feature importance (if available)
        plt.subplot(2, 2, 4)
        try:
            importance = reg_heft.get_feature_importance(top_n=10)
            features = list(importance.keys())
            values = list(importance.values())
            
            plt.barh(features, values)
            plt.title('Top 10 Feature Importance')
            plt.xlabel('Importance')
        except Exception as e:
            plt.text(0.5, 0.5, f'Feature importance\nnot available:\n{str(e)}', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Feature Importance (Unavailable)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'ml_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Generate text report
        report_path = os.path.join(self.output_dir, 'ml_pipeline_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# ML Pipeline Results Report\n\n")
            
            f.write("## Dataset Generation (Dataset V1)\n")
            f.write(f"- **Samples**: {results['dataset']['samples']:,}\n")
            f.write(f"- **Features**: {results['dataset']['features']}\n")
            f.write(f"- **Generation Time**: {results['dataset']['generation_time']:.2f}s\n\n")
            
            f.write("## Model Training Results (Model V1)\n")
            f.write("| Model | Test R² | Test MSE | Training Time |\n")
            f.write("|-------|---------|----------|---------------|\n")
            
            for model_name, model_data in results['models'].items():
                r2 = model_data['test_metrics']['r2']
                mse = model_data['test_metrics']['mse']
                time_taken = model_data['training_time']
                marker = "*" if model_name == best_model_type else ""
                f.write(f"| {model_name} {marker} | {r2:.4f} | {mse:.4f} | {time_taken:.2f}s |\n")
            
            f.write(f"\n**Best Model**: {best_model_type}\n\n")
            
            f.write("## Scheduling Performance Evaluation\n")
            f.write("| Algorithm | Avg Makespan | Improvement | Avg Time |\n")
            f.write("|-----------|--------------|-------------|----------|\n")
            
            for alg_name, alg_data in results['evaluation'].items():
                avg_makespan = alg_data['stats']['avg_makespan']
                avg_time = alg_data['stats']['avg_time'] * 1000  # Convert to ms
                improvement = alg_data['stats'].get('improvement_percent', 0)
                
                if improvement != 0:
                    improvement_str = f"{improvement:+.2f}%"
                else:
                    improvement_str = "baseline"
                
                f.write(f"| {alg_name} | {avg_makespan:.2f} | {improvement_str} | {avg_time:.2f}ms |\n")
            
            f.write("\n## Key Findings\n")
            
            # Performance analysis
            ml_alg_name = [name for name in results['evaluation'].keys() if 'Regression' in name][0]
            ml_stats = results['evaluation'][ml_alg_name]['stats']
            improvement = ml_stats.get('improvement_percent', 0)
            
            if improvement > 0:
                f.write(f"**Positive Result**: Regression HEFT achieved {improvement:.2f}% improvement over original HEFT\n")
            elif improvement > -5:
                f.write(f"**Comparable Performance**: Regression HEFT performed within 5% of original HEFT ({improvement:.2f}%)\n")
            else:
                f.write(f"**Lower Performance**: Regression HEFT underperformed by {abs(improvement):.2f}%\n")

            f.write(f"- Best ML model: **{best_model_type}** with R² = {results['models'][best_model_type]['test_metrics']['r2']:.3f}\n")
            f.write(f"- Average training time: {sum(m['training_time'] for m in results['models'].values()) / len(results['models']):.2f}s per model\n")
            
            # Speed analysis
            original_time = results['evaluation']['Original HEFT']['stats']['avg_time'] * 1000
            ml_time = results['evaluation'][ml_alg_name]['stats']['avg_time'] * 1000
            
            if ml_time < original_time:
                f.write(f"**Speed Improvement**: {original_time/ml_time:.1f}x faster than original HEFT\n")
            else:
                f.write(f"**Speed Trade-off**: {ml_time/original_time:.1f}x slower than original HEFT\n")

        print(f"Analysis report saved to: {report_path}")


def main():
    """Run the complete ML pipeline demo."""
    demo = MLPipelineDemo()
    results = demo.run_complete_pipeline()
    print(len(results))
    print("\n Demo completed successfully!")
    print("\nQuick Summary:")
    #print(f"- Generated {results['dataset']['samples']} training samples")
    #print(f"- Trained {len(results['models'])} different models")
    #print(f"- Evaluated on {len(results['evaluation']['Original HEFT']['makespans'])} test DAGs")
    
    # Find best performing algorithm
    best_alg = min(
        results['evaluation'].items(),
        key=lambda x: x[1]['stats']['avg_makespan']
    )
    
    print(f"- Best performing algorithm: {best_alg[0]}")
    print(f"- Average makespan: {best_alg[1]['stats']['avg_makespan']:.2f}")


if __name__ == "__main__":
    main()