"""
Smart ML-HEFT Tester: Demonstrates efficient model reuse.

This script shows how to:
1. Check if a trained model exists
2. Load existing model OR train a new one
3. Save the model for future use
4. Use the model to get predictions quickly
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.ml import RegressionHEFT, ExpertDatasetGenerator
from src.algorithms.heft import HEFTAlgorithm
from src.core.workflow_dag import WorkflowDAG
from src.utils.dag_generator import DAGGenerator
from src.utils.visualizer import Visualizer
import time

def get_or_train_model(model_path="models/trained_xgboost.joblib", force_retrain=False):
    """
    Get a trained model - either load existing or train new one.
    
    Args:
        model_path: Path to save/load the model
        force_retrain: If True, retrain even if model exists
        
    Returns:
        RegressionHEFT: Trained model ready for use
    """
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Check if model exists and load it
    if os.path.exists(model_path) and not force_retrain:
        print(f"📁 Loading existing model from {model_path}")
        ml_heft = RegressionHEFT()
        ml_heft.load_model(model_path)
        print(f"✅ Model loaded successfully!")
        return ml_heft
    
    # Train new model
    print(f"🤖 Training new model (will save to {model_path})")
    ml_heft = RegressionHEFT(model_type="xgboost")
    
    # Generate training data
    print("📊 Generating training dataset...")
    dataset_gen = ExpertDatasetGenerator()
    training_data = dataset_gen.generate_dataset_v1(
        num_dags=20,  # Reasonable size for quick training
        ql_runs_per_dag=5  # Fewer runs for speed
    )
    
    # Train the model
    print("🎓 Training model...")
    training_start = time.time()
    ml_heft.train(training_data, save_model_path=model_path)
    training_time = time.time() - training_start
    
    print(f"✅ Model trained and saved in {training_time:.2f} seconds")
    return ml_heft

def quick_prediction_demo(ml_heft, num_tests=5):
    """
    Demonstrate quick predictions on multiple DAGs.
    
    Args:
        ml_heft: Trained RegressionHEFT model
        num_tests: Number of test DAGs to generate
    """
    print(f"\n🚀 Quick Prediction Demo ({num_tests} test DAGs)")
    print("="*60)
    
    dag_generator = DAGGenerator()
    heft_original = HEFTAlgorithm()
    
    results = []
    
    for i in range(num_tests):
        # Generate test DAG
        E, W, c = dag_generator.generate_random_dag(
            num_tasks=8 + i*2,  # Vary DAG size
            num_processors=3,
            edge_probability=0.4,
            random_seed=100 + i
        )
        dag = WorkflowDAG(E, W, c)
        
        # Quick ML prediction (no training needed!)
        ml_start = time.time()
        ml_result = ml_heft.schedule(dag)
        ml_time = time.time() - ml_start
        
        # Original HEFT for comparison
        heft_start = time.time()
        heft_result = heft_original.schedule(dag)
        heft_time = time.time() - heft_start
        
        # Store results
        results.append({
            'dag_size': dag.num_tasks,
            'ml_makespan': ml_result.makespan,
            'heft_makespan': heft_result.makespan,
            'ml_time': ml_time * 1000,  # Convert to ms
            'heft_time': heft_time * 1000
        })
        
        difference = abs(ml_result.makespan - heft_result.makespan)
        percent_diff = (difference / heft_result.makespan) * 100
        
        print(f"DAG {i+1} ({dag.num_tasks} tasks): "
              f"HEFT={heft_result.makespan:.1f}, "
              f"ML-HEFT={ml_result.makespan:.1f}, "
              f"diff={percent_diff:.1f}%, "
              f"time={ml_time*1000:.1f}ms")
    
    # Summary statistics
    print(f"\n📊 Summary Statistics:")
    avg_diff = sum(abs(r['ml_makespan'] - r['heft_makespan']) / r['heft_makespan'] * 100 for r in results) / len(results)
    avg_ml_time = sum(r['ml_time'] for r in results) / len(results)
    avg_heft_time = sum(r['heft_time'] for r in results) / len(results)
    
    print(f"  Average difference: {avg_diff:.2f}%")
    print(f"  Average ML-HEFT time: {avg_ml_time:.1f}ms")
    print(f"  Average HEFT time: {avg_heft_time:.1f}ms")
    print(f"  Speed ratio: {avg_ml_time/avg_heft_time:.1f}x slower")

def visualize_schedule_comparison(dag, ml_result, heft_result, save_path="visualizations"):
    """
    Create Gantt chart visualizations comparing ML-HEFT and Original HEFT.
    
    Args:
        dag: The workflow DAG
        ml_result: RegressionHEFT schedule result
        heft_result: Original HEFT schedule result
        save_path: Directory to save visualization files
    """
    print(f"\n📊 Creating Gantt Chart Visualizations...")
    
    # Create visualizations directory
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize visualizer
    visualizer = Visualizer()
    
    try:
        # Create ML-HEFT Gantt chart
        ml_chart_path = os.path.join(save_path, "ml_heft_gantt.png")
        visualizer.visualize_gantt_chart(
            dag, 
            ml_result, 
            title=f"ML-HEFT Schedule (Makespan: {ml_result.makespan:.2f})",
            save_path=ml_chart_path
        )
        print(f"✅ ML-HEFT Gantt chart saved: {ml_chart_path}")
        
        # Create Original HEFT Gantt chart
        heft_chart_path = os.path.join(save_path, "original_heft_gantt.png")
        visualizer.visualize_gantt_chart(
            dag, 
            heft_result, 
            title=f"Original HEFT Schedule (Makespan: {heft_result.makespan:.2f})",
            save_path=heft_chart_path
        )
        print(f"✅ Original HEFT Gantt chart saved: {heft_chart_path}")
        
        # Create algorithm comparison
        comparison_path = os.path.join(save_path, "algorithm_comparison.png")
        visualizer.compare_algorithms(
            dag, 
            {"ML-HEFT": ml_result, "Original HEFT": heft_result},
            save_path=comparison_path
        )
        print(f"✅ Algorithm comparison saved: {comparison_path}")
        
        # Also create DAG visualization
        dag_path = os.path.join(save_path, "dag_structure.png")
        visualizer.visualize_dag(
            dag, 
            title=f"DAG Structure ({dag.num_tasks} tasks, {dag.num_processors} processors)",
            save_path=dag_path
        )
        print(f"✅ DAG structure saved: {dag_path}")
        
        print(f"\n📁 All visualizations saved in: {save_path}/")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {str(e)}")
        print("💡 Tip: Make sure matplotlib is installed and you have display capabilities")

def interactive_visualization_demo(ml_heft):
    """
    Interactive demo that lets user choose a DAG and see visualizations.
    
    Args:
        ml_heft: Trained RegressionHEFT model
    """
    print(f"\n🎨 Interactive Visualization Demo")
    print("="*60)
    
    dag_generator = DAGGenerator()
    heft_original = HEFTAlgorithm()
    
    # Let user choose DAG type
    print("Choose a DAG type to visualize:")
    print("1. Random DAG (small)")
    print("2. Random DAG (medium)")
    print("3. Gaussian Elimination")
    print("4. Epigenomics")
    print("5. Laplace")
    
    try:
        choice = input("Enter choice (1-5, default=1): ").strip() or "1"
    except:
        choice = "1"
    
    # Generate DAG based on choice
    try:
        if choice == "1":
            E, W, c = dag_generator.generate_random_dag(8, 3, 0.4, 300)
            dag_name = "Random DAG (Small)"
        elif choice == "2":
            E, W, c = dag_generator.generate_random_dag(15, 4, 0.3, 301)
            dag_name = "Random DAG (Medium)"
        elif choice == "3":
            E, W, c = dag_generator.generate_gaussian_elimination_dag(4, 3, 302)
            dag_name = "Gaussian Elimination"
        elif choice == "4":
            E, W, c = dag_generator.generate_epigenomics_dag(3, 3, 303)
            dag_name = "Epigenomics"
        elif choice == "5":
            E, W, c = dag_generator.generate_laplace_dag(3, 3, 304)
            dag_name = "Laplace"
        else:
            E, W, c = dag_generator.generate_random_dag(8, 3, 0.4, 300)
            dag_name = "Random DAG (Small)"
            
        dag = WorkflowDAG(E, W, c)
        print(f"Generated {dag_name}: {dag.num_tasks} tasks, {dag.num_processors} processors")
        
        # Schedule with both algorithms
        print("🔄 Scheduling with both algorithms...")
        ml_result = ml_heft.schedule(dag)
        heft_result = heft_original.schedule(dag)
        
        # Show results
        print(f"\n📊 Scheduling Results:")
        print(f"  ML-HEFT makespan:      {ml_result.makespan:.2f}")
        print(f"  Original HEFT makespan: {heft_result.makespan:.2f}")
        
        difference = abs(ml_result.makespan - heft_result.makespan)
        percent_diff = (difference / heft_result.makespan) * 100
        print(f"  Difference:             {difference:.2f} ({percent_diff:.1f}%)")
        
        # Create visualizations
        vis_dir = f"visualizations/{dag_name.lower().replace(' ', '_')}"
        visualize_schedule_comparison(dag, ml_result, heft_result, vis_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Error in visualization demo: {str(e)}")
        return False

def benchmark_different_dag_types(ml_heft):
    """
    Test the model on different types of DAGs.
    
    Args:
        ml_heft: Trained RegressionHEFT model
    """
    print(f"\n🔬 Benchmark on Different DAG Types")
    print("="*60)
    
    dag_generator = DAGGenerator()
    heft_original = HEFTAlgorithm()
    
    # Test different DAG types
    dag_types = [
        ("Random DAG", lambda: dag_generator.generate_random_dag(10, 3, 0.3, 200)),
        ("Gaussian Elimination", lambda: dag_generator.generate_gaussian_elimination_dag(3, 3, 201)),
        ("Epigenomics", lambda: dag_generator.generate_epigenomics_dag(3, 3, 202)),
        ("Laplace", lambda: dag_generator.generate_laplace_dag(3, 3, 203))
    ]
    
    for dag_name, dag_func in dag_types:
        try:
            E, W, c = dag_func()
            dag = WorkflowDAG(E, W, c)
            
            # Get predictions
            ml_result = ml_heft.schedule(dag)
            heft_result = heft_original.schedule(dag)
            
            difference = abs(ml_result.makespan - heft_result.makespan)
            percent_diff = (difference / heft_result.makespan) * 100
            
            print(f"{dag_name:20}: "
                  f"HEFT={heft_result.makespan:.1f}, "
                  f"ML-HEFT={ml_result.makespan:.1f}, "
                  f"diff={percent_diff:.1f}%")
                  
        except Exception as e:
            print(f"{dag_name:20}: Error - {str(e)}")

def main():
    """
    Test the model on different types of DAGs.
    
    Args:
        ml_heft: Trained RegressionHEFT model
    """
    print(f"\n🔬 Benchmark on Different DAG Types")
    print("="*60)
    
    dag_generator = DAGGenerator()
    heft_original = HEFTAlgorithm()
    
    # Test different DAG types
    dag_types = [
        ("Random DAG", lambda: dag_generator.generate_random_dag(10, 3, 0.3, 200)),
        ("Gaussian Elimination", lambda: dag_generator.generate_gaussian_elimination_dag(3, 3, 201)),
        ("Epigenomics", lambda: dag_generator.generate_epigenomics_dag(3, 3, 202)),
        ("Laplace", lambda: dag_generator.generate_laplace_dag(3, 3, 203))
    ]
    
    for dag_name, dag_func in dag_types:
        try:
            E, W, c = dag_func()
            dag = WorkflowDAG(E, W, c)
            
            # Get predictions
            ml_result = ml_heft.schedule(dag)
            heft_result = heft_original.schedule(dag)
            
            difference = abs(ml_result.makespan - heft_result.makespan)
            percent_diff = (difference / heft_result.makespan) * 100
            
            print(f"{dag_name:20}: "
                  f"HEFT={heft_result.makespan:.1f}, "
                  f"ML-HEFT={ml_result.makespan:.1f}, "
                  f"diff={percent_diff:.1f}%")
                  
        except Exception as e:
            print(f"{dag_name:20}: Error - {str(e)}")

def main():
    """Main demonstration function."""
    print("🧠 Smart ML-HEFT Tester: Model Reuse Demo")
    print("="*60)
    
    # Step 1: Get trained model (load or train)
    model_path = "models/smart_xgboost_model.joblib"
    
    print("Choose an option:")
    print("1. Use existing model (if available)")
    print("2. Force retrain new model")
    
    try:
        choice = input("Enter choice (1 or 2, default=1): ").strip() or "1"
        force_retrain = (choice == "2")
    except:
        force_retrain = False
    
    # Get or train model
    ml_heft = get_or_train_model(model_path, force_retrain=force_retrain)
    
    # Ask user what they want to do
    print(f"\nWhat would you like to do?")
    print("1. Quick predictions demo")
    print("2. Interactive visualization demo")
    print("3. Benchmark different DAG types")
    print("4. All of the above")
    
    try:
        demo_choice = input("Enter choice (1-4, default=4): ").strip() or "4"
    except:
        demo_choice = "4"
    
    # Execute chosen demos
    if demo_choice in ["1", "4"]:
        quick_prediction_demo(ml_heft, num_tests=5)
    
    if demo_choice in ["2", "4"]:
        interactive_visualization_demo(ml_heft)
    
    if demo_choice in ["3", "4"]:
        benchmark_different_dag_types(ml_heft)
    
    print(f"\n✅ Demo complete! Model saved at: {model_path}")
    print("💡 Next time you run this script, it will load instantly!")
    print("📊 Check the 'visualizations/' folder for Gantt charts!")

if __name__ == "__main__":
    main()