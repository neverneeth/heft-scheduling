import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.ml import RegressionHEFT, ExpertDatasetGenerator
from src.algorithms.heft import HEFTAlgorithm
from src.core.workflow_dag import WorkflowDAG
from src.utils.dag_generator import DAGGenerator
import time

def main():
    print("🔬 ML-HEFT Model Testing")
    print("="*50)
    
    # Step 1: Create test DAG
    dag_generator = DAGGenerator()
    E, W, c = dag_generator.generate_random_dag(
        num_tasks=10,
        num_processors=3,
        edge_probability=0.3,
        random_seed=42
    )
    dag = WorkflowDAG(E, W, c)
    print(f"Generated test DAG: {dag.num_tasks} tasks, {dag.num_processors} processors")

    # Step 2: Smart model loading/training
    model_path = "models/mltester_model.joblib"
    
    if os.path.exists(model_path):
        print(f"\n📁 Loading existing model from {model_path}")
        ml_heft = RegressionHEFT()
        ml_heft.load_model(model_path)
        print("✅ Model loaded instantly!")
        training_time = 0  # No training needed
    else:
        print(f"\n🤖 Training new model (will save to {model_path})")
        ml_heft = RegressionHEFT(model_type="xgboost")
        
        # Generate training data
        dataset_gen = ExpertDatasetGenerator()
        training_data = dataset_gen.generate_dataset_v1(num_dags=10)
        
        # Train and save the model
        training_start = time.time()
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        ml_heft.train(training_data, save_model_path=model_path)
        training_time = time.time() - training_start
        print(f"✅ Model trained and saved in {training_time:.2f} seconds")

    # Step 3: Compare ML-HEFT vs Original HEFT
    print(f"\n{'='*50}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*50}")
    
    # Original HEFT
    heft_original = HEFTAlgorithm()
    heft_start = time.time()
    heft_result = heft_original.schedule(dag)
    heft_time = time.time() - heft_start
    
    # ML-HEFT (using pre-trained model)
    ml_start = time.time()
    ml_result = ml_heft.schedule(dag)
    ml_time = time.time() - ml_start
    
    # Results comparison
    print(f"Original HEFT makespan: {heft_result.makespan:.2f} (in {heft_time*1000:.2f}ms)")
    print(f"ML-HEFT makespan:      {ml_result.makespan:.2f} (in {ml_time*1000:.2f}ms)")
    
    difference = abs(ml_result.makespan - heft_result.makespan)
    percent_diff = (difference / heft_result.makespan) * 100
    
    print(f"Absolute difference:    {difference:.2f}")
    print(f"Percentage difference:  {percent_diff:.2f}%")
    
    if ml_result.makespan < heft_result.makespan:
        print("🎉 ML-HEFT performed better!")
    elif ml_result.makespan > heft_result.makespan:
        print("📈 Original HEFT performed better")
    else:
        print("🤝 Both algorithms achieved the same makespan")
    
    # Speed comparison
    speed_ratio = ml_time / heft_time if heft_time > 0 else float('inf')
    if speed_ratio > 1:
        print(f"⏱️ Original HEFT was {speed_ratio:.1f}x faster")
    else:
        print(f"⚡ ML-HEFT was {1/speed_ratio:.1f}x faster")
    
    # Model persistence info
    if training_time > 0:
        print(f"\n💾 Model saved to {model_path}")
        print("💡 Next time this script runs, it will load instantly!")
    else:
        print(f"\n⚡ Used pre-trained model - no training time needed!")
    
    return ml_result.makespan, heft_result.makespan

if __name__ == "__main__":
    main()