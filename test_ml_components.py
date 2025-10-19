"""
Quick test of the ML module components.

This script tests the basic functionality of our ML components
before running the full pipeline demo.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml import TaskFeatureExtractor, RegressionHEFT
from src.utils.dag_generator import DAGGenerator
from src.core.workflow_dag import WorkflowDAG


def test_feature_extraction():
    """Test the feature extraction functionality."""
    print("🧪 Testing Feature Extraction...")
    
    # Generate a simple test DAG
    edges, costs, comm = DAGGenerator.generate_random_dag(
        num_tasks=8,
        num_processors=3,
        edge_probability=0.4,
        random_seed=42
    )
    dag = WorkflowDAG(edges, costs, comm)
    
    # Extract features
    extractor = TaskFeatureExtractor()
    
    for i, task in enumerate(dag.task_list[:3]):  # Test first 3 tasks
        features = extractor.extract_task_features(dag, task)
        print(f"  Task {task}: {len(features)} features extracted")
        
        # Show some example features
        if i == 0:
            example_features = list(features.items())[:5]
            for feature, value in example_features:
                print(f"    {feature}: {value:.3f}")
    
    print("✅ Feature extraction working!")
    return True


def test_regression_heft_initialization():
    """Test that RegressionHEFT can be initialized."""
    print("\n🧪 Testing RegressionHEFT Initialization...")
    
    try:
        # Test different model types
        model_types = ["xgboost", "random_forest", "linear"]
        
        for model_type in model_types:
            reg_heft = RegressionHEFT(model_type=model_type)
            print(f"  ✅ {model_type} model initialized successfully")
            
        return True
    except Exception as e:
        print(f"  ❌ Error initializing models: {e}")
        return False


def test_dag_generation():
    """Test that we can generate various types of DAGs."""
    print("\n🧪 Testing DAG Generation...")
    
    dag_types = [
        ("Random", lambda: DAGGenerator.generate_random_dag(10, 3, 0.3)),
        ("Layered", lambda: DAGGenerator.generate_layered_dag(3, 3, 3)),
        ("Fork-Join", lambda: DAGGenerator.generate_fork_join_dag(2, 6, 2, 3)),
        ("Gaussian", lambda: DAGGenerator.generate_gaussian_elimination_dag(4, 3)),
        ("Epigenomics", lambda: DAGGenerator.generate_epigenomics_dag(2, 3))
    ]
    
    for name, generator in dag_types:
        try:
            result = generator()
            if isinstance(result, tuple):
                edges, costs, comm = result
                dag = WorkflowDAG(edges, costs, comm)
            else:
                dag = result
            
            print(f"  ✅ {name}: {dag.num_tasks} tasks, {len(dag.graph.edges())} edges")
            
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")
            return False
    
    return True


def main():
    """Run all quick tests."""
    print("🚀 Quick ML Module Test")
    print("=" * 40)
    
    tests = [
        test_dag_generation,
        test_feature_extraction, 
        test_regression_heft_initialization
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! ML module is ready for use.")
        print("\n🚀 Ready to run the full ML pipeline demo:")
        print("   python examples/ml_pipeline_demo.py")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == len(tests)


if __name__ == "__main__":
    main()