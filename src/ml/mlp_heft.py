"""
MLP-based HEFT Algorithm: Uses a neural network to predict task priorities.

This module implements a HEFT variant that uses a Multi-Layer Perceptron (MLP)
regression model to predict upward ranks instead of calculating them heuristically.
"""

import os
import warnings
import joblib
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.core.workflow_dag import WorkflowDAG
from src.core.schedule_result import ScheduleResult
from src.algorithms.base import SchedulingAlgorithm
from src.ml.feature_extractor import TaskFeatureExtractor


class MLPHEFT(SchedulingAlgorithm):
    """
    HEFT algorithm using MLP (neural network) regression to predict task priorities.
    
    Instead of calculating upward ranks heuristically, this algorithm
    uses a trained MLP model to predict the upward rank of each task
    based on extracted features.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the MLP HEFT algorithm.
        
        Args:
            model_path: Path to saved model file.
        """
        super().__init__("MLP-HEFT")
        self.algorithm_name = "MLP-HEFT"
        self.model_type = "mlp"
        self.model = None
        self.scaler = None
        self.feature_extractor = TaskFeatureExtractor()
        self.feature_names = None
        self.is_trained = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def train(
        self, 
        dataset_data, 
        test_size: float = 0.2,
        tune_hyperparameters: bool = True,
        save_model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the MLP regression model on the expert dataset.
        
        Args:
            dataset_data: Either a path to a CSV dataset (str) or a DataFrame directly.
            test_size: Fraction of data to use for testing.
            tune_hyperparameters: Whether to perform hyperparameter tuning.
            save_model_path: Path to save the trained model.
            
        Returns:
            A dictionary with training results and metrics.
        """
        print(f"Training {self.model_type.upper()} regression model")
        
        if isinstance(dataset_data, str):
            df = pd.read_csv(dataset_data)
        elif isinstance(dataset_data, pd.DataFrame):
            df = dataset_data.copy()
        else:
            raise ValueError("dataset_data must be either a file path (str) or pandas DataFrame")
        print(f"Loaded dataset: {len(df)} samples, {len(df.columns)} columns")
        
        X, y, feature_names = self._prepare_training_data(df)
        self.feature_names = feature_names
        
        print(f"Features: {len(feature_names)}")
        print(f"Target: upward_rank (range: {y.min():.2f} - {y.max():.2f})")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if tune_hyperparameters:
            print("Tuning hyperparameters for MLP...")
            self.model, best_params = self._tune_hyperparameters(X_train_scaled, y_train)
        else:
            self.model = self._create_model()
            best_params = None
        
        print("Training final MLP model...")
        self.model.fit(X_train_scaled, y_train)
        
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        results = {
            'model_type': self.model_type,
            'best_params': best_params,
            'feature_count': len(feature_names),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_metrics': {
                'mse': mean_squared_error(y_train, train_pred),
                'mae': mean_absolute_error(y_train, train_pred),
                'r2': r2_score(y_train, train_pred)
            },
            'test_metrics': {
                'mse': mean_squared_error(y_test, test_pred),
                'mae': mean_absolute_error(y_test, test_pred),
                'r2': r2_score(y_test, test_pred)
            }
        }
        
        self.is_trained = True
        
        if save_model_path:
            self.save_model(save_model_path)
            print(f"Model saved to: {save_model_path}")
        
        self._print_training_results(results)
        
        return results
    
    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare feature matrix and target vector from the dataset."""
        non_feature_cols = [
            'dag_id', 'dag_type', 'task_id', 'expert_priority', 
            'expert_makespan', 'expert_algorithm', 'upward_rank'
        ]
        feature_cols = [col for col in df.columns if col not in non_feature_cols]
        X = df[feature_cols].values
        y = df['upward_rank'].values
        X = np.nan_to_num(X)
        return X, y, feature_cols

    def _create_model(self) -> MLPRegressor:
        """Create a default MLP regression model."""
        return MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42,
            early_stopping=True
        )

    def _tune_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Tuple[MLPRegressor, Dict]:
        """Perform hyperparameter tuning for the MLP model."""
        model = MLPRegressor(max_iter=500, random_state=42, early_stopping=True)
        param_grid = {
            'hidden_layer_sizes': [(50, 25), (100, 50), (100, 100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
        }
        
        grid_search = GridSearchCV(
            model, param_grid, cv=3,  # 3-fold CV is faster for MLP
            scoring='neg_mean_squared_error', 
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        return grid_search.best_estimator_, grid_search.best_params_

    def schedule(self, dag: WorkflowDAG) -> ScheduleResult:
        """
        Schedule tasks using MLP-predicted priorities.
        
        Args:
            dag: The workflow DAG to schedule.
            
        Returns:
            A ScheduleResult object with task assignments and timing.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained or loaded before scheduling.")
        
        task_priorities = self._calculate_regression_priorities(dag)
        
        processor_schedules = {f'p{i}': [] for i in range(dag.num_processors)}
        processor_end_times = {f'p{i}': 0 for i in range(dag.num_processors)}
        task_schedule = {}
        scheduled_tasks = set()
        
        while len(scheduled_tasks) < dag.num_tasks:
            ready_tasks = [
                task for task in dag.task_list 
                if task not in scheduled_tasks and 
                all(pred in scheduled_tasks for pred in dag.get_predecessors(task))
            ]
            
            if not ready_tasks:
                raise RuntimeError("Execution error: No ready tasks found, but not all tasks are scheduled. Check DAG for cycles.")
            
            highest_priority_task = max(ready_tasks, key=lambda task: task_priorities.get(task, 0))
            
            best_processor, start_time, end_time = self._find_best_processor(
                dag, highest_priority_task, processor_schedules, 
                processor_end_times, task_schedule
            )
            
            processor_schedules[best_processor].append({
                'task': highest_priority_task, 'start': start_time, 'finish': end_time
            })
            processor_end_times[best_processor] = end_time
            task_schedule[highest_priority_task] = {
                'processor': best_processor, 'start_time': start_time, 'finish_time': end_time,
                'priority': task_priorities.get(highest_priority_task, 0)
            }
            scheduled_tasks.add(highest_priority_task)
        
        makespan = max(processor_end_times.values()) if processor_end_times else 0
        
        return ScheduleResult(
            algorithm_name=self.algorithm_name,
            makespan=makespan,
            task_schedule=task_schedule,
            processor_schedules=processor_schedules,
            metadata={
                'model_type': self.model_type,
                'num_tasks': dag.num_tasks,
                'num_processors': dag.num_processors
            }
        )

    def _calculate_regression_priorities(self, dag: WorkflowDAG) -> Dict[str, float]:
        """Calculate task priorities using the trained MLP model."""
        task_priorities = {}
        
        feature_vectors = []
        for task in dag.task_list:
            features = self.feature_extractor.extract_task_features(dag, task)
            feature_vectors.append([features.get(name, 0.0) for name in self.feature_names])
            
        X = np.array(feature_vectors)
        if self.scaler:
            X = self.scaler.transform(X)
        
        predicted_ranks = self.model.predict(X)
        
        for i, task in enumerate(dag.task_list):
            task_priorities[task] = predicted_ranks[i]
            
        return task_priorities

    def _find_best_processor(
        self, 
        dag: WorkflowDAG, 
        task: str,
        processor_schedules: Dict,
        processor_end_times: Dict,
        task_schedule: Dict
    ) -> Tuple[str, float, float]:
        """Find the best processor (earliest finish time) for a task."""
        best_processor = None
        best_finish_time = float('inf')
        best_start_time = float('inf')

        for processor in dag.processor_list:
            # Earliest time the processor is free + data transfer time from predecessors
            est = self._calculate_est(dag, task, processor, task_schedule)
            
            # Find an insertion slot that accommodates the EST
            start_time, finish_time = self._find_earliest_slot(
                dag, task, processor, processor_schedules[processor], est
            )

            if finish_time < best_finish_time:
                best_finish_time = finish_time
                best_start_time = start_time
                best_processor = processor
        
        return best_processor, best_start_time, best_finish_time

    def _calculate_est(self, dag: WorkflowDAG, task: str, processor: str, task_schedule: Dict) -> float:
        """Calculate the Earliest Start Time for a task on a given processor."""
        max_pred_finish_time = 0.0
        for pred in dag.get_predecessors(task):
            pred_info = task_schedule[pred]
            pred_finish = pred_info['finish_time']
            
            # Add communication cost if predecessors are on different processors
            if pred_info['processor'] != processor:
                pred_finish += dag.get_communication_cost(pred, task)
            
            max_pred_finish_time = max(max_pred_finish_time, pred_finish)
            
        return max_pred_finish_time

    def _find_earliest_slot(
        self, 
        dag: WorkflowDAG, 
        task: str, 
        processor: str,
        schedule: List, 
        earliest_start: float
    ) -> Tuple[float, float]:
        """Find the earliest available slot for a task on a processor's schedule."""
        computation_cost = dag.get_computation_cost(task, int(processor[1:]))
        
        if not schedule:
            return earliest_start, earliest_start + computation_cost
        
        sorted_schedule = sorted(schedule, key=lambda x: x['start'])
        
        # Try to fit before the first task
        if earliest_start + computation_cost <= sorted_schedule[0]['start']:
            return earliest_start, earliest_start + computation_cost
        
        # Try to fit between existing tasks
        for i in range(len(sorted_schedule) - 1):
            slot_start = max(earliest_start, sorted_schedule[i]['finish'])
            if slot_start + computation_cost <= sorted_schedule[i+1]['start']:
                return slot_start, slot_start + computation_cost
        
        # Append at the end of the schedule
        last_task_finish = sorted_schedule[-1]['finish']
        start_time = max(earliest_start, last_task_finish)
        return start_time, start_time + computation_cost

    def save_model(self, filepath: str):
        """Save the trained model and scaler."""
        if not self.is_trained:
            raise RuntimeError("No trained model to save.")
        
        model_data = {
            'model': self.model, 'scaler': self.scaler, 'model_type': self.model_type,
            'feature_names': self.feature_names, 'algorithm_name': self.algorithm_name
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)

    def load_model(self, filepath: str):
        """Load a trained model and scaler."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data.get('model_type', 'mlp')
        self.feature_names = model_data['feature_names']
        self.algorithm_name = model_data.get('algorithm_name', 'MLP-HEFT')
        self.is_trained = True
        print(f"Successfully loaded {self.model_type} model from {filepath}")

    def _print_training_results(self, results: Dict):
        """Print training results in a formatted way."""
        print("\n" + "="*50)
        print("MLP TRAINING RESULTS")
        print("="*50)
        print(f"Model Type: {results['model_type']}")
        if results['best_params']:
            print(f"\nBest Parameters:")
            for param, value in results['best_params'].items():
                print(f"  {param}: {value}")
        print(f"\nTest Metrics:")
        print(f"  R-squared: {results['test_metrics']['r2']:.4f}")
        print(f"  MSE: {results['test_metrics']['mse']:.4f}")
        print(f"  MAE: {results['test_metrics']['mae']:.4f}")
        print("="*50)
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Feature importance is not directly available for MLP models.
        This method is provided for API consistency but will raise a warning.
        """
        warnings.warn("`get_feature_importance` is not supported for MLPRegressor. "
                      "Consider using permutation importance for model inspection.")
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
        
        # Return an empty dictionary as there are no importances to show
        return {}