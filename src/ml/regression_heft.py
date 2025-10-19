"""
Regression-based HEFT Algorithm: Uses ML to predict task priorities.

This module implements a HEFT variant that uses regression models to predict
upward ranks instead of calculating them heuristically.
"""

import os
import pickle
import joblib
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import xgboost as xgb

from src.core.workflow_dag import WorkflowDAG
from src.core.schedule_result import ScheduleResult
from src.algorithms.base import SchedulingAlgorithm
from .feature_extractor import TaskFeatureExtractor


class RegressionHEFT(SchedulingAlgorithm):
    """
    HEFT algorithm using regression to predict task priorities.
    
    Instead of calculating upward ranks heuristically, this algorithm
    uses a trained regression model to predict the upward rank of each task
    based on extracted features.
    """
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = "xgboost"):
        """
        Initialize the Regression HEFT algorithm.
        
        Args:
            model_path: Path to saved model file
            model_type: Type of regression model to use
        """
        super().__init__("Regression-HEFT")
        self.algorithm_name = "Regression-HEFT"
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_extractor = TaskFeatureExtractor()
        self.feature_names = None
        self.is_trained = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    def load_model(self, model_path: str):
        """Load a pre-trained model from file."""
        try:
            import joblib
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.model_type = model_data['model_type']
            self.feature_names = model_data['feature_names']
            self.is_trained = True  # This is the key fix!
            
            print(f"Successfully loaded {self.model_type} model from {model_path}")
            print(f"Model trained on {len(self.feature_names)} features")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")


    def train(
        self, 
        dataset_data, 
        test_size: float = 0.2,
        tune_hyperparameters: bool = True,
        save_model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the regression model on expert dataset.
        
        Args:
            dataset_data: Either a path to CSV dataset (str) or DataFrame directly
            test_size: Fraction of data to use for testing
            tune_hyperparameters: Whether to perform hyperparameter tuning
            save_model_path: Path to save the trained model
            
        Returns:
            Dictionary with training results and metrics
        """
        print(f"🚀 Training {self.model_type} regression model")
        
        # Load dataset - handle both DataFrame and file path
        if isinstance(dataset_data, str):
            df = pd.read_csv(dataset_data)
        elif isinstance(dataset_data, pd.DataFrame):
            df = dataset_data.copy()
        else:
            raise ValueError("dataset_data must be either a file path (str) or pandas DataFrame")
        print(f"📊 Loaded dataset: {len(df)} samples, {len(df.columns)} columns")
        
        # Prepare features and target
        X, y, feature_names = self._prepare_training_data(df)
        self.feature_names = feature_names
        
        print(f"📈 Features: {len(feature_names)}")
        print(f"🎯 Target: upward_rank (range: {y.min():.2f} - {y.max():.2f})")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=None
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train model
        if tune_hyperparameters:
            print("🔧 Tuning hyperparameters...")
            self.model, best_params = self._tune_hyperparameters(X_train_scaled, y_train)
        else:
            self.model = self._create_model()
            best_params = None
        
        # Train final model
        print("🎓 Training final model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
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
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=5, scoring='neg_mean_squared_error'
        )
        results['cv_mse'] = -cv_scores.mean()
        results['cv_mse_std'] = cv_scores.std()
        
        self.is_trained = True
        
        # Save model
        if save_model_path:
            self.save_model(save_model_path)
            print(f"💾 Model saved to: {save_model_path}")
        
        # Print results
        self._print_training_results(results)
        
        return results
    
    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare feature matrix and target vector from dataset."""
        # Remove non-feature columns
        non_feature_cols = [
            'dag_id', 'dag_type', 'task_id', 'expert_priority', 
            'expert_makespan', 'expert_algorithm', 'upward_rank'
        ]
        
        feature_cols = [col for col in df.columns if col not in non_feature_cols]
        
        X = df[feature_cols].values
        y = df['upward_rank'].values
        
        # Handle any missing values
        X = np.nan_to_num(X)
        
        return X, y, feature_cols
    
    def _create_model(self):
        """Create a regression model based on model_type."""
        if self.model_type == "xgboost":
            return xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == "linear":
            return LinearRegression()
        elif self.model_type == "ridge":
            return Ridge(alpha=1.0)
        elif self.model_type == "lasso":
            return Lasso(alpha=1.0)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _tune_hyperparameters(self, X, y):
        """Perform hyperparameter tuning."""
        if self.model_type == "xgboost":
            model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.2]
            }
        elif self.model_type == "random_forest":
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
        elif self.model_type == "gradient_boosting":
            model = GradientBoostingRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.2]
            }
        elif self.model_type == "ridge":
            model = Ridge()
            param_grid = {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
        elif self.model_type == "lasso":
            model = Lasso(max_iter=2000)
            param_grid = {
                'alpha': [0.01, 0.1, 1.0, 10.0]
            }
        else:
            # No tuning for linear regression
            return self._create_model(), {}
        
        grid_search = GridSearchCV(
            model, param_grid, cv=5, 
            scoring='neg_mean_squared_error', 
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def schedule(self, dag: WorkflowDAG) -> ScheduleResult:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before scheduling")
        
        # Calculate task priorities using regression
        task_priorities = self._calculate_regression_priorities(dag)
        
        # Initialize tracking variables
        processor_schedules = {f'p{i}': [] for i in range(dag.num_processors)}
        processor_end_times = {f'p{i}': 0 for i in range(dag.num_processors)}
        task_schedule = {}
        scheduled_tasks = set()
        
        # Continue until all tasks are scheduled
        while len(scheduled_tasks) < dag.num_tasks:
            # Get tasks that are ready to be scheduled (all predecessors scheduled)
            ready_tasks = [
                task for task in dag.task_list 
                if task not in scheduled_tasks and 
                all(pred in scheduled_tasks for pred in dag.get_predecessors(task))
            ]
            
            if not ready_tasks:
                # This should never happen in a valid DAG
                raise RuntimeError("No ready tasks but not all tasks scheduled")
            
            # Find highest priority ready task
            highest_priority_task = max(
                ready_tasks,
                key=lambda task: task_priorities.get(task, 0)
            )
            
            # Find best processor for this task
            best_processor, start_time, end_time = self._find_best_processor(
                dag, highest_priority_task, processor_schedules, 
                processor_end_times, task_schedule
            )
            
            # Add task to schedule
            processor_schedules[best_processor].append({
                'task': highest_priority_task,
                'start': start_time,
                'finish': end_time,
                'duration': end_time - start_time
            })
            
            processor_end_times[best_processor] = end_time
            
            task_schedule[highest_priority_task] = {
                'processor': best_processor,
                'start_time': start_time,
                'finish_time': end_time,
                'execution_time': end_time - start_time,
                'priority': task_priorities.get(highest_priority_task, 0)
            }
            
            # Mark task as scheduled
            scheduled_tasks.add(highest_priority_task)
        
        # Calculate makespan
        makespan = max(processor_end_times.values())
        
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
        """Calculate task priorities using the trained regression model."""
        task_priorities = {}
        
        for task in dag.task_list:
            # Extract features for this task
            features = self.feature_extractor.extract_task_features(dag, task)
            
            # Prepare feature vector
            feature_vector = []
            for feature_name in self.feature_names:
                feature_vector.append(features.get(feature_name, 0.0))
            
            # Predict priority using model
            X = np.array([feature_vector])
            if self.scaler:
                X = self.scaler.transform(X)
            
            predicted_rank = self.model.predict(X)[0]
            task_priorities[task] = predicted_rank
        
        return task_priorities
    
    def _find_best_processor(
        self, 
        dag: WorkflowDAG, 
        task: str,
        processor_schedules: Dict,
        processor_end_times: Dict,
        task_schedule: Dict
    ) -> Tuple[str, float, float]:
        """Find the best processor and timing for a task."""
        best_processor = None
        best_start_time = float('inf')
        best_end_time = float('inf')
        
        for processor in dag.processor_list:
            # Calculate earliest start time on this processor
            earliest_start = processor_end_times[processor]
            
            # Consider predecessor constraints
            for pred in dag.get_predecessors(task):
                if pred in task_schedule:
                    pred_info = task_schedule[pred]
                    if pred_info['processor'] == processor:
                        # Same processor: no communication cost
                        earliest_start = max(earliest_start, pred_info['finish_time'])
                    else:
                        # Different processor: add communication cost
                        comm_cost = dag.get_communication_cost(pred, task)
                        earliest_start = max(earliest_start, pred_info['finish_time'] + comm_cost)

            # Try to find an earlier slot (insertion-based scheduling)
            start_time, end_time = self._find_earliest_slot(
                dag, task, processor, processor_schedules[processor], earliest_start
            )
            
            # Check if this is better than current best
            if end_time < best_end_time:
                best_processor = processor
                best_start_time = start_time
                best_end_time = end_time
        
        return best_processor, best_start_time, best_end_time
    
    def _find_earliest_slot(
        self, 
        dag: WorkflowDAG, 
        task: str, 
        processor: str,
        schedule: List, 
        earliest_start: float
    ) -> Tuple[float, float]:
        """Find the earliest available slot for a task on a processor."""
        computation_cost = dag.get_computation_cost(task, int(processor[1:]))
        
        if not schedule:
            # Empty processor schedule
            return earliest_start, earliest_start + computation_cost
        
        # Sort existing tasks by start time
        sorted_schedule = sorted(schedule, key=lambda x: x['start'])
        
        # Try to fit before first task
        if earliest_start + computation_cost <= sorted_schedule[0]['start']:
            return earliest_start, earliest_start + computation_cost
        
        # Try to fit between existing tasks
        for i in range(len(sorted_schedule) - 1):
            current_task = sorted_schedule[i]
            next_task = sorted_schedule[i + 1]
            
            slot_start = max(earliest_start, current_task['finish'])
            slot_end = slot_start + computation_cost
            
            if slot_end <= next_task['start']:
                return slot_start, slot_end
        
        # Append to end
        last_task = sorted_schedule[-1]
        slot_start = max(earliest_start, last_task['finish'])
        return slot_start, slot_start + computation_cost
    
    def save_model(self, filepath: str):
        """Save the trained model and scaler."""
        if not self.is_trained:
            raise RuntimeError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'algorithm_name': self.algorithm_name
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
        self.model_type = model_data.get('model_type', 'unknown')
        self.feature_names = model_data['feature_names']
        self.algorithm_name = model_data.get('algorithm_name', 'Regression-HEFT')
        self.is_trained = True
    
    def _print_training_results(self, results: Dict):
        """Print training results in a formatted way."""
        print("\n" + "="*50)
        print("🎯 TRAINING RESULTS")
        print("="*50)
        
        print(f"Model Type: {results['model_type']}")
        print(f"Features: {results['feature_count']}")
        print(f"Train Samples: {results['train_samples']}")
        print(f"Test Samples: {results['test_samples']}")
        
        if results['best_params']:
            print(f"\nBest Parameters:")
            for param, value in results['best_params'].items():
                print(f"  {param}: {value}")
        
        print(f"\nTraining Metrics:")
        print(f"  MSE: {results['train_metrics']['mse']:.4f}")
        print(f"  MAE: {results['train_metrics']['mae']:.4f}")
        print(f"  R²:  {results['train_metrics']['r2']:.4f}")
        
        print(f"\nTest Metrics:")
        print(f"  MSE: {results['test_metrics']['mse']:.4f}")
        print(f"  MAE: {results['test_metrics']['mae']:.4f}")
        print(f"  R²:  {results['test_metrics']['r2']:.4f}")
        
        print(f"\nCross-Validation:")
        print(f"  MSE: {results['cv_mse']:.4f} ± {results['cv_mse_std']:.4f}")
        
        print("="*50)
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get feature importance from the trained model."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
        
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # Linear models
            importances = np.abs(self.model.coef_)
        else:
            raise ValueError("Model doesn't support feature importance")
        
        # Create importance dictionary
        importance_dict = dict(zip(self.feature_names, importances))
        
        # Sort by importance
        sorted_importance = sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return dict(sorted_importance[:top_n])


def main():
    """Example usage of the Regression HEFT algorithm."""
    # This would typically be run after generating a dataset
    print("Regression HEFT Algorithm")
    print("This module requires a trained dataset to function.")
    print("Run dataset_generator.py first to create training data.")


if __name__ == "__main__":
    main()