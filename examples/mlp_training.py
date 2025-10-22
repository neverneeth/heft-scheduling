from src.ml.mlp_heft import MLPHEFT
# Make sure you have a 'src' folder with the necessary structure for this import to work
# If not, you may need to adjust the import path.

def main():
    """
    This script demonstrates how to train the MLPHEFT model on a dataset,
    let it test itself, and save the resulting model.
    """
    # 1. Define the paths for your dataset and the output model file.
    # I am assuming your dataset is a CSV file. If not, please adjust.
    dataset_path = "/ml_final_output/datasets/dataset_v1.csv"
    model_export_path = '/ml_final_output/models/mlp_heft_model.joblib'

    print(f"Starting training process...")
    print(f"Dataset path: {dataset_path}")
    print(f"Model will be saved to: {model_export_path}")

    # 2. Initialize the MLP_HEFT algorithm.
    mlp_scheduler = MLPHEFT()

    # 3. Call the train method.
    # This single function call performs all the requested actions:
    #   - It loads the data from `dataset_path`.
    #   - It automatically splits the data into a training set and a testing set
    #     (the `test_size=0.2` default means 20% is used for testing).
    #   - It trains the MLP model on the training data.
    #   - It evaluates the trained model on the unseen testing data and prints the metrics.
    #   - It saves the final trained model and the scaler to the `save_model_path`.
    training_results = mlp_scheduler.train(
        dataset_data=dataset_path,
        tune_hyperparameters=True,  # Set to False for faster, less-optimized training
        save_model_path=model_export_path
    )

    print("\n--- Training and Evaluation Complete ---")
    print(f"The trained model has been successfully exported to: {model_export_path}")
    
    # You can now inspect the performance metrics from the returned dictionary.
    print("\nFinal Test Metrics:")
    test_metrics = training_results.get('test_metrics', {})
    r2 = test_metrics.get('r2', 'N/A')
    mse = test_metrics.get('mse', 'N/A')
    print(f"  - R-squared: {r2:.4f}")
    print(f"  - Mean Squared Error: {mse:.4f}")

    # To use the model later, you would do this:
    # print("\nExample of loading the model for future use:")
    # loaded_scheduler = MLPHEFT(model_path=model_export_path)
    # Now `loaded_scheduler` is ready to schedule a DAG.
    # e.g., loaded_scheduler.schedule(my_dag)

if __name__ == "__main__":
    main()