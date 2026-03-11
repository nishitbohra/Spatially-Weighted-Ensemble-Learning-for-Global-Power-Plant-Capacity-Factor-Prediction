import logging
import pandas as pd
import numpy as np
from pathlib import Path
from src.models import ModelPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 70)
    logger.info("TRAINING KAN MODEL")
    logger.info("=" * 70)
    
    # Load raw data and run feature engineering
    from src.preprocessing import Preprocessor
    from src.feature_engineering import FeatureEngineer
    
    raw_data_path = Path('data/raw/global_power_plant_database.csv')
    logger.info(f"Loading raw data from {raw_data_path}")
    
    preprocessor = Preprocessor()
    df_raw = preprocessor.load_data(raw_data_path)
    df_clean = preprocessor.clean_data(df_raw)
    df = preprocessor.filter_valid_generation(df_clean)
    logger.info(f"Data loaded and cleaned: {len(df)} rows")
    
    # Feature engineering
    logger.info("Running feature engineering...")
    feature_engineer = FeatureEngineer()
    df = feature_engineer.engineer_all_features(df)
    logger.info(f"Feature engineering complete: {df.shape[1]} columns")
    
    # Initialize model pipeline
    model_pipeline = ModelPipeline()
    
    # Create train/val/test splits
    logger.info("Creating train/val/test splits")
    X_train, X_val, X_test, y_train, y_val, y_test = model_pipeline.create_stratified_split(df)
    
    # Train KAN
    logger.info("\nStarting KAN training...")
    kan_model = model_pipeline.train_kan(X_train, y_train, X_val, y_val)
    
    if kan_model is not None:
        # Evaluate KAN
        logger.info("\nEvaluating KAN model on test set...")
        metrics = model_pipeline.evaluate_model(kan_model, X_test, y_test, 'KAN')
        
        logger.info("\n" + "=" * 70)
        logger.info("KAN MODEL RESULTS")
        logger.info("=" * 70)
        logger.info(f"MAE:  {metrics['MAE']:.4f}")
        logger.info(f"RMSE: {metrics['RMSE']:.4f}")
        logger.info(f"R²:   {metrics['R2']:.4f}")
        logger.info("=" * 70)
        
        # Save KAN model
        results_dir = Path('results/models')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save using torch (KAN models can't be pickled with joblib)
        import torch
        kan_path = results_dir / 'kan_model.pth'
        try:
            torch.save(kan_model.state_dict(), kan_path)
            logger.info(f"\n✓ KAN model state dict saved to {kan_path}")
        except Exception as e:
            logger.warning(f"Could not save KAN model: {e}")
            logger.info("KAN model trained but not saved due to serialization issues")
        
        # Update model comparison CSV
        comparison_path = Path('results/model_comparison.csv')
        if comparison_path.exists():
            df_results = pd.read_csv(comparison_path)
            
            # Add or update KAN results
            kan_row = pd.DataFrame({
                'Model': ['KAN'],
                'MAE': [metrics['MAE']],
                'RMSE': [metrics['RMSE']],
                'R2': [metrics['R2']]
            })
            
            # Remove existing KAN row if present
            df_results = df_results[df_results['Model'] != 'KAN']
            
            # Add new KAN row
            df_results = pd.concat([df_results, kan_row], ignore_index=True)
            
            # Sort by R2 descending
            df_results = df_results.sort_values('R2', ascending=False)
            
            # Save updated results
            df_results.to_csv(comparison_path, index=False)
            logger.info(f"✓ Model comparison updated at {comparison_path}")
            
            logger.info("\nUpdated Model Comparison:")
            print(df_results.to_string(index=False))
    else:
        logger.error("KAN training failed!")
    
    logger.info("\n" + "=" * 70)
    logger.info("KAN TRAINING COMPLETE")
    logger.info("=" * 70)

if __name__ == '__main__':
    main()
