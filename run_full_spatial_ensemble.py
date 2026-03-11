"""
Run Full Spatial Ensemble on All Training Samples

This script trains the spatial ensemble on ALL 23,130 training samples
(no sampling/interpolation) and evaluates on the full test set.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

from src.preprocessing import Preprocessor
from src.feature_engineering import FeatureEngineer
from src.models import ModelPipeline
from src.spatial_ensemble import SpatiallyWeightedEnsemble

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*70)
    logger.info("FULL SPATIAL ENSEMBLE EVALUATION (ALL SAMPLES)")
    logger.info("="*70)
    logger.info("")
    
    # Load and prepare data
    logger.info("[STEP 1] Loading and preparing data...")
    base_path = Path(__file__).parent
    data_path = base_path / 'data' / 'raw' / 'global_power_plant_database.csv'
    
    preprocessor = Preprocessor()
    df = preprocessor.load_data(str(data_path))
    df = preprocessor.clean_data(df)
    df = preprocessor.filter_valid_generation(df)
    
    engineer = FeatureEngineer()
    df = engineer.engineer_all_features(df)
    
    logger.info(f"Final dataset: {len(df)} power plants")
    logger.info("")
    
    # Prepare features
    logger.info("[STEP 2] Preparing features...")
    pipeline = ModelPipeline()
    X, y = pipeline.prepare_features(df)
    logger.info("")
    
    # Split data
    logger.info("[STEP 3] Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.create_stratified_split(df)
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info("")
    
    # Extract coordinates
    coords_train = df.loc[X_train.index, ['latitude', 'longitude']].values
    coords_test = df.loc[X_test.index, ['latitude', 'longitude']].values
    
    logger.info(f"Coordinates extracted:")
    logger.info(f"  Train coords: {coords_train.shape}")
    logger.info(f"  Test coords: {coords_test.shape}")
    logger.info("")
    
    # Train spatial ensemble on FULL training set
    logger.info("[STEP 4] Training Spatially-Weighted Ensemble on ALL samples...")
    logger.info(f"  Training on {len(X_train)} samples (NO SAMPLING)")
    logger.info("="*70)
    
    model = SpatiallyWeightedEnsemble(
        base_models=None,  # Uses default: RF, GB, Ridge
        spatial_bandwidth='adaptive',
        n_neighbors=50,
        include_spatial_lag=True
    )
    
    import time
    start_time = time.time()
    model.fit(X_train.values, y_train.values, coords=coords_train)
    training_time = time.time() - start_time
    
    logger.info("="*70)
    logger.info(f"✓ Training completed in {training_time:.2f}s ({training_time/60:.2f} min)")
    logger.info("")
    
    # Get model contributions
    logger.info("[STEP 5] Analyzing model contributions...")
    try:
        # Sample a subset for contribution analysis to avoid memory issues
        sample_size = min(1000, len(X_train))
        sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train.values[sample_indices]
        coords_sample = coords_train[sample_indices]
        
        contributions = model.get_model_contributions(X_sample, coords_sample)
        logger.info("  Base model weights:")
        for name, weight in contributions.items():
            logger.info(f"    {name}: {weight:.4f} ({weight*100:.2f}%)")
    except Exception as e:
        logger.warning(f"  Could not compute model contributions: {e}")
        contributions = {}
    logger.info("")
    
    # Predict on test set
    logger.info("[STEP 6] Evaluating on test set...")
    start_time = time.time()
    y_pred = model.predict(X_test.values, coords=coords_test)
    prediction_time = time.time() - start_time
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    logger.info(f"  Prediction time: {prediction_time:.2f}s")
    logger.info("")
    logger.info("="*70)
    logger.info("FINAL RESULTS - FULL SPATIAL ENSEMBLE")
    logger.info("="*70)
    logger.info(f"  R² Score:  {r2:.4f}")
    logger.info(f"  MAE:       {mae:.4f}")
    logger.info(f"  RMSE:      {rmse:.4f}")
    logger.info("="*70)
    logger.info("")
    
    # Save results
    results_dir = base_path / 'results' / 'spatial_ensemble_full'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("[STEP 7] Saving results...")
    
    # Save metrics
    results_df = pd.DataFrame([{
        'Model': 'SpatialEnsemble_Full',
        'TrainingSamples': len(X_train),
        'TestSamples': len(X_test),
        'R2': r2,
        'MAE': mae,
        'RMSE': rmse,
        'TrainingTime_sec': training_time,
        'PredictionTime_sec': prediction_time,
        'RF_weight': contributions.get('rf', np.nan),
        'GB_weight': contributions.get('gb', np.nan),
        'Ridge_weight': contributions.get('ridge', np.nan)
    }])
    
    results_df.to_csv(results_dir / 'full_ensemble_results.csv', index=False)
    logger.info(f"  ✓ Results saved to {results_dir / 'full_ensemble_results.csv'}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Predictions vs Actual
    axes[0].scatter(y_test, y_pred, alpha=0.3, s=10)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Capacity Factor', fontsize=12)
    axes[0].set_ylabel('Predicted Capacity Factor', fontsize=12)
    axes[0].set_title(f'Spatial Ensemble: R² = {r2:.4f}', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Model contributions (only if available)
    if contributions:
        names = list(contributions.keys())
        weights = list(contributions.values())
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        
        axes[1].bar(names, weights, color=colors[:len(names)], alpha=0.7, edgecolor='black')
        axes[1].set_ylabel('Average Weight', fontsize=12)
        axes[1].set_title('Base Model Contributions', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        for i, (name, weight) in enumerate(zip(names, weights)):
            axes[1].text(i, weight + 0.02, f'{weight:.3f}', 
                        ha='center', va='bottom', fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'Contributions not available', 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Base Model Contributions', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'full_ensemble_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ Visualization saved to {results_dir / 'full_ensemble_evaluation.png'}")
    
    # Save model
    try:
        import joblib
        model_path = results_dir / 'spatial_ensemble_full.pkl'
        joblib.dump(model, model_path)
        logger.info(f"  ✓ Model saved to {model_path}")
    except Exception as e:
        logger.warning(f"  Could not save model: {e}")
    
    logger.info("")
    logger.info("="*70)
    logger.info("✓ FULL SPATIAL ENSEMBLE EVALUATION COMPLETE")
    logger.info("="*70)

if __name__ == '__main__':
    main()
