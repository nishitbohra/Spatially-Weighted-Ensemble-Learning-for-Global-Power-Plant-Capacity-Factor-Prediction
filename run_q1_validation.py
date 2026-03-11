"""
Q1 Journal Validation Pipeline
Implements critical statistical rigor for journal submission

Run this AFTER your existing pipeline to generate:
1. Cross-validation results with confidence intervals
2. Statistical significance tests
3. Ablation study
4. Naive baseline comparisons
5. Publication-quality comparison tables

Usage:
    python run_q1_validation.py
"""

import logging
import time
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from scipy import stats

sys.path.append(str(Path(__file__).parent / 'src'))

from src.preprocessing import Preprocessor
from src.feature_engineering import FeatureEngineer
from src.models import ModelPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('q1_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def cross_validate_with_ci(model, X, y, model_name, n_folds=10):
    """
    Perform k-fold cross-validation with confidence intervals
    
    Returns:
        DataFrame with mean, std, and 95% CI for MAE, RMSE, R²
    """
    logger.info(f"Running {n_folds}-fold CV for {model_name}...")
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    mae_scores = -cross_val_score(model, X, y, cv=kfold, 
                                   scoring='neg_mean_absolute_error', n_jobs=-1)
    rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=kfold,
                                           scoring='neg_mean_squared_error', n_jobs=-1))
    r2_scores = cross_val_score(model, X, y, cv=kfold, 
                                scoring='r2', n_jobs=-1)
    
    results = {
        'Model': model_name,
        'MAE_mean': mae_scores.mean(),
        'MAE_std': mae_scores.std(),
        'MAE_CI_lower': np.percentile(mae_scores, 2.5),
        'MAE_CI_upper': np.percentile(mae_scores, 97.5),
        'RMSE_mean': rmse_scores.mean(),
        'RMSE_std': rmse_scores.std(),
        'R2_mean': r2_scores.mean(),
        'R2_std': r2_scores.std(),
        'R2_CI_lower': np.percentile(r2_scores, 2.5),
        'R2_CI_upper': np.percentile(r2_scores, 97.5)
    }
    
    logger.info(f"{model_name} - R² = {results['R2_mean']:.4f} ± {results['R2_std']:.4f} "
                f"[95% CI: {results['R2_CI_lower']:.4f}, {results['R2_CI_upper']:.4f}]")
    
    return results, r2_scores


def statistical_comparison(scores1, scores2, name1, name2):
    """
    Perform statistical significance test between two models
    
    Returns:
        Dictionary with t-test and Wilcoxon test results
    """
    logger.info(f"Statistical comparison: {name1} vs {name2}")
    
    # Paired t-test
    t_stat, t_pval = stats.ttest_rel(scores1, scores2)
    
    # Wilcoxon signed-rank test (non-parametric)
    w_stat, w_pval = stats.wilcoxon(scores1, scores2)
    
    # Effect size (Cohen's d)
    mean_diff = np.mean(scores1 - scores2)
    pooled_std = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    significant = t_pval < 0.05
    interpretation = 'significantly better' if significant and mean_diff > 0 else \
                    'significantly worse' if significant and mean_diff < 0 else \
                    'not significantly different'
    
    logger.info(f"Result: {name1} is {interpretation} than {name2}")
    logger.info(f"  t-test p-value: {t_pval:.4f}, Cohen's d: {cohens_d:.3f}")
    
    return {
        'Model_1': name1,
        'Model_2': name2,
        't_statistic': t_stat,
        't_pvalue': t_pval,
        'wilcoxon_pvalue': w_pval,
        'cohens_d': cohens_d,
        'significant': significant,
        'interpretation': interpretation
    }


def ablation_study(model, X, y, feature_names):
    """
    Perform ablation study by removing feature groups
    
    Returns:
        DataFrame with ablation results
    """
    logger.info("Running ablation study...")
    
    # Define feature groups
    feature_groups = {
        'Spatial': ['latitude', 'longitude'],
        'Capacity': ['capacity_mw', 'log_capacity_mw'],
        'Temporal': ['plant_age'],
        'Regional': ['regional_renewable_share'],
        'Fuel_Type': [col for col in feature_names if col.startswith('primary_fuel_')],
        'Continent': [col for col in feature_names if col.startswith('continent_')]
    }
    
    # Baseline (all features)
    baseline_r2 = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1).mean()
    logger.info(f"Baseline R² (all features): {baseline_r2:.4f}")
    
    ablation_results = []
    
    for group_name, group_features in feature_groups.items():
        # Find indices of features to remove
        remove_indices = [i for i, fname in enumerate(feature_names) 
                         if fname in group_features]
        
        if not remove_indices:
            logger.info(f"  {group_name}: No matching features found")
            continue
        
        # Create dataset without these features
        X_ablated = np.delete(X.values if hasattr(X, 'values') else X, 
                             remove_indices, axis=1)
        
        # Evaluate
        r2_ablated = cross_val_score(model, X_ablated, y, cv=5, 
                                     scoring='r2', n_jobs=-1).mean()
        
        r2_drop = baseline_r2 - r2_ablated
        importance_pct = (r2_drop / baseline_r2) * 100 if baseline_r2 > 0 else 0
        
        ablation_results.append({
            'Feature_Group': group_name,
            'N_Features': len(remove_indices),
            'Baseline_R2': baseline_r2,
            'Ablated_R2': r2_ablated,
            'R2_Drop': r2_drop,
            'Relative_Importance_%': importance_pct
        })
        
        logger.info(f"  {group_name}: R² drop = {r2_drop:.4f} ({importance_pct:.1f}% importance)")
    
    return pd.DataFrame(ablation_results).sort_values('R2_Drop', ascending=False)


def naive_baseline_comparison(y_train, y_test, X_train, X_test):
    """
    Compare against naive baselines
    
    Returns:
        DataFrame with baseline performance
    """
    logger.info("Computing naive baseline comparisons...")
    
    baselines = []
    
    # 1. Mean baseline
    mean_pred = np.full_like(y_test, y_train.mean())
    baselines.append({
        'Method': 'Mean Baseline',
        'MAE': mean_absolute_error(y_test, mean_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, mean_pred)),
        'R2': r2_score(y_test, mean_pred)
    })
    logger.info(f"  Mean Baseline: R² = {baselines[-1]['R2']:.4f}")
    
    # 2. Median baseline
    median_pred = np.full_like(y_test, y_train.median())
    baselines.append({
        'Method': 'Median Baseline',
        'MAE': mean_absolute_error(y_test, median_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, median_pred)),
        'R2': r2_score(y_test, median_pred)
    })
    logger.info(f"  Median Baseline: R² = {baselines[-1]['R2']:.4f}")
    
    # 3. Linear regression (simplest ML baseline)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    baselines.append({
        'Method': 'Linear Regression',
        'MAE': mean_absolute_error(y_test, lr_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred)),
        'R2': r2_score(y_test, lr_pred)
    })
    logger.info(f"  Linear Regression: R² = {baselines[-1]['R2']:.4f}")
    
    return pd.DataFrame(baselines)


def main():
    start_time = time.time()
    
    logger.info("="*70)
    logger.info("Q1 JOURNAL VALIDATION PIPELINE")
    logger.info("="*70)
    
    base_dir = Path(__file__).parent
    data_raw = base_dir / 'data' / 'raw' / 'global_power_plant_database.csv'
    results_dir = base_dir / 'results' / 'q1_validation'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    logger.info("\n[STEP 1] LOADING DATA")
    preprocessor = Preprocessor()
    df = preprocessor.load_data(data_raw)
    df = preprocessor.clean_data(df)
    df = preprocessor.filter_valid_generation(df)
    
    logger.info("\n[STEP 2] FEATURE ENGINEERING")
    engineer = FeatureEngineer()
    df = engineer.engineer_all_features(df)
    
    # Prepare features
    logger.info("\n[STEP 3] PREPARING FEATURES")
    model_pipeline = ModelPipeline()
    X, y = model_pipeline.prepare_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = model_pipeline.create_stratified_split(df)
    
    feature_names = X.columns.tolist() if hasattr(X, 'columns') else \
                    [f'feature_{i}' for i in range(X.shape[1])]
    
    logger.info(f"Dataset: {len(X)} samples, {len(feature_names)} features")
    
    # Initialize models
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb
    
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=15, 
                                             random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, 
                                    learning_rate=0.1, random_state=42, n_jobs=-1)
    }
    
    # ========================================================================
    # CRITICAL TASK 1: CROSS-VALIDATION WITH CONFIDENCE INTERVALS
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("[CRITICAL TASK 1] CROSS-VALIDATION WITH CONFIDENCE INTERVALS")
    logger.info("="*70)
    
    cv_results = []
    cv_scores = {}
    
    for name, model in models.items():
        result, scores = cross_validate_with_ci(model, X, y, name, n_folds=10)
        cv_results.append(result)
        cv_scores[name] = scores
    
    cv_df = pd.DataFrame(cv_results)
    cv_path = results_dir / 'cross_validation_results.csv'
    cv_df.to_csv(cv_path, index=False)
    logger.info(f"\n✓ Cross-validation results saved to {cv_path}")
    
    # ========================================================================
    # CRITICAL TASK 2: STATISTICAL SIGNIFICANCE TESTS
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("[CRITICAL TASK 2] STATISTICAL SIGNIFICANCE TESTS")
    logger.info("="*70)
    
    comparisons = []
    model_names = list(cv_scores.keys())
    
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            comp = statistical_comparison(
                cv_scores[model_names[i]],
                cv_scores[model_names[j]],
                model_names[i],
                model_names[j]
            )
            comparisons.append(comp)
    
    comp_df = pd.DataFrame(comparisons)
    comp_path = results_dir / 'statistical_comparisons.csv'
    comp_df.to_csv(comp_path, index=False)
    logger.info(f"\n✓ Statistical comparisons saved to {comp_path}")
    
    # ========================================================================
    # CRITICAL TASK 3: ABLATION STUDY
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("[CRITICAL TASK 3] ABLATION STUDY")
    logger.info("="*70)
    
    # Use Random Forest for ablation (best model)
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, 
                                     random_state=42, n_jobs=-1)
    
    ablation_df = ablation_study(rf_model, X, y, feature_names)
    ablation_path = results_dir / 'ablation_study.csv'
    ablation_df.to_csv(ablation_path, index=False)
    logger.info(f"\n✓ Ablation study saved to {ablation_path}")
    
    # ========================================================================
    # CRITICAL TASK 4: NAIVE BASELINE COMPARISONS
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("[CRITICAL TASK 4] NAIVE BASELINE COMPARISONS")
    logger.info("="*70)
    
    baseline_df = naive_baseline_comparison(y_train, y_test, X_train, X_test)
    baseline_path = results_dir / 'naive_baselines.csv'
    baseline_df.to_csv(baseline_path, index=False)
    logger.info(f"\n✓ Naive baselines saved to {baseline_path}")
    
    # ========================================================================
    # SUMMARY REPORT
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("Q1 VALIDATION SUMMARY")
    logger.info("="*70)
    
    # Best model from CV
    best_model_idx = cv_df['R2_mean'].idxmax()
    best_model_row = cv_df.iloc[best_model_idx]
    
    logger.info(f"\n✓ BEST MODEL: {best_model_row['Model']}")
    logger.info(f"  R² = {best_model_row['R2_mean']:.4f} ± {best_model_row['R2_std']:.4f}")
    logger.info(f"  95% CI: [{best_model_row['R2_CI_lower']:.4f}, {best_model_row['R2_CI_upper']:.4f}]")
    logger.info(f"  MAE = {best_model_row['MAE_mean']:.4f} ± {best_model_row['MAE_std']:.4f}")
    
    # Most important feature group
    top_feature_group = ablation_df.iloc[0]
    logger.info(f"\n✓ MOST IMPORTANT FEATURE GROUP: {top_feature_group['Feature_Group']}")
    logger.info(f"  Contributes {top_feature_group['Relative_Importance_%']:.1f}% to model performance")
    logger.info(f"  R² drops from {top_feature_group['Baseline_R2']:.4f} to "
                f"{top_feature_group['Ablated_R2']:.4f} without it")
    
    # Best baseline
    best_baseline = baseline_df.loc[baseline_df['R2'].idxmax()]
    improvement = ((best_model_row['R2_mean'] - best_baseline['R2']) / 
                   abs(best_baseline['R2'])) * 100
    logger.info(f"\n✓ IMPROVEMENT OVER BEST BASELINE ({best_baseline['Method']}):")
    logger.info(f"  Baseline R²: {best_baseline['R2']:.4f}")
    logger.info(f"  Our model R²: {best_model_row['R2_mean']:.4f}")
    logger.info(f"  Improvement: {improvement:.1f}%")
    
    total_time = time.time() - start_time
    logger.info(f"\n✓ Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    logger.info("\n" + "="*70)
    logger.info("FILES GENERATED:")
    logger.info("="*70)
    logger.info(f"1. {cv_path}")
    logger.info(f"2. {comp_path}")
    logger.info(f"3. {ablation_path}")
    logger.info(f"4. {baseline_path}")
    logger.info("\n✓ Q1 validation pipeline completed successfully!")
    logger.info("="*70)


if __name__ == '__main__':
    main()
