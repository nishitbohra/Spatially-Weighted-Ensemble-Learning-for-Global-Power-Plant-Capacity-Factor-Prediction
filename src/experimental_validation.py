"""
Rigorous experimental validation for Q1 journal submission.

Includes:
- K-fold cross-validation
- Statistical significance testing
- Ablation studies
- Baseline comparisons
- Confidence intervals
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')


class ExperimentalValidator:
    """
    Performs rigorous statistical validation for academic publication.
    """
    
    def __init__(self, n_folds=10, n_bootstrap=1000, alpha=0.05):
        """
        Args:
            n_folds: Number of folds for cross-validation
            n_bootstrap: Number of bootstrap samples for CI
            alpha: Significance level for tests
        """
        self.n_folds = n_folds
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        
    def cross_validate_models(self, models_dict, X, y):
        """
        Perform k-fold cross-validation for all models.
        
        Args:
            models_dict: Dictionary of {name: model}
            X: Feature matrix
            y: Target variable
            
        Returns:
            DataFrame with CV results and confidence intervals
        """
        logger.info(f"Running {self.n_folds}-fold cross-validation")
        
        cv_results = []
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        for model_name, model in models_dict.items():
            logger.info(f"Cross-validating {model_name}...")
            
            # MAE scores
            mae_scores = -cross_val_score(
                model, X, y, 
                cv=kfold, 
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            
            # RMSE scores
            rmse_scores = np.sqrt(-cross_val_score(
                model, X, y,
                cv=kfold,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            ))
            
            # R² scores
            r2_scores = cross_val_score(
                model, X, y,
                cv=kfold,
                scoring='r2',
                n_jobs=-1
            )
            
            cv_results.append({
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
            })
        
        results_df = pd.DataFrame(cv_results)
        logger.info("Cross-validation completed")
        
        return results_df
    
    def statistical_comparison(self, model1_scores, model2_scores, 
                               model1_name='Model 1', model2_name='Model 2'):
        """
        Perform statistical significance test between two models.
        
        Uses paired t-test and Wilcoxon signed-rank test.
        
        Returns:
            Dictionary with test results
        """
        logger.info(f"Statistical comparison: {model1_name} vs {model2_name}")
        
        # Paired t-test
        t_stat, t_pval = stats.ttest_rel(model1_scores, model2_scores)
        
        # Wilcoxon signed-rank test (non-parametric)
        w_stat, w_pval = stats.wilcoxon(model1_scores, model2_scores)
        
        # Effect size (Cohen's d)
        mean_diff = np.mean(model1_scores - model2_scores)
        pooled_std = np.sqrt((np.var(model1_scores) + np.var(model2_scores)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        results = {
            'model1': model1_name,
            'model2': model2_name,
            't_statistic': t_stat,
            't_pvalue': t_pval,
            'wilcoxon_statistic': w_stat,
            'wilcoxon_pvalue': w_pval,
            'cohens_d': cohens_d,
            'significant': t_pval < self.alpha,
            'interpretation': 'significantly different' if t_pval < self.alpha else 'not significantly different'
        }
        
        logger.info(f"Result: {results['interpretation']} (p={t_pval:.4f})")
        
        return results
    
    def ablation_study(self, model, X, y, feature_names):
        """
        Perform ablation study by removing feature groups.
        
        Measures importance of feature categories.
        
        Returns:
            DataFrame with ablation results
        """
        logger.info("Running ablation study")
        
        # Define feature groups
        feature_groups = {
            'Spatial': ['latitude', 'longitude'],
            'Capacity': ['capacity_mw', 'log_capacity_mw'],
            'Temporal': ['plant_age', 'commissioning_year'],
            'Regional': ['regional_renewable_share'],
            'Fuel_Type': [col for col in feature_names if col.startswith('primary_fuel_')],
            'Continent': [col for col in feature_names if col.startswith('continent_')]
        }
        
        # Baseline (all features)
        baseline_r2 = cross_val_score(model, X, y, cv=5, scoring='r2').mean()
        
        ablation_results = []
        
        for group_name, group_features in feature_groups.items():
            # Find indices of features to remove
            remove_indices = [i for i, fname in enumerate(feature_names) 
                            if fname in group_features]
            
            if not remove_indices:
                continue
            
            # Create dataset without these features
            X_ablated = np.delete(X, remove_indices, axis=1)
            
            # Evaluate
            r2_ablated = cross_val_score(model, X_ablated, y, cv=5, scoring='r2').mean()
            
            ablation_results.append({
                'Feature_Group': group_name,
                'Baseline_R2': baseline_r2,
                'Ablated_R2': r2_ablated,
                'R2_Drop': baseline_r2 - r2_ablated,
                'Importance_Percentage': ((baseline_r2 - r2_ablated) / baseline_r2) * 100
            })
            
            logger.info(f"{group_name}: R² drop = {baseline_r2 - r2_ablated:.4f}")
        
        return pd.DataFrame(ablation_results).sort_values('R2_Drop', ascending=False)
    
    def naive_baseline_comparison(self, y_train, y_test):
        """
        Compare against naive baselines.
        
        Baselines:
        1. Mean prediction
        2. Median prediction
        3. Stratified mean (by fuel type)
        
        Returns:
            DataFrame with baseline performance
        """
        logger.info("Computing naive baselines")
        
        baselines = []
        
        # Mean baseline
        mean_pred = np.full_like(y_test, y_train.mean())
        baselines.append({
            'Baseline': 'Mean',
            'MAE': mean_absolute_error(y_test, mean_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, mean_pred)),
            'R2': r2_score(y_test, mean_pred)
        })
        
        # Median baseline
        median_pred = np.full_like(y_test, y_train.median())
        baselines.append({
            'Baseline': 'Median',
            'MAE': mean_absolute_error(y_test, median_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, median_pred)),
            'R2': r2_score(y_test, median_pred)
        })
        
        return pd.DataFrame(baselines)
    
    def bootstrap_confidence_intervals(self, y_true, y_pred, metric='mae'):
        """
        Compute bootstrap confidence intervals for metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metric: 'mae', 'rmse', or 'r2'
            
        Returns:
            Dictionary with mean and CI
        """
        logger.info(f"Computing bootstrap CI for {metric}")
        
        n = len(y_true)
        bootstrap_scores = []
        
        for _ in range(self.n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n, n, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            # Compute metric
            if metric == 'mae':
                score = mean_absolute_error(y_true_boot, y_pred_boot)
            elif metric == 'rmse':
                score = np.sqrt(mean_squared_error(y_true_boot, y_pred_boot))
            elif metric == 'r2':
                score = r2_score(y_true_boot, y_pred_boot)
            
            bootstrap_scores.append(score)
        
        bootstrap_scores = np.array(bootstrap_scores)
        
        return {
            'metric': metric,
            'mean': bootstrap_scores.mean(),
            'std': bootstrap_scores.std(),
            'ci_lower': np.percentile(bootstrap_scores, 2.5),
            'ci_upper': np.percentile(bootstrap_scores, 97.5)
        }
    
    def plot_cv_comparison(self, cv_results_df, save_path):
        """
        Create publication-quality plot of CV results with error bars.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        models = cv_results_df['Model']
        
        # MAE plot
        axes[0].errorbar(
            models, 
            cv_results_df['MAE_mean'],
            yerr=cv_results_df['MAE_std'],
            fmt='o-',
            capsize=5,
            linewidth=2,
            markersize=8
        )
        axes[0].set_ylabel('MAE', fontsize=14)
        axes[0].set_title('Mean Absolute Error\n(10-Fold CV)', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # RMSE plot
        axes[1].errorbar(
            models,
            cv_results_df['RMSE_mean'],
            yerr=cv_results_df['RMSE_std'],
            fmt='s-',
            capsize=5,
            linewidth=2,
            markersize=8,
            color='orange'
        )
        axes[1].set_ylabel('RMSE', fontsize=14)
        axes[1].set_title('Root Mean Squared Error\n(10-Fold CV)', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
        
        # R² plot with CI
        axes[2].errorbar(
            models,
            cv_results_df['R2_mean'],
            yerr=[[cv_results_df['R2_mean'] - cv_results_df['R2_CI_lower']],
                  [cv_results_df['R2_CI_upper'] - cv_results_df['R2_mean']]],
            fmt='^-',
            capsize=5,
            linewidth=2,
            markersize=8,
            color='green'
        )
        axes[2].set_ylabel('R² Score', fontsize=14)
        axes[2].set_title('Coefficient of Determination\n(10-Fold CV, 95% CI)', fontsize=14)
        axes[2].grid(True, alpha=0.3)
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"CV comparison plot saved to {save_path}")
    
    def generate_statistical_report(self, results_dict, save_path):
        """
        Generate comprehensive statistical report for paper.
        
        Args:
            results_dict: Dictionary with all experimental results
            save_path: Path to save markdown report
        """
        report = []
        report.append("# Statistical Validation Report")
        report.append("\n## Cross-Validation Results\n")
        
        if 'cv_results' in results_dict:
            report.append(results_dict['cv_results'].to_markdown(index=False))
        
        report.append("\n## Pairwise Model Comparisons\n")
        
        if 'comparisons' in results_dict:
            for comp in results_dict['comparisons']:
                report.append(f"\n### {comp['model1']} vs {comp['model2']}")
                report.append(f"- t-test p-value: {comp['t_pvalue']:.4f}")
                report.append(f"- Cohen's d: {comp['cohens_d']:.3f}")
                report.append(f"- Result: {comp['interpretation']}")
        
        report.append("\n## Ablation Study\n")
        
        if 'ablation' in results_dict:
            report.append(results_dict['ablation'].to_markdown(index=False))
        
        with open(save_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Statistical report saved to {save_path}")
