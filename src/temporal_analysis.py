"""
Temporal Analysis: Multi-Year Capacity Factor Prediction

This module implements temporal analysis leveraging all available years
of generation data (2013-2019) to:
1. Extract temporal trends and patterns
2. Engineer temporal features (growth rates, volatility, trends)
3. Predict capacity factors with temporal context
4. Analyze year-over-year variability

Key Innovation:
- Uses longitudinal data to capture temporal dynamics
- Engineers features from historical trajectories
- Accounts for weather variability and operational changes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class TemporalFeatureEngineer:
    """
    Extract temporal features from multi-year generation data
    """
    
    def __init__(self, years=None):
        """
        Parameters
        ----------
        years : list, optional
            Years to consider. Default is 2013-2017 (available in GPPD).
        """
        if years is None:
            self.years = list(range(2013, 2018))  # 2013-2017 (5 years available)
        else:
            self.years = years
            
        self.generation_columns = [
            f'estimated_generation_gwh_{year}' for year in self.years
        ]
        
    def compute_temporal_features(self, df):
        """
        Compute temporal features from multi-year data
        
        Parameters
        ----------
        df : DataFrame
            Must contain estimated_generation_gwh_YYYY columns and capacity_mw
            
        Returns
        -------
        df : DataFrame
            Original DataFrame with added temporal features
        """
        logger.info("Computing temporal features from multi-year data...")
        
        # Capacity factors for each year
        cf_cols = []
        for year in self.years:
            gen_col = f'estimated_generation_gwh_{year}'
            cf_col = f'cf_{year}'
            
            if gen_col in df.columns:
                # Compute annual capacity factor
                df[cf_col] = df[gen_col] / (df['capacity_mw'] * 8.76)
                # Clip to [0, 1.1] to handle minor data issues
                df[cf_col] = df[cf_col].clip(0, 1.1)
                cf_cols.append(cf_col)
        
        logger.info(f"Computed capacity factors for {len(cf_cols)} years")
        
        # Drop rows with insufficient temporal data
        initial_rows = len(df)
        df = df.dropna(subset=cf_cols, thresh=len(cf_cols)//2)  # At least 50% of years
        logger.info(f"Retained {len(df)}/{initial_rows} plants with sufficient temporal data")
        
        # Temporal statistics
        cf_array = df[cf_cols].values
        
        # Mean capacity factor
        df['cf_mean'] = np.nanmean(cf_array, axis=1)
        
        # Temporal variability
        df['cf_std'] = np.nanstd(cf_array, axis=1)
        df['cf_cv'] = df['cf_std'] / (df['cf_mean'] + 1e-10)  # Coefficient of variation
        
        # Min and max
        df['cf_min'] = np.nanmin(cf_array, axis=1)
        df['cf_max'] = np.nanmax(cf_array, axis=1)
        df['cf_range'] = df['cf_max'] - df['cf_min']
        
        # Trend (linear regression over time)
        logger.info("Computing temporal trends...")
        trends = []
        r_squared = []
        
        for idx, row in df.iterrows():
            values = row[cf_cols].values  # Convert to numpy array
            valid_mask = pd.notna(values)  # Use pandas notna for safety
            
            if valid_mask.sum() >= 3:  # Need at least 3 points for trend
                years_valid = np.array(self.years)[valid_mask]
                values_valid = values[valid_mask]
                
                try:
                    # Linear regression
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        years_valid, values_valid
                    )
                    trends.append(slope)
                    r_squared.append(r_value ** 2)
                except (ValueError, AttributeError) as e:
                    # Handle edge cases
                    trends.append(0)
                    r_squared.append(0)
            else:
                trends.append(0)
                r_squared.append(0)
        
        df['cf_trend'] = trends
        df['cf_trend_r2'] = r_squared
        
        # Identify stable vs. volatile plants
        df['is_stable'] = (df['cf_cv'] < 0.1).astype(int)
        df['is_volatile'] = (df['cf_cv'] > 0.3).astype(int)
        
        # Year-over-year changes
        if len(cf_cols) >= 2:
            recent_change = df[cf_cols[-1]] - df[cf_cols[-2]]
            df['cf_recent_change'] = recent_change
        
        logger.info(f"Temporal features added: {len([c for c in df.columns if 'cf_' in c])}")
        
        # Log summary statistics
        logger.info(f"  Mean CF: {df['cf_mean'].mean():.3f} ± {df['cf_mean'].std():.3f}")
        logger.info(f"  CF Volatility (CV): {df['cf_cv'].mean():.3f}")
        logger.info(f"  Stable plants: {df['is_stable'].sum()} ({df['is_stable'].mean()*100:.1f}%)")
        logger.info(f"  Volatile plants: {df['is_volatile'].sum()} ({df['is_volatile'].mean()*100:.1f}%)")
        
        return df
    
    def get_temporal_feature_names(self):
        """
        Get list of temporal feature names
        """
        return [
            'cf_mean', 'cf_std', 'cf_cv', 'cf_min', 'cf_max', 'cf_range',
            'cf_trend', 'cf_trend_r2', 'is_stable', 'is_volatile', 'cf_recent_change'
        ]


def temporal_analysis_summary(df, output_dir=None):
    """
    Generate comprehensive temporal analysis summary
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with temporal features
    output_dir : Path, optional
        Directory to save output files
    """
    logger.info("\n" + "="*70)
    logger.info("TEMPORAL ANALYSIS SUMMARY")
    logger.info("="*70)
    
    # Year coverage statistics
    years = list(range(2013, 2018))  # 2013-2017 (available years)
    gen_cols = [f'estimated_generation_gwh_{year}' for year in years]
    
    coverage = {}
    for year, col in zip(years, gen_cols):
        if col in df.columns:
            n_valid = df[col].notna().sum()
            coverage[year] = {
                'n_plants': n_valid,
                'pct': n_valid / len(df) * 100,
                'total_capacity_mw': df[df[col].notna()]['capacity_mw'].sum()
            }
    
    logger.info("\nYear-by-year coverage:")
    for year, stats in coverage.items():
        logger.info(f"  {year}: {stats['n_plants']:,} plants ({stats['pct']:.1f}%), "
                   f"{stats['total_capacity_mw']:,.0f} MW")
    
    # Temporal patterns by fuel type
    logger.info("\nTemporal patterns by fuel type:")
    for fuel in df['primary_fuel'].value_counts().head(10).index:
        subset = df[df['primary_fuel'] == fuel]
        if len(subset) > 10:
            logger.info(f"\n  {fuel}:")
            logger.info(f"    Mean CF: {subset['cf_mean'].mean():.3f}")
            logger.info(f"    Volatility (CV): {subset['cf_cv'].mean():.3f}")
            logger.info(f"    Trend (slope): {subset['cf_trend'].mean():.4f}/year")
            logger.info(f"    Stable plants: {subset['is_stable'].mean()*100:.1f}%")
    
    # Regional temporal patterns
    if 'continent' in df.columns:
        logger.info("\nTemporal patterns by continent:")
        for continent in df['continent'].unique():
            subset = df[df['continent'] == continent]
            if len(subset) > 10:
                logger.info(f"\n  {continent}:")
                logger.info(f"    Mean CF: {subset['cf_mean'].mean():.3f}")
                logger.info(f"    Volatility: {subset['cf_cv'].mean():.3f}")
                logger.info(f"    Trend: {subset['cf_trend'].mean():.4f}/year")
    
    # Create visualizations if output directory provided
    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 1. Temporal coverage heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        coverage_data = []
        for year in years:
            col = f'estimated_generation_gwh_{year}'
            if col in df.columns:
                by_fuel = df.groupby('primary_fuel')[col].apply(lambda x: x.notna().sum())
                coverage_data.append(by_fuel)
        
        if coverage_data:
            coverage_df = pd.concat(coverage_data, axis=1, keys=years).T
            coverage_df = coverage_df[coverage_df.sum(axis=0).sort_values(ascending=False).head(15).index]
            
            sns.heatmap(coverage_df, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax)
            ax.set_title('Data Coverage by Fuel Type and Year', fontweight='bold')
            ax.set_xlabel('Fuel Type')
            ax.set_ylabel('Year')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'temporal_coverage.png', dpi=300, bbox_inches='tight')
            logger.info(f"\n✓ Saved: temporal_coverage.png")
            plt.close()
        
        # 2. Capacity factor trends by fuel
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        top_fuels = df['primary_fuel'].value_counts().head(8).index
        
        for idx, fuel in enumerate(top_fuels):
            if idx >= 4:
                break
            ax = axes[idx]
            
            subset = df[df['primary_fuel'] == fuel]
            cf_cols = [f'cf_{year}' for year in years if f'cf_{year}' in subset.columns]
            
            if cf_cols:
                yearly_means = subset[cf_cols].mean()
                yearly_stds = subset[cf_cols].std()
                
                x = [int(col.split('_')[1]) for col in cf_cols]
                
                ax.plot(x, yearly_means, 'o-', linewidth=2, markersize=8, label='Mean')
                ax.fill_between(x,
                               yearly_means - yearly_stds,
                               yearly_means + yearly_stds,
                               alpha=0.3, label='± 1 std')
                
                ax.set_title(f'{fuel}', fontweight='bold')
                ax.set_xlabel('Year')
                ax.set_ylabel('Capacity Factor')
                ax.grid(alpha=0.3)
                ax.legend()
                ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'cf_temporal_trends_by_fuel.png', dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved: cf_temporal_trends_by_fuel.png")
        plt.close()
        
        # 3. Volatility distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # CV distribution
        ax = axes[0]
        ax.hist(df['cf_cv'].clip(0, 1), bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(df['cf_cv'].median(), color='red', linestyle='--', 
                  label=f'Median: {df["cf_cv"].median():.3f}')
        ax.set_xlabel('Coefficient of Variation', fontsize=12)
        ax.set_ylabel('Number of Plants', fontsize=12)
        ax.set_title('Temporal Volatility Distribution', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Trend distribution
        ax = axes[1]
        ax.hist(df['cf_trend'].clip(-0.1, 0.1), bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(df['cf_trend'].median(), color='red', linestyle='--',
                  label=f'Median: {df["cf_trend"].median():.4f}')
        ax.set_xlabel('CF Trend (change/year)', fontsize=12)
        ax.set_ylabel('Number of Plants', fontsize=12)
        ax.set_title('Capacity Factor Trends (2013-2019)', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'temporal_volatility_trends.png', dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved: temporal_volatility_trends.png")
        plt.close()
        
        # 4. Stable vs Volatile comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # By fuel type
        ax = axes[0]
        stability_by_fuel = df.groupby('primary_fuel').agg({
            'is_stable': 'mean',
            'is_volatile': 'mean',
            'capacity_mw': 'count'
        }).rename(columns={'capacity_mw': 'count'})
        
        stability_by_fuel = stability_by_fuel[stability_by_fuel['count'] >= 50].sort_values('is_stable')
        
        x = np.arange(len(stability_by_fuel))
        width = 0.35
        
        ax.barh(x - width/2, stability_by_fuel['is_stable'], width, label='Stable', alpha=0.8)
        ax.barh(x + width/2, stability_by_fuel['is_volatile'], width, label='Volatile', alpha=0.8)
        
        ax.set_yticks(x)
        ax.set_yticklabels(stability_by_fuel.index)
        ax.set_xlabel('Proportion', fontsize=12)
        ax.set_title('Stability by Fuel Type', fontweight='bold')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        # By renewable status
        ax = axes[1]
        if 'is_renewable' in df.columns:
            stability_renewable = df.groupby('is_renewable')[['is_stable', 'is_volatile']].mean()
            
            x = np.arange(2)
            width = 0.35
            
            ax.bar(x - width/2, stability_renewable['is_stable'], width, 
                  label='Stable', alpha=0.8)
            ax.bar(x + width/2, stability_renewable['is_volatile'], width,
                  label='Volatile', alpha=0.8)
            
            ax.set_xticks(x)
            ax.set_xticklabels(['Fossil', 'Renewable'])
            ax.set_ylabel('Proportion', fontsize=12)
            ax.set_title('Stability: Renewable vs Fossil', fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'stability_comparison.png', dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved: stability_comparison.png")
        plt.close()
    
    logger.info("\n" + "="*70)
    logger.info("✓ Temporal analysis completed")
    logger.info("="*70)


if __name__ == "__main__":
    # Test with sample data
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    n_plants = 100
    
    df = pd.DataFrame({
        'capacity_mw': np.random.uniform(10, 1000, n_plants),
        'primary_fuel': np.random.choice(['Solar', 'Wind', 'Gas', 'Coal'], n_plants),
        'continent': np.random.choice(['Asia', 'Europe', 'North America'], n_plants)
    })
    
    # Add generation data for multiple years
    for year in range(2013, 2020):
        base_cf = np.random.uniform(0.2, 0.8, n_plants)
        noise = np.random.normal(0, 0.05, n_plants)
        cf = np.clip(base_cf + noise, 0, 1)
        df[f'estimated_generation_gwh_{year}'] = cf * df['capacity_mw'] * 8.76
    
    # Test temporal feature engineering
    engineer = TemporalFeatureEngineer()
    df = engineer.compute_temporal_features(df)
    
    print("\nTemporal features created:")
    for col in engineer.get_temporal_feature_names():
        if col in df.columns:
            print(f"  {col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}")
