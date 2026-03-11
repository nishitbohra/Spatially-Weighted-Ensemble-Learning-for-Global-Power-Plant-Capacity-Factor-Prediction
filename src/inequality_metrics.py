import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')


class InequalityAnalyzer:
    
    def __init__(self):
        self.gini_coefficient = None
        self.theil_index = None
        
    def compute_gini_coefficient(self, values):
        logger.info("Computing Gini coefficient")
        
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        
        gini = (2 * np.sum((np.arange(1, n+1) * sorted_values))) / (n * cumsum[-1]) - (n + 1) / n
        
        self.gini_coefficient = gini
        logger.info(f"Gini coefficient: {gini:.4f}")
        
        return gini
    
    def plot_lorenz_curve(self, values, save_path):
        logger.info("Plotting Lorenz curve")
        
        sorted_values = np.sort(values)
        cumsum = np.cumsum(sorted_values)
        cumsum_normalized = cumsum / cumsum[-1]
        
        lorenz_x = np.linspace(0, 1, len(cumsum_normalized))
        lorenz_y = np.concatenate([[0], cumsum_normalized])
        lorenz_x_plot = np.concatenate([[0], lorenz_x])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Equality')
        ax.plot(lorenz_x_plot, lorenz_y, 'b-', linewidth=2, label='Lorenz Curve')
        ax.fill_between(lorenz_x_plot, lorenz_y, lorenz_x_plot, alpha=0.3, color='red', label='Inequality Area')
        
        ax.set_xlabel('Cumulative Share of Power Plants', fontsize=12)
        ax.set_ylabel('Cumulative Share of Capacity', fontsize=12)
        ax.set_title(f'Lorenz Curve for Power Plant Capacity\nGini Coefficient: {self.gini_coefficient:.4f}', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Lorenz curve saved to {save_path}")
        
        return fig
    
    def compute_theil_index(self, df, group_column='continent', value_column='capacity_mw'):
        logger.info(f"Computing Theil index with grouping by {group_column}")
        
        total_capacity = df[value_column].sum()
        total_count = len(df)
        
        overall_mean = total_capacity / total_count
        
        theil_within = 0
        theil_between = 0
        
        group_stats = []
        
        for group, group_df in df.groupby(group_column):
            group_capacity = group_df[value_column].sum()
            group_count = len(group_df)
            group_mean = group_capacity / group_count
            
            group_share_capacity = group_capacity / total_capacity
            group_share_count = group_count / total_count
            
            if group_share_count > 0 and group_share_capacity > 0:
                theil_between += group_share_capacity * np.log(group_share_capacity / group_share_count)
            
            for value in group_df[value_column]:
                if value > 0 and group_mean > 0:
                    theil_within += (value / total_capacity) * np.log(value / group_mean)
            
            group_stats.append({
                'group': group,
                'count': group_count,
                'total_capacity': group_capacity,
                'mean_capacity': group_mean,
                'share_of_total': group_share_capacity
            })
        
        theil_total = theil_within + theil_between
        
        self.theil_index = {
            'total': theil_total,
            'within': theil_within,
            'between': theil_between,
            'within_percentage': (theil_within / theil_total * 100) if theil_total > 0 else 0,
            'between_percentage': (theil_between / theil_total * 100) if theil_total > 0 else 0
        }
        
        logger.info(f"Theil Index (Total): {theil_total:.4f}")
        logger.info(f"Theil Within Groups: {theil_within:.4f} ({self.theil_index['within_percentage']:.1f}%)")
        logger.info(f"Theil Between Groups: {theil_between:.4f} ({self.theil_index['between_percentage']:.1f}%)")
        
        return self.theil_index, pd.DataFrame(group_stats)
    
    def compute_country_rankings(self, df):
        logger.info("Computing country rankings by capacity")
        
        country_rankings = df.groupby('country_long').agg({
            'capacity_mw': ['sum', 'mean', 'count'],
            'capacity_factor': 'mean',
            'is_renewable': lambda x: (x * df.loc[x.index, 'capacity_mw']).sum() / df.loc[x.index, 'capacity_mw'].sum()
        }).round(3)
        
        country_rankings.columns = ['total_capacity_mw', 'mean_capacity_mw', 'plant_count', 
                                    'mean_capacity_factor', 'renewable_share']
        
        country_rankings = country_rankings.sort_values('total_capacity_mw', ascending=False).head(50)
        country_rankings.reset_index(inplace=True)
        
        logger.info(f"Top 50 country rankings computed")
        
        return country_rankings
    
    def run_inequality_analysis(self, df, figures_dir, results_dir):
        logger.info("Starting inequality analysis")
        
        figures_dir = Path(figures_dir)
        results_dir = Path(results_dir)
        
        gini = self.compute_gini_coefficient(df['capacity_mw'].values)
        
        self.plot_lorenz_curve(df['capacity_mw'].values, figures_dir / 'lorenz_curve.png')
        
        theil_results, group_stats = self.compute_theil_index(df, 'continent', 'capacity_mw')
        
        theil_df = pd.DataFrame([{
            'Metric': 'Theil Index (Total)',
            'Value': theil_results['total']
        }, {
            'Metric': 'Theil Within Groups',
            'Value': theil_results['within']
        }, {
            'Metric': 'Theil Between Groups',
            'Value': theil_results['between']
        }, {
            'Metric': 'Within Group Percentage',
            'Value': theil_results['within_percentage']
        }, {
            'Metric': 'Between Group Percentage',
            'Value': theil_results['between_percentage']
        }])
        
        theil_path = results_dir / 'theil_decomposition.csv'
        theil_df.to_csv(theil_path, index=False)
        logger.info(f"Theil decomposition saved to {theil_path}")
        
        country_rankings = self.compute_country_rankings(df)
        rankings_path = results_dir / 'country_rankings.csv'
        country_rankings.to_csv(rankings_path, index=False)
        logger.info(f"Country rankings saved to {rankings_path}")
        
        summary = {
            'gini': gini,
            'theil': theil_results,
            'top_countries': country_rankings.head(10)
        }
        
        logger.info("Inequality analysis completed")
        
        return summary
