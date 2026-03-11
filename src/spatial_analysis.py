import logging
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from libpysal.weights import KNN
from esda.moran import Moran

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')


class SpatialAnalyzer:
    
    def __init__(self):
        self.weights = None
        
    def compute_morans_i(self, df, variable):
        logger.info(f"Computing Moran's I for {variable}")
        
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
            crs='EPSG:4326'
        )
        
        logger.info("Building KNN spatial weights (k=8)")
        # Suppress the disconnected components warning - it's expected for isolated power plants
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='The weights matrix is not fully connected')
            self.weights = KNN.from_dataframe(gdf, k=8)
        
        # Log information about disconnected components
        n_components = self.weights.n_components
        if n_components > 1:
            logger.info(f"Spatial weights computed with {n_components} disconnected components (expected for isolated plants)")
        
        moran = Moran(gdf[variable], self.weights)
        
        logger.info(f"Moran's I: {moran.I:.4f}")
        logger.info(f"p-value: {moran.p_sim:.4f}")
        logger.info(f"z-score: {moran.z_sim:.4f}")
        
        return {
            'moran_i': moran.I,
            'p_value': moran.p_sim,
            'z_score': moran.z_sim,
            'moran_object': moran
        }
    
    def plot_morans_scatter(self, moran_result, variable, save_path):
        logger.info(f"Creating Moran scatter plot for {variable}")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        moran = moran_result['moran_object']
        
        x = moran.z
        
        try:
            w_matrix = moran.w.sparse.toarray() if hasattr(moran.w.sparse, 'toarray') else moran.w.full()[0]
            y = np.dot(w_matrix, moran.z.reshape(-1, 1)).flatten()
        except:
            sample_size = min(5000, len(x))
            indices = np.random.choice(len(x), sample_size, replace=False)
            x = x[indices]
            w_subset = moran.w.sparse[indices][:, indices]
            y = w_subset.dot(moran.z[indices].reshape(-1, 1)).flatten()
            logger.info(f"Sampled {sample_size} points for visualization due to memory constraints")
        
        ax.scatter(x, y, alpha=0.5, s=20)
        ax.axhline(0, color='red', linestyle='--', linewidth=1)
        ax.axvline(0, color='red', linestyle='--', linewidth=1)
        
        slope = moran.I
        x_line = np.array([x.min(), x.max()])
        y_line = slope * x_line
        ax.plot(x_line, y_line, 'b-', linewidth=2, label=f"Slope = {slope:.3f}")
        
        ax.set_xlabel(f'Standardized {variable}', fontsize=12)
        ax.set_ylabel(f'Spatial Lag of {variable}', fontsize=12)
        ax.set_title(f"Moran's I Scatter Plot\nI = {moran.I:.4f}, p = {moran.p_sim:.4f}", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Moran scatter plot saved to {save_path}")
        
        return fig
    
    def compute_regional_stats(self, df):
        logger.info("Computing regional statistics")
        
        regional_stats = df.groupby('continent').agg({
            'capacity_factor': 'mean',
            'capacity_mw': 'sum',
            'is_renewable': lambda x: (x * df.loc[x.index, 'capacity_mw']).sum() / df.loc[x.index, 'capacity_mw'].sum()
        }).round(3)
        
        regional_stats.columns = ['mean_capacity_factor', 'total_capacity_mw', 'renewable_share']
        regional_stats = regional_stats.sort_values('total_capacity_mw', ascending=False)
        
        logger.info(f"Regional statistics computed for {len(regional_stats)} continents")
        
        return regional_stats
    
    def compute_country_rankings(self, df):
        logger.info("Computing country rankings by installed capacity")
        
        country_rankings = df.groupby('country_long').agg({
            'capacity_mw': 'sum',
            'capacity_factor': 'mean',
            'is_renewable': lambda x: (x * df.loc[x.index, 'capacity_mw']).sum() / df.loc[x.index, 'capacity_mw'].sum(),
            'country': 'count'
        }).round(3)
        
        country_rankings.columns = ['total_capacity_mw', 'mean_capacity_factor', 'renewable_share', 'plant_count']
        country_rankings = country_rankings.sort_values('total_capacity_mw', ascending=False).head(20)
        
        logger.info(f"Top 20 countries by capacity computed")
        
        return country_rankings
