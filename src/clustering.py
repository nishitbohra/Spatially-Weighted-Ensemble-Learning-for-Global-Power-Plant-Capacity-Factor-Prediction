import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import hdbscan

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')


class ClusterAnalyzer:
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.clusterer = None
        
    def perform_clustering(self, df, min_cluster_size=50, min_samples=10):
        logger.info(f"Starting HDBSCAN clustering with min_cluster_size={min_cluster_size}, min_samples={min_samples}")
        
        features = df[['latitude', 'longitude', 'log_capacity_mw']].copy()
        features = features.replace([np.inf, -np.inf], np.nan).dropna()
        
        features_scaled = self.scaler.fit_transform(features)
        
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            core_dist_n_jobs=1
        )
        
        cluster_labels = self.clusterer.fit_predict(features_scaled)
        
        df['cluster'] = cluster_labels
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        logger.info(f"Clustering completed: {n_clusters} clusters found, {n_noise} noise points")
        
        mask = cluster_labels != -1
        if mask.sum() > 1 and len(set(cluster_labels[mask])) > 1:
            silhouette_avg = silhouette_score(features_scaled[mask], cluster_labels[mask])
            logger.info(f"Silhouette score (excluding noise): {silhouette_avg:.3f}")
        else:
            silhouette_avg = None
            logger.info("Cannot compute silhouette score: insufficient valid clusters")
        
        return df, silhouette_avg
    
    def get_cluster_summary(self, df):
        logger.info("Generating cluster summary")
        
        summary = df[df['cluster'] != -1].groupby('cluster').agg({
            'country': 'count',
            'primary_fuel': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Mixed',
            'capacity_mw': 'mean',
            'latitude': 'mean',
            'longitude': 'mean'
        }).round(3)
        
        summary.columns = ['plant_count', 'dominant_fuel', 'mean_capacity_mw', 'centroid_lat', 'centroid_lon']
        summary = summary.sort_values('plant_count', ascending=False)
        summary.reset_index(inplace=True)
        summary['cluster_id'] = summary['cluster']
        
        logger.info(f"Cluster summary generated for {len(summary)} clusters")
        
        return summary
    
    def plot_cluster_map(self, df, title, save_path):
        logger.info(f"Creating cluster map: {title}")
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        noise = df[df['cluster'] == -1]
        clusters = df[df['cluster'] != -1]
        
        if len(noise) > 0:
            ax.scatter(noise['longitude'], noise['latitude'], 
                      c='gray', s=5, alpha=0.3, label='Noise')
        
        if len(clusters) > 0:
            scatter = ax.scatter(clusters['longitude'], clusters['latitude'],
                               c=clusters['cluster'], s=10, alpha=0.6, 
                               cmap='tab20', edgecolors='none')
            plt.colorbar(scatter, ax=ax, label='Cluster ID')
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Cluster map saved to {save_path}")
        
        return fig
    
    def run_clustering_analysis(self, df_all, df_renewable, df_fossil, figures_dir):
        logger.info("Running clustering analysis for all plant types")
        
        figures_dir = Path(figures_dir)
        
        logger.info("Clustering all plants")
        df_all_clustered, silhouette_all = self.perform_clustering(df_all.copy())
        summary_all = self.get_cluster_summary(df_all_clustered)
        self.plot_cluster_map(df_all_clustered, 
                             'HDBSCAN Clustering: All Power Plants', 
                             figures_dir / 'cluster_map_all.png')
        
        logger.info("Clustering renewable plants")
        df_renewable_clustered, silhouette_renewable = self.perform_clustering(df_renewable.copy())
        summary_renewable = self.get_cluster_summary(df_renewable_clustered)
        self.plot_cluster_map(df_renewable_clustered, 
                             'HDBSCAN Clustering: Renewable Power Plants', 
                             figures_dir / 'cluster_map_renewable.png')
        
        logger.info("Clustering fossil fuel plants")
        df_fossil_clustered, silhouette_fossil = self.perform_clustering(df_fossil.copy())
        summary_fossil = self.get_cluster_summary(df_fossil_clustered)
        self.plot_cluster_map(df_fossil_clustered, 
                             'HDBSCAN Clustering: Fossil Fuel Power Plants', 
                             figures_dir / 'cluster_map_fossil.png')
        
        results = {
            'all': {'df': df_all_clustered, 'summary': summary_all, 'silhouette': silhouette_all},
            'renewable': {'df': df_renewable_clustered, 'summary': summary_renewable, 'silhouette': silhouette_renewable},
            'fossil': {'df': df_fossil_clustered, 'summary': summary_fossil, 'silhouette': silhouette_fossil}
        }
        
        logger.info("Clustering analysis completed for all plant types")
        
        return results
