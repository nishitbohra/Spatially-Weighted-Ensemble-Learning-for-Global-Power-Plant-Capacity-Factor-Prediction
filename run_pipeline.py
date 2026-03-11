import logging
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent / 'src'))

from src.preprocessing import Preprocessor
from src.feature_engineering import FeatureEngineer
from src.spatial_analysis import SpatialAnalyzer
from src.clustering import ClusterAnalyzer
from src.models import ModelPipeline
from src.inequality_metrics import InequalityAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    start_time = time.time()
    
    logger.info("="*70)
    logger.info("GLOBAL POWER PLANT ML RESEARCH PIPELINE")
    logger.info("="*70)
    
    base_dir = Path(__file__).parent
    data_raw = base_dir / 'data' / 'raw' / 'global_power_plant_database.csv'
    data_processed = base_dir / 'data' / 'processed' / 'cleaned_gppd.csv'
    figures_dir = base_dir / 'figures'
    results_dir = base_dir / 'results'
    
    figures_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    step_start = time.time()
    logger.info("\n[STEP 1] DATA PREPROCESSING")
    preprocessor = Preprocessor()
    df = preprocessor.load_data(data_raw)
    df = preprocessor.clean_data(df)
    df = preprocessor.filter_valid_generation(df)
    preprocessor.save_processed(df, data_processed)
    logger.info(f"Step 1 completed in {time.time() - step_start:.2f} seconds")
    
    step_start = time.time()
    logger.info("\n[STEP 2] FEATURE ENGINEERING")
    engineer = FeatureEngineer()
    df = engineer.engineer_all_features(df)
    logger.info(f"Step 2 completed in {time.time() - step_start:.2f} seconds")
    
    step_start = time.time()
    logger.info("\n[STEP 3] SPATIAL ANALYSIS")
    spatial_analyzer = SpatialAnalyzer()
    
    moran_result = spatial_analyzer.compute_morans_i(df, 'capacity_factor')
    spatial_analyzer.plot_morans_scatter(
        moran_result, 
        'capacity_factor', 
        figures_dir / 'morans_i_scatter.png'
    )
    
    regional_stats = spatial_analyzer.compute_regional_stats(df)
    regional_stats.to_csv(results_dir / 'regional_statistics.csv')
    logger.info(f"Regional statistics saved")
    
    country_rankings = spatial_analyzer.compute_country_rankings(df)
    country_rankings.to_csv(results_dir / 'country_capacity_rankings.csv')
    logger.info(f"Country rankings saved")
    
    logger.info(f"Step 3 completed in {time.time() - step_start:.2f} seconds")
    
    step_start = time.time()
    logger.info("\n[STEP 4] CLUSTERING ANALYSIS")
    cluster_analyzer = ClusterAnalyzer()
    
    df_renewable = df[df['is_renewable'] == 1].copy()
    df_fossil = df[df['is_renewable'] == 0].copy()
    
    clustering_results = cluster_analyzer.run_clustering_analysis(
        df, df_renewable, df_fossil, figures_dir
    )
    
    clustering_results['all']['summary'].to_csv(results_dir / 'cluster_summary_all.csv', index=False)
    clustering_results['renewable']['summary'].to_csv(results_dir / 'cluster_summary_renewable.csv', index=False)
    clustering_results['fossil']['summary'].to_csv(results_dir / 'cluster_summary_fossil.csv', index=False)
    
    logger.info(f"Step 4 completed in {time.time() - step_start:.2f} seconds")
    
    step_start = time.time()
    logger.info("\n[STEP 5] MACHINE LEARNING MODELS")
    model_pipeline = ModelPipeline()
    
    results_df, X_test, y_test = model_pipeline.run_model_comparison(df, results_dir)
    logger.info(f"\nModel Comparison Results:\n{results_df}")
    
    model_pipeline.generate_shap_analysis(X_test, figures_dir)
    
    logger.info(f"Step 5 completed in {time.time() - step_start:.2f} seconds")
    
    step_start = time.time()
    logger.info("\n[STEP 6] INEQUALITY ANALYSIS")
    inequality_analyzer = InequalityAnalyzer()
    
    inequality_summary = inequality_analyzer.run_inequality_analysis(df, figures_dir, results_dir)
    
    logger.info(f"Step 6 completed in {time.time() - step_start:.2f} seconds")
    
    total_time = time.time() - start_time
    
    logger.info("\n" + "="*70)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("="*70)
    logger.info(f"Plants analyzed: {len(df):,}")
    logger.info(f"Features engineered: {len([col for col in df.columns])}")
    logger.info(f"Models trained: 4 (RandomForest, XGBoost, TabNet, KAN)")
    logger.info(f"Figures generated: {len(list(figures_dir.glob('*.png')))}")
    logger.info(f"Results saved: {len(list(results_dir.glob('*.csv')))}")
    logger.info(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info("="*70)
    logger.info("Pipeline completed successfully!")
    logger.info("="*70)


if __name__ == '__main__':
    main()
