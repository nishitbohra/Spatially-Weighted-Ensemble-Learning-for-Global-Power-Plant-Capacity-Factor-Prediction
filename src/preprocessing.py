import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Preprocessor:
    
    def __init__(self):
        self.df = None
        
    def load_data(self, path):
        path = Path(path)
        logger.info(f"Loading data from {path}")
        self.df = pd.read_csv(path, low_memory=False)
        logger.info(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df
    
    def clean_data(self, df):
        logger.info(f"Starting data cleaning. Initial rows: {len(df)}")
        initial_rows = len(df)
        
        df = df.drop_duplicates()
        logger.info(f"Duplicates removed: {initial_rows - len(df)} rows")
        
        critical_columns = ['capacity_mw', 'latitude', 'longitude', 'primary_fuel']
        df = df.dropna(subset=critical_columns)
        logger.info(f"Rows after removing missing critical columns: {len(df)}")
        
        numeric_columns = ['capacity_mw', 'latitude', 'longitude', 'estimated_generation_gwh_2017']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"Data cleaning completed. Final rows: {len(df)}")
        logger.info(f"Total rows removed: {initial_rows - len(df)}")
        
        return df
    
    def filter_valid_generation(self, df):
        logger.info(f"Filtering valid generation data. Initial rows: {len(df)}")
        initial_rows = len(df)
        
        df = df[
            (df['estimated_generation_gwh_2017'] > 0) & 
            (df['capacity_mw'] > 0)
        ].copy()
        
        logger.info(f"Valid generation filter applied: {len(df)} rows remaining")
        logger.info(f"Rows filtered out: {initial_rows - len(df)}")
        
        logger.info(f"Capacity range: {df['capacity_mw'].min():.2f} - {df['capacity_mw'].max():.2f} MW")
        logger.info(f"Generation range: {df['estimated_generation_gwh_2017'].min():.2f} - {df['estimated_generation_gwh_2017'].max():.2f} GWh")
        
        return df
    
    def save_processed(self, df, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Processed data saved to {path}")
        logger.info(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
