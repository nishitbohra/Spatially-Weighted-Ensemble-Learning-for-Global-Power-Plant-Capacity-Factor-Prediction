import logging
import pandas as pd
import numpy as np
from pathlib import Path
import pycountry_convert as pc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    
    RENEWABLE_TYPES = ['Solar', 'Wind', 'Hydro', 'Biomass', 'Geothermal', 'Waste', 'Wave and Tidal']
    
    def __init__(self):
        self.df = None
        
    def compute_capacity_factor(self, df):
        logger.info("Computing capacity factor")
        
        df['capacity_factor'] = df['estimated_generation_gwh_2017'] / (df['capacity_mw'] * 8.76)
        df['capacity_factor'] = df['capacity_factor'].clip(0, 1.1)
        
        logger.info(f"Capacity factor computed. Mean: {df['capacity_factor'].mean():.3f}")
        logger.info(f"Capacity factor range: {df['capacity_factor'].min():.3f} - {df['capacity_factor'].max():.3f}")
        
        return df
    
    def add_renewable_classification(self, df):
        logger.info("Adding renewable classification")
        
        df['is_renewable'] = df['primary_fuel'].isin(self.RENEWABLE_TYPES).astype(int)
        
        df['energy_category'] = df['primary_fuel'].apply(
            lambda x: 'Renewable' if x in self.RENEWABLE_TYPES else 'Fossil'
        )
        
        renewable_count = df['is_renewable'].sum()
        logger.info(f"Renewable plants: {renewable_count} ({renewable_count/len(df)*100:.1f}%)")
        
        return df
    
    def compute_plant_age(self, df):
        logger.info("Computing plant age")
        
        df['age_missing'] = df['commissioning_year'].isna().astype(int)
        
        median_year_by_fuel = df.groupby('primary_fuel')['commissioning_year'].median()
        df['commissioning_year'] = df.apply(
            lambda row: median_year_by_fuel[row['primary_fuel']] 
            if pd.isna(row['commissioning_year']) 
            else row['commissioning_year'],
            axis=1
        )
        
        df['plant_age'] = 2019 - df['commissioning_year']
        
        logger.info(f"Plant age computed. Mean age: {df['plant_age'].mean():.1f} years")
        logger.info(f"Missing commissioning years imputed: {df['age_missing'].sum()}")
        
        return df
    
    def add_continent_mapping(self, df):
        logger.info("Adding continent mapping")
        
        def get_continent(country_name):
            try:
                country_code = pc.country_name_to_country_alpha2(country_name, cn_name_format="default")
                continent_code = pc.country_alpha2_to_continent_code(country_code)
                continent_name = pc.convert_continent_code_to_continent_name(continent_code)
                return continent_name
            except:
                return 'Unknown'
        
        df['continent'] = df['country_long'].apply(get_continent)
        
        logger.info(f"Continent distribution:\n{df['continent'].value_counts()}")
        
        return df
    
    def add_regional_renewable_share(self, df):
        logger.info("Computing regional renewable share")
        
        regional_stats = df.groupby('continent').agg({
            'capacity_mw': 'sum',
            'is_renewable': lambda x: (x * df.loc[x.index, 'capacity_mw']).sum()
        }).reset_index()
        
        regional_stats['regional_renewable_share'] = (
            regional_stats['is_renewable'] / regional_stats['capacity_mw']
        )
        
        df = df.merge(
            regional_stats[['continent', 'regional_renewable_share']], 
            on='continent', 
            how='left'
        )
        
        logger.info("Regional renewable share added")
        
        return df
    
    def add_log_capacity(self, df):
        logger.info("Adding log capacity feature")
        
        df['log_capacity_mw'] = np.log1p(df['capacity_mw'])
        
        logger.info(f"Log capacity range: {df['log_capacity_mw'].min():.2f} - {df['log_capacity_mw'].max():.2f}")
        
        return df
    
    def engineer_all_features(self, df):
        logger.info("Starting feature engineering pipeline")
        
        df = self.compute_capacity_factor(df)
        df = self.add_renewable_classification(df)
        df = self.compute_plant_age(df)
        df = self.add_continent_mapping(df)
        df = self.add_regional_renewable_share(df)
        df = self.add_log_capacity(df)
        
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['capacity_factor'])
        
        logger.info(f"Feature engineering completed. Final shape: {df.shape}")
        
        return df
