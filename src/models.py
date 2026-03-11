import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
from src.spatial_ensemble import SpatiallyWeightedEnsemble
try:
    from kan import KAN
except ImportError:
    try:
        from pykan import KAN
    except ImportError:
        KAN = None
import shap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')

np.random.seed(42)
torch.manual_seed(42)


class ModelPipeline:
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_features(self, df):
        logger.info("Preparing features for modeling")
        
        numeric_features = ['capacity_mw', 'latitude', 'longitude', 'plant_age', 
                           'log_capacity_mw', 'regional_renewable_share']
        
        categorical_features = ['primary_fuel', 'continent']
        
        df_model = df[numeric_features + categorical_features + ['capacity_factor']].copy()
        df_model = df_model.dropna()
        
        df_encoded = pd.get_dummies(df_model, columns=categorical_features, drop_first=False)
        
        X = df_encoded.drop('capacity_factor', axis=1)
        y = df_encoded['capacity_factor']
        
        self.feature_names = X.columns.tolist()
        
        logger.info(f"Features prepared: {X.shape[1]} features, {len(y)} samples")
        
        return X, y
    
    def create_stratified_split(self, df):
        logger.info("Creating stratified train/val/test split")
        
        X, y = self.prepare_features(df)
        
        # Try stratification by continent only, but check if all continents have at least 2 samples
        df_temp = df.loc[X.index].copy()
        continent_counts = df_temp['continent'].value_counts()
        
        # If any continent has fewer than 2 samples, don't use stratification
        if (continent_counts < 2).any():
            logger.info("Some strata too small, using simple random split")
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.15, random_state=42, stratify=None
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.1765, random_state=42, stratify=None
            )
        else:
            logger.info("Using stratified split by continent")
            stratify_key = df_temp['continent']
            
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.15, random_state=42, stratify=stratify_key
            )
            
            stratify_key_temp = stratify_key.loc[X_temp.index]
            # Check again for the temp split
            if (stratify_key_temp.value_counts() < 2).any():
                logger.info("Temp strata too small, using simple split for val")
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=0.1765, random_state=42, stratify=None
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=0.1765, random_state=42, stratify=stratify_key_temp
                )
        
        logger.info(f"Train size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
        logger.info(f"Val size: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
        logger.info(f"Test size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_random_forest(self, X_train, y_train):
        logger.info("Training Random Forest Regressor")
        
        n_estimators = 100
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            warm_start=True,
            verbose=0
        )
        
        # Train incrementally to show progress
        step = 10
        for i in range(step, n_estimators + 1, step):
            model.n_estimators = i
            model.fit(X_train, y_train)
            progress = (i / n_estimators) * 100
            print(f"\rRandom Forest Training Progress: {progress:.0f}% ({i}/{n_estimators} trees)", end='', flush=True)
        
        print()  # New line after progress
        self.models['RandomForest'] = model
        logger.info("Random Forest training completed")
        
        return model
    
    def train_spatial_ensemble(self, X_train, y_train, coords_train):
        """Train the spatially-weighted ensemble model."""
        logger.info("Training Spatially-Weighted Ensemble (Novel Method)")
        
        model = SpatiallyWeightedEnsemble(
            base_models=None,  # Uses default: RF, GB, Ridge
            spatial_bandwidth='adaptive',
            n_neighbors=50,
            include_spatial_lag=True
        )
        
        model.fit(X_train.values, y_train.values, coords=coords_train)
        self.models['SpatialEnsemble'] = model
        logger.info("Spatial Ensemble training completed")
        
        return model
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        logger.info("Training XGBoost Regressor")
        
        # Custom callback for progress tracking
        class XGBoostProgressCallback(xgb.callback.TrainingCallback):
            def __init__(self, n_estimators):
                self.n_estimators = n_estimators
                
            def after_iteration(self, model, epoch, evals_log):
                progress = ((epoch + 1) / self.n_estimators) * 100
                print(f"\rXGBoost Training Progress: {progress:.0f}% (Tree {epoch + 1}/{self.n_estimators})", end='', flush=True)
                return False
        
        n_estimators = 100
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            callbacks=[XGBoostProgressCallback(n_estimators)]
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        print()  # New line after progress
        self.models['XGBoost'] = model
        logger.info("XGBoost training completed")
        
        return model
    
    def train_tabnet(self, X_train, y_train, X_val, y_val):
        logger.info("Training TabNet Regressor")
        
        # Convert to numpy arrays with proper dtypes and ensure writability
        X_train_np = X_train.values.copy().astype(np.float32)
        y_train_np = y_train.values.copy().astype(np.float32).reshape(-1, 1)
        X_val_np = X_val.values.copy().astype(np.float32)
        y_val_np = y_val.values.copy().astype(np.float32).reshape(-1, 1)
        
        # Custom callback for real-time epoch progress
        from pytorch_tabnet.callbacks import Callback
        
        class ProgressCallback(Callback):
            def __init__(self, max_epochs):
                self.max_epochs = max_epochs
                self.current_epoch = 0
                
            def on_epoch_end(self, epoch, logs=None):
                self.current_epoch += 1
                progress = (self.current_epoch / self.max_epochs) * 100
                print(f"\rTabNet Training Progress: {progress:.0f}% (Epoch {self.current_epoch}/{self.max_epochs})", end='', flush=True)
        
        max_epochs = 50
        model = TabNetRegressor(
            n_d=8,
            n_a=8,
            n_steps=3,
            gamma=1.3,
            n_independent=2,
            n_shared=2,
            seed=42,
            verbose=0
        )
        
        progress_callback = ProgressCallback(max_epochs)
        
        try:
            model.fit(
                X_train_np, y_train_np,
                eval_set=[(X_val_np, y_val_np)],
                max_epochs=max_epochs,
                patience=10,
                batch_size=256,
                virtual_batch_size=128,
                callbacks=[progress_callback]
            )
            print()  # New line after progress
        except Exception as e:
            print()  # New line after progress
            logger.warning(f"TabNet callback issue, but training may have succeeded: {e}")
        
        self.models['TabNet'] = model
        logger.info("TabNet training completed")
        
        return model
    
    def train_kan(self, X_train, y_train, X_val, y_val):
        logger.info("Training KAN (Kolmogorov-Arnold Network)")
        
        if KAN is None:
            logger.warning("KAN not available, skipping KAN training")
            return None
        
        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Try to set PyTorch to use a more compatible LAPACK backend
        # This may help avoid Intel MKL lstsq issues
        original_num_threads = torch.get_num_threads()
        try:
            # Limit threads to potentially avoid MKL issues
            torch.set_num_threads(1)
            logger.info("Set PyTorch to single-threaded mode to avoid MKL conflicts")
        except:
            pass
        
        # Try multiple strategies to train KAN
        strategies = [
            {
                'name': 'Adam with moderate sample and grid',
                'sample_size': 3000,
                'val_sample_size': 600,
                'optimizer': 'Adam',
                'steps': 200,
                'lr': 0.001,
                'lamb': 0.0,  # No regularization initially
                'grid': 5,  # Reasonable grid resolution
                'hidden_size': 10,
                'update_grid': False  # Don't update grid to avoid lstsq issues
            },
            {
                'name': 'Adam with small sample and fine grid',
                'sample_size': 2000,
                'val_sample_size': 400,
                'optimizer': 'Adam',
                'steps': 150,
                'lr': 0.001,
                'lamb': 0.0,
                'grid': 7,  # Finer grid
                'hidden_size': 12,
                'update_grid': False
            },
            {
                'name': 'LBFGS with moderate sample',
                'sample_size': 2000,
                'val_sample_size': 400,
                'optimizer': 'LBFGS',
                'steps': 20,
                'lr': 1.0,
                'lamb': 0.0,
                'grid': 5,
                'hidden_size': 10,
                'update_grid': False
            },
        ]
        
        for strategy in strategies:
            try:
                logger.info(f"Trying KAN training strategy: {strategy['name']}")
                
                # Sample subset for KAN (it's computationally expensive)
                sample_size = min(strategy['sample_size'], len(X_train))
                indices = np.random.choice(len(X_train), sample_size, replace=False)
                X_train_sample = X_train_scaled[indices]
                y_train_sample = y_train.values[indices].reshape(-1, 1)
                
                val_sample_size = min(strategy['val_sample_size'], len(X_val))
                val_indices = np.random.choice(len(X_val), val_sample_size, replace=False)
                X_val_sample = X_val_scaled[val_indices]
                y_val_sample = y_val.values[val_indices].reshape(-1, 1)
                
                # Ensure data is contiguous and properly formatted
                X_train_sample = np.ascontiguousarray(X_train_sample, dtype=np.float32)
                y_train_sample = np.ascontiguousarray(y_train_sample, dtype=np.float32)
                X_val_sample = np.ascontiguousarray(X_val_sample, dtype=np.float32)
                y_val_sample = np.ascontiguousarray(y_val_sample, dtype=np.float32)
                
                # Check for NaN/Inf values
                if np.any(~np.isfinite(X_train_sample)) or np.any(~np.isfinite(y_train_sample)):
                    logger.warning("NaN or Inf detected in training data, skipping strategy")
                    continue
                
                model = KAN(
                    width=[X_train.shape[1], strategy['hidden_size'], 1],
                    grid=strategy['grid'],
                    k=3,
                    seed=42
                )
                
                dataset = {
                    'train_input': torch.tensor(X_train_sample, dtype=torch.float32),
                    'train_label': torch.tensor(y_train_sample, dtype=torch.float32),
                    'test_input': torch.tensor(X_val_sample, dtype=torch.float32),
                    'test_label': torch.tensor(y_val_sample, dtype=torch.float32)
                }
                
                # Validate tensors
                for key, tensor in dataset.items():
                    if not torch.isfinite(tensor).all():
                        raise ValueError(f"Non-finite values in {key}")
                
                # Train with progress tracking
                steps = strategy['steps']
                update_grid = strategy.get('update_grid', False)
                print(f"KAN Training ({strategy['name']}): ", end='', flush=True)
                
                # Train without grid updates to avoid lstsq calls
                if not update_grid:
                    # Disable grid updates by training with update_grid=False
                    model.fit(
                        dataset, 
                        opt=strategy['optimizer'], 
                        steps=steps, 
                        lamb=strategy['lamb'],
                        lr=strategy.get('lr', 1.0),
                        update_grid=False
                    )
                else:
                    model.fit(
                        dataset, 
                        opt=strategy['optimizer'], 
                        steps=steps, 
                        lamb=strategy['lamb'],
                        lr=strategy.get('lr', 1.0)
                    )
                
                print(f" ✓ Completed")
                self.models['KAN'] = model
                logger.info(f"KAN training completed with strategy '{strategy['name']}' (trained on {sample_size} samples)")
                
                # Restore thread count
                try:
                    torch.set_num_threads(original_num_threads)
                except:
                    pass
                
                return model
                
            except Exception as e:
                print(f" ✗ Failed")
                logger.warning(f"KAN training strategy '{strategy['name']}' failed: {str(e)[:100]}")
                continue
        
        # Restore thread count
        try:
            torch.set_num_threads(original_num_threads)
        except:
            pass
        
        # All strategies failed
        logger.error("All KAN training strategies failed, skipping KAN")
        return None
    
    def evaluate_model(self, model, X, y, model_name):
        logger.info(f"Evaluating {model_name}")
        
        if model_name == 'TabNet':
            X_np = X.values.astype(np.float32)
            y_pred = model.predict(X_np).flatten()
        elif model_name == 'KAN':
            X_scaled = self.scaler.transform(X)
            y_pred = model.forward(torch.tensor(X_scaled, dtype=torch.float32)).detach().numpy().flatten()
        else:
            y_pred = model.predict(X)
        
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        logger.info(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'predictions': y_pred}
    
    def run_model_comparison(self, df, results_dir):
        logger.info("Starting model comparison pipeline")
        
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        models_dir = results_dir / 'models'
        
        # Check if core models already exist
        models_exist = (
            (models_dir / 'random_forest.pkl').exists() and
            (models_dir / 'xgboost_model.json').exists() and
            (models_dir / 'tabnet_model.zip').exists()
        )
        
        # Always create the split for evaluation
        X_train, X_val, X_test, y_train, y_val, y_test = self.create_stratified_split(df)
        
        if models_exist:
            logger.info("✓ Pre-trained models found. Loading existing models to skip retraining...")
            self.load_models(models_dir)
        else:
            logger.info("No pre-trained models found. Training new models...")
            
            rf_model = self.train_random_forest(X_train, y_train)
            xgb_model = self.train_xgboost(X_train, y_train, X_val, y_val)
            tabnet_model = self.train_tabnet(X_train, y_train, X_val, y_val)
            
            # Save models immediately after training
            self.save_models(results_dir)
            logger.info("✓ Models trained and saved successfully")
        
        # KAN is experimental and trained separately (optional)
        if KAN is not None and not (models_dir / 'kan_model.pkl').exists():
            logger.info("Training KAN model (experimental)...")
            try:
                kan_model = self.train_kan(X_train, y_train, X_val, y_val)
                if kan_model is not None:
                    # Save KAN separately
                    import joblib
                    joblib.dump(kan_model, models_dir / 'kan_model.pkl')
                    logger.info("KAN model saved")
            except Exception as e:
                logger.warning(f"KAN training failed: {e}")
        elif (models_dir / 'kan_model.pkl').exists():
            logger.info("Loading pre-trained KAN model...")
            try:
                import joblib
                self.models['KAN'] = joblib.load(models_dir / 'kan_model.pkl')
                logger.info("✓ KAN model loaded")
            except Exception as e:
                logger.warning(f"Failed to load KAN model: {e}")
        
        # Evaluate all models
        results = []
        
        for model_name, model in self.models.items():
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            results.append({
                'Model': model_name,
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'R2': metrics['R2']
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('R2', ascending=False)
        
        results_path = results_dir / 'model_comparison.csv'
        results_df.to_csv(results_path, index=False)
        logger.info(f"Model comparison results saved to {results_path}")
        
        return results_df, X_test, y_test
    
    def load_models(self, models_dir):
        """Load pre-trained models from disk to avoid retraining."""
        import joblib
        
        models_dir = Path(models_dir)
        
        # Load Random Forest
        if (models_dir / 'random_forest.pkl').exists():
            self.models['RandomForest'] = joblib.load(models_dir / 'random_forest.pkl')
            logger.info("✓ Random Forest model loaded")
        
        # Load XGBoost
        if (models_dir / 'xgboost_model.json').exists():
            model = xgb.XGBRegressor()
            model.load_model(str(models_dir / 'xgboost_model.json'))
            self.models['XGBoost'] = model
            logger.info("✓ XGBoost model loaded")
        
        # Load TabNet
        if (models_dir / 'tabnet_model.zip').exists():
            model = TabNetRegressor()
            model.load_model(str(models_dir / 'tabnet_model.zip'))
            self.models['TabNet'] = model
            logger.info("✓ TabNet model loaded")
        
        logger.info(f"Total models loaded: {len(self.models)}")
    
    def save_models(self, results_dir):
        logger.info("Saving trained models")
        
        models_dir = results_dir / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        
        import joblib
        
        if 'RandomForest' in self.models:
            joblib.dump(self.models['RandomForest'], models_dir / 'random_forest.pkl')
            logger.info("Random Forest model saved")
        
        if 'XGBoost' in self.models:
            # Use get_booster() method for saving XGBoost models
            self.models['XGBoost'].get_booster().save_model(str(models_dir / 'xgboost_model.json'))
            logger.info("XGBoost model saved")
        
        if 'TabNet' in self.models:
            self.models['TabNet'].save_model(str(models_dir / 'tabnet_model'))
            logger.info("TabNet model saved")
        
        logger.info(f"Models saved to {models_dir}")
    
    def generate_shap_analysis(self, X_test, figures_dir):
        logger.info("Generating SHAP analysis for XGBoost")
        
        if 'XGBoost' not in self.models:
            logger.warning("XGBoost model not found, skipping SHAP analysis")
            return
        
        explainer = shap.TreeExplainer(self.models['XGBoost'])
        shap_values = explainer.shap_values(X_test)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, show=False, max_display=15)
        
        save_path = Path(figures_dir) / 'shap_summary.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SHAP summary plot saved to {save_path}")
        
        return fig
