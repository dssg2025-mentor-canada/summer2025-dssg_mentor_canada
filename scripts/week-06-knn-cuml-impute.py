import cudf
import cupy as cp
import pandas as pd
import numpy as np
from cuml.experimental.preprocessing import KNNImputer
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load and prep data
def load_data(file_path):
    """
    Load dataset from a Feather file into a cuDF DataFrame.
    Replace common missing value placeholders (e.g., '', 'NA') with pd.NA.
    """
    try:
        logging.info("Loading dataset...")
        # Load Feather file with pandas, then convert to cuDF
        df = pd.read_feather(file_path)
        na_values = ['', 'NA', 'missing']
        df = df.replace(na_values, pd.NA)
        # Convert to cuDF and ensure float32 for GPU compatibility
        gdf = cudf.from_pandas(df).astype('float32')
        logging.info(f"Dataset loaded with shape: {gdf.shape}")
        return gdf
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

# Step 2: Preprocess data (no scaling for pre-standardized continuous variables! already done!)
def preprocess_data(gdf, missing_threshold=0.9):
    """
    Optionally drop columns with high missingness.
    Assume continuous variables are pre-standardized; do not scale binary variables.
    """
    try:
        logging.info("Preprocessing data...")
        # Drop columns with high missingness
        missing_rates = gdf.isna().mean()
        high_missing_cols = missing_rates[missing_rates > missing_threshold].index
        if len(high_missing_cols) > 0:
            logging.warning(f"Dropping {len(high_missing_cols)} columns with >{missing_threshold*100}% missing values: {high_missing_cols}")
            gdf = gdf.drop(columns=high_missing_cols)
        
        # Convert to CuPy array for cuML
        data = gdf.to_cupy()
        logging.info("Data preprocessing completed.")
        return data, gdf.columns
    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        raise

# Step 3: Perform k-NN imputation with cuML
def impute_missing_cuml(data, k=5):
    """
    Use cuML KNNImputer to impute missing values on GPU.
    
    Args:
        data (cupy.ndarray): Input data with missing values (NaN).
        k (int): Number of nearest neighbors for imputation.
    
    Returns:
        cupy.ndarray: Imputed data with missing values filled.
    """
    try:
        logging.info(f"Starting k-NN imputation with cuML, k={k}")
        # Initialize KNNImputer
        imputer = KNNImputer(n_neighbors=k, missing_values=np.nan)
        # Perform imputation
        imputed_data = imputer.fit_transform(data)
        logging.info("cuML k-NN imputation completed.")
        return imputed_data
    except Exception as e:
        logging.error(f"Error in cuML k-NN imputation: {e}")
        raise

# Step 4: Post-process and save results (no inverse scaling)
def postprocess_and_save(imputed_data, columns, output_path):
    """
    Convert imputed data to Pandas DataFrame and save as CSV.
    """
    try:
        logging.info("Post-processing and saving results...")
        # Convert CuPy array to Pandas DataFrame
        imputed_df = pd.DataFrame(cp.asnumpy(imputed_data), columns=columns)
        imputed_df.to_csv(output_path, index=False)
        logging.info(f"Imputed dataset saved to {output_path}")
        return imputed_df
    except Exception as e:
        logging.error(f"Error in post-processing: {e}")
        raise

# Main workflow
def main():
    file_path = '../../dssg-2025-mentor-canada/Data/ohe_unimputed_train.feather'
    output_path = '../../dssg-2025-mentor-canada/Data/faiss_knn_imputed_dataset.csv'
    k_neighbors = 5
    missing_threshold = 0.9
    
    gdf = load_data(file_path)
    data, columns = preprocess_data(gdf, missing_threshold)
    imputed_data = impute_missing_cuml(data, k=k_neighbors)
    imputed_df = postprocess_and_save(imputed_data, columns, output_path)
    
    return imputed_df

if __name__ == "__main__":
    main()