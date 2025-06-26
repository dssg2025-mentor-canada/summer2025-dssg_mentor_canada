import pandas as pd
import numpy as np
import faiss
import logging

# Set up logging to capture warnings/informational messages, and errors
# Also helps to track which columns are being dropped
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1: Load and prepare the dataset
def load_data(file_path):
    """
    Load dataset from Feather file.
    Replace common missing value placeholders (e.g., '', 'NA') with pd.NA for pandas:
    """
    try:
        logging.info("Loading dataset...") # Initialize logging info

        df = pd.read_feather(file_path)
        na_values = ['', 'NA', 'missing']
        df = df.replace(na_values, pd.NA)

        df = df.astype('float32') # Ensures all are numeric
        
        logging.info(f"Dataset loaded with shape: {df.shape}") # Update logging info
        return df

    except Exception as e:
        logging.error(f"Error loading data: {e}") # Logging error info if occurs (e.g., in .astype)
        raise

# Step 2: Preprocess the data (no scaling for pre-standardized continuous variables)
def preprocess_data(df, missing_threshold = 0.90, passthrough_prefixes = None):
    """
    Drop columns with defined high missingness.

    From previous preprocessing:
    - Our continuous variables are already pre-standardize, so we don't scale metric variables.
    - We also don't standardize any dummy-coded or ordinal variables.

    df: Pandas DataFrame.
    missing_threshold: Threshold for dropping columns (default: 0.97).
    passthrough_prefix: String prefix to match passthrough columns (e.g., 'QS4_19_CURRENTME').
    """
    try:
        logging.info("Preprocessing data...")
        # Handle passthrough columns by prefix
        passthrough_data = None
        passthrough_cols = []
        if passthrough_prefixes:
            # Collect columns matching any prefix
            passthrough_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in passthrough_prefixes)]
            if passthrough_cols:
                logging.info(f"Passthrough columns: {passthrough_cols}")
                passthrough_data = df[passthrough_cols].copy()
                df = df.drop(columns=passthrough_cols)
            else:
                logging.warning("No columns matched the passthrough prefixes.")
        # Identify reliable features (<10% missing) for passthrough imputation
        reliable_cols = df.columns[df.isna().mean() < 0.1]
        reliable_features = df[reliable_cols].copy()
        
        # Drop columns with high missingness
        missing_rates = df.isna().mean()
        high_missing_cols = missing_rates[missing_rates > missing_threshold].index
        if len(high_missing_cols) > 0:
            logging.warning(f"Dropping {len(high_missing_cols)} columns with >{missing_threshold*100}% missing values: {high_missing_cols}")
            df = df.drop(columns=high_missing_cols)
        
        # Convert to NumPy array for features
        data = df.values
        feature_cols = df.columns
        logging.info("Data preprocessing completed.")
        return data, feature_cols, passthrough_data, reliable_features
    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        raise

# Step 3: Perform k-NN imputation with FAISS
def impute_missing_faiss(data, k = 5):
    """
    Use FAISS to find k nearest neighbors and impute missing values.
    
    Args:
        data (np.ndarray): Input data with missing values (nan).
        k (int): Number of nearest neighbors for imputation.
    
    Returns:
        np.ndarray: Imputed data with missing values filled.
    """
    try:
        logging.info(f"Starting k-NN imputation with FAISS, k={k}")
        data_for_faiss = np.where(np.isnan(data), 0, data)
        data_for_faiss = np.ascontiguousarray(data_for_faiss) # Ensures a C-contiguous array!
        n, d = data.shape
        
        index = faiss.IndexFlatL2(d)
        index = faiss.IndexFlatIP(d) if d > 256 else index
        index.add(data_for_faiss)
        
        distances, indices = index.search(data_for_faiss, k + 1)
        
        imputed_data = data.copy()
        
        for i in range(n):
            missing_idx = np.where(np.isnan(data[i]))[0]
            if missing_idx.size > 0:
                neighbor_indices = indices[i, 1:k+1]
                neighbor_values = data[neighbor_indices]
                neighbor_distances = distances[i, 1:k+1]
                weights = 1 / (neighbor_distances + 1e-6)
                weights = weights / np.sum(weights)
                
                for idx in missing_idx:
                    valid_neighbors = ~np.isnan(neighbor_values[:, idx])
                    if valid_neighbors.any():
                        imputed_data[i, idx] = np.sum(weights[valid_neighbors] * neighbor_values[valid_neighbors, idx]) / np.sum(weights[valid_neighbors])
                    else:
                        col_mean = np.nanmean(data[:, idx])
                        imputed_data[i, idx] = col_mean if not np.isnan(col_mean) else 0
        
        logging.info("FAISS k-NN imputation completed.")
        return imputed_data
    except Exception as e:
        logging.error(f"Error in FAISS k-NN imputation: {e}")
        raise

# Step 4: Impute passthrough columns
def impute_passthrough(passthrough_data, feature_data, k=5):
    """
    Impute missing values in passthrough columns using FAISS k-NN with imputed features.
    
    Args:
        passthrough_data: Pandas DataFrame of passthrough columns.
        feature_data: Pandas DataFrame of imputed features.
        k: Number of nearest neighbors.
    
    Returns:
        Pandas DataFrame with imputed passthrough columns.
    """
    try:
        if passthrough_data is None:
            return None
        logging.info(f"Imputing passthrough columns with k={k}")
        passthrough_imputed = passthrough_data.copy()
        data_for_faiss = feature_data.fillna(0).values
        data_for_faiss = np.ascontiguousarray(data_for_faiss.astype('float32'))
        n, d = data_for_faiss.shape
        
        index = faiss.IndexFlatL2(d)
        index = faiss.IndexFlatIP(d) if d > 256 else index
        index.add(data_for_faiss)
        
        distances, indices = index.search(data_for_faiss, k + 1)
        
        for col in passthrough_imputed.columns:
            missing_idx = passthrough_imputed[col].isna()
            if missing_idx.any():
                logging.info(f"Imputing column {col} with {missing_idx.sum()} missing values")
                is_binary = passthrough_data[col].nunique(dropna=True) <= 2
                for i in np.where(missing_idx)[0]:
                    neighbor_indices = indices[i, 1:k+1]
                    neighbor_values = passthrough_data[col].iloc[neighbor_indices]
                    neighbor_distances = distances[i, 1:k+1]
                    weights = 1 / (neighbor_distances + 1e-6)
                    weights = weights / np.sum(weights)
                    
                    valid_neighbors = ~neighbor_values.isna()
                    if valid_neighbors.any():
                        imputed_value = np.sum(weights[valid_neighbors] * neighbor_values[valid_neighbors]) / np.sum(weights[valid_neighbors])
                        if is_binary:
                            imputed_value = round(imputed_value)
                        passthrough_imputed.loc[i, col] = imputed_value
                    else:
                        col_mean = passthrough_data[col].mean()
                        imputed_value = col_mean if not np.isnan(col_mean) else 0
                        if is_binary:
                            imputed_value = round(imputed_value)
                        passthrough_imputed.loc[i, col] = imputed_value
                if missing_idx.sum() / len(passthrough_data) > 0.9:
                    logging.warning(f"Column {col} has >90% missing values; imputation may be unreliable.")
        
        logging.info("Passthrough imputation completed.")
        return passthrough_imputed
    except Exception as e:
        logging.error(f"Error in passthrough imputation: {e}")
        raise


# Step 5: Post-process and save results
def postprocess_and_save(imputed_data, feature_cols, passthrough_data, imputed_features_df, output_path, k=5):
    """
    Combine imputed features with imputed passthrough columns and save as CSV.
    
    Args:
        imputed_data: NumPy array of imputed features.
        feature_cols: List of feature column names.
        passthrough_data: Pandas DataFrame of passthrough columns (or None).
        imputed_features_df: Pandas DataFrame of imputed features.
        output_path: Path to save the CSV.
        k: Number of neighbors for passthrough imputation.
    
    Returns:
        Pandas DataFrame of the final dataset.
    """
    try:
        logging.info("Post-processing and saving results...")
        # Convert imputed features to DataFrame
        imputed_df = pd.DataFrame(imputed_data, columns=feature_cols)
        # Impute passthrough columns
        if passthrough_data is not None:
            passthrough_data = impute_passthrough(passthrough_data, imputed_features_df, k)
            imputed_df = pd.concat([imputed_df, passthrough_data], axis=1)
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
    missing_threshold = 0.97

    # List of prefixes for passthrough columns:
    passthrough_prefixes = ['QS1_19_HIGHSCHOOL', 'QS4_19_CURRENTME', 'QS4_18_CURRENTOR', 
                            'QS1_20_HIGHSCHOOL', 'QS1_21_FURTHEDUCA', 'QS1_22_HIGHESTEDU', 
                            'QS1_23_YEARCOMPLE', 'QS1_25_EMPLOYMENT', 'QS1_26_EMPLOYMENT',
                            'QS1_27_PLANNINGRE',  'QS1_28_EMPLOYMENT', 'QS4_4_EDUCATIONALEXPEC', 
                            'QS4_5_SATEDU','QS4_6_DISAPPOINTED', # these are self-motivation variable.
                            'QS4_7_SOCIALCAPITAL', 'QS4_8_HELPSEEKING', 'QS4_9_MENTALHEALTH',
                            'QS4_10_MENTALWELLBE', 'QS4_11_BELONGING', 'QS4_12_TRUST',
                            'QS4_16_FORMALVOL', 'QS4_25_FUTUREMEN', 'QS4_21_MENTORING',
                            'QS4_17_SERVEDASM', 'QS4_18_CURRENTOR', 'QS4_22_PASTMENTO', 
                            'QS4_1_MEANINGFULPERSON', 'QS4_13_LIFEEVE']  
    
    
    df = load_data(file_path)
    data, feature_cols, passthrough_data, reliable_features = preprocess_data(df, missing_threshold, passthrough_prefixes)
    imputed_data = impute_missing_faiss(data, k=k_neighbors)
    imputed_features_df = pd.DataFrame(imputed_data, columns=feature_cols)
    imputed_df = postprocess_and_save(imputed_data, feature_cols, passthrough_data, imputed_features_df, output_path, k_neighbors)
    
    return imputed_df

if __name__ == "__main__":
    main()
    