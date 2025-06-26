import pandas as pd
import numpy as np
import faiss
import logging
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1: Load and prepare the dataset
def load_data(file_path):
    try:
        logging.info("Loading dataset...")
        df = pd.read_feather(file_path)
        na_values = ['', 'NA', 'missing']
        df = df.replace(na_values, pd.NA)
        df = df.astype('float32')
        logging.info(f"Dataset loaded with shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

# Step 2: Preprocess the data
def preprocess_data(df, missing_threshold=0.90, passthrough_prefixes=None):
    try:
        logging.info("Preprocessing data...")
        passthrough_data = None
        passthrough_cols = []
        if passthrough_prefixes:
            passthrough_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in passthrough_prefixes)]
            if passthrough_cols:
                logging.info(f"Passthrough columns: {passthrough_cols}")
                passthrough_data = df[passthrough_cols].copy()
                df = df.drop(columns=passthrough_cols)
            else:
                logging.warning("No columns matched the passthrough prefixes.")
        reliable_cols = df.columns[df.isna().mean() < 0.1]
        reliable_features = df[reliable_cols].copy()
        missing_rates = df.isna().mean()
        high_missing_cols = missing_rates[missing_rates > missing_threshold].index
        if len(high_missing_cols) > 0:
            logging.warning(f"Dropping {len(high_missing_cols)} columns with >{missing_threshold*100}% missing values: {high_missing_cols}")
            df = df.drop(columns=high_missing_cols)
        data = df.values
        feature_cols = df.columns
        logging.info("Data preprocessing completed.")
        return data, feature_cols, passthrough_data, reliable_features
    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        raise

# Step 3: Perform k-NN imputation with FAISS
def impute_missing_faiss(data, k=5, index=None):
    try:
        logging.info(f"Starting k-NN imputation with FAISS, k={k}")
        data_for_faiss = np.where(np.isnan(data), 0, data)
        data_for_faiss = np.ascontiguousarray(data_for_faiss)
        n, d = data.shape
        
        if index is None:
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
        return imputed_data, index
    except Exception as e:
        logging.error(f"Error in FAISS k-NN imputation: {e}")
        raise

# Step 4: Impute passthrough columns
def impute_passthrough(passthrough_data, feature_data, k=5, index=None):
    try:
        if passthrough_data is None:
            return None
        logging.info(f"Imputing passthrough columns with k={k}")
        passthrough_imputed = passthrough_data.copy()
        data_for_faiss = feature_data.fillna(0).values
        data_for_faiss = np.ascontiguousarray(data_for_faiss.astype('float32'))
        n, d = data_for_faiss.shape
        
        if index is None:
            index = faiss.IndexFlatL2(d)
            index = faiss.IndexFlatIP(d) if d > 256 else index
            index.add(data_for_faiss)
        
        distances, indices = index.search(data_for_faiss, k + 1)
        
        for col in passthrough_imputed.columns:
            missing_idx = passthrough_imputed[col].isna()
            if missing_idx.any():
                logging.info(f"Imputing column {col} with {missing_idx.sum()} missing values ({100 * missing_idx.sum() / len(passthrough_data):.2f}%)")
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
                
                # Log sample imputations after imputation
                if is_binary:
                    sample_idx = np.where(missing_idx)[0][:5]
                    for i in sample_idx:
                        logging.info(f"Sample imputation for {col} at index {i}: {passthrough_imputed.loc[i, col]}")
                
                if missing_idx.sum() / len(passthrough_data) > 0.9:
                    logging.warning(f"Column {col} has >90% missing values; imputation may be unreliable.")
        
        logging.info("Passthrough imputation completed.")
        return passthrough_imputed, index
    except Exception as e:
        logging.error(f"Error in passthrough imputation: {e}")
        raise


# Step 5: Post-process and save results
def postprocess_and_save(imputed_data, feature_cols, passthrough_data, imputed_features_df, output_path, k=5, index=None):
    try:
        logging.info("Post-processing and saving results...")
        imputed_df = pd.DataFrame(imputed_data, columns=feature_cols)
        if passthrough_data is not None:
            passthrough_data, _ = impute_passthrough(passthrough_data, imputed_features_df, k, index)
            imputed_df = pd.concat([imputed_df, passthrough_data], axis=1)
        imputed_df.to_csv(output_path, index=False)
        logging.info(f"Imputed dataset saved to {output_path}")
        return imputed_df
    except Exception as e:
        logging.error(f"Error in post-processing: {e}")
        raise

# Step 6: Tune k with cross-validation
def tune_k(data, feature_cols, passthrough_data, reliable_features, k_values=[5, 10, 20, 40, 60], mask_fraction=0.1, random_seed=42):
    try:
        logging.info("Starting k tuning...")
        np.random.seed(random_seed)
        results = []
        
        data_for_faiss = np.where(np.isnan(data), 0, data)
        data_for_faiss = np.ascontiguousarray(data_for_faiss)
        n, d = data.shape
        index = faiss.IndexFlatL2(d)
        index = faiss.IndexFlatIP(d) if d > 256 else index
        index.add(data_for_faiss)
        
        mask = np.zeros_like(data, dtype=bool)
        for col in range(data.shape[1]):
            non_missing_idx = np.where(~np.isnan(data[:, col]))[0]
            n_mask = int(mask_fraction * len(non_missing_idx))
            if n_mask > 0:
                mask_idx = np.random.choice(non_missing_idx, size=n_mask, replace=False)
                mask[mask_idx, col] = True
        
        ground_truth = data[mask]
        masked_data = data.copy()
        masked_data[mask] = np.nan
        
        for k in k_values:
            logging.info(f"Evaluating k={k}")
            imputed_data, _ = impute_missing_faiss(masked_data, k, index)
            imputed_features_df = pd.DataFrame(imputed_data, columns=feature_cols)
            
            mse = mean_squared_error(ground_truth, imputed_data[mask])
            
            acc = None
            if passthrough_data is not None:
                passthrough_imputed, _ = impute_passthrough(passthrough_data, imputed_features_df, k, index)
                binary_cols = [col for col in passthrough_data.columns if passthrough_data[col].nunique(dropna=True) <= 2]
                logging.info(f"Number of binary passthrough columns: {len(binary_cols)}")
                if binary_cols:
                    acc_scores = []
                    for col in binary_cols:
                        mask_col = passthrough_data[col].notna() & passthrough_imputed[col].notna()
                        logging.info(f"Column {col}: {mask_col.sum()} valid entries for accuracy")
                        if mask_col.sum() > 0:
                            acc_scores.append(accuracy_score(passthrough_data[col][mask_col], passthrough_imputed[col][mask_col]))
                    acc = np.mean(acc_scores) if acc_scores else None
                    if acc_scores:
                        logging.info(f"Binary accuracy scores: {acc_scores}")
            
            results.append({'k': k, 'mse': mse, 'binary_acc': acc})
            logging.info(f"k={k}: MSE={mse:.6f}, Binary Accuracy={acc if acc is not None else 'N/A'}")
        
        best_k = min(results, key=lambda x: x['mse'])['k']
        logging.info(f"Best k={best_k} based on MSE")
        return best_k, results
    except Exception as e:
        logging.error(f"Error in k tuning: {e}")
        raise
    
# Main workflow
def main():
    file_path = '../../dssg-2025-mentor-canada/Data/ohe_unimputed_train.feather'
    output_path = '../../dssg-2025-mentor-canada/Data/faiss_tuned_knn_imputed_dataset.csv'
    missing_threshold = 0.97
    k_values = [5, 10, 20, 40, 60]
    
    passthrough_prefixes = [
        'QS1_19_HIGHSCHOOL', 'QS4_19_CURRENTME', 'QS4_18_CURRENTOR',
        'QS1_20_HIGHSCHOOL', 'QS1_21_FURTHEDUCA', 'QS1_22_HIGHESTEDU',
        'QS1_23_YEARCOMPLE', 'QS1_25_EMPLOYMENT', 'QS1_26_EMPLOYMENT',
        'QS1_27_PLANNINGRE', 'QS1_28_EMPLOYMENT', 'QS4_4_EDUCATIONALEXPEC',
        'QS4_5_SATEDU', 'QS4_6_DISAPPOINTED', 'QS4_7_SOCIALCAPITAL',
        'QS4_8_HELPSEEKING', 'QS4_9_MENTALHEALTH', 'QS4_10_MENTALWELLBE',
        'QS4_11_BELONGING', 'QS4_12_TRUST', 'QS4_16_FORMALVOL',
        'QS4_25_FUTUREMEN', 'QS4_21_MENTORING', 'QS4_17_SERVEDASM',
        'QS4_18_CURRENTOR', 'QS4_22_PASTMENTO', 'QS4_1_MEANINGFULPERSON',
        'QS4_13_LIFEEVE'
    ]
    
    df = load_data(file_path)
    data, feature_cols, passthrough_data, reliable_features = preprocess_data(df, missing_threshold, passthrough_prefixes)
    
    # Tune k
    best_k, results = tune_k(data, feature_cols, passthrough_data, reliable_features, k_values)
    
    # Impute with best k
    imputed_data, index = impute_missing_faiss(data, best_k)
    imputed_features_df = pd.DataFrame(imputed_data, columns=feature_cols)
    imputed_df = postprocess_and_save(imputed_data, feature_cols, passthrough_data, imputed_features_df, output_path, best_k, index)
    
    return imputed_df, best_k, results

if __name__ == "__main__":
    imputed_df, best_k, results = main()
    print(f"Best k: {best_k}")
    print("Tuning results:")
    for result in results:
        print(f"k={result['k']}: MSE={result['mse']:.6f}, Binary Accuracy={result['binary_acc'] if result['binary_acc'] is not None else 'N/A'}")