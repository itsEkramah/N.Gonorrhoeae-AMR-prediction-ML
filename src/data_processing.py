import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

def load_data(antibiotic, data_dir='DATA'):
    """
    Loads metadata and unitig data for a specific antibiotic.
    
    Args:
        antibiotic (str): One of 'Azithromycin', 'Ciprofloxacin', 'Cefixime'.
        data_dir (str): Path to the data directory.
        
    Returns:
        pd.DataFrame: Combined dataframe with metadata and unitigs.
    """
    abx_map = {
        'Azithromycin': 'azm',
        'Ciprofloxacin': 'cip',
        'Cefixime': 'cfx'
    }
    code = abx_map.get(antibiotic)
    if not code:
        raise ValueError(f"Unknown antibiotic: {antibiotic}. Must be one of {list(abx_map.keys())}")
        
    # Load metadata
    metadata_path = f'{data_dir}/metadata.csv'
    try:
        metadata = pd.read_csv(metadata_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    # Load unitigs
    rtab_path = f'{data_dir}/{code}_sr_gwas_filtered_unitigs.Rtab'
    try:
        # Transpose so samples are rows, unitigs are columns
        # The file seems to be space-separated based on inspection
        unitigs = pd.read_csv(rtab_path, sep=r'\s+', index_col=0).T
    except FileNotFoundError:
        raise FileNotFoundError(f"Unitig file not found at {rtab_path}")
        
    # Check index name consistency
    # metadata has 'Sample_ID' column. unitigs index is sample IDs.
    
    # Merge
    # We use inner join to keep only samples present in both
    combined = metadata.set_index('Sample_ID').join(unitigs, how='inner')
    
    return combined, code

def preprocess_data(data, target_col, metadata_cols=['Country', 'Continent']):
    """
    Preprocesses the data: handles missing values, encodes categorical features.
    
    Args:
        data (pd.DataFrame): Combined dataframe.
        target_col (str): Name of the target column (e.g., 'azm_sr').
        metadata_cols (list): List of categorical metadata columns to encode.
        
    Returns:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
    """
    # 1. Handle Missing Target
    # Drop rows where target is NaN
    data = data.dropna(subset=[target_col]).copy()
    
    # 2. Handle Missing Features
    # For categorical metadata, we can treat NaN as a separate category or drop.
    # Given the prompt "Handle missing data by either imputation or removal", 
    # let's fill categorical NaNs with 'Unknown' to avoid data loss, 
    # as metadata might be missing but genomic data is valuable.
    for col in metadata_cols:
        if col in data.columns:
            data[col] = data[col].fillna('Unknown')
            
    # 3. One-Hot Encoding
    # We only encode specified metadata columns. 
    # Unitigs are assumed to be already binary (or numeric) and don't need OHE.
    data = pd.get_dummies(data, columns=metadata_cols, drop_first=True) # drop_first to avoid multicollinearity? 
    # Trees handle it fine, but linear models prefer it. Let's keep it simple.
    
    # 4. Define X and y
    y = data[target_col]
    
    # X should exclude target and other non-feature columns
    # We need to be careful to exclude other phenotype columns present in metadata
    # A safe way is to select unitigs + encoded metadata columns
    
    # Identify unitig columns: they are the ones that came from the Rtab file.
    # In the merged dataframe, they are the columns that are NOT from metadata.csv original columns.
    # But we don't have the original metadata columns list easily available unless we reload or hardcode.
    # Alternatively, we can drop known non-feature columns.
    
    exclude_cols = [
        'Year', 'Beta.lactamase', 'Azithromycin', 'Ciprofloxacin', 'Ceftriaxone', 
        'Cefixime', 'Tetracycline', 'Penicillin', 'NG_MAST', 'Group', 
        'azm_mic', 'cip_mic', 'cro_mic', 'cfx_mic', 'tet_mic', 'pen_mic', 
        'log2_azm_mic', 'log2_cip_mic', 'log2_cro_mic', 'log2_cfx_mic', 'log2_tet_mic', 'log2_pen_mic',
        'azm_sr', 'cip_sr', 'cro_sr', 'cfx_sr', 'tet_sr', 'pen_sr'
    ]
    
    # Also exclude the target_col (already in exclude_cols but just in case)
    
    feature_cols = [c for c in data.columns if c not in exclude_cols]
    
    X = data[feature_cols]
    
    # Ensure all features are numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0) # Fill any remaining NaNs (e.g. from coercion) with 0
    
    return X, y

def get_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Splits data into training and testing sets using stratified split.
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
