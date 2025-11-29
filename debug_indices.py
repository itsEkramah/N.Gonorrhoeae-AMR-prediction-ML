import pandas as pd

def check_indices():
    metadata = pd.read_csv('DATA/metadata.csv')
    print("Metadata Sample_IDs (first 5):")
    print(metadata['Sample_ID'].head().tolist())
    
    rtab = pd.read_csv('DATA/azm_sr_gwas_filtered_unitigs.Rtab', sep='\t', index_col=0).T
    print("Unitig Indices (first 5):")
    print(rtab.index[:5].tolist())
    
    # Check intersection
    common = set(metadata['Sample_ID']).intersection(set(rtab.index))
    print(f"Number of common IDs: {len(common)}")

if __name__ == "__main__":
    check_indices()
