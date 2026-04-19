import os
import pandas as pd

def process_drug_pairs():
    """
    Reads drug_smiles and DDI data, merges them, and saves the result.
    """
    raw_dir = os.path.join('data', 'raw')
    processed_dir = os.path.join('data', 'processed')
    
    # Load files
    print("Loading drug_smiles.csv ...")
    smiles_df = pd.read_csv(os.path.join(raw_dir, 'drug_smiles.csv'))
    
    print("Loading ddis.csv ...")
    ddis_df = pd.read_csv(os.path.join(raw_dir, 'ddis.csv'))
    
    # We only need the 'drug_id' and 'smiles' columns from the smiles dataframe
    smiles_subset = smiles_df[['drug_id', 'smiles']]
    
    # Merge for drug1
    merged_df = ddis_df.merge(
        smiles_subset, 
        left_on='d1', 
        right_on='drug_id', 
        how='left'
    ).rename(columns={'smiles': 'smiles1'}).drop(columns=['drug_id'])
    
    # Merge for drug2
    merged_df = merged_df.merge(
        smiles_subset, 
        left_on='d2', 
        right_on='drug_id', 
        how='left'
    ).rename(columns={'smiles': 'smiles2'}).drop(columns=['drug_id'])
    
    # Select the required columns: drug1, drug2, smiles1, smiles2, interaction_label
    # d1 -> drug1, d2 -> drug2, type -> interaction_label
    merged_df = merged_df.rename(columns={
        'd1': 'drug1',
        'd2': 'drug2',
        'type': 'interaction_label'
    })
    
    final_df = merged_df[['drug1', 'drug2', 'smiles1', 'smiles2', 'interaction_label']]
    
    # Save processed pairs
    output_path = os.path.join(processed_dir, 'drug_pairs.csv')
    print(f"Saving merged pairs to {output_path} ...")
    final_df.to_csv(output_path, index=False)
    
    print("\nFirst 5 rows of processed data:")
    print(final_df.head())

if __name__ == '__main__':
    process_drug_pairs()
