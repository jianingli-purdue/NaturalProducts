from utils import load_data, convert_to_array
import pandas as pd
import argparse

# if __name__ == '__main__':
#         parser = argparse.ArgumentParser(description='Visualize chem space after dimensionality reduction with tSNE.')
#         parser.add_argument('--data_folder', type=str, default='./data', help='Path to the data folder.')
#         parser.add_argument('--encoding', type=str, default='chemformer', help='ML Encoding used for converting SMILES to vectors.')
#         args = parser.parse_args()

#         data_folder = args.data_folder
#         encoding = args.encoding
        
        

encoding='chemformer'
data_folder = 'C:\\Users\\anton\\Desktop\\c\\JianingLi\\data'
selection = "Eukaryota-Viridiplantae-Streptophyta-Magnoliopsida-Celastraceae-"
reference_species = "Eukaryota-Viridiplantae-Streptophyta-Magnoliopsida-Celastraceae-Tripterygium-Tripterygium wilfordii"
reference_smiles = "CC(=O)OC1C2C(OC(=O)c3ccccc3)C(OC(=O)c3ccccc3)C3(C)C(OC(=O)c4ccccc4)C(OC(=O)c4ccccc4)CC(C)C13OC2(C)C"
reference_smiles = "CC1(C)O[C@]23[C@H](OC(=O)c4cccnc4)[C@H]1C[C@@H](OC(=O)C=Cc1ccccc1)[C@]2(CO)[C@@H](OC(=O)c1ccccc1)CC[C@]3(C)O"

colname_w_smiles='canonical_smiles'  # name of the column in the input file, may be not canonicalized smiles
encoding_columns='latent_vector'   # if features (aka embeddings) are already available in the input file, specify the column name here

filename_from_encoding = {
        'chemformer': 'Lotus_fulldata_latent_matrix_Chemformer.csv',
        'smitrans': 'Lotus_fulldata_latent_matrix_SMILES-TRANSFORMER.csv',
        'SELformer': 'Lotus_fulldata_SELformer.csv',
        'nyan': 'Lotus_fulldata_latent_matrix_nyan.csv',
        'molvae': 'Lotus.csv',
}

if encoding in filename_from_encoding:
        data_file = f'{data_folder}/{filename_from_encoding[encoding]}'
else:
        raise ValueError(f"Unknown encoding {encoding}")

df = load_data(file_path=data_file, colname_w_smiles=colname_w_smiles, colname_w_features=encoding_columns, top_rows=1000, compute_ECFP_fingerprints=False)
df = df[df['taxonomic_chain'].str.startswith(selection)].copy()
df[encoding_columns] = df[encoding_columns].apply(convert_to_array)
print(df[encoding_columns].head())

df_reference_smiles = df[df[colname_w_smiles] == reference_smiles]
if df_reference_smiles.empty:
    raise ValueError(f"Reference SMILES '{reference_smiles}' not found in the dataset.")
vec0 = df_reference_smiles['latent_vector'].values[0]

vecs = df['latent_vector'].values