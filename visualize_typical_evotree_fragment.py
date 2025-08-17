from html import parser
from utils import load_data, convert_to_array
import argparse
import numpy as np
import json

# if __name__ == '__main__':
#         parser = argparse.ArgumentParser(description='Visualize a part of the evolutionary tree relative to a given species and SMILES.')
#         parser.add_argument('--data_folder', type=str, default='./data', help='Path to the data folder, which contains the csv file with embeddings.')
#         parser.add_argument('--encoding', type=str, default='chemformer', help='ML Encoding used for converting SMILES to vectors.')
#         parser.add_argument('--td_selection', type=int, default=3, help='Max taxonomic distance to consider for selection of datapoints to build the part of the tree.')
#         parser.add_argument('--size_threshold', type=int, default=15, help='Minimum number of molecules in a current species to include it into analysis.')
#         parser.add_argument('--species_for_comparisons', type=str, default='Eukaryota-Viridiplantae-Streptophyta-Magnoliopsida-Asteraceae-Petasites-Petasites japonicus', help='Species for comparisons.')
#         parser.add_argument('--smiles_for_comparisons', type=str, default='CSC=CC(=O)OC1CC(C)C2(C)CC(=C(C)C)C(=O)CC2C1', help=' SMILES from the species for comparisons.')
#         args = parser.parse_args()

#         data_folder = args.data_folder
#         encoding = args.encoding
#         td_selection = args.td_selection
#         species_for_comparisons = args.species_for_comparisons
#         smiles_for_comparisons = args.smiles_for_comparisons
#         size_threshold = args.size_threshold

data_folder = 'C:\\Users\\anton\\Desktop\\c\\JianingLi\\data'
encoding = 'chemformer'
td_selection = 3
species_for_comparisons = 'Eukaryota-Viridiplantae-Streptophyta-Magnoliopsida-Asteraceae-Petasites-Petasites japonicus'
smiles_for_comparisons = 'CSC=CC(=O)OC1CC(C)C2(C)CC(=C(C)C)C(=O)CC2C1'
size_threshold = 15

parts = species_for_comparisons.split('-')
taxonomic_chain_ref_short = '-'.join(parts[-3:])
selection = '-'.join(parts[:-td_selection])
ref_class = parts[-4] if len(parts) >= 4 else None
ref_family = parts[-3] if len(parts) >= 3 else None
ref_genus = parts[-2] if len(parts) >= 2 else None
ref_species = parts[-1] if len(parts) >= 1 else None

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

data_file = f'{data_folder}/magnoliopsida_latent_matrix_Chemformer.csv'

df = load_data(file_path=data_file, colname_w_smiles=colname_w_smiles, colname_w_features=encoding_columns, top_rows=None, compute_ECFP_fingerprints=False)
df = df[df['taxonomic_chain'].str.startswith(selection)].copy()
df = df.reset_index(drop=True)
df[encoding_columns] = df[encoding_columns].apply(convert_to_array)
print(df[encoding_columns].head())

# Keep only species with the number of NPs above a threshold
file_with_n_molecules_per_taxonomic_chain = f'{data_folder}/n_molecules_per_taxonomic_chain.json'
with open(file_with_n_molecules_per_taxonomic_chain, 'r') as f:
        nsmiles_per_taxonomic_chain = json.loads(f.read())

df = df[df['taxonomic_chain'].apply(lambda x: nsmiles_per_taxonomic_chain[x]) >= size_threshold].copy()


# In each species, select the molecule the closest to the given SMILES for comparisons
df_smiles_for_comparisons = df[df[colname_w_smiles] == smiles_for_comparisons]
if df_smiles_for_comparisons.empty:
        raise ValueError(f"SMILES from the species for comparisons '{smiles_for_comparisons}' not found in the dataset.")

vec0 = df_smiles_for_comparisons['latent_vector'].values[0]
print("SMILES from the species for comparisons:")
print(f"Full vector: {vec0[0]:.3f}, {vec0[1]:.3f}, {vec0[2]:.3f}, ..., {vec0[-1]:.3f}")
df['r0'] = df[encoding_columns].apply(lambda x: np.linalg.norm(np.array(x) - vec0))
indices_for_min_r0 = df.groupby('taxonomic_chain')['r0'].idxmin()
dfvicinity = df.loc[indices_for_min_r0].reset_index(drop=True)
dfvicinity = dfvicinity.sort_values(by='r0', ascending=False) # when the median value below is between two specific values, this sorting ensures the choice of the bigger r0, and therefore doen't overestimate the performance


df_td1 = dfvicinity[(dfvicinity['genus'] == ref_genus) & (dfvicinity['species'] != ref_species)].copy().reset_index(drop=True)
df_td2 = dfvicinity[(dfvicinity['family'] == ref_family) & (dfvicinity['genus'] != ref_genus)].copy().reset_index(drop=True)
df_td3 = dfvicinity[(dfvicinity['classx'] == ref_class) & (dfvicinity['family'] != ref_family)].copy().reset_index(drop=True)

chem_dist_td1 = df_td1['r0'].median()
idx_median_td1 = df_td1['r0'].sub(chem_dist_td1).abs().idxmin()
chem_dist_td1, smiles_median_td1, tax_chain_dt1 = df_td1[['r0', colname_w_smiles, 'taxonomic_chain']].values[idx_median_td1] # redo chem dist to make sure it's one of r0s, and not the mean of two central r0s
tax_chain_dt1_short = '-'.join(tax_chain_dt1.split('-')[-3:])
print(f"td1: chem distance in full space: {chem_dist_td1:.3f}, {tax_chain_dt1_short}, SMILES: {smiles_median_td1}")

chem_dist_td2 = df_td2['r0'].median()
idx_median_td2 = df_td2['r0'].sub(chem_dist_td2).abs().idxmin()
chem_dist_td2, smiles_median_td2, tax_chain_dt2 = df_td2[['r0', colname_w_smiles, 'taxonomic_chain']].values[idx_median_td2]
tax_chain_dt2_short = '-'.join(tax_chain_dt2.split('-')[-3:])
print(f"td2: chem distance in full space: {chem_dist_td2:.3f}, {tax_chain_dt2_short}, SMILES: {smiles_median_td2}")

chem_dist_td3 = df_td3['r0'].median()
idx_median_td3 = df_td3['r0'].sub(chem_dist_td3).abs().idxmin()
chem_dist_td3, smiles_median_td3, tax_chain_dt3 = df_td3[['r0', colname_w_smiles, 'taxonomic_chain']].values[idx_median_td3]
tax_chain_dt3_short = '-'.join(tax_chain_dt3.split('-')[-3:])
print(f"td3: chem distance in full space: {chem_dist_td3:.3f}, {tax_chain_dt3_short}, SMILES: {smiles_median_td3}")

# draw chem structures from smiles_median_td1,2,3
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
mol_0 = Chem.MolFromSmiles(smiles_for_comparisons)
mol_1 = Chem.MolFromSmiles(smiles_median_td1)
mol_2 = Chem.MolFromSmiles(smiles_median_td2)
mol_3 = Chem.MolFromSmiles(smiles_median_td3)
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].imshow(Draw.MolToImage(mol_0))
axes[0, 0].set_title("Initial molecule")
axes[0, 0].axis('off')
axes[0, 1].imshow(Draw.MolToImage(mol_1))
axes[0, 1].set_title("TD1")
axes[0, 1].axis('off')
axes[1, 0].imshow(Draw.MolToImage(mol_2))
axes[1, 0].set_title("TD2")
axes[1, 0].axis('off')
axes[1, 1].imshow(Draw.MolToImage(mol_3))
axes[1, 1].set_title("TD3")
axes[1, 1].axis('off')
plt.show()