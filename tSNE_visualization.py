from utils import load_data, convert_to_array
import pandas as pd
import argparse
from sklearn.manifold import TSNE
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
# reference_smiles = "CC(=O)OCC12C(OC(=O)C=Cc3ccccc3)CCC(C)(O)C13OC(C)(C)C(C(OC(C)=O)C2OC(=O)c1ccccc1)C3OC(C)=O"
# reference_smiles = "CC(=O)O[C@H]1[C@@H]2OC(=O)[C@@H](C)[C@H](C)c3ccncc3C(=O)OC[C@]3(C)O[C@@]4([C@H](OC(C)=O)[C@H]3[C@@H](OC=O)[C@@H](OC(C)=O)[C@]4(COC(=O)c3ccoc3)[C@H]1OC(C)=O)C2(C)O"
# reference_smiles = "CC(C)c1ccc2c(c1O)CC[C@H]1C3=COC(=O)C3=CC[C@]21C"
# reference_smiles = "COc1c(C(C)C)cc(O)c2c1CC[C@H]1C3=C(CC[C@]21C)C(=O)OC3"

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

df = load_data(file_path=data_file, colname_w_smiles=colname_w_smiles, colname_w_features=encoding_columns, top_rows=None, compute_ECFP_fingerprints=False)
df = df[df['taxonomic_chain'].str.startswith(selection)].copy()
df[encoding_columns] = df[encoding_columns].apply(convert_to_array)
print(df[encoding_columns].head())

df_reference_smiles = df[df[colname_w_smiles] == reference_smiles]
if df_reference_smiles.empty:
    raise ValueError(f"Reference SMILES '{reference_smiles}' not found in the dataset.")
vec0 = df_reference_smiles['latent_vector'].values[0]

vecs = df['latent_vector'].values
X = np.vstack(vecs)
print(X.shape)

# run tsne analysis
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X)
vecs_tsne = X_embedded.tolist()
df['tsne_1'] = [v[0] for v in vecs_tsne]
df['tsne_2'] = [v[1] for v in vecs_tsne]

parts = reference_species.split('-')
genus_prefix = '-'.join(parts[:-1]) + '-'
family_prefix = '-'.join(parts[:-2]) + '-'
colors = []
for idx, row in df.iterrows():
        if row['taxonomic_chain'] == reference_species and row[colname_w_smiles] == reference_smiles:
                colors.append('black')
        elif row['taxonomic_chain'] == reference_species:
                colors.append('gray')
        elif row['taxonomic_chain'].startswith(genus_prefix):
                colors.append('green')
        elif row['taxonomic_chain'].startswith(family_prefix):
                colors.append('red')
        else:
                colors.append('yellow')
category_order = [
        'Other families',
        'Other genus in family',
        'Other species in genus',
        'Reference species, other SMILES',
        'Reference species & SMILES'
]
color_map = {
        'Reference species & SMILES': 'black',
        'Reference species, other SMILES': 'black',
        'Other species in genus': 'green',
        'Other genus in family': 'red',
        'Other families': 'yellow'
}

categories = []
for idx, row in df.iterrows():
        if row['taxonomic_chain'] == reference_species and row[colname_w_smiles] == reference_smiles:
                categories.append('Reference species & SMILES')
        elif row['taxonomic_chain'] == reference_species:
                categories.append('Reference species, other SMILES')
        elif row['taxonomic_chain'].startswith(genus_prefix):
                categories.append('Other species in genus')
        elif row['taxonomic_chain'].startswith(family_prefix):
                categories.append('Other genus in family')
        else:
                categories.append('Other families')

# Plot each category in order, so "Other families" is at the bottom
plt.figure(figsize=(10, 8))
for cat in category_order:
        idxs = [i for i, c in enumerate(categories) if c == cat]
        size = 60 if cat == 'Reference species & SMILES' else 10
        plt.scatter(
                [vecs_tsne[i][0] for i in idxs],
                [vecs_tsne[i][1] for i in idxs],
                c=color_map[cat],
                alpha=0.7,
                label=cat,
                edgecolors='none',
                s=size,
                zorder=category_order.index(cat)  # ensure later categories are drawn on top
        )

handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[cat], markersize=8, label=cat)
        for cat in reversed(category_order)
]
plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.xlabel('tSNE-1')
plt.ylabel('tSNE-2')
plt.tight_layout()
plt.show()

# compute importance of PCs
pca = PCA()
pca.fit(X)
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
print("Explained variance ratio per PC:", explained_variance[0:10])
print("Cumulative explained variance:", cumulative_variance[0:10])

# vicinity of vec0
df['r0'] = df[encoding_columns].apply(lambda x: np.linalg.norm(np.array(x) - vec0))
indices_for_min_r0 = df.groupby('taxonomic_chain')['r0'].idxmin()
dfvicinity = df.loc[indices_for_min_r0].reset_index(drop=True)

parts = reference_species.split('-')
genus_prefix = '-'.join(parts[:-1]) + '-'
family_prefix = '-'.join(parts[:-2]) + '-'
colors = []
for idx, row in dfvicinity.iterrows():
        if row['taxonomic_chain'] == reference_species and row[colname_w_smiles] == reference_smiles:
                colors.append('black')
        # elif row['taxonomic_chain'] == reference_species:
        #         colors.append('gray')
        elif row['taxonomic_chain'].startswith(genus_prefix):
                colors.append('green')
        elif row['taxonomic_chain'].startswith(family_prefix):
                colors.append('red')
        else:
                colors.append('yellow')
category_order = [
        'Other families',
        'Other genus in family',
        'Other species in genus',
        # 'Reference species, other SMILES',
        'Reference species & SMILES'
]
color_map = {
        'Reference species & SMILES': 'black',
        # 'Reference species, other SMILES': 'black',
        'Other species in genus': 'green',
        'Other genus in family': 'red',
        'Other families': 'yellow'
}

categories = []
for idx, row in dfvicinity.iterrows():
        if row['taxonomic_chain'] == reference_species and row[colname_w_smiles] == reference_smiles:
                categories.append('Reference species & SMILES')
        # elif row['taxonomic_chain'] == reference_species:
        #         categories.append('Reference species, other SMILES')
        elif row['taxonomic_chain'].startswith(genus_prefix):
                categories.append('Other species in genus')
        elif row['taxonomic_chain'].startswith(family_prefix):
                categories.append('Other genus in family')
        else:
                categories.append('Other families')

# Plot each category in order, so "Other families" is at the bottom
plt.figure(figsize=(10, 8))
for cat in category_order:
        idxs = [i for i, c in enumerate(categories) if c == cat]
        size = 60 if cat == 'Reference species & SMILES' else 30
        plt.scatter(
                [dfvicinity['tsne_1'][i] for i in idxs],
                [dfvicinity['tsne_2'][i] for i in idxs],
                c=color_map[cat],
                alpha=0.7,
                label=cat,
                edgecolors='none',
                s=size,
                zorder=category_order.index(cat)  # ensure later categories are drawn on top
        )

handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[cat], markersize=8, label=cat)
        for cat in reversed(category_order)
]
plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.xlabel('tSNE-1')
plt.ylabel('tSNE-2')
plt.tight_layout()
plt.show()

dfvicinity.sort_values(by='r0')[['family', 'genus', 'species', colname_w_smiles, 'r0']]

# td1, chem distance in 1024D space
dfvicinity[(dfvicinity['genus'] == 'Tripterygium') & (dfvicinity['species']!='Tripterygium wilfordii')]['r0'].median()
# td2, chem distance in 1024D space
dfvicinity[dfvicinity['genus'] != 'Tripterygium']['r0'].median()

# td1, distance in tSNE space
df_reference_smiles = df[df[colname_w_smiles] == reference_smiles]
if df_reference_smiles.empty:
    raise ValueError(f"Reference SMILES '{reference_smiles}' not found in the dataset.")
vec0_tsne = df_reference_smiles['tsne_1'].values[0], df_reference_smiles['tsne_2'].values[0]
dfvicinity['r_tsne'] = dfvicinity.apply(lambda x: np.linalg.norm(np.array((x['tsne_1'], x['tsne_2'])) - np.array(vec0_tsne)), axis=1)
dfvicinity[(dfvicinity['genus'] == 'Tripterygium') & (dfvicinity['species']!='Tripterygium wilfordii')]['r_tsne'].median()
# td2, chem distance in 1024D space
dfvicinity[dfvicinity['genus'] != 'Tripterygium']['r_tsne'].median()