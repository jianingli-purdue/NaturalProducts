from utils import load_data, convert_to_array, plot_tsne_results
import argparse
from sklearn.manifold import TSNE
import numpy as np
from sklearn.decomposition import PCA

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Visualize chem space after dimensionality reduction with tSNE.')
        parser.add_argument('--data_folder', type=str, default='./data', help='Path to the data folder, which contains the csv file with embeddings.')
        parser.add_argument('--encoding', type=str, default='chemformer', help='ML Encoding used for converting SMILES to vectors.')
        parser.add_argument('--output_folder', type=str, default='./tSNE_plots', help='Path to the output folder for tSNE plots.')
        parser.add_argument('--td_selection', type=int, default=3, help='Max taxonomic distance to consider for selection of datapoints to build tSNE map.')
        parser.add_argument('--size_threshold', type=int, default=15, help='Minimum number of molecules in a current species to include it into analysis.')
        parser.add_argument('--reference_species', type=str, default='Eukaryota-Viridiplantae-Streptophyta-Magnoliopsida-Celastraceae-Tripterygium-Tripterygium wilfordii', help='Reference species.')
        parser.add_argument('--reference_smiles', type=str, default='CC(=O)OC1C2C(OC(=O)c3ccccc3)C(OC(=O)c3ccccc3)C3(C)C(OC(=O)c4ccccc4)C(OC(=O)c4ccccc4)CC(C)C13OC2(C)C', help='Reference SMILES from the reference species.')
        args = parser.parse_args()

        data_folder = args.data_folder
        encoding = args.encoding
        output_folder = args.output_folder
        td_selection = args.td_selection
        reference_species = args.reference_species
        reference_smiles = args.reference_smiles
        size_threshold = args.size_threshold

        # data_folder = 'C:\\Users\\anton\\Desktop\\c\\JianingLi\\data'
        # output_folder = 'C:\\Users\\anton\\Desktop\\c\\JianingLi\\tSNE_plots'

        parts = reference_species.split('-')
        taxonomic_chain_ref_short = '-'.join(parts[-3:])
        selection = '-'.join(parts[:-td_selection])

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
        df = df.reset_index(drop=True)
        df[encoding_columns] = df[encoding_columns].apply(convert_to_array)
        print(df[encoding_columns].head())

        vecs = df['latent_vector'].values
        X = np.vstack(vecs)
        print(f"Shape of X (feature matrix): {X.shape}")

        # compute importance of PCs
        pca = PCA()
        pca.fit(X)
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        print("Explained variance ratio per PC:", explained_variance[0:10])
        print("Cumulative explained variance:", cumulative_variance[0:10])

        # run tSNE analysis
        tsne = TSNE(n_components=2, random_state=42)
        X_embedded = tsne.fit_transform(X)
        vecs_tsne = X_embedded.tolist()
        df['tsne_1'] = [v[0] for v in vecs_tsne]
        df['tsne_2'] = [v[1] for v in vecs_tsne]
        plot_tsne_results(df, reference_species=reference_species, dotsize='small', png_filename=f'{output_folder}/tsne_ref_{taxonomic_chain_ref_short}.png')

        # Parse reference_species for the names of class, family, genus, species
        parts = reference_species.split('-')
        ref_class = parts[-4] if len(parts) >= 4 else None
        ref_family = parts[-3] if len(parts) >= 3 else None
        ref_genus = parts[-2] if len(parts) >= 2 else None
        ref_species = parts[-1] if len(parts) >= 1 else None

        # In each species, select the molecule the closest to the given reference SMILES
        df_reference_smiles = df[df[colname_w_smiles] == reference_smiles]
        if df_reference_smiles.empty:
                raise ValueError(f"Reference SMILES '{reference_smiles}' not found in the dataset.")
        
        # When we built the tSNE map, we used all SMILES from `selection`, but now we'll keep only SMILES from species with at least `size_threshold` NPs
        file_with_n_molecules_per_taxonomic_chain = f'{data_folder}/n_molecules_per_taxonomic_chain.json'
        import json
        with open(file_with_n_molecules_per_taxonomic_chain, 'r') as f:
                nsmiles_per_taxonomic_chain = json.loads(f.read())

        df = df[df['taxonomic_chain'].apply(lambda x: nsmiles_per_taxonomic_chain[x]) >= size_threshold].copy()
                
        vec0 = df_reference_smiles['latent_vector'].values[0]
        vec0_tsne = df_reference_smiles['tsne_1'].values[0], df_reference_smiles['tsne_2'].values[0]
        print("Reference SMILES:")
        print(f"Full vector: {vec0[0]:.3f}, {vec0[1]:.3f}, {vec0[2]:.3f}, ..., {vec0[-1]:.3f}")
        print(f"tSNE vector: {vec0_tsne[0]:.3f}, {vec0_tsne[1]:.3f}")
        df['r0'] = df[encoding_columns].apply(lambda x: np.linalg.norm(np.array(x) - vec0))
        indices_for_min_r0 = df.groupby('taxonomic_chain')['r0'].idxmin()
        dfvicinity = df.loc[indices_for_min_r0].reset_index(drop=True)
        dfvicinity['r_tsne'] = dfvicinity.apply(lambda x: np.linalg.norm(np.array((x['tsne_1'], x['tsne_2'])) - np.array(vec0_tsne)), axis=1)
        rs_for_td1 = dfvicinity[(dfvicinity['genus'] == ref_genus) & (dfvicinity['species'] != ref_species)]['r_tsne']
        print(f"Distances in tSNE space for td1: {rs_for_td1}")
        radii = [
                rs_for_td1.median(),
                dfvicinity[(dfvicinity['family'] == ref_family) & (dfvicinity['genus'] != ref_genus)]['r_tsne'].median(),
                dfvicinity[(dfvicinity['classx'] == ref_class) & (dfvicinity['family'] != ref_family)]['r_tsne'].median(),
        ]
        plot_tsne_results(dfvicinity, reference_species=reference_species, 
                        dotsize='large', center_of_circles=vec0_tsne, radii=radii,
                        png_filename=f'{output_folder}/tsne_ref_{taxonomic_chain_ref_short}_smiles_{reference_smiles}.png')
        # dfvicinity.sort_values(by='r0')[['family', 'genus', 'species', colname_w_smiles, 'r0']]

        print(f"td1, td2, and td3 vs distance in 2D tSNE space: {radii}")
        print("td1, td2, and td3 vs chem distance in 1024D space:")
        print(
                dfvicinity[(dfvicinity['genus'] == ref_genus) & (dfvicinity['species'] != ref_species)]['r0'].median(),
                dfvicinity[(dfvicinity['family'] == ref_family) & (dfvicinity['genus'] != ref_genus)]['r0'].median(),
                dfvicinity[(dfvicinity['classx'] == ref_class) & (dfvicinity['family'] != ref_family)]['r0'].median(),
        )
