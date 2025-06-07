"""
Main execution script for natural products analysis.

This script performs the complete analysis pipeline:
1. Loads molecular data with either Chemformer or SMILES-Transformer embeddings
2. Processes each reference taxonomic chain
3. Computes chemical and taxonomic distances
4. Generates visualizations and statistical analyses
5. Saves results to specified directories

The script can be run from command line with arguments:
    --data_folder: Directory containing input data
    --encoding: Type of molecular encoding ('chemformer' or 'smitrans')
    --upper_limit_ref_size: Maximum number of molecules for reference species
    --lower_limit_ref_size: Minimum number of molecules for reference species
"""

# run this when you change utils.py to reload functions from utils.py
# import importlib
# import utils
# importlib.reload(utils)

from utils import load_data, run_all, convert_to_array, draw_pairs_of_molecules
import pandas as pd
import argparse

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Run calculations for p-value distributions.')
        parser.add_argument('--data_folder', type=str, default='./data', help='Path to the data folder.')
        parser.add_argument('--encoding', type=str, default='chemformer', help='ML Encoding used for converting SMILES to vectors.')
        parser.add_argument('--upper_limit_ref_size', type=int, default=1000, help='Include reference species with number of molecules less than this value.')
        parser.add_argument('--lower_limit_ref_size', type=int, default=180, help='Include reference species with number of molecules greater or equal than this value.')
        parser.add_argument('--size_threshold', type=int, default=15, help='Minimum number of molecules for a current species.')
        parser.add_argument('--min_size_threshold', type=int, default=20, help='Minimum number of molecules for a current species to be included in slow low-level computations.')
        parser.add_argument('--percentiles', type=str, default='10,25,40,50,60,75,90', help='Percentiles to compute.')
        parser.add_argument('--tdistance1', type=int, default=1, help='One of the taxonomic distances for statistical comparisons.')
        parser.add_argument('--tdistance2', type=int, default=3, help='The other taxonomic distance for statistical comparisons.')
        args = parser.parse_args()

        data_folder = args.data_folder
        encoding = args.encoding
        percentiles = list(map(int, args.percentiles.split(',')))
        
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
        df[encoding_columns] = df[encoding_columns].apply(convert_to_array)
        print(df[encoding_columns].head())

        df_taxonomic_chain_ref = pd.read_csv('./statistics_on_n_molecules_per_taxonomic_chain.csv')
        nmol_lower_cutoff = args.lower_limit_ref_size
        nmol_upper_cutoff = args.upper_limit_ref_size
        taxonomic_chains_ref = df_taxonomic_chain_ref[
                (df_taxonomic_chain_ref['nmol'] < nmol_upper_cutoff) & 
                (df_taxonomic_chain_ref['nmol'] >= nmol_lower_cutoff)
        ]['taxonomic_chain'].values

        for taxonomic_chain_ref in taxonomic_chains_ref:
                print(f"Starting with taxonomic chain: {taxonomic_chain_ref}")
                try:
                        run_all(df,
                                encoding=encoding,
                                encoding_columns=encoding_columns,
                                ref_chain=taxonomic_chain_ref.split('-'),
                                max_taxonomic_distance=3,   # 2 for debugging, 3 for production runs
                                size_threshold=args.size_threshold,
                                min_size_threshold=args.min_size_threshold,
                                distance_metric='Euclidean',
                                percentiles=percentiles,
                                tdistance1=args.tdistance1,
                                tdistance2=args.tdistance2,
                                verbose=True,
                                calc_stats_chemical_distances_vs_taxonomic_distances=False,
                                save_dataframes_to_folder=data_folder,
                                save_data_for_percentiles_to_folder=f"{data_folder}/data_for_percentiles",
                                enforce_low_level_recomputations=False,
                        )
                except Exception as e:
                        print(f"Error for {taxonomic_chain_ref}:")
                        print(e)
                        print()
                        pass
