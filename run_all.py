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
# from utils import load_data, run_all, convert_to_array

from utils import load_data, run_all, convert_to_array
import pandas as pd
import argparse

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Run calculations for p-value distributions.')
        parser.add_argument('--data_folder', type=str, default='./data', help='Path to the data folder.')
        parser.add_argument('--dataset', type=str, default='coconut', help='Name of the dataset to use.')
        parser.add_argument('--encoding', type=str, default='ecfp', help='ML Encoding used for converting SMILES to vectors.')
        parser.add_argument('--upper_limit_ref_size', type=int, default=10000, help='Include reference species with number of molecules less than this value.')
        parser.add_argument('--lower_limit_ref_size', type=int, default=1000, help='Include reference species with number of molecules greater or equal than this value.')
        parser.add_argument('--size_threshold', type=int, default=20, help='Minimum number of molecules for a current species.')
        parser.add_argument('--min_size_threshold', type=int, default=20, help='Minimum number of molecules for a current species to be included in slow low-level computations.')
        parser.add_argument('--percentiles', type=str, default='10,25,40,50,60,75,90', help='Percentiles to compute.')
        parser.add_argument('--evo_distance_type', type=str, default='continuous', help='Type of evolutionary distance: discrete for taxonomic distance, continuous for time-calibrated evolutionary distance.')
        parser.add_argument('--tdistance1', type=int, default=1, help='One of the taxonomic distances for statistical comparisons.')
        parser.add_argument('--tdistance2', type=int, default=3, help='The other taxonomic distance for statistical comparisons.')
        parser.add_argument("--evo_distances", default="./data/all_species_distances_upper_triangle_evo_distance_upto200.csv", help="Path to the csv file with evolutionary distances between species")
        parser.add_argument("--max_evo_distance", type=float, default=200., help="Maximum evolutionary distance to consider for computing chemical distances.")
        args = parser.parse_args()

        data_folder = args.data_folder
        dataset = args.dataset
        encoding = args.encoding
        percentiles = list(map(int, args.percentiles.split(',')))
        size_threshold = args.size_threshold
        min_size_threshold = args.min_size_threshold
        evo_distance_type = args.evo_distance_type
        evo_distances_file = args.evo_distances
        max_evo_distance = args.max_evo_distance
        nmol_lower_cutoff = args.lower_limit_ref_size
        nmol_upper_cutoff = args.upper_limit_ref_size
        tdistance1 = args.tdistance1
        tdistance2 = args.tdistance2
        
        # data_folder = './data'
        # dataset = 'coconut'
        # encoding = 'ecfp'
        # percentiles = [10, 25, 40, 50, 60, 75, 90]
        # size_threshold = 20
        # min_size_threshold = 20
        # evo_distance_type = 'discrete'#"continuous"
        # nmol_lower_cutoff =1000
        # nmol_upper_cutoff =10000
        # tdistance1 = 2
        # tdistance2 = 3
        # evo_distances_file = "./data/all_species_distances_upper_triangle_evo_distance_upto200.csv"
        # max_evo_distance = 200.
        
        colname_w_smiles='canonical_smiles'  # name of the column in the input file, may be not canonicalized smiles
        
        if dataset == 'lotus':
                filename_from_encoding = {
                        'chemformer': 'Lotus_fulldata_latent_matrix_Chemformer.csv',
                        'smitrans': 'Lotus_fulldata_latent_matrix_SMILES-TRANSFORMER.csv',
                        'SELformer': 'Lotus_fulldata_SELformer.csv',
                        'nyan': 'Lotus_fulldata_latent_matrix_nyan.csv',
                        'molvae': 'Lotus.csv',
                }
                taxonomic_levels = ['superkingdom', 'kingdom', 'phylum', 'classx', 'family', 'genus', 'species']    
                encoding_columns = 'latent_vector'   # if features (aka embeddings) are already available in the input file, specify the column name here
        elif dataset == 'coconut':
                filename_from_encoding = {
                        'ecfp': 'coconut_on_tree_51w_no_metal_and_salt.csv',
                }
                taxonomic_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'clean_species_name']
                encoding_columns = None
        else:
                raise ValueError(f"Unknown dataset {dataset}")

        if encoding in filename_from_encoding:
                data_file = f'{data_folder}/{filename_from_encoding[encoding]}'
        else:
                raise ValueError(f"Unknown encoding {encoding}")
        
        df = load_data(
                file_path=data_file, 
                colname_w_smiles=colname_w_smiles,
                colname_w_features=(encoding_columns if encoding != 'ecfp' else None), 
                top_rows=None, 
                compute_ECFP_fingerprints=(encoding=='ecfp'),
                taxonomic_levels = taxonomic_levels
                )
        
        if encoding == 'ecfp':
                encoding_columns = "ECFP6_2048"
        else:
                df[encoding_columns] = df[encoding_columns].apply(convert_to_array)
        print(df[encoding_columns].head())

        df_taxonomic_chain_ref = pd.read_csv('./statistics_on_n_molecules_per_taxonomic_chain.csv')
        taxonomic_chains_ref = df_taxonomic_chain_ref[
                (df_taxonomic_chain_ref['nmol'] < nmol_upper_cutoff) & 
                (df_taxonomic_chain_ref['nmol'] >= nmol_lower_cutoff)
        ]['taxonomic_chain'].values
        
        if evo_distance_type == 'discrete':
                td_params = {
                        'evo_distance_type': 'discrete',
                        'max_taxonomic_distance': 3,   # 2 for debugging, 3 for production runs
                        'tdistance1': tdistance1,
                        'tdistance2': tdistance2,
                        }
        elif evo_distance_type == 'continuous':                
                df_evo_distances = pd.read_csv(evo_distances_file, usecols=["distance", "tax_lineage_name_1", "tax_lineage_name_2"])
                df_evo_distances = df_evo_distances[df_evo_distances["distance"] <= max_evo_distance]
                td_params = {
                        'evo_distance_type': 'continuous',
                        'max_evo_distance': max_evo_distance,
                        'df_evo_distances': df_evo_distances,
                        }
                
        for taxonomic_chain_ref in taxonomic_chains_ref:
                print(f"Starting with taxonomic chain: {taxonomic_chain_ref}")
                try:
                        run_all(df,
                                encoding=encoding,
                                encoding_columns=encoding_columns,
                                ref_chain=taxonomic_chain_ref.split('-'),
                                td_params = td_params,
                                size_threshold=size_threshold,
                                min_size_threshold=min_size_threshold,
                                distance_metric=('Euclidean' if encoding != 'ecfp' else 'Tanimoto'),
                                percentiles=percentiles,                                
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
