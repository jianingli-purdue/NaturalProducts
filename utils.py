import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Draw
import matplotlib.pyplot as plt
from scipy import stats
import ast
import math
import json


def convert_to_array(x):
    """
    Convert a string representation of a list/array into a numpy array.
    
    Args:
        x (str): String representation of a list or array
        
    Returns:
        numpy.ndarray or None: Converted array if successful, None if conversion fails
    """
    try:
        return np.array(ast.literal_eval(x), dtype=float)
    except (ValueError, SyntaxError):
        print(f"Could not convert {x}")
        return None  # Return None if the conversion fails

def load_data(
        file_path='./Lotus.csv',
        colname_w_smiles='smiles', # name of the column in the input file, may be not canonicalized smiles
        colname_w_features=None,   # if features (aka embeddings) are already available in the input file, specify the column name here
        top_rows=None,             # if None, the whole csv file will be loaded, otherwise the given number of top rows
        compute_ECFP_fingerprints=False,
):
    """
    Load and preprocess molecular data from a CSV file.
    
    Args:
        file_path (str): Path to the input CSV file
        colname_w_smiles (str): Column name containing SMILES strings
        colname_w_features (str, optional): Column name containing pre-computed molecular features
        top_rows (int, optional): Number of rows to load (None for all)
        compute_ECFP_fingerprints (bool): Whether to compute Morgan fingerprints
        
    Returns:
        pandas.DataFrame: Processed dataframe containing molecular data and taxonomic information
    """
    taxonomic_levels = ['superkingdom', 'kingdom', 'phylum', 'classx', 'family', 'genus', 'species']
    cols_to_load = [colname_w_smiles] + taxonomic_levels
    if colname_w_features:
        cols_to_load += [colname_w_features]

    df = pd.read_csv(file_path, usecols=cols_to_load, nrows=top_rows)
    df = df.dropna(subset=taxonomic_levels[6]).copy()
    df['taxonomic_chain'] = df[taxonomic_levels].astype(str).agg('-'.join, axis=1)
    unique_taxonomies = df['taxonomic_chain'].unique()
    smiles_colname = 'canonicalized_smiles'
    df[smiles_colname] = df[colname_w_smiles].apply(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True))
    print(f"The size of the dataset {len(df)}, {len(unique_taxonomies)} unique species found")
    print(f"{sum(df[smiles_colname] != df[colname_w_smiles])} molecules had non-canonical SMILES, changing them to canonical.")

    if compute_ECFP_fingerprints:
        radius = 3
        nBits = 2048
        dict_SMILES_to_fp = dict()
        for smi in df[smiles_colname].unique():
            try:
                mol = Chem.MolFromSmiles(smi)
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
                dict_SMILES_to_fp[smi] = np.array(fp)
            except:
                print(f"WARNING: couldn't compute fingerprints for SMILES {smi}, skipping it")
                dict_SMILES_to_fp[smi] = np.nan

        fp_colname = f'ECFP{2 * radius}_{nBits}'
        df[fp_colname] = df[smiles_colname].apply(lambda smi: dict_SMILES_to_fp[smi])
        nbefore = len(df)
        df = df.dropna(subset=fp_colname).copy()
        print(f"After skipping rows without fingerprints, the dataset size decreases from {nbefore} to {len(df)}")

    return df


def chemical_distance(df, taxonomic_chain, taxonomic_chain_ref,
                      size_threshold=20,
                      encoding_columns='ECFP6_2048',
                      distance_metric='Tanimoto',
                      smiles_colname='canonicalized_smiles',
                      save_data_for_percentiles_to_folder=None,
                      encoding='ECFP6_2048',
                      ):
    """
    Calculate chemical distances between molecules from two taxonomic groups.
    
    Args:
        df (pandas.DataFrame): Input dataframe with molecular data
        taxonomic_chain (str): Target taxonomic chain
        taxonomic_chain_ref (str): Reference taxonomic chain
        size_threshold (int): Minimum number of molecules required
        encoding_columns (str): Column containing molecular encodings
        distance_metric (str): Type of distance metric ('Tanimoto', 'Cosine', or 'Euclidean')
        smiles_colname (str): Column name containing SMILES strings
        save_data_for_percentiles_to_folder (str, optional): Directory to save percentile data
        encoding (str): Type of molecular encoding used
        
    Returns:
        tuple: (25th percentile distance, reference SMILES, current SMILES, 
               50th percentile distance, reference SMILES, current SMILES)
    """
    # check whether there are enough different molecules in the current set
    set_current = set(df[df['taxonomic_chain'] == taxonomic_chain][smiles_colname])
    if len(set_current) < size_threshold:
        return None, None, None, None, None, None

    # Extract relevant embeddings
    vecs_current = df[df['taxonomic_chain'] == taxonomic_chain][encoding_columns]
    smiles_current = df[df['taxonomic_chain'] == taxonomic_chain][smiles_colname]
    vecs_reference = df[df['taxonomic_chain'] == taxonomic_chain_ref][encoding_columns]
    smiles_reference = df[df['taxonomic_chain'] == taxonomic_chain_ref][smiles_colname]

    # Create a list to store distances between the current molecule and the reference set
    distances_df = pd.DataFrame(columns=['smi_current', 'smi_ref', 'distance'], dtype=object)

    # Loop over the current set molecules and find the closest molecule in the reference set for each
    for vec_current, smi_current in zip(vecs_current, smiles_current):
        closest_distance = float('Inf')
        closest_smi_ref = None

        for vec_reference, smi_reference in zip(vecs_reference, smiles_reference):
            if distance_metric == 'Tanimoto':
                dot_product = np.dot(vec_current, vec_reference)
                distance = 1. - dot_product / (np.sum(vec_current) + np.sum(vec_reference) - dot_product)
            elif distance_metric == 'Cosine':
                distance = 1. - np.dot(vec_current, vec_reference) / (np.linalg.norm(vec_current) * np.linalg.norm(vec_reference))
            elif distance_metric == 'Euclidean':
                distance = math.dist(vec_current, vec_reference)
            else:
                print(f"Error: distance_metric {distance_metric} is not implemented yet")
                return None, None, None, None, None, None
            if distance < closest_distance:
                closest_distance = distance
                closest_smi_ref = smi_reference

        distances_df = pd.concat([distances_df, 
                                  pd.DataFrame([{'smi_current': smi_current, 'smi_ref': closest_smi_ref, 'distance': closest_distance}])
                                  ], ignore_index=True)

    # Extract the percentile distances and the corresponding SMILES
    distances_df = distances_df.sort_values(by='distance').reset_index(drop=True)
    if save_data_for_percentiles_to_folder:
        os.makedirs(save_data_for_percentiles_to_folder, exist_ok=True)
        taxonomic_chain_ref_short = '-'.join(taxonomic_chain_ref.split('-')[-3:])
        taxonomic_chain_short = '-'.join(taxonomic_chain.split('-')[-3:])
        distances_df.to_csv(f'{save_data_for_percentiles_to_folder}/data_for_percentiles_{taxonomic_chain_ref_short}_{taxonomic_chain_short}_{encoding}.csv', index=False)
    idx_50 = int(len(distances_df) * 0.5)
    idx_25 = int(len(distances_df) * 0.25)
    smi_current_50, smi_ref_50, distance_percentile_50 = distances_df.loc[idx_50, ['smi_current', 'smi_ref', 'distance']]
    smi_current_25, smi_ref_25, distance_percentile_25 = distances_df.loc[idx_25, ['smi_current', 'smi_ref', 'distance']]

    return distance_percentile_25, smi_ref_25, smi_current_25, distance_percentile_50, smi_ref_50, smi_current_50


def chemical_distances_vs_taxonomic_distances(
        df,
        ref_chain,
        max_taxonomic_distance=2,
        size_threshold=20,
        encoding_columns='ECFP6_2048',
        distance_metric='Tanimoto',
        taxonomic_levels = ['superkingdom', 'kingdom', 'phylum', 'classx', 'family', 'genus', 'species'],
        save_data_for_percentiles_to_folder=None,
        encoding='ECFP6_2048',
):
    """
    Analyze relationship between chemical and taxonomic distances.
    
    Args:
        df (pandas.DataFrame): Input dataframe with molecular data
        ref_chain (list): Reference taxonomic chain
        max_taxonomic_distance (int): Maximum taxonomic distance to consider
        size_threshold (int): Minimum number of molecules required
        encoding_columns (str): Column containing molecular encodings
        distance_metric (str): Type of distance metric
        taxonomic_levels (list): List of taxonomic level names
        save_data_for_percentiles_to_folder (str, optional): Directory to save percentile data
        encoding (str): Type of molecular encoding used
        
    Returns:
        pandas.DataFrame: DataFrame containing chemical distances for different taxonomic distances
    """
    taxonomic_chain_ref = '-'.join(ref_chain[0:7])
    chemical_distances_vs_taxonomic_distances = []
    for taxonomic_distance in range(max_taxonomic_distance + 1):
        # get the list of all taxonomic_chains separated from the ref species by this taxonomic_distance
        df['mask'] = True
        for level in range(7 - taxonomic_distance):
            df['mask'] = (df[taxonomic_levels[level]] == ref_chain[level]) & df['mask']
        if taxonomic_distance > 0:
            level = 7 - taxonomic_distance
            df['mask'] = (df[taxonomic_levels[level]] != ref_chain[level]) & df['mask']
        taxonomic_chains = df[df['mask']]['taxonomic_chain'].unique()

        # if len(taxonomic_chains)>0:
        for taxonomic_chain in taxonomic_chains:
            distance_percentile_25, smi_ref_25, smi_current_25, distance_percentile_50, smi_ref_50, smi_current_50 = chemical_distance(df,
                                                                                   taxonomic_chain,
                                                                                   taxonomic_chain_ref,
                                                                                   size_threshold=size_threshold,
                                                                                   encoding_columns=encoding_columns,
                                                                                   distance_metric=distance_metric,
                                                                                   save_data_for_percentiles_to_folder=save_data_for_percentiles_to_folder,
                                                                                   encoding=encoding,
                                                                                   )
            if distance_percentile_25 is not None:
                chemical_distances_vs_taxonomic_distances.append(
                    [taxonomic_chain_ref, taxonomic_chain, taxonomic_distance, distance_percentile_25, smi_ref_25, smi_current_25, distance_percentile_50, smi_ref_50, smi_current_50])

    return pd.DataFrame(chemical_distances_vs_taxonomic_distances,
                     columns=['taxonomic_chain_ref', 'taxonomic_chain', 'taxonomic_distance', 'distance_percentile_25', 'smi_ref_25', 'smi_current_25', 'distance_percentile_50', 'smi_ref_50', 'smi_current_50'])


def stats_chemical_distances_vs_taxonomic_distances(
        df_chemical_distances_vs_taxonomic_distances, taxonomic_chain_ref,
        max_taxonomic_distance=2,
        verbose=True,
):
    """
    Calculate statistical measures for chemical distances at different taxonomic distances.
    
    Args:
        df_chemical_distances_vs_taxonomic_distances (pandas.DataFrame): Input distance data
        taxonomic_chain_ref (str): Reference taxonomic chain
        max_taxonomic_distance (int): Maximum taxonomic distance to consider
        verbose (bool): Whether to print detailed statistics
        
    Returns:
        pandas.DataFrame: Statistical summary of chemical distances
    """
    df_stats_chemical_distances_vs_taxonomic_distances = pd.DataFrame()

    if verbose:
        print("taxonomic_distance: distance_percentile_50_ave (distance_percentile_50_std), distance_percentile_25_ave (distance_percentile_25_std)")
    for taxonomic_distance in range(max_taxonomic_distance + 1):
        dfc = df_chemical_distances_vs_taxonomic_distances[(
                df_chemical_distances_vs_taxonomic_distances['taxonomic_chain_ref'] == taxonomic_chain_ref
                                                           ) & (
                df_chemical_distances_vs_taxonomic_distances['taxonomic_distance'] == taxonomic_distance
        )]
        distance_percentile_50_ave, distance_percentile_50_std = dfc['distance_percentile_50'].mean(), dfc['distance_percentile_50'].std()
        distance_percentile_25_ave, distance_percentile_25_std = dfc['distance_percentile_25'].mean(), dfc['distance_percentile_25'].std()
        df_stats_chemical_distances_vs_taxonomic_distances = pd.concat([
            df_stats_chemical_distances_vs_taxonomic_distances,
            pd.DataFrame(
                [[taxonomic_distance, distance_percentile_50_ave, distance_percentile_50_std, distance_percentile_25_ave, distance_percentile_25_std]],
                columns=['taxonomic_distance', 'distance_percentile_50_ave', 'distance_percentile_50_std', 'distance_percentile_25_ave', 'distance_percentile_25_std'])
        ])
        if verbose:
            print(
            f"{taxonomic_distance}: {distance_percentile_50_ave:.3f} ({distance_percentile_50_std:.3f}), {distance_percentile_25_ave:.3f} ({distance_percentile_25_std:.3f})")

    return df_stats_chemical_distances_vs_taxonomic_distances


def visualize_chemical_distances_vs_taxonomic_distances(dfc, df_stats_chemical_distances_vs_taxonomic_distances, save_png=None):
    """
    Create visualization plots for chemical distances vs taxonomic distances.
    
    Args:
        dfc (pandas.DataFrame): Chemical distance data
        df_stats_chemical_distances_vs_taxonomic_distances (pandas.DataFrame): Statistical summary data
        save_png (str, optional): Path to save the generated plots
    """
    plt.figure(figsize=(10, 5))
    plt.scatter(dfc['taxonomic_distance'], dfc['distance_percentile_50'], marker='_', color='r')
    plt.errorbar(df_stats_chemical_distances_vs_taxonomic_distances['taxonomic_distance'],
                 df_stats_chemical_distances_vs_taxonomic_distances['distance_percentile_50_ave'],
                 yerr=df_stats_chemical_distances_vs_taxonomic_distances['distance_percentile_50_std'],
                 color='k', capsize=5, marker='o', ms=10,
                 )
    plt.xlabel('Taxonomic distance')
    plt.xlim(0, 4.5)
    # plt.ylim(0, 1)
    plt.xticks([0, 1, 2, 3, 4])
    plt.ylabel('distance_percentile_50')
    plt.savefig(save_png.replace('.png', '_percentile50.png')) if save_png else plt.show()

    plt.figure(figsize=(10, 5))
    plt.scatter(dfc['taxonomic_distance'], dfc['distance_percentile_25'], marker='_', color='b')
    plt.errorbar(df_stats_chemical_distances_vs_taxonomic_distances['taxonomic_distance'],
                 df_stats_chemical_distances_vs_taxonomic_distances['distance_percentile_25_ave'],
                 yerr=df_stats_chemical_distances_vs_taxonomic_distances['distance_percentile_25_std'],
                 color='k', capsize=5, marker='o', ms=10,
                 )
    plt.xlabel('Taxonomic distance')
    plt.xlim(0, 4.5)
    # plt.ylim(0, 1)
    plt.xticks([0, 1, 2, 3, 4])
    plt.ylabel('distance_percentile_25')
    plt.savefig(save_png.replace('.png', '_percentile25.png')) if save_png else plt.show()


def run_all(
        df,
        ref_chain,
        max_taxonomic_distance=2,   # 2 for debugging, 3 for production runs
        size_threshold=20,
        min_size_threshold=10,
        encoding='ECFP6_2048',
        encoding_columns='ECFP6_2048',
        distance_metric='Tanimoto',
        verbose=True,
        tdistance1=2,
        tdistance2=3,
        visualize=True,
        save_dataframes_to_folder=None,
        save_data_for_percentiles_to_folder=None,
        enforce_low_level_recomputations=False,
):
    """
    Run the complete analysis pipeline.
    
    Args:
        df (pandas.DataFrame): Input dataframe with molecular data
        ref_chain (list): Reference taxonomic chain
        max_taxonomic_distance (int): Maximum taxonomic distance to analyze
        size_threshold (int): Minimum molecules for analysis
        min_size_threshold (int): Absolute minimum molecules required
        encoding (str): Type of molecular encoding
        encoding_columns (str): Column containing molecular encodings
        distance_metric (str): Type of distance metric
        verbose (bool): Whether to print detailed output
        tdistance1 (int): First taxonomic distance for statistical comparison
        tdistance2 (int): Second taxonomic distance for statistical comparison
        visualize (bool): Whether to generate visualizations
        save_dataframes_to_folder (str, optional): Directory to save results
        save_data_for_percentiles_to_folder (str, optional): Directory to save percentile data
        enforce_low_level_recomputations (bool): Whether to force recomputation of existing results
        
    Returns:
        pandas.DataFrame: Welch's t-test statistics
    """
    if size_threshold < min_size_threshold:
        raise ValueError(f"size_threshold {size_threshold} can't be less than min_size_threshold {min_size_threshold}")
    taxonomic_chain_ref = '-'.join(ref_chain[0:7])
    short_taxonomic_chain_ref = '-'.join(ref_chain[-3:])
    
    curr_folder = f'{save_dataframes_to_folder}/chem_vs_tax_dist_csv_files_{encoding}_ms{min_size_threshold}'
    os.makedirs(curr_folder, exist_ok=True)
    curr_csv_file = f'{curr_folder}/chem_vs_tax_dist_{taxonomic_chain_ref}.csv'
    if not enforce_low_level_recomputations and save_dataframes_to_folder and os.path.exists(curr_csv_file):
        df_chemical_distances_vs_taxonomic_distances = pd.read_csv(curr_csv_file)
    else:
        df_chemical_distances_vs_taxonomic_distances = chemical_distances_vs_taxonomic_distances(
            df, ref_chain,
            max_taxonomic_distance=max_taxonomic_distance,  # 2 for debugging, 3 for production runs
            size_threshold=min_size_threshold,
            encoding_columns=encoding_columns,
            distance_metric=distance_metric,
            save_data_for_percentiles_to_folder=save_data_for_percentiles_to_folder,
            encoding=encoding,
        )
        if save_dataframes_to_folder:
            df_chemical_distances_vs_taxonomic_distances.to_csv(curr_csv_file, index=False)
            
    # select only those rows where size >= size_threshold
    file_with_n_molecules_per_taxonomic_chain = f'{save_dataframes_to_folder}/n_molecules_per_taxonomic_chain.json'
    nsmiles_per_taxonomic_chain = {}
    if os.path.exists(file_with_n_molecules_per_taxonomic_chain):
        with open(file_with_n_molecules_per_taxonomic_chain, 'r') as f:
            nsmiles_per_taxonomic_chain = json.loads(f.read())
            # nsmiles_per_taxonomic_chain = {k: v for k, v in nsmiles_per_taxonomic_chain.items()}
    if len(nsmiles_per_taxonomic_chain) == 0:    
        nsmiles_per_taxonomic_chain = {}
        taxonomic_chains = df['taxonomic_chain'].unique()
        for taxonomic_chain in taxonomic_chains:
            set_current = set(df[df['taxonomic_chain'] == taxonomic_chain]['canonicalized_smiles'])
            nsmiles_per_taxonomic_chain[taxonomic_chain] = len(set_current)
        with open(file_with_n_molecules_per_taxonomic_chain, 'w') as f:
            f.write(json.dumps(nsmiles_per_taxonomic_chain))
    df_above_threshold = df_chemical_distances_vs_taxonomic_distances[
        df_chemical_distances_vs_taxonomic_distances['taxonomic_chain'].apply(lambda x: nsmiles_per_taxonomic_chain[x]) >= size_threshold
        ].copy()
    
    # calculate statistics for chemical distances vs taxonomic distances
    df_stats_chemical_distances_vs_taxonomic_distances = stats_chemical_distances_vs_taxonomic_distances(
        df_above_threshold,
        taxonomic_chain_ref=taxonomic_chain_ref,
        max_taxonomic_distance=max_taxonomic_distance,
        verbose=verbose
    )
    df_stats_chemical_distances_vs_taxonomic_distances = df_stats_chemical_distances_vs_taxonomic_distances.dropna(subset=['distance_percentile_50_ave'])
    if save_dataframes_to_folder:
        curr_folder = f'{save_dataframes_to_folder}/stat_chem_vs_tax_dist_csv_files_{encoding}_s{size_threshold}'
        os.makedirs(curr_folder, exist_ok=True)
        df_stats_chemical_distances_vs_taxonomic_distances.to_csv(f'{curr_folder}/stat_chem_vs_tax_dist_{taxonomic_chain_ref}.csv', index=False)

    # calculate Welch's t-test for difference between taxonomic distances tdistance1 and tdistance2 (by default, 2 and 3)
    data_Wstat = [taxonomic_chain_ref]
    columns_Wstat = ['taxonomic_chain_ref']

    for percentile in ['25', '50']:
        sample1 = df_above_threshold[
            df_above_threshold['taxonomic_distance'] == tdistance1
            ][f"distance_percentile_{percentile}"]
        sample2 = df_above_threshold[
            df_above_threshold['taxonomic_distance'] == tdistance2
            ][f"distance_percentile_{percentile}"]
        st = stats.ttest_ind(sample2, sample1, equal_var=False, alternative='greater')
        data_Wstat += [st.statistic, st.pvalue, st.df]
        columns_Wstat += [f'Wtstat_{percentile}', f'Wt_pvalue_{percentile}', f'Wtdof_{percentile}']

    df_Welch_stat = pd.DataFrame([data_Wstat], columns=columns_Wstat)
    if save_dataframes_to_folder:
        curr_folder = f'{save_dataframes_to_folder}/Welch_stat_chem_vs_tax_dist_csv_files_{encoding}_s{size_threshold}'
        os.makedirs(curr_folder, exist_ok=True)
        df_Welch_stat.to_csv(f'{curr_folder}/Welch_stat_chem_vs_tax_dist_{taxonomic_chain_ref}.csv', index=False)

    if visualize and len(df_stats_chemical_distances_vs_taxonomic_distances) > 0:
        visualize_chemical_distances_vs_taxonomic_distances(
            df_above_threshold,
            df_stats_chemical_distances_vs_taxonomic_distances,
            save_png=f'{save_dataframes_to_folder}/stat_chem_vs_tax_dist_csv_files_{encoding}_s{size_threshold}/stat_chem_vs_tax_dist_{short_taxonomic_chain_ref}.png'
        )

    return df_Welch_stat


def draw_pairs_of_molecules(smi_curr, smi_ref, save_typical_molecules_png=None):
    """
    Draw a pair of molecules side by side for comparison.
    
    Args:
        smi_curr (str): SMILES string of current molecule
        smi_ref (str): SMILES string of reference molecule
        save_typical_molecules_png (str, optional): Path to save the generated image
    """
    mol_ref = Chem.MolFromSmiles(smi_ref)
    mol_curr = Chem.MolFromSmiles(smi_curr)    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(Draw.MolToImage(mol_curr))
    axes[0].set_title("Current")
    axes[0].axis('off')
    axes[1].imshow(Draw.MolToImage(mol_ref))
    axes[1].set_title("Reference")
    axes[1].axis('off')
    if save_typical_molecules_png:
        os.makedirs(os.path.dirname(save_typical_molecules_png), exist_ok=True)
        plt.savefig(save_typical_molecules_png)
    else:
        plt.show()
