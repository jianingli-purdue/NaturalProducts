import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from scipy import stats
import argparse
from mpmath import mp
import json


def plot_kde(ax, data, color, label, p_value_legend, range_min=-40, range_max=0):
    """
    Plot kernel density estimation of p-value distributions.
    
    Args:
        ax: matplotlib axis object
        data (array-like): Data points for KDE
        color: Color for the plot
        label (str): Label for the legend
        p_value_legend (float): P-value threshold for legend annotation
        range_min (float): Minimum x-axis value
        range_max (float): Maximum x-axis value
        
    Returns:
        str: Legend label with percentage below threshold
    """
    x_values = np.linspace(range_min, range_max, 1000)
    kde_values = gaussian_kde(data).evaluate(x_values)
    ax.plot(x_values, kde_values, color=color, linewidth=2)
    percentage_below_threshold = (data < np.log10(p_value_legend)).mean() * 100
    return f'{label} ({percentage_below_threshold:.1f}% p-value < {p_value_legend})'

def visualize_p_value(df, variable, filter_criteria, p_value_legend=0.01, save_png=None):
    """
    Create visualization of p-value distributions for different parameters.
    
    Args:
        df (pandas.DataFrame): Input data containing p-values
        variable (str): Variable to analyze ('percentile', 'embedding', or 'size')
        filter_criteria (dict): Criteria for filtering the data
        p_value_legend (float): P-value threshold for annotations
        save_png (str, optional): Path to save the generated plot
    """
    keyword = filter_criteria['taxonomic_chain_ref']
   
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.rc('font', family='DejaVu Sans', size=14, weight='bold')

    values_of_variable = [25, 50] if variable == 'percentile' else df[variable].unique()
    colors = plt.get_cmap('jet', len(values_of_variable))
    legend_labels = []
    
    for i, val in enumerate(values_of_variable):
        if variable == 'percentile':
            subset = df[(df['size'] == filter_criteria['size']) & (df['embedding'] == filter_criteria['embedding'])]
        elif variable == 'embedding':
            subset = df[(df['size'] == filter_criteria['size']) & (df['embedding'] == val)]
        else:
            subset = df[(df['size'] == val) & (df['embedding'] == filter_criteria['embedding'])]
        data = subset[f'log_Wt_pvalue_{filter_criteria["percentile"]}'] if variable != 'percentile' else subset[f'log_Wt_pvalue_{val}']
        label = f'Percentile {val}' if variable == 'percentile' else val if variable == 'embedding' else f'Size {val}'
        if len(data) > 1:
            legend_labels.append(plot_kde(ax, data, colors(i), label, p_value_legend))
    
    title = f'{keyword}, Different {variable.capitalize()}s' if keyword else f'All Taxonomic Chains, Different {variable.capitalize()}s'
    plt.title(title, fontsize=16, weight='bold')
    ax.axvline(x=np.log10(p_value_legend), color='gray', linestyle='-', linewidth=1)
    plt.xlabel('Log10(p-value)', fontsize=14, weight='bold')
    plt.ylabel('Frequency', fontsize=14, weight='bold')
    plt.legend(title=variable.capitalize(), labels=legend_labels)
    ax.grid(False)
    plt.savefig(save_png) if save_png else plt.show()


if __name__ == '__main__':
    """
    Main execution flow for p-value histogram analysis.
    
    This script performs the following operations:
    1. Loads and processes data from CSV files
    2. Combines data from multiple sources if needed
    3. Calculates p-values and their logarithms
    4. Generates visualizations for different taxonomic groups
    5. Performs statistical analysis including Welch's t-test
    6. Saves results to files

    Command-line Arguments:
        --data_folder (str): Path to data directory (default: './data/')
        --data_filename (str): Name of the data file (default: 'combined_data_Welch_stat.csv')
        --upper_limit_ref_size (int): Maximum reference species size (default: 1000)
        --lower_limit_ref_size (int): Minimum reference species size (default: 180)
        --encoding (str): ML Encoding type (default: 'chemformer')
        --size_threshold (int): Minimum molecules per species (default: 20)
        --min_size_threshold (int): Absolute minimum molecules required (default: 20)

    Output:
        - Combined data CSV files
        - P-value distribution plots
        - Statistical analysis results
        - Lists of unique taxonomic chains
    """
    parser = argparse.ArgumentParser(description='Visualize p-value distributions.')
    parser.add_argument('--data_folder', type=str, default='./data/', help='Path to the data folder.')
    parser.add_argument('--data_filename', type=str, default='combined_data_Welch_stat.csv', help='Name of the data file.')
    parser.add_argument('--upper_limit_ref_size', type=int, default=1000, help='Include reference species with number of molecules less than this value.')
    parser.add_argument('--lower_limit_ref_size', type=int, default=180, help='Include reference species with number of molecules greater or equal than this value.')
    parser.add_argument('--encoding', type=str, default='chemformer', help='ML Encoding used for converting SMILES to vectors.')
    parser.add_argument('--size_threshold', type=int, default=20, help='Minimum number of molecules in a current species to include it into analysis.')
    parser.add_argument('--min_size_threshold', type=int, default=20, help='Minimum number of molecules in a species for which raw data were precomputed.')
    args = parser.parse_args()

    data_folder = args.data_folder
    data_filename = args.data_filename
    encoding = args.encoding
    size_threshold = args.size_threshold
    min_size_threshold = args.min_size_threshold
    
    df_stat_refspecies = pd.read_csv('./statistics_on_n_molecules_per_taxonomic_chain.csv')
    included_refspecies = df_stat_refspecies[(df_stat_refspecies['nmol'] < args.upper_limit_ref_size) & (df_stat_refspecies['nmol'] >= args.lower_limit_ref_size)]['taxonomic_chain'].values
    
    if data_filename is None:
        encodings = ['chemformer']
        sizes = [20, 25, 30]
        combined_df = None
        for e in encodings:
            for s in sizes:
                curr_folder_path = f'{data_folder}/Welch_stat_chem_vs_tax_dist_csv_files_{e}_s{s}'
                for file_name in os.listdir(curr_folder_path):
                    if file_name.startswith("Welch_stat_chem_vs_tax_dist_") and file_name.endswith(".csv"):
                        file_path = os.path.join(curr_folder_path, file_name)
                        dfc = pd.read_csv(file_path).dropna(subset=['Wt_pvalue_25', 'Wt_pvalue_50'])
                        dfc['embedding'] = e
                        dfc['size'] = s
                        if len(dfc) > 0:
                            combined_df = pd.concat([combined_df, dfc]) if combined_df is not None else dfc
                            
        combined_df.to_csv(f'{data_folder}/{data_filename}', index=False)

    combined_df = pd.read_csv(f'{data_folder}/{data_filename}')
    combined_df = combined_df.dropna(subset=['Wt_pvalue_25', 'Wt_pvalue_50'])
    combined_df = combined_df[combined_df['taxonomic_chain_ref'].isin(included_refspecies)].copy()
    for p in [25, 50]:
        combined_df[f'log_Wt_pvalue_{p}'] = np.log10(combined_df[f'Wt_pvalue_{p}'])
        
    # build plots for distributions of pvalue of individual reference species as a function of various settings
    for keyword in [
        'Magnoliopsida',
        'Pinopsida',
        'Viridiplantae', 
        'Fungi',
        '',
        ]:
        for variable, filter_criteria in [
            ('embedding', {'percentile': 50, 'size': size_threshold, 'taxonomic_chain_ref': keyword}),
            ('percentile', {'embedding': encoding, 'size': size_threshold, 'taxonomic_chain_ref': keyword}),
            ('size', {'embedding': encoding, 'percentile': 50, 'taxonomic_chain_ref': keyword})
        ]:
            fixed_parameters = '-'.join(f'{key}_{value}' for key, value in filter_criteria.items()).replace('taxonomic_chain_ref_', '')
            if keyword == '':
                fixed_parameters += 'all'
            curr_folder_path = f'{data_folder}/pngs/'
            os.makedirs(curr_folder_path, exist_ok=True)
            save_png = f'{curr_folder_path}/various_{variable}s_fixed_{fixed_parameters}.png'
            dfc = combined_df[combined_df['taxonomic_chain_ref'].str.contains(keyword)].copy() if keyword else combined_df.copy()
            visualize_p_value(dfc, variable, filter_criteria, save_png=save_png)
            
    # generate the table with pvalues for chemical distances concatenated together, regardless of ref species, for different taxonomic distances and selected keywords
    file_with_n_molecules_per_taxonomic_chain = f'{data_folder}/n_molecules_per_taxonomic_chain.json'
    with open(file_with_n_molecules_per_taxonomic_chain, 'r') as f:
        nsmiles_per_taxonomic_chain = json.loads(f.read())
    
    combined_df = None
    curr_folder_path = f'{data_folder}/chem_vs_tax_dist_csv_files_{encoding}_ms{min_size_threshold}'
    for file_name in os.listdir(curr_folder_path):
        if file_name.startswith("chem_vs_tax_dist_") and file_name.endswith(".csv"):
            file_path = os.path.join(curr_folder_path, file_name)
            dfc = pd.read_csv(file_path, usecols=['taxonomic_chain_ref', 'taxonomic_chain', 'taxonomic_distance', 'distance_percentile_25', 'distance_percentile_50'])
            dfc = dfc.replace([np.inf, -np.inf], np.nan).dropna(subset=['distance_percentile_25', 'distance_percentile_50'])
            dfc = dfc[dfc['taxonomic_chain_ref'].isin(included_refspecies)]
            dfc = dfc[dfc['taxonomic_chain'].apply(lambda x: nsmiles_per_taxonomic_chain[x]) >= size_threshold].copy()
            if len(dfc) > 0:                
                combined_df = pd.concat([combined_df, dfc]) if combined_df is not None else dfc
                
    suffix = f"{encoding}_s{size_threshold}_refs{args.lower_limit_ref_size}-{args.upper_limit_ref_size}"
    combined_df.to_csv(f'{data_folder}/combined_data_Welch_stat_{suffix}.csv', index=False)
    with open(f'{data_folder}/unique_taxonomic_chain_refs_{suffix}.txt', 'w') as f:
        for item in combined_df['taxonomic_chain_ref'].unique():
            f.write(f"{item}\n")
                
    df_Welch_stat_combined = None
    for tdistance1, tdistance2 in [
        [2, 3],
        [1, 2],
    ]:
        for keyword in [
            'Magnoliopsida',
            'Pinopsida', 'Viridiplantae', 'Fungi', '']:
            dfc = combined_df[combined_df['taxonomic_chain_ref'].str.contains(keyword)]
            
            data_Wstat = []
            columns_Wstat = []

            for percentile in ['25', '50']:
                sample1 = dfc[
                    dfc['taxonomic_distance'] == tdistance1
                    ][f"distance_percentile_{percentile}"]
                sample2 = dfc[
                    dfc['taxonomic_distance'] == tdistance2
                    ][f"distance_percentile_{percentile}"]
                st = stats.ttest_ind(sample2, sample1, equal_var=False, alternative='greater')
                Wtstat, Wtpvalue, Wtdof = st.statistic, st.pvalue, st.df
                t = mp.mpf(Wtstat)
                nu = mp.mpf(Wtdof)
                x2 = nu / (t**2 + nu)
                pvalue_high_acc = mp.betainc(nu/2, mp.one/2, x2=x2, regularized=True)/2
                # print(f"p-value: {Wtpvalue} vs {pvalue_high_acc}")
                data_Wstat += [Wtstat, pvalue_high_acc, Wtdof]
                columns_Wstat += [f'Wtstat_{percentile}', f'Wt_pvalue_{percentile}', f'Wtdof_{percentile}']

            df_Welch_stat = pd.DataFrame([data_Wstat], columns=columns_Wstat)
            df_Welch_stat['keyword'] = keyword
            df_Welch_stat['tdistances'] = f'{tdistance1}vs{tdistance2}'
            df_Welch_stat_combined = pd.concat([df_Welch_stat_combined, df_Welch_stat]) if df_Welch_stat_combined is not None else df_Welch_stat
    df_Welch_stat_combined.to_csv(f'{data_folder}/combined_Welch_stat_{encoding}_s{size_threshold}.csv', index=False)
