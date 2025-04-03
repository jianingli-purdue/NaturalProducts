import pandas as pd
import argparse
import os
from utils import draw_pairs_of_molecules

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--data_folder', type=str, default='./data/chem_vs_tax_dist_csv_files_chemformer_ms10', help='Path to the data folder.')
        parser.add_argument('--keyword', type=str, default='', help='keyword to filter the files in the data folder')
        parser.add_argument('--tdistance', type=int, default=2, help='Taxonomic distance')
        parser.add_argument('--percentile', type=int, default=50, help='Percentile')
        parser.add_argument('--n_pairs', type=int, default=3, help='Number of typical pairs to visualize.')
        args = parser.parse_args()

        data_folder = args.data_folder
        td = args.tdistance
        percentile = args.percentile
        n_typical_pairs = args.n_pairs
        keyword = args.keyword


        csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv') and keyword in f]

        for csv_file in csv_files:
                taxonomic_chain_ref = os.path.splitext(csv_file)[0].replace('.csv', '').replace('chem_vs_tax_dist_', '')
                print(f"Starting with reference taxonomic chain: {taxonomic_chain_ref}")
                df_chemical_distances_vs_taxonomic_distances = pd.read_csv(os.path.join(data_folder, csv_file))
                if len(df_chemical_distances_vs_taxonomic_distances) == 0:
                        print(f"No data for {taxonomic_chain_ref}")
                        continue

                dfc = df_chemical_distances_vs_taxonomic_distances[df_chemical_distances_vs_taxonomic_distances['taxonomic_distance'] == td].copy()
                distance_percentile_ave = dfc[f'distance_percentile_{percentile}'].mean()
                dfc['abs_difference_from_average_distance'] = (dfc[f'distance_percentile_{percentile}'] - distance_percentile_ave).abs()
                dfc = dfc.sort_values(by='abs_difference_from_average_distance', ascending=False).reset_index(drop=True)
                for i, row in dfc.head(n_typical_pairs).iterrows():
                        smi_ref = getattr(row, f'smi_ref_{percentile}')
                        smi_curr = getattr(row, f'smi_current_{percentile}')
                        short_taxonomic_chain = '-'.join(row.taxonomic_chain.split('-')[-3:])
                        chem_distance = getattr(row, f'distance_percentile_{percentile}')
                        print(f"chem dist {chem_distance:.2f}, {short_taxonomic_chain}, SMILES reference {smi_ref}, SMILES current {smi_curr}")
                        draw_pairs_of_molecules(smi_curr, smi_ref, save_typical_molecules_png=f'{data_folder}/../typical_pairs_of_molecules/{taxonomic_chain_ref}_td{td}_{i}_chd{chem_distance:.1f}_{short_taxonomic_chain}.png')
