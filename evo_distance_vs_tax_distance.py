import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from utils import taxonomic_distance_from_lineages


def main():
    parser = argparse.ArgumentParser(description="Plot evolutionary distance vs taxonomic distance.")
    parser.add_argument("--csv", default="./data/all_species_distances_upper_triangle.csv", help="Path to all_species_distances_upper_triangle.csv")
    parser.add_argument("--out", default="./data/evo_vs_tax_distance.png", help="Output plot filename")
    parser.add_argument("--selection", type=str, default="", help="Optional selection condition for filtering the data, e.g., taxonomic_lineage_[12] contain \"Magnoliopsida\"")
    parser.add_argument("--plot", choices=["distribution", "points"], default="distribution", help="Plot distributions (violin) or individual datapoints")
    args = parser.parse_args()
    input_csv = Path(args.csv)
    output_png = Path(args.out)
    selection_condition = args.selection

    # input_csv = "./data/all_species_distances_upper_triangle.csv"
    # output_png = "./data/evo_vs_tax_distance.png"
    # selection_condition = ""
    # taxonomic_levels = ['superkingdom', 'kingdom', 'phylum', 'classx', 'family', 'genus', 'species'] # Lotus
    # taxonomic_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species'] # Coconut

    df = pd.read_csv(input_csv, usecols=["distance", "tax_lineage_name_1", "tax_lineage_name_2"])
    df = df[df['tax_lineage_name_1'].str.contains(selection_condition) | df['tax_lineage_name_2'].str.contains(selection_condition)] if selection_condition else df
    df["tax_distance"] = df.apply(lambda r: taxonomic_distance_from_lineages(r["tax_lineage_name_1"], r["tax_lineage_name_2"]), axis=1)
    df = df.dropna(subset=["distance", "tax_distance"])

    plt.figure(figsize=(6, 5))
    plt.scatter([0], [0], s=6)
    max_tax_distance = int(df["tax_distance"].max())
    tax_distances = list(range(max_tax_distance + 1))
    if args.plot == "points":
        plt.scatter(df["distance"], df["tax_distance"], s=1, c="C0")
    else:
        grouped_distances = [df.loc[df["tax_distance"] == d, "distance"].values for d in tax_distances]
        non_empty = [(d, values) for d, values in zip(tax_distances, grouped_distances) if len(values) > 0]
        if non_empty:
            plot_positions, plot_values = zip(*non_empty)
            plt.violinplot(
                plot_values,
                positions=plot_positions,
                vert=False,
                showmeans=False,
                showmedians=True,
                showextrema=False,
                widths=0.8,
            )
            
    plt.xlabel("Evolutionary distance")
    plt.ylabel("Taxonomic distance")
    plt.yticks(tax_distances)
    plt.ylim(-0.5, max_tax_distance + 0.5)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    
    print("Statistics:")
    print("tax_distance,count,mean,median,std")
    for tax_distance in tax_distances:
        subset = df[df["tax_distance"] == tax_distance]["distance"]
        if len(subset) > 0:
            print(f"{tax_distance},{len(subset)},{subset.mean()},{subset.median()},{subset.std()}")
            
if __name__ == "__main__":
    main()
