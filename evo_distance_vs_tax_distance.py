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
    args = parser.parse_args()
    input_csv = Path(args.csv)
    output_png = Path(args.out)
    selection_condition = args.selection

    # input_csv = "./data/all_species_distances_upper_triangle.csv"
    # output_png = "./data/evo_vs_tax_distance.png"
    # taxonomic_levels = ['superkingdom', 'kingdom', 'phylum', 'classx', 'family', 'genus', 'species'] # Lotus
    # taxonomic_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species'] # Coconut

    df = pd.read_csv(input_csv, usecols=["distance", "tax_lineage_name_1", "tax_lineage_name_2"])
    df = df[df['tax_lineage_name_1'].str.contains(selection_condition) | df['tax_lineage_name_2'].str.contains(selection_condition)] if selection_condition else df
    df["tax_distance"] = df.apply(lambda r: taxonomic_distance_from_lineages(r["tax_lineage_name_1"], r["tax_lineage_name_2"]), axis=1)
    df = df.dropna(subset=["distance", "tax_distance"])

    plt.figure(figsize=(6, 5))
    plt.scatter(df["distance"], df["tax_distance"], s=6)
    plt.scatter([0], [0], s=6)
    plt.xlabel("Evolutionary distance")
    plt.ylabel("Taxonomic distance")
    plt.yticks(range(int(df["tax_distance"].max()) + 2))
    plt.ylim(0, df["tax_distance"].max()*1.05)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)

if __name__ == "__main__":
    main()
