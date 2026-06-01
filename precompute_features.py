"""
סקריפט חד-פעמי: מחשב פיצ'רים מרחביים לכל קשת ברשת OSMnx ושומר ל-data/edges_features.parquet.

הרץ פעם אחת לפני אימון המודל:
    python precompute_features.py

זמן ריצה משוער: 2–5 דקות (תלוי במחשב).
"""
import time

from src.routing import load_graph
from src.spatial import compute_edge_features

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    print("Loading street graph...")
    G = load_graph()
    print(f"  {G.number_of_edges():,} edges, {G.number_of_nodes():,} nodes\n")

    compute_edge_features(G)

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.1f} seconds")
