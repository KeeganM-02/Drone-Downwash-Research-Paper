import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------
# CONFIG: set your base path here
# Example: r"C:\Users\YourName\Documents\SprayProject"
# -------------------------
base_path = r"C:\Users\keega\Desktop\WSP_Images"
figures_path = os.path.join(base_path, "Drone_Figures")

# Create folder if it doesn't exist
os.makedirs(figures_path, exist_ok=True)

# -------------------------
# Load CSV
# -------------------------
df = pd.read_csv("wsp_analysis_results.csv")

# Clean column names (remove spaces, %, (), /, etc.)
df.columns = (
    df.columns.str.strip()
              .str.replace(r"[()%/]", "", regex=True)
              .str.replace(" ", "_")
)

print("Cleaned column names:", df.columns.tolist())

# -------------------------
# Filter: Keep only drone flights
# Assumes backpack trial is labeled differently (e.g., "Backpack", "BP", etc.)
# If your Height column has ONLY "5ft, 10ft, 15ft" for drone flights, this will work:
# -------------------------
drone_df = df[df["Height"].isin(["5ft", "10ft", "15ft"])].copy()

# -------------------------
# Metrics to plot
# -------------------------
metrics = [
    "Coverage_Rate",
    "Coverage_Density",
    "Deposition",
    "Droplet_Count"
]

# Set style
sns.set_theme(style="whitegrid", font_scale=1.3)

# -------------------------
# 1. Coverage across Heights & Canopy
# -------------------------
for metric in metrics:
    plt.figure(figsize=(10,6))
    sns.pointplot(data=drone_df, x="Height", y=metric, hue="Canopy_Position",
                  dodge=True, markers="o", capsize=.1, errwidth=1.5, palette="Set1")
    plt.title(f"{metric.replace('_', ' ')} vs Height (by Canopy Position)", fontsize=16)
    plt.ylabel(metric.replace("_", " "))
    plt.xlabel("Drone Flight Height")
    plt.legend(title="Canopy Position")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, f"{metric}_Height_vs_Canopy.png"), dpi=300)
    plt.close()

# -------------------------
# 2. Canopy Layer comparison at each Height
# -------------------------
for metric in metrics:
    plt.figure(figsize=(10,6))
    sns.boxplot(data=drone_df, x="Canopy_Position", y=metric, hue="Height", palette="Set3")
    plt.title(f"{metric.replace('_', ' ')} by Canopy Layer & Height", fontsize=16)
    plt.ylabel(metric.replace("_", " "))
    plt.xlabel("Canopy Position")
    plt.legend(title="Drone Height")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, f"{metric}_Canopy_vs_Height.png"), dpi=300)
    plt.close()

# -------------------------
# 3. Variability across Plants
# -------------------------
for metric in metrics:
    plt.figure(figsize=(12,6))
    sns.boxplot(data=drone_df, x="Plant", y=metric, hue="Height", palette="Set2")
    plt.title(f"{metric.replace('_', ' ')} by Plant & Height", fontsize=16)
    plt.ylabel(metric.replace("_", " "))
    plt.xlabel("Plant")
    plt.legend(title="Drone Height")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, f"{metric}_Plant_vs_Height.png"), dpi=300)
    plt.close()

print(f"Drone-only plots saved in: {figures_path}")
