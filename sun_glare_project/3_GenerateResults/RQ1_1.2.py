import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_prediction_counts(normal_csv, glare_csv, glare_half_csv):

    df_normal = pd.read_csv(normal_csv)
    df_glare = pd.read_csv(glare_csv)
    df_glare_half = pd.read_csv(glare_half_csv)

    classes = sorted(list(
        set(df_normal["prediction"]) |
        set(df_glare["prediction"]) |
        set(df_glare_half["prediction"])
    ))

    normal_counts = [(df_normal["prediction"] == cls).sum() for cls in classes]
    glare_counts = [(df_glare["prediction"] == cls).sum() for cls in classes]
    glare_half_counts = [(df_glare_half["prediction"] == cls).sum() for cls in classes]

    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(12, 6))

    plt.bar(x - width, normal_counts, width,
            label="W/o Sun Glare", color="#1f77b4")

    plt.bar(x, glare_counts, width,
            label="Full Sun Glare", color="#ff7f0e")

    plt.bar(x + width, glare_half_counts, width,
            label="Half Sun Glare", color="#ffa500")

    plt.xticks(x, classes, rotation=45)
    plt.ylabel("Number of Predictions")
    plt.title("Normal vs Glare — Prediction Count per Traffic Sign Class")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("RQ1.png")
    plt.show()



plot_prediction_counts(
    normal_csv="../Results/csv/predictions.csv",
    glare_csv="../Results/csv/predictions_glare.csv",
    glare_half_csv="../Results/csv/predictions_glare_half.csv",


)
