import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV
df = pd.read_csv("./Results/csv/predictions_glare.csv")

# Filter only misclassified samples
misclassified = df[df["ground_truth"] != df["prediction"]]

# Count how many times each true label was misclassified
error_counts = misclassified["ground_truth"].value_counts().sort_values(ascending=False)

# Plot
plt.figure(figsize=(10, 6))
error_counts.plot(kind='bar', color='blue', edgecolor='black')
plt.title("Misclassification Count per Ground Truth Class")
plt.xlabel("Ground Truth Label")
plt.ylabel("Number of Misclassifications")
plt.grid(alpha=0.5)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("./Results/figures/test1.png")
plt.show()

# plt.figure(figsize=(8, 5))
# plt.hist(misclassified["confidence"], bins=20, color='orange', edgecolor='black')
# plt.title("Confidence Distribution of Misclassified Samples")
# plt.xlabel("Confidence")
# plt.ylabel("Frequency")
# plt.grid(alpha=0.5)
# plt.savefig("./Results/figures/test2.png")
# plt.show()


# plt.figure(figsize=(10,6))
# sns.boxplot(data=misclassified, x="ground_truth", y="confidence")
# plt.xticks(rotation=45, ha='right')
# plt.title("Confidence Levels of Misclassified Samples per True Class")
# plt.tight_layout()
# plt.grid(alpha=0.5)
# plt.savefig("./Results/figures/test3.png")
# plt.show()

# Compute total misclassifications per ground truth class
misclassified = df[df["ground_truth"] != df["prediction"]]
top_classes = misclassified["ground_truth"].value_counts().nlargest(5).index.tolist()

# Filter only rows where ground truth or prediction is in top 5
subset_df = df[(df["ground_truth"].isin(top_classes)) | (df["prediction"].isin(top_classes))]

# Compute confusion matrix
cm = confusion_matrix(
    subset_df["ground_truth"],
    subset_df["prediction"],
    labels=top_classes
)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=top_classes)
disp.plot(ax=ax, cmap="Reds", colorbar=True)
plt.title("Zoomed-in Confusion Matrix for Top 5 Misclassified Classes")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("./Results/figures/test4.png")
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_misclassifications(df, plot_type="bar"):

    # Filter only incorrect predictions
    wrong = df[df["ground_truth"] != df["prediction"]]

    # Count misclassifications per (ground truth → prediction)
    mis_counts = (
        wrong.groupby(["ground_truth", "prediction"])
             .size()
             .reset_index(name="count")
    )

    if mis_counts.empty:
        print("No misclassifications found.")
        return

    # PLOT AS BAR CHART
    if plot_type == "bar":
        plt.figure(figsize=(10, 5))

        # Build label like "stop → speedlimit"
        mis_counts["pair"] = mis_counts["ground_truth"] + " → " + mis_counts["prediction"]

        plt.bar(mis_counts["pair"], mis_counts["count"])
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Misclassification Type")
        plt.ylabel("Count")
        plt.title("Traffic Sign Misclassification Counts")
        plt.tight_layout()
        plt.savefig("oma.png")
        plt.show()
        return


    print("Invalid plot_type. Use 'bar' or 'radar'.")

def plot_confusion_matrix(df, normalize=False):
  
    # Collect all classes
    classes = sorted(list(set(df["ground_truth"]) | set(df["prediction"])))

    # Build confusion matrix
    matrix = pd.crosstab(df["ground_truth"], df["prediction"],
                         rownames=["Ground Truth"], colnames=["Prediction"],
                         dropna=False)

    # Ensure full class axis
    matrix = matrix.reindex(index=classes, columns=classes, fill_value=0)

    # Normalize rows if requested
    if normalize:
        matrix = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0)

    plt.figure(figsize=(9, 7))

    # High-contrast colormap
    cmap = plt.cm.Blues

    # Draw heatmap
    plt.imshow(matrix, cmap=cmap)

    # Axis labels
    plt.xticks(np.arange(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(np.arange(len(classes)), classes)

    # Annotate each cell with readable text
    max_val = matrix.values.max()

    for i in range(len(classes)):
        for j in range(len(classes)):
            value = matrix.iloc[i, j]
            if normalize:
                text = f"{value:.2f}"
            else:
                text = f"{int(value)}"

            # Pick text color based on background intensity
            brightness = matrix.iloc[i, j] / max_val if max_val > 0 else 0
            color = "white" if brightness > 0.5 else "black"

            plt.text(j, i, text, ha="center", va="center",
                     color=color, fontsize=11, fontweight="bold")

    plt.title("Confusion Matrix (Ground Truth vs Prediction)", fontsize=14)
    cbar = plt.colorbar()
    cbar.set_label("Frequency" if not normalize else "Proportion")
    plt.xlabel("Prediction")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    plt.savefig("kire.png")
    plt.show()

plot_misclassifications(df)
plot_confusion_matrix(df)

