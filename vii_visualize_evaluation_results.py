import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# RESULTS WITHOUT CROSS-ENCODER
# ============================================================

without_cross_encoder = [
    # BM25
    ["BM25 | fixed_token", 0.75, 0.85625, 1.0],
    ["BM25 | recursive", 0.73125, 0.85, 0.99375],
    ["BM25 | sentence", 0.7375, 0.8625, 0.99375],
    ["BM25 | semantic", 0.74375, 0.83125, 0.98125],

    # Dense - fixed_token
    ["OpenAI | fixed_token", 0.83125, 0.9625, 1.0],
    ["BioBERT | fixed_token", 0.5625, 0.70625, 0.95625],
    ["BGE | fixed_token", 0.73125, 0.8375, 1.0],
    ["MedCPT | fixed_token", 0.54375, 0.71875, 0.96875],

    # Dense - recursive
    ["OpenAI | recursive", 0.8625, 0.95, 1.0],
    ["BioBERT | recursive", 0.5875, 0.70625, 0.95625],
    ["BGE | recursive", 0.73125, 0.81875, 1.0],
    ["MedCPT | recursive", 0.525, 0.7125, 0.975],

    # Dense - sentence
    ["OpenAI | sentence", 0.86875, 0.9625, 0.99375],
    ["BioBERT | sentence", 0.5875, 0.725, 0.95],
    ["BGE | sentence", 0.75, 0.875, 0.99375],
    ["MedCPT | sentence", 0.6375, 0.76875, 0.95625],

    # Dense - semantic
    ["OpenAI | semantic", 0.89375, 0.96875, 1.0],
    ["BioBERT | semantic", 0.55, 0.68125, 0.9125],
    ["BGE | semantic", 0.81875, 0.875, 1.0],
    ["MedCPT | semantic", 0.5375, 0.69375, 0.95625],
]


# ============================================================
# RESULTS WITH CROSS-ENCODER
# ============================================================

with_cross_encoder = [
    # BM25 + Cross-Encoder
    ["BM25 + Cross-Encoder | fixed_token", 0.2375, 0.3375, 0.75],
    ["BM25 + Cross-Encoder | recursive", 0.2125, 0.34375, 0.725],
    ["BM25 + Cross-Encoder | sentence", 0.225, 0.3625, 0.825],
    ["BM25 + Cross-Encoder | semantic", 0.24375, 0.39375, 0.85],

    # Dense + Cross-Encoder - fixed_token
    ["OpenAI + Cross-Encoder | fixed_token", 0.9125, 0.9625, 1.0],
    ["BioBERT + Cross-Encoder | fixed_token", 0.85625, 0.9125, 0.99375],
    ["BGE + Cross-Encoder | fixed_token", 0.91875, 0.95625, 1.0],
    ["MedCPT + Cross-Encoder | fixed_token", 0.8875, 0.93125, 0.98125],

    # Dense + Cross-Encoder - recursive
    ["OpenAI + Cross-Encoder | recursive", 0.8625, 0.95625, 1.0],
    ["BioBERT + Cross-Encoder | recursive", 0.81875, 0.9125, 0.975],
    ["BGE + Cross-Encoder | recursive", 0.8625, 0.95, 1.0],
    ["MedCPT + Cross-Encoder | recursive", 0.84375, 0.9375, 0.9875],

    # Dense + Cross-Encoder - sentence
    ["OpenAI + Cross-Encoder | sentence", 0.85, 0.93125, 0.99375],
    ["BioBERT + Cross-Encoder | sentence", 0.76875, 0.8375, 0.94375],
    ["BGE + Cross-Encoder | sentence", 0.84375, 0.93125, 0.9875],
    ["MedCPT + Cross-Encoder | sentence", 0.83125, 0.9, 0.96875],

    # Dense + Cross-Encoder - semantic
    ["OpenAI + Cross-Encoder | semantic", 0.875, 0.9375, 1.0],
    ["BioBERT + Cross-Encoder | semantic", 0.83125, 0.88125, 0.95625],
    ["BGE + Cross-Encoder | semantic", 0.86875, 0.93125, 1.0],
    ["MedCPT + Cross-Encoder | semantic", 0.825, 0.88125, 0.9625],
]


# ============================================================
# HYBRID RESULTS
# BM25 + Dense Retrieval
# ============================================================

hybrid = [
    # fixed_token
    ["BM25 + OpenAI | fixed_token", 0.3125, 0.5125, 1.0],
    ["BM25 + BioBERT | fixed_token", 0.3, 0.48125, 0.95625],
    ["BM25 + BGE | fixed_token", 0.3125, 0.55625, 0.975],
    ["BM25 + MedCPT | fixed_token", 0.325, 0.525, 0.98125],

    # recursive
    ["BM25 + OpenAI | recursive", 0.31875, 0.49375, 1.0],
    ["BM25 + BioBERT | recursive", 0.35, 0.4875, 0.94375],
    ["BM25 + BGE | recursive", 0.31875, 0.5375, 0.975],
    ["BM25 + MedCPT | recursive", 0.33125, 0.54375, 0.96875],

    # sentence
    ["BM25 + OpenAI | sentence", 0.33125, 0.475, 1.0],
    ["BM25 + BioBERT | sentence", 0.3375, 0.475, 0.91875],
    ["BM25 + BGE | sentence", 0.30625, 0.475, 0.99375],
    ["BM25 + MedCPT | sentence", 0.33125, 0.4625, 0.98125],

    # semantic
    ["BM25 + OpenAI | semantic", 0.225, 0.50625, 1.0],
    ["BM25 + BioBERT | semantic", 0.24375, 0.4375, 0.9125],
    ["BM25 + BGE | semantic", 0.2625, 0.50625, 0.99375],
    ["BM25 + MedCPT | semantic", 0.20625, 0.33125, 0.9375],
]


# ============================================================
# HEATMAP FUNCTION
# ============================================================

def create_heatmap(results, title, filename):

    df = pd.DataFrame(
        results,
        columns=[
            "Configuration",
            "Recall@5",
            "Recall@10",
            "Recall@50"
        ]
    )

    df = df.set_index("Configuration")

    fig, ax = plt.subplots(
        figsize=(10, max(7, len(df) * 0.55))
    )

    # Keep all three heatmaps on the same 0-1 scale
    image = ax.imshow(
        df.values,
        aspect="auto",
        vmin=0,
        vmax=1
    )

    # X-axis labels
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns)

    # Y-axis labels
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)

    # Display recall values inside cells
    for row in range(len(df.index)):
        for col in range(len(df.columns)):
            ax.text(
                col,
                row,
                f"{df.iloc[row, col]:.3f}",
                ha="center",
                va="center"
            )

    # Color scale
    cbar = plt.colorbar(image, ax=ax)
    cbar.set_label("Recall")

    ax.set_title(title)
    ax.set_xlabel("Evaluation Metric")
    ax.set_ylabel("Retrieval Configuration")

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# CREATE HEATMAP 1
# WITHOUT CROSS-ENCODER
# ============================================================

create_heatmap(
    without_cross_encoder,
    "Retrieval Performance Without Cross-Encoder",
    "without_cross_encoder_heatmap.png"
)


# ============================================================
# CREATE HEATMAP 2
# WITH CROSS-ENCODER
# ============================================================

create_heatmap(
    with_cross_encoder,
    "Retrieval Performance With Cross-Encoder",
    "with_cross_encoder_heatmap.png"
)


# ============================================================
# CREATE HEATMAP 3
# HYBRID RANKING
# ============================================================

create_heatmap(
    hybrid,
    "Hybrid Retrieval Performance: BM25 + Dense Retrieval",
    "hybrid_heatmap.png"
)