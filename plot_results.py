"""
plot_results.py · Gráfico da avaliação da pesquisa (lê results/retrieval_results.csv, não recalcula nada)
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/retrieval_results.csv")
labels = {
    "bm25": "BM25",
    "dense_general": "Dense (geral)",
    "dense_biomed": "Dense (biomédico)",
    "hybrid_bm25+general": "Híbrido\nBM25+geral",
    "hybrid_bm25+biomed": "Híbrido\nBM25+biomédico",
    "hybrid_biomed+rerank": "Híbrido biomédico\n+ reranker",
}
df["label"] = df["method"].map(labels).fillna(df["method"])

fig, ax = plt.subplots(figsize=(11, 5))
x = range(len(df))
w = 0.38
b1 = ax.bar([i - w / 2 for i in x], df["recall@1"], w, label="Recall@1")
b2 = ax.bar([i + w / 2 for i in x], df["mrr"], w, label="MRR")
ax.bar_label(b1, fmt="%.3f", fontsize=8)
ax.bar_label(b2, fmt="%.3f", fontsize=8)
ax.set_xticks(list(x))
ax.set_xticklabels(df["label"], fontsize=9)
ax.set_ylim(0.8, 1.0)
ax.set_ylabel("score")
ax.set_title("Recuperação do artigo certo em 10 000 resumos (1 000 perguntas PubMedQA)")
ax.legend()
fig.tight_layout()
fig.savefig("results/retrieval_chart.png", dpi=150)
print("Gráfico guardado em results/retrieval_chart.png")