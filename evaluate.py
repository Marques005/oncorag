"""
evaluate.py · Avalia os métodos de pesquisa com Recall@k e MRR

Para cada pergunta sabemos qual é o artigo certo (gold_pubid), por isso a avaliação é automática:
- Recall@k: em que % das perguntas o artigo certo aparece nos k primeiros resultados
- MRR:      média de 1/posição do artigo certo (1.º lugar = 1, 2.º = 0,5, ... ; fora do top-k = 0)
"""
import os, time
import pandas as pd
from retrievers import BM25Retriever, DenseRetriever, HybridRetriever, RerankRetriever

K = 20
os.makedirs("results", exist_ok=True)

docs = pd.read_parquet("data/corpus.parquet")
queries = pd.read_parquet("data/queries.parquet")
doc_text = dict(zip(docs.pubid, docs.text))


def rank_of_gold(results, gold):
    """Posição (1, 2, 3...) do artigo certo nos resultados, ou None se não estiver no top-k."""
    for pos, (pid, _) in enumerate(results, start=1):
        if pid == gold:
            return pos
    return None


def evaluate(retriever):
    t0 = time.time()
    all_results = retriever.search(queries["question"].tolist(), k=K)
    secs = time.time() - t0

    ranks = [rank_of_gold(res, gold) for res, gold in zip(all_results, queries["gold_pubid"])]
    r = pd.Series(ranks, dtype="float")
    metrics = {
        "method": retriever.name,
        "recall@1": (r <= 1).mean(),
        "recall@5": (r <= 5).mean(),
        "recall@20": (r <= 20).mean(),
        "mrr": (1 / r).fillna(0).mean(),
        "search_s": secs,
    }
    per_query = queries.assign(rank=ranks, top1=[res[0][0] for res in all_results])
    return metrics, per_query


def show_failures(per_query, n=3):
    """Mostra casos em que o artigo certo NÃO ficou em 1.º lugar: é aqui que se aprende."""
    fails = per_query[per_query["rank"] != 1].head(n)
    for _, row in fails.iterrows():
        pos = "fora do top-20" if pd.isna(row["rank"]) else f"{int(row['rank'])}.º lugar"
        print(f"\n  ❓ {row.question}")
        print(f"     Artigo certo ficou em: {pos}")
        print(f"     1.º resultado: {doc_text[row.top1][:200]}...")


if __name__ == "__main__":
    bm25 = BM25Retriever(docs)
    dense_general = DenseRetriever(docs, "sentence-transformers/all-MiniLM-L6-v2", name="dense_general")
    dense_biomed = DenseRetriever(docs, "NeuML/pubmedbert-base-embeddings", name="dense_biomed")

    hybrid_general = HybridRetriever([bm25, dense_general], name="hybrid_bm25+general")
    hybrid_biomed = HybridRetriever([bm25, dense_biomed], name="hybrid_bm25+biomed")

    retrievers = [
        bm25,
        dense_general,
        dense_biomed,
        hybrid_general,
        hybrid_biomed,
        # Passo 5: cross-encoder (treinado em pesquisas web, MS MARCO) a reordenar o top-20 do melhor híbrido
        RerankRetriever(hybrid_biomed, docs, "cross-encoder/ms-marco-MiniLM-L-6-v2",
                        name="hybrid_biomed+rerank", candidates=20),
    ]

    all_metrics, ranks_by_method = [], {}
    for r in retrievers:
        m, per_query = evaluate(r)
        all_metrics.append(m)
        ranks_by_method[r.name] = per_query["rank"]
        per_query.to_csv(f"results/{r.name}_ranks.csv", index=False)

    table = pd.DataFrame(all_metrics).round(3)
    table.to_csv("results/retrieval_results.csv", index=False)
    print("\n================ RESULTADOS ================")
    print(table.to_string(index=False))

    # Complementaridade: os métodos erram nas MESMAS perguntas ou em perguntas diferentes?
    rk = pd.DataFrame(ranks_by_method)
    bm25_top1 = rk["bm25"] == 1
    for name in rk.columns.drop("bm25"):
        dense_top1 = rk[name] == 1
        print(f"\n[{name} vs bm25]")
        print(f"  Só o BM25 acerta em 1.º:      {(bm25_top1 & ~dense_top1).sum()}")
        print(f"  Só o {name} acerta em 1.º: {(~bm25_top1 & dense_top1).sum()}")
        print(f"  Nenhum acerta em 1.º:         {(~bm25_top1 & ~dense_top1).sum()}")