"""
rag.py · Teste de ponta a ponta: pesquisa + LLM

Nas 100 perguntas do P1, o gpt-oss-20b responde em 4 condições:
  1. sem_contexto   -> só a pergunta                                  (P1: 34%)
  2. rag_barato     -> top-3 do híbrido BM25 + embeddings gerais
  3. rag_melhor     -> top-3 do híbrido BM25 + biomédico + reranker
  4. artigo_certo   -> o resumo certo (pesquisa perfeita = teto)      (P1: 66%)

Também regista se o artigo certo estava no contexto, para separar erros de PESQUISA de erros de GERAÇÃO.
"""
import os, re, time
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv
from retrievers import BM25Retriever, DenseRetriever, HybridRetriever, RerankRetriever

load_dotenv()
MODEL = "openai/gpt-oss-20b"
TOP_K = 3
SYSTEM = "You are a biomedical expert. Reply with exactly one word: yes, no, or maybe."

client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"])
docs = pd.read_parquet("data/corpus.parquet")
queries = pd.read_parquet("data/queries.parquet")
queries = queries[queries["in_p1"]].reset_index(drop=True)          # as mesmas 100 perguntas do P1
doc_text = dict(zip(docs.pubid, docs.text))
questions = queries["question"].tolist()

# ---------- Pesquisa (feita uma vez para as 100 perguntas) ----------
bm25 = BM25Retriever(docs)
dense_general = DenseRetriever(docs, "sentence-transformers/all-MiniLM-L6-v2", name="dense_general")
dense_biomed = DenseRetriever(docs, "NeuML/pubmedbert-base-embeddings", name="dense_biomed")
cheap = HybridRetriever([bm25, dense_general], name="hybrid_bm25+general")
best = RerankRetriever(HybridRetriever([bm25, dense_biomed], name="hybrid_bm25+biomed"),
                       docs, "cross-encoder/ms-marco-MiniLM-L-6-v2", name="hybrid_biomed+rerank")

retrieved = {
    "rag_barato": [[pid for pid, _ in r] for r in cheap.search(questions, k=TOP_K)],
    "rag_melhor": [[pid for pid, _ in r] for r in best.search(questions, k=TOP_K)],
}


# ---------- Geração ----------
def make_prompt(q, ctx_ids):
    if not ctx_ids:
        return f"Question: {q}\nAnswer with only one word: yes, no or maybe."
    ctx = "\n\n".join(f"[{i}] {doc_text[pid]}" for i, pid in enumerate(ctx_ids, start=1))
    return f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer with only one word: yes, no or maybe."


def parse(text):
    m = re.search(r"\b(yes|no|maybe)\b", (text or "").lower())
    return m.group(1) if m else "invalid"


def ask(prompt, retries=5):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                max_tokens=512, temperature=0, reasoning_effort="low",
            )
            return r.choices[0].message.content or ""
        except Exception as e:                                    
            if attempt == retries - 1:
                return f"ERRO: {e}"
            time.sleep(2 ** attempt)


rows = []
for i, q in queries.iterrows():
    conditions = {
        "sem_contexto": [],
        "rag_barato": retrieved["rag_barato"][i],
        "rag_melhor": retrieved["rag_melhor"][i],
        "artigo_certo": [q.gold_pubid],
    }
    for cond, ctx_ids in conditions.items():
        text = ask(make_prompt(q.question, ctx_ids))
        pred = "erro" if text.startswith("ERRO") else parse(text)
        rows.append(dict(qid=q.qid, condition=cond, gold=q.answer, pred=pred,
                         correct=pred == q.answer, gold_in_context=q.gold_pubid in ctx_ids, raw=text))
    print(f"{i + 1}/{len(queries)} feito")

# ---------- Resultados ----------
df = pd.DataFrame(rows)
df.to_csv("results/rag_results.csv", index=False)

order = ["sem_contexto", "rag_barato", "rag_melhor", "artigo_certo"]
summary = df.groupby("condition").agg(
    accuracy=("correct", "mean"),
    gold_in_context=("gold_in_context", "mean"),
    invalid=("pred", lambda p: (p == "invalid").sum()),
    erros=("pred", lambda p: (p == "erro").sum()),
).reindex(order).round(3)
summary.to_csv("results/rag_summary.csv")
print("\n================ RAG DE PONTA A PONTA ================")
print(summary.to_string())

# De onde vêm os erros? (só nas condições RAG)
print("\nAccuracy consoante o artigo certo estava ou não no contexto:")
rag = df[df.condition.isin(["rag_barato", "rag_melhor"])]
print(rag.groupby(["condition", "gold_in_context"])["correct"].agg(["mean", "count"]).round(3).to_string())

ax = summary["accuracy"].plot(kind="bar", rot=0, figsize=(8, 5), color=["#999", "#4a9", "#2a7", "#555"])
ax.bar_label(ax.containers[0], fmt="%.2f")
ax.set_ylim(0, 1)
ax.set_ylabel("accuracy")
ax.set_title(f"PubMedQA (n=100), {MODEL}: sem contexto → RAG → artigo certo")
ax.figure.tight_layout()
ax.figure.savefig("results/rag_accuracy.png", dpi=150)
print("\nGráfico guardado em results/rag_accuracy.png")