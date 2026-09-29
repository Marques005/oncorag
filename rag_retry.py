"""
rag_retry.py · Repete SÓ as perguntas que deram erro no rag.py (limite de pedidos da API)

- Refaz a pesquisa (rápido; o reranker demora ~1-2 min) para reconstruir os contextos
- Volta a perguntar ao LLM apenas as linhas com pred == "erro", com mais calma:
    * espera 2 s entre pedidos
    * se a API disser "try again in Xs", espera exatamente esse tempo
    * se o limite for DIÁRIO, pára e guarda o progresso (corre de novo amanhã)
- Recalcula as tabelas e o gráfico
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
PAUSE = 2.0  # segundos entre pedidos

client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"])
docs = pd.read_parquet("data/corpus.parquet")
queries = pd.read_parquet("data/queries.parquet")
queries = queries[queries["in_p1"]].reset_index(drop=True)
doc_text = dict(zip(docs.pubid, docs.text))

df = pd.read_csv("results/rag_results.csv", dtype={"qid": str})
todo = df.index[df["pred"] == "erro"].tolist()
print(f"Linhas com erro a repetir: {len(todo)}")

if todo:
    # ---------- Reconstruir os contextos (mesma pesquisa do rag.py) ----------
    questions = queries["question"].tolist()
    bm25 = BM25Retriever(docs)
    dense_general = DenseRetriever(docs, "sentence-transformers/all-MiniLM-L6-v2", name="dense_general")
    dense_biomed = DenseRetriever(docs, "NeuML/pubmedbert-base-embeddings", name="dense_biomed")
    cheap = HybridRetriever([bm25, dense_general], name="hybrid_bm25+general")
    best = RerankRetriever(HybridRetriever([bm25, dense_biomed], name="hybrid_bm25+biomed"),
                           docs, "cross-encoder/ms-marco-MiniLM-L-6-v2", name="hybrid_biomed+rerank")
    ctx_by_cond = {
        "rag_barato": dict(zip(queries.qid, [[p for p, _ in r] for r in cheap.search(questions, k=TOP_K)])),
        "rag_melhor": dict(zip(queries.qid, [[p for p, _ in r] for r in best.search(questions, k=TOP_K)])),
    }
    qinfo = queries.set_index("qid")


def make_prompt(q, ctx_ids):
    if not ctx_ids:
        return f"Question: {q}\nAnswer with only one word: yes, no or maybe."
    ctx = "\n\n".join(f"[{i}] {doc_text[pid]}" for i, pid in enumerate(ctx_ids, start=1))
    return f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer with only one word: yes, no or maybe."


def parse(text):
    m = re.search(r"\b(yes|no|maybe)\b", (text or "").lower())
    return m.group(1) if m else "invalid"


def wait_time(msg, attempt):
    """Lê o 'try again in 1m2.5s' da mensagem de erro; se não houver, espera 2^attempt (máx. 60 s)."""
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", msg)
    if m:
        return int(m.group(1) or 0) * 60 + float(m.group(2)) + 1
    return min(2 ** attempt, 60)


class DailyLimit(Exception):
    pass


def ask_patient(prompt, retries=8):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                max_tokens=512, temperature=0, reasoning_effort="low",
            )
            return r.choices[0].message.content or ""
        except Exception as e:
            msg = str(e)
            if "per day" in msg:
                raise DailyLimit(msg)
            secs = wait_time(msg, attempt)
            print(f"   limite de pedidos, a esperar {secs:.0f}s...")
            time.sleep(secs)
    return "ERRO: demasiadas tentativas"


done = 0
try:
    for n, idx in enumerate(todo, start=1):
        row = df.loc[idx]
        q = qinfo.loc[row.qid]
        cond = row.condition
        if cond == "sem_contexto":
            ctx_ids = []
        elif cond == "artigo_certo":
            ctx_ids = [q.gold_pubid]
        else:
            ctx_ids = ctx_by_cond[cond][row.qid]
        text = ask_patient(make_prompt(q.question, ctx_ids))
        pred = "erro" if text.startswith("ERRO") else parse(text)
        df.loc[idx, ["pred", "correct", "raw"]] = [pred, pred == row.gold, text]
        done += 1
        print(f"{n}/{len(todo)} repetido ({cond}): {pred}")
        if n % 10 == 0:
            df.to_csv("results/rag_results.csv", index=False)   # guarda o progresso
        time.sleep(PAUSE)
except DailyLimit as e:
    print(f"\n⚠️ Limite DIÁRIO da API atingido. Progresso guardado ({done} repetidos). Corre de novo amanhã.\n{e}")

df.to_csv("results/rag_results.csv", index=False)

# ---------- Resultados ----------
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