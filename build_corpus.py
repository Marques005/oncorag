"""
Passo 1 · Construir o corpus e as perguntas

- Corpus: 1 000 artigos do PubMedQA anotado (os "certos") + N_DISTRACTORS do PubMedQA não anotado (distração)
- Perguntas: as 1 000 do PubMedQA anotado, cada uma com o pubid do seu artigo certo
- Marca as 100 perguntas usadas no P1 (mesma seed) para o teste de ponta a ponta
"""
import os
import pandas as pd
from datasets import load_dataset

N_DISTRACTORS = 9000
SEED = 42
os.makedirs("data", exist_ok=True)

print("A carregar o PubMedQA...")
lab = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
unl = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled", split="train").shuffle(seed=SEED).select(range(N_DISTRACTORS))


def to_doc(ex, source):
    # Só o resumo (contexts); a pergunta NÃO entra no documento, senão seria batota
    return {"pubid": str(ex["pubid"]), "text": " ".join(ex["context"]["contexts"]), "source": source}


docs = pd.DataFrame([to_doc(e, "labeled") for e in lab] + [to_doc(e, "distractor") for e in unl])
docs = docs.drop_duplicates("pubid").reset_index(drop=True)

# As mesmas 100 perguntas do P1: shuffle(seed=42) + as primeiras 100
p1_ids = set(str(x) for x in lab.shuffle(seed=SEED).select(range(100))["pubid"])

queries = pd.DataFrame({
    "qid": [str(x) for x in lab["pubid"]],
    "question": lab["question"],
    "gold_pubid": [str(x) for x in lab["pubid"]],
    "answer": lab["final_decision"],
})
queries["in_p1"] = queries["qid"].isin(p1_ids)

docs.to_parquet("data/corpus.parquet", index=False)
queries.to_parquet("data/queries.parquet", index=False)

# ---- Verificações ----
words = docs["text"].str.split().str.len()
print(f"\nCorpus: {len(docs)} documentos ({(docs.source == 'labeled').sum()} certos + {(docs.source == 'distractor').sum()} distração)")
print(f"Palavras por documento: média {words.mean():.0f}, máximo {words.max()}")
print(f"Perguntas: {len(queries)} ({queries.in_p1.sum()} marcadas como P1)")
print(f"Todos os artigos certos estão no corpus? {queries.gold_pubid.isin(docs.pubid).all()}")
print("\nExemplo:")
q = queries.iloc[0]
print("  Pergunta:", q.question)
print("  Artigo certo:", docs.loc[docs.pubid == q.gold_pubid, "text"].iloc[0][:300], "...")