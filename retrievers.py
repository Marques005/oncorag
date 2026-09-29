"""
retrievers.py · Módulo de pesquisa reutilizável (vai ser uma ferramenta do agente no P3)

Todos os retrievers têm a mesma interface:
    r = Retriever(docs_df, ...)            # docs_df com colunas: pubid, text
    r.search(["pergunta 1", ...], k=20)    # -> lista (uma por pergunta) de [(pubid, score), ...] por ordem

Passo 2: BM25Retriever    (pesquisa por palavras)
Passo 3: DenseRetriever   (embeddings, pesquisa por significado)
Passo 4: HybridRetriever  (junta as duas com RRF)
Passo 5: RerankRetriever  (cross-encoder por cima de outro retriever)
"""
import os
import numpy as np
import bm25s


class BM25Retriever:
    """Pesquisa por palavras.

    Pontua cada documento pelas palavras que partilha com a pergunta, dando mais peso
    a palavras raras no corpus (IDF) e descontando documentos muito longos.
    """

    name = "bm25"

    def __init__(self, docs_df):
        self.ids = docs_df["pubid"].tolist()
        # tokenize: minúsculas, separa em palavras e remove stopwords inglesas ("the", "of", "is"...)
        corpus_tokens = bm25s.tokenize(docs_df["text"].tolist(), stopwords="en")
        self.model = bm25s.BM25()
        self.model.index(corpus_tokens)   # índice invertido: palavra -> documentos onde aparece

    def search(self, queries, k=20):
        q_tokens = bm25s.tokenize(queries, stopwords="en")
        idx, scores = self.model.retrieve(q_tokens, k=k)
        return [
            [(self.ids[i], float(s)) for i, s in zip(row_i, row_s)]
            for row_i, row_s in zip(idx, scores)
        ]


class DenseRetriever:
    """Pesquisa por significado (bi-encoder).

    1. Cada documento é transformado num vetor (embedding) UMA vez e guardado em disco.
    2. Cada pergunta é transformada num vetor no momento da pesquisa.
    3. Os documentos mais próximos (similaridade de cosseno) são devolvidos.
    """

    def __init__(self, docs_df, model_name, name, cache_dir="indexes", batch_size=32):
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.ids = docs_df["pubid"].tolist()
        self.model = SentenceTransformer(model_name)

        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{name}.npy")
        if os.path.exists(path):
            print(f"[{name}] A carregar embeddings guardados de {path}")
            self.emb = np.load(path)
        else:
            print(f"[{name}] A calcular embeddings de {len(self.ids)} documentos (só na 1.ª vez)...")
            self.emb = self.model.encode(
                docs_df["text"].tolist(),
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,   # vetores com comprimento 1 -> produto interno = cosseno
                convert_to_numpy=True,
            )
            np.save(path, self.emb)
        print(f"[{name}] {self.emb.shape[0]} vetores de dimensão {self.emb.shape[1]}")

    def search(self, queries, k=20):
        q = self.model.encode(queries, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
        sims = q @ self.emb.T                                   # (n_perguntas × n_docs): cosseno
        top = np.argpartition(-sims, k, axis=1)[:, :k]          # os k maiores (ainda desordenados)
        results = []
        for qi, cand in enumerate(top):
            order = cand[np.argsort(-sims[qi, cand])]           # ordena só esses k
            results.append([(self.ids[i], float(sims[qi, i])) for i in order])
        return results


class HybridRetriever:
    """Junta vários retrievers com Reciprocal Rank Fusion (RRF).

    Cada método dá a sua lista ordenada. Cada documento recebe 1/(k_rrf + posição) em cada lista,
    e as pontuações somam-se. Um documento bem colocado em VÁRIAS listas sobe para o topo.
    Usa só as posições (não os scores), por isso não é preciso pôr BM25 e cosseno na mesma escala.
    """

    def __init__(self, retrievers, name, k_rrf=60, candidates=100):
        self.retrievers = retrievers
        self.name = name
        self.k_rrf = k_rrf              # 60 é o valor clássico do artigo original do RRF
        self.candidates = candidates    # quantos resultados pedir a cada método antes de fundir

    def search(self, queries, k=20):
        lists = [r.search(queries, k=self.candidates) for r in self.retrievers]
        results = []
        for qi in range(len(queries)):
            fused = {}
            for method_results in lists:
                for rank, (pid, _) in enumerate(method_results[qi], start=1):
                    fused[pid] = fused.get(pid, 0.0) + 1.0 / (self.k_rrf + rank)
            top = sorted(fused.items(), key=lambda x: -x[1])[:k]
            results.append(top)
        return results


class RerankRetriever:
    """Pesquisa em 2 fases: triagem rápida + reordenação precisa.

    1. O retriever de 1.ª fase (rápido, bi-encoder/BM25) devolve os `candidates` melhores.
    2. Um cross-encoder lê cada par (pergunta, documento) JUNTOS e dá uma nota de relevância.
    3. Os candidatos são reordenados por essa nota.

    Nota: só REORDENA os candidatos. Se o artigo certo não estiver no top `candidates`
    da 1.ª fase, o reranker não o consegue recuperar.
    """

    def __init__(self, first_stage, docs_df, model_name, name, candidates=20, batch_size=32):
        from sentence_transformers import CrossEncoder

        self.first_stage = first_stage
        self.name = name
        self.candidates = candidates
        self.batch_size = batch_size
        self.text = dict(zip(docs_df["pubid"], docs_df["text"]))
        self.model = CrossEncoder(model_name, max_length=512)

    def search(self, queries, k=20):
        cands = self.first_stage.search(queries, k=self.candidates)
        pairs = [(q, self.text[pid]) for q, cand in zip(queries, cands) for pid, _ in cand]
        print(f"[{self.name}] A reordenar {len(pairs)} pares (pergunta, documento)...")
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=True)

        results, i = [], 0
        for cand in cands:
            n = len(cand)
            scored = [(pid, float(s)) for (pid, _), s in zip(cand, scores[i:i + n])]
            i += n
            results.append(sorted(scored, key=lambda x: -x[1])[:k])
        return results