"""Retrieval: semantic, lexical, fusion, reranking (Phases 3, 5, 6, 7).

Plain-Python implementations kept deliberately transparent:

* ``semantic``  — pgvector cosine similarity over chunk embeddings
* ``lexical``   — PostgreSQL full-text search over ``content_tsv``
* ``fusion``    — Reciprocal Rank Fusion of ranked lists
* ``rerank``    — cross-encoder reranking (pretrained, then fine-tuned)
* ``filters``   — metadata predicates (service / version / environment / type / date)
"""
