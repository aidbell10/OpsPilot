"""Knowledge ingestion & chunking (Phase 3).

Loads synthetic knowledge-base documents, splits them into token-bounded
chunks with configurable overlap, attaches denormalised metadata, and persists
``documents`` + ``document_chunks`` rows. Embedding happens in a separate pass.
"""
