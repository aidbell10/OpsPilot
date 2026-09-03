"""Context construction, structured LLM generation, citation verification (Phase 3).

* build an evidence context from retrieved chunks (each tagged with its real id)
* prompt the LLM for a Pydantic-validated investigation result
* verify every citation resolves to a real retrieved chunk; drop / flag the rest
* return an explicit "insufficient evidence" result when support is inadequate

Retrieved document text is always treated as untrusted data, never instructions.
"""
