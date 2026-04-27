"""FakeCollection: in-memory ChromaDB collection substitute for unit tests.

Supports the subset of the Chroma ``get()`` API used by ChromaFsBackend:
``ids``, ``where`` (``$in`` operator), ``where_document`` (``$contains`` /
``$regex``), and ``include``.
"""

from __future__ import annotations

import re
from typing import Any


class FakeCollection:
    """In-memory stand-in for ``chromadb.Collection``.

    Args:
        docs: List of document dicts.  Each dict MUST have:
            - ``id``: str — the document ID
            - ``document``: str — the document text
            - any additional keys are treated as metadata fields
    """

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self._docs = docs

    def get(
        self,
        *,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        docs = list(self._docs)

        if ids is not None:
            docs = [d for d in docs if d["id"] in ids]

        if where is not None:
            docs = [d for d in docs if _match_where(d, where)]

        if where_document is not None:
            docs = [d for d in docs if _match_where_document(d.get("document", ""), where_document)]

        include = include or ["ids", "documents", "metadatas"]
        result: dict[str, Any] = {}
        if "ids" in include:
            result["ids"] = [d["id"] for d in docs]
        if "documents" in include:
            result["documents"] = [d.get("document", "") for d in docs]
        if "metadatas" in include:
            result["metadatas"] = [
                {k: v for k, v in d.items() if k not in ("id", "document")}
                for d in docs
            ]
        return result


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------


def _match_where(doc: dict[str, Any], where: dict[str, Any]) -> bool:
    """Return True when the document satisfies all ``where`` clauses."""
    for field, cond in where.items():
        val = doc.get(field)
        if isinstance(cond, dict):
            for op, operand in cond.items():
                if op == "$in" and val not in operand:
                    return False
                if op == "$eq" and val != operand:
                    return False
                if op == "$ne" and val == operand:
                    return False
        elif val != cond:
            return False
    return True


def _match_where_document(document: str, where_document: dict[str, Any]) -> bool:
    """Return True when the document text satisfies all ``where_document`` clauses."""
    for op, pattern in where_document.items():
        if op == "$contains" and pattern not in document:
            return False
        if op == "$regex" and not re.search(pattern, document):
            return False
    return True
