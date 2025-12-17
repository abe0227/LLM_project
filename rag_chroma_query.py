"""rag_chroma_query.py

Reusable ChromaDB query utilities for your RAG project.

- Uses e5-large-v2 embeddings (SentenceTransformers).
- Applies E5 prefixes:
    - query: <text>  for queries
    - passage: <text> for documents (documents should already have this in your DB, but query side always uses query:)
- Uses normalized embeddings (recommended for E5).
- Supports step-gated retrieval via metadata filter: where={"pipeline_step": "StepX_..."}

Typical usage in other scripts:

    from rag_chroma_query import RagChromaQuery

    rq = RagChromaQuery(
        db_path="./chroma_db",
        collection_name="playwriting_rag_e5_v1",
        model_name="intfloat/e5-large-v2",
    )

    hits = rq.query("Help me define the premise", step="Step1_Premise", k=6)
    for h in hits:
        print(h["id"], h["distance"], h["metadata"].get("category"))

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import chromadb
from sentence_transformers import SentenceTransformer


Json = Dict[str, Any]


@dataclass
class RagHit:
    """A single retrieval result."""
    id: str
    distance: float
    document: Optional[str]
    metadata: Json


class RagChromaQuery:
    """Query helper for a Chroma collection using E5-large-v2 embeddings."""

    def __init__(
        self,
        db_path: str,
        collection_name: str,
        model_name: str = "intfloat/e5-large-v2",
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            db_path: Path to your persistent Chroma directory.
            collection_name: The collection name you ingested into.
            model_name: SentenceTransformers model name (default: intfloat/e5-large-v2).
            device: Optional. e.g., "cuda" or "cpu". If None, SentenceTransformer decides.
        """
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name=collection_name)

        # SentenceTransformer handles device internally; you can pass device if you want.
        if device:
            self.model = SentenceTransformer(model_name, device=device)
        else:
            self.model = SentenceTransformer(model_name)

    @staticmethod
    def _e5_query_prefix(text: str) -> str:
        text = (text or "").strip()
        return "query: " + text

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query using E5 conventions (query: prefix + L2 normalize)."""
        q = self._e5_query_prefix(query)
        vec = self.model.encode([q], normalize_embeddings=True)
        return vec[0].tolist()

    def query(
        self,
        query: str,
        step: Optional[str] = None,
        k: int = 6,
        where_extra: Optional[Json] = None,
        include: Sequence[str] = ("documents", "metadatas", "distances"),
    ) -> List[RagHit]:
        """
        Query the collection.

        Args:
            query: Natural language query.
            step: If provided, enforces step-gated retrieval: where={"pipeline_step": step}.
            k: Number of results.
            where_extra: Optional additional metadata filters to AND with pipeline_step.
            include: Chroma include fields.

        Returns:
            List[RagHit]
        """
        q_emb = self.embed_query(query)

        where: Optional[Json] = None
        if step and where_extra:
            # Chroma doesn't support nested AND in older APIs consistently, so we flatten where.
            where = {"pipeline_step": step, **where_extra}
        elif step:
            where = {"pipeline_step": step}
        elif where_extra:
            where = dict(where_extra)

        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            where=where,
            include=list(include),
        )

        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0] if "distances" in res else [None] * len(ids)
        docs = res.get("documents", [[]])[0] if "documents" in res else [None] * len(ids)
        metas = res.get("metadatas", [[]])[0] if "metadatas" in res else [{}] * len(ids)

        hits: List[RagHit] = []
        for i, _id in enumerate(ids):
            hits.append(
                RagHit(
                    id=_id,
                    distance=float(dists[i]) if dists[i] is not None else float("nan"),
                    document=docs[i] if docs else None,
                    metadata=metas[i] if metas else {},
                )
            )
        return hits

    def query_many_steps(
        self,
        query: str,
        steps: Sequence[str],
        k_per_step: int = 4,
        include: Sequence[str] = ("documents", "metadatas", "distances"),
    ) -> Dict[str, List[RagHit]]:
        """Convenience: run the same query against multiple steps (each step is filtered separately)."""
        out: Dict[str, List[RagHit]] = {}
        for step in steps:
            out[step] = self.query(query=query, step=step, k=k_per_step, include=include)
        return out

    def get_by_id(
        self,
        ids: Union[str, Sequence[str]],
        include: Sequence[str] = ("documents", "metadatas"),
    ) -> Dict[str, Any]:
        """Fetch raw stored items by id(s)."""
        if isinstance(ids, str):
            ids = [ids]
        return self.collection.get(ids=list(ids), include=list(include))


def pretty_print(hits: Sequence[RagHit], max_chars: int = 280) -> None:
    """Small helper to view results quickly in the console."""
    for rank, h in enumerate(hits, start=1):
        title = h.metadata.get("title") or ""
        cat = h.metadata.get("category") or ""
        step = h.metadata.get("pipeline_step") or ""
        src = h.metadata.get("source_name") or ""
        snippet = (h.document or "").replace("\n", " ")
        if len(snippet) > max_chars:
            snippet = snippet[: max_chars - 1] + "…"

        print(f"#{rank}  id={h.id}")
        print(f"    distance={h.distance:.4f}  step={step}  category={cat}  source={src}")
        if title:
            print(f"    title={title}")
        if snippet:
            print(f"    doc={snippet}")
        print()


if __name__ == "__main__":
    # Example quick test (edit paths/collection as needed)
    rq = RagChromaQuery(
        db_path="./chroma_db",
        collection_name="playwriting_rag_e5_v1",
        model_name="intfloat/e5-large-v2",
    )

    hits = rq.query(
        "My dialogue feels on-the-nose. How do I add subtext without losing clarity?",
        step="Step6_Details",
        k=6,
    )
    pretty_print(hits)
