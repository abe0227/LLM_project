"""query_cli.py

Interactive CLI for querying your ChromaDB RAG collection.

User inputs:
- query text
- pipeline step (as integer 1~6)
- top-k (number of results)

Then it prints the retrieved card *content* extracted from stored document text.

This script uses RagChromaQuery (E5-large-v2) from rag_chroma_query.py.

Usage:
  python query_cli.py --db ./chroma_db --collection playwriting_rag_e5_v1

Enter blank query to exit.
"""

from __future__ import annotations

import argparse
import re
from typing import Optional

from rag_chroma_query import RagChromaQuery


# Extract just the "Content: ..." region from our stored document string
CONTENT_RE = re.compile(
    r"\bContent:\s*(?P<content>.*?)(?:\n(?:Checklist Items:|Symptoms:|Root cause:|Fix:|$))",
    re.DOTALL,
)


STEP_MAP = {
    1: "Step1_Premise",
    2: "Step2_Premise_Consistency",
    3: "Step3_Act_Structure",
    4: "Step4_Character_Arc",
    5: "Step5_Conflict_Escalation",
    6: "Step6_Details",
}

STEP_DESC = {
    1: "建立故事核心前提 Premise（Theme / Protagonist / Goal / Conflict / Stakes / 1-sentence premise）",
    2: "檢查 Premise 是否內部矛盾（動機/行動矛盾、主題與行動衝突、goal 不清、stakes 不足、靠巧合或外力解決）",
    3: "建立三幕結構（Act1 Setup / Act2 Confrontation + Midpoint / Act3 Resolution + Payoff）",
    4: "角色弧線檢查（Arc 類型、是否支持 Theme、是否與 Conflict 衝突）",
    5: "衝突與節奏升級（initial → midpoint → crisis、escalating stakes、reversals）",
    6: "細節檢查（場景/對白/角色一致性/格式/節奏快慢等）",
}


def extract_content(document: Optional[str]) -> str:
    """Extract the 'Content:' section from our stored document text."""
    if not document:
        return ""
    doc = document.replace("\r\n", "\n").replace("\r", "\n")
    m = CONTENT_RE.search(doc)
    if m:
        return m.group("content").strip()
    return doc.replace("passage: ", "", 1).strip()


def print_step_help() -> None:
    print("Pipeline steps (enter 1~6):")
    for i in range(1, 7):
        print(f"  {i}. {STEP_MAP[i]}  —  {STEP_DESC[i]}")
    print()


def parse_step(step_input: str) -> str:
    s = (step_input or "").strip()
    if not s:
        return STEP_MAP[1]  # default Step 1
    try:
        n = int(s)
    except ValueError:
        raise ValueError("Pipeline step must be an integer 1~6.")
    if n not in STEP_MAP:
        raise ValueError("Pipeline step must be in 1~6.")
    return STEP_MAP[n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="./chroma_db", help="Chroma persistent DB directory")
    ap.add_argument("--collection", default="playwriting_rag_e5_v1", help="Chroma collection name")
    ap.add_argument("--model", default="intfloat/e5-large-v2", help="SentenceTransformers model name")
    ap.add_argument("--device", default=None, help="Optional: cpu/cuda")
    args = ap.parse_args()

    rq = RagChromaQuery(
        db_path=args.db,
        collection_name=args.collection,
        model_name=args.model,
        device=args.device,
    )

    print("=" * 88)
    print("RAG Query CLI (E5-large-v2 + ChromaDB)")
    print(f"DB: {args.db}")
    print(f"Collection: {args.collection}")
    print("輸入空白 query 直接結束。\n")
    print_step_help()

    default_k = 6

    while True:
        query = input("Query text: ").strip()
        if not query:
            print("Bye.")
            break

        step_raw = input("Pipeline step (1~6, default=1, type 'help' to show steps): ").strip()
        if step_raw.lower() in {"help", "h", "?"}:
            print_step_help()
            step_raw = input("Pipeline step (1~6, default=1): ").strip()

        try:
            step = parse_step(step_raw)
        except ValueError as e:
            print(f"Step input error: {e}  -> using default step=1\n")
            step = STEP_MAP[1]

        k_str = input(f"Top-k (default {default_k}): ").strip()
        try:
            k = int(k_str) if k_str else default_k
        except ValueError:
            print("Top-k 不是整數，已使用預設值。\n")
            k = default_k

        hits = rq.query(query=query, step=step, k=k)

        print("\n" + "-" * 88)
        print(f"Results: {len(hits)}  (step={step})")
        print("-" * 88)

        if not hits:
            print("(沒有找到結果)\n")
            continue

        for i, h in enumerate(hits, start=1):
            content = extract_content(h.document)
            title = h.metadata.get("title") or ""
            category = h.metadata.get("category") or ""
            source = h.metadata.get("source_name") or ""

            print(f"#{i}  id={h.id}  distance={h.distance:.4f}")
            if title:
                print(f"    title: {title}")
            print(f"    category: {category}  source: {source}")
            print("    content:")
            print("    " + content.replace("\n", "\n    "))
            print()

        print("=" * 88 + "\n")


if __name__ == "__main__":
    main()
