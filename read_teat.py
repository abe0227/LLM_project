"""read_teat.py

Smoke-test script to query your ChromaDB RAG collection using RagChromaQuery (E5-large-v2).

It runs the test queries previously proposed (Step1~Step6 + a step-gating check).

Usage:
  python read_teat.py --db ./chroma_db --collection playwriting_rag_e5_v1

If you omit args, it uses defaults below.
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

from rag_chroma_query import RagChromaQuery, pretty_print


Test = Tuple[str, str, str]  # (name, step, query)


def build_tests() -> List[Test]:
    return [
        # Step 1: Premise
        (
            "S1-1 Premise: theme/logline readiness",
            "Step1_Premise",
            "Help me define the premise: theme, protagonist, goal, conflict, stakes. Also check if the theme is clear and logline-ready.",
        ),
        (
            "S1-2 Premise: add a unique twist",
            "Step1_Premise",
            "My premise feels familiar. How can I add a unique twist without changing the core theme?",
        ),

        # Step 2: Premise consistency
        (
            "S2-1 Premise contradictions: motivation vs action, coincidence/deus ex",
            "Step2_Premise_Consistency",
            "Find contradictions in this premise: the protagonist wants X but repeatedly chooses actions that undermine X. Also check if the resolution relies on coincidence or outside intervention.",
        ),
        (
            "S2-2 Premise contradictions: tone promise mismatch",
            "Step2_Premise_Consistency",
            "Does my premise promise a comedy but the core conflict forces a bleak tragedy? Help me spot tone contradictions.",
        ),

        # Step 3: Act structure
        (
            "S3-1 Act structure: 3 acts + midpoint + payoff",
            "Step3_Act_Structure",
            "Build a three-act structure for this premise: Act 1 setup, Act 2 confrontation (define midpoint), Act 3 resolution (check payoff).",
        ),
        (
            "S3-2 Act structure: ending unfinished or too neat",
            "Step3_Act_Structure",
            "My ending feels either unfinished or too neat. What are the common structure problems and how to fix the payoff?",
        ),

        # Step 4: Character arc
        (
            "S4-1 Character arc type + supports theme",
            "Step4_Character_Arc",
            "Determine whether the protagonist arc is positive, negative, or flat, and whether the arc supports the theme and conflicts with the external goal.",
        ),
        (
            "S4-2 Character arc: protagonist feels flat",
            "Step4_Character_Arc",
            "My protagonist feels flat. What character flaw and self-recognition/reversal moment would make the arc feel earned?",
        ),

        # Step 5: Conflict escalation
        (
            "S5-1 Conflict escalation chain",
            "Step5_Conflict_Escalation",
            "Evaluate conflict escalation: initial conflict → midpoint shift → crisis. Do stakes escalate and do we have enough reversals?",
        ),
        (
            "S5-2 Conflict escalation: increase jeopardy without contrivance",
            "Step5_Conflict_Escalation",
            "My story feels too easy for the protagonist. How can I increase jeopardy and obstacles without contrivance?",
        ),

        # Step 6: Details
        (
            "S6-1 Details: on-the-nose dialogue",
            "Step6_Details",
            "My dialogue is on-the-nose. How do I add subtext and cut exposition without losing clarity?",
        ),
        (
            "S6-2 Details: formatting mistakes (O.S/O.C/CONT'D)",
            "Step6_Details",
            "Check common screenplay formatting mistakes: CONT'D usage, O.S vs O.C labels, and overusing transitions or camera directions.",
        ),

        # Step-gating verification
        (
            "GateCheck: ask for dialogue rewrite but force Step1 retrieval",
            "Step1_Premise",
            "Rewrite my dialogue to be less on-the-nose and more natural.",
        ),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="./chroma_db", help="Path to Chroma persistent directory")
    ap.add_argument("--collection", default="playwriting_rag_e5_v1", help="Chroma collection name")
    ap.add_argument("--model", default="intfloat/e5-large-v2", help="SentenceTransformers model name")
    ap.add_argument("--k", type=int, default=6, help="Top-k results per query")
    args = ap.parse_args()

    rq = RagChromaQuery(
        db_path=args.db,
        collection_name=args.collection,
        model_name=args.model,
    )

    tests = build_tests()

    for name, step, q in tests:
        print("=" * 88)
        print(f"{name}")
        print(f"step={step}")
        print(f"query={q}")
        print("-" * 88)
        hits = rq.query(query=q, step=step, k=args.k)
        pretty_print(hits)

        # Special check: confirm pipeline_step in top hit metadata for the gate test
        if name.startswith("GateCheck") and hits:
            top_step = hits[0].metadata.get("pipeline_step")
            print(f"[GateCheck] top_hit.pipeline_step = {top_step}\n")

    print("=" * 88)
    print("Done.")


if __name__ == "__main__":
    main()
