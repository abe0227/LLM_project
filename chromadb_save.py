import json
import chromadb
from sentence_transformers import SentenceTransformer

JSONL_PATH = "rag_expanded_single_step_cards_en_nolang.jsonl"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "playwriting_rag_e5_v1"

model = SentenceTransformer("intfloat/e5-large-v2")  # 1024 dims :contentReference[oaicite:3]{index=3}

def e5_embed(texts):
    # E5 建議 normalize_embeddings=True :contentReference[oaicite:4]{index=4}
    return model.encode(texts, normalize_embeddings=True).tolist()

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}  # normalize + cosine 是常見搭配
)

def build_document(card: dict) -> str:
    parts = []
    if card.get("title"):   parts.append(f"Title: {card['title']}")
    if card.get("content"): parts.append(f"Content: {card['content']}")

    items = card.get("items") or []
    if items:
        parts.append("Checklist Items:")
        parts.extend([f"- {x}" for x in items])

    symptoms = card.get("symptoms") or []
    if symptoms:
        parts.append("Symptoms:")
        parts.extend([f"- {x}" for x in symptoms])

    if card.get("root_cause"):
        parts.append(f"Root cause: {card['root_cause']}")

    fix = card.get("fix") or []
    if fix:
        parts.append("Fix:")
        parts.extend([f"- {x}" for x in fix])

    # E5 passage 前綴 :contentReference[oaicite:5]{index=5}
    return "passage: " + "\n".join(parts).strip()

def build_metadata(card: dict) -> dict:
    src = card.get("source") or {}
    tags = card.get("tags") or []
    tags_str = ",".join([t for t in tags if isinstance(t, str)])

    return {
        "type": card.get("type", ""),
        "category": card.get("category", ""),
        "pipeline_step": card.get("pipeline_step", ""),  # 核心 filter 欄
        "source_name": src.get("name", ""),
        "source_url": src.get("url", ""),
        "tags": tags_str
    }

batch = 200
ids, docs, metas = [], [], []

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        card = json.loads(line)
        ids.append(card["id"])
        docs.append(build_document(card))
        metas.append(build_metadata(card))

        if len(ids) >= batch:
            embs = e5_embed(docs)
            collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
            ids, docs, metas = [], [], []

if ids:
    embs = e5_embed(docs)
    collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

print("Done. Count =", collection.count())
