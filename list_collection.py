import argparse
import chromadb

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="./chroma_db")
args = ap.parse_args()

client = chromadb.PersistentClient(path=args.db)

# 新版/舊版 chroma API 可能不同，這段同時兼容常見情況
try:
    cols = client.list_collections()
    print("Collections:")
    for c in cols:
        # 有些版本回傳物件，有些回傳 dict/str
        name = getattr(c, "name", None) or (c.get("name") if isinstance(c, dict) else str(c))
        print("-", name)
except Exception as e:
    print("Failed to list collections:", e)
