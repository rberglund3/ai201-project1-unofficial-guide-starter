import json
import random

def run_diagnostics():
    # Load the chunks you just generated
    with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Total chunks loaded: {len(chunks)}\n")

    # DIAGNOSTIC 1: Empty chunks check
    empty_chunks = [c for c in chunks if len(c['text'].strip()) == 0]
    if empty_chunks:
        print(f"⚠️ WARNING: Found {len(empty_chunks)} empty chunks!")
    else:
        print("✅ No empty chunks found.")

    # DIAGNOSTIC 2: Same length check
    lengths = [len(c['text']) for c in chunks]
    unique_lengths = len(set(lengths))
    if unique_lengths <= 2 and len(chunks) > 5:
        print("⚠️ WARNING: Chunks are suspiciously identical in length. The text splitter might be slicing mid-word.")
    else:
        print(f"✅ Chunk lengths vary naturally ({unique_lengths} different lengths found).")

    print("\n--- 🎲 5 RANDOM CHUNKS FOR MANUAL REVIEW 🎲 ---")

    # Grab 5 random chunks
    sample_chunks = random.sample(chunks, min(5, len(chunks)))

    for i, chunk in enumerate(sample_chunks, 1):
        text = chunk['text']
        source = chunk['metadata'].get('source', 'UNKNOWN SOURCE')

        print(f"\n[ Chunk {i} ]")
        print(f"📁 Source: {source}")
        print(f"📏 Length: {len(text)} characters")
        print(f"📝 Text:\n{text}")
        print("-" * 50)

if __name__ == "__main__":
    run_diagnostics()
