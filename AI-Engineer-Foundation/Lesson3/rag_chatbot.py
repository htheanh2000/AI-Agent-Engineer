import os
import chromadb
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_corpus(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def split_into_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 50):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - chunk_overlap

    return chunks


def build_vector_store(chunks):
    client_chroma = chromadb.Client()
    collection = client_chroma.create_collection(name="noi_quy_cong_ty")

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": "noi_quy_cong_ty", "chunk_id": i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
    )

    return collection


def retrieve_relevant_chunks(collection, question: str, top_k: int = 3):
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
    )
    docs = results["documents"][0]
    return docs


def answer_question_with_context(question: str, context_chunks):
    context_text = "\n\n---\n\n".join(context_chunks)

    system_prompt = (
        "Bạn là chatbot nội bộ của công ty. "
        "Bạn chỉ được trả lời dựa trên nội dung nội quy công ty được cung cấp dưới đây. "
        "Nếu không tìm thấy câu trả lời, hãy nói 'Tôi không chắc theo nội quy hiện tại.'"
    )

    user_content = (
        f"Nội quy công ty:\n{context_text}\n\n"
        f"Câu hỏi của nhân viên: {question}\n\n"
        "Trả lời rõ ràng, ngắn gọn bằng tiếng Việt."
    )

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    return resp.choices[0].message.content


if __name__ == "__main__":
    # 1. Load & index
    corpus = load_corpus("noi_quy_cong_ty.txt")
    chunks = split_into_chunks(corpus, chunk_size=500, chunk_overlap=50)
    collection = build_vector_store(chunks)

    print("=== Chatbot Nội bộ – Nội quy Công ty (RAG) ===")
    print("Gõ câu hỏi về nội quy công ty. Enter để thoát.")

    # 2. Loop hỏi–đáp
    while True:
        question = input("\n👤 Bạn: ").strip()
        if not question:
            break

        context_chunks = retrieve_relevant_chunks(collection, question, top_k=3)
        answer = answer_question_with_context(question, context_chunks)

        print("\n🤖 Bot:", answer)
