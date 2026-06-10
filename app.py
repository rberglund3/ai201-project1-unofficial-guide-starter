import os
import gradio as gr
from dotenv import load_dotenv
from groq import Groq
from embed_and_retrieve import retrieve

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in your .env file.")

client = Groq(api_key=GROQ_API_KEY)

def ask_housing_guide(question: str, selected_source: str) -> tuple[str, str]:
    # 1. Dynamic Dropdown Logic
    if selected_source == "Search All":
        actual_filter = None
        search_k = 12         # Throw a massive net for the whole database
        use_mmr_flag = True   # Force diversity so one building doesn't dominate
    else:
        actual_filter = selected_source
        search_k = 4          # Laser-focused net for specific buildings
        use_mmr_flag = False  # Pure mathematical match

    # Run the retrieval with our dynamic settings
    retrieval_response = retrieve(
        question,
        k=search_k,
        source_filter=actual_filter,
        use_mmr=use_mmr_flag
    )

    documents = retrieval_response.get('documents', [[]])[0] if retrieval_response else []
    metadatas = retrieval_response.get('metadatas', [[]])[0] if retrieval_response else []

    if not documents:
        return "I don't have enough information in my database to answer that question.", "No sources found."

    # 2. Programmatically aggregate sources
    unique_sources = set()
    context_blocks = []

    for doc, meta in zip(documents, metadatas):
        source_name = meta.get('source', 'Unknown Document')
        chunk_idx = meta.get('chunk_index', 'N/A')
        unique_sources.add(source_name)
        context_blocks.append(f"--- START CHUNK (Source: {source_name}, Position: {chunk_idx}) ---\n{doc}\n--- END CHUNK ---")

    context_str = "\n\n".join(context_blocks)

    # 3. System Prompt
    system_prompt = (
        "You are an expert, highly objective assistant helping Georgia Tech students find off-campus housing.\n"
        "Your task is to answer the user's question using ONLY the provided document chunks below.\n\n"
        "CRITICAL RULES:\n"
        "1. Base your answer strictly on the text provided. Do NOT use outside knowledge.\n"
        "2. If the provided document chunks do not contain the answer, you must state exactly: "
        "'I am sorry, but the provided documents do not contain enough information to answer this question.'\n"
        "3. Do not try to make up a plausible answer or guess."
    )

    user_prompt = f"CONTEXT DOCUMENTS:\n{context_str}\n\nUSER QUESTION: {question}"

    # 4. Generate Answer
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        answer = f"❌ Error communicating with Groq API: {str(e)}"

    # 5. Format sources
    sources_output = "\n".join(f"• {source}" for source in sorted(unique_sources))

    return answer, sources_output

# ========================================================================
# GRADIO WEB INTERFACE
# ========================================================================
with gr.Blocks(title="GT Off-Campus Housing Companion") as demo:
    gr.Markdown("# 🐝 Georgia Tech Off-Campus Housing Unofficial Assistant")
    gr.Markdown(
        "Ask questions about off-campus student apartments. Use the dropdown to filter by a specific building to ensure accurate pricing and details!"
    )

    with gr.Row():
        with gr.Column(scale=2):
            inp = gr.Textbox(label="Your Housing Question", lines=2)

            # The new Dropdown menu
            source_dropdown = gr.Dropdown(
                choices=[
                    "Search All",
                    "rambler",
                    "the_standard",
                    "square_on_fifth",
                    "inspire",
                    "reddit_home_park",
                    "Floor_Plans _ Hub_Atlanta.html"
                ],
                value="Search All",
                label="Filter by Source (Optional)"
            )

            btn = gr.Button("Submit Query", variant="primary")

        with gr.Column(scale=3):
            answer_box = gr.Textbox(label="Grounded AI Answer", lines=10, interactive=False)
            sources_box = gr.Textbox(label="Programmatic Source Verification", lines=3, interactive=False)

    # Connect both inputs (the textbox AND the dropdown) to the function
    btn.click(ask_housing_guide, inputs=[inp, source_dropdown], outputs=[answer_box, sources_box])
    inp.submit(ask_housing_guide, inputs=[inp, source_dropdown], outputs=[answer_box, sources_box])

if __name__ == "__main__":
    demo.launch()
