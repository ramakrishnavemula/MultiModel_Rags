# 🧠 Multimodal Retrieval-Augmented Generation (RAG)

This repository demonstrates two end-to-end **Multimodal RAG pipelines** — one built with **CLIP** and another with **Unstructured + Tesseract** — capable of understanding and answering questions across **text, images, and tables** in PDF documents.

---

## 🚀 Features

* **Multimodal Understanding**: Supports text, image, and table extraction.
* **Two Architectures**:

  * 🧩 **CLIP-based RAG** — uses unified embeddings for images and text.
  * 🧾 **Unstructured + Tesseract RAG** — parses complex PDFs with text, images, and structured tables.
* **LLM Integration**: Uses GPT-4o for multimodal reasoning.
* **Vector Search**: Employs Chroma and FAISS for semantic retrieval.
* **Scalable**: Can be extended to include audio or video inputs.

---

### In Google Colab:

```bash
!pip install "unstructured[all-docs]" langchain_chroma langchain langchain-community langchain-openai
!pip install pymupdf pillow torch torchvision transformers faiss-cpu
!sudo apt-get install -y poppler-utils tesseract-ocr
```
Usage:

1️⃣ CLIP-based RAG

Extracts image and text embeddings jointly for visual and textual understanding(python multimodel_rag_clip.py)

2️⃣ Unstructured + Tesseract RAG

Parses tables, text, and images from PDFs for deep contextual understanding(python multimodelrag_unstructured_tesseract.py)

Both systems store embeddings in FAISS / ChromaDB and use **GPT-4o** for answering queries.

---

🧠 Example Query
“What does the chart in the document indicate about performance trends?”

The model retrieves relevant text and image sections, interprets both modalities, and generates a contextual answer.


## 📈 Future Enhancements

* Add **audio (speech-to-text)** and **video (frame extraction)** support
* Integrate **LangGraph** for agentic retrieval orchestration
* Deploy via **Vertex AI** or **Streamlit** interface
