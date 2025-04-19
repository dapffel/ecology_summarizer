# 🌿 Ecology Summarizer

`ecology_summarizer` is a domain-specific Python package for generating structured summaries of ecological research papers. It uses GPT (via OpenAI API) and vector-based memory (FAISS) to create accurate, context-aware summaries, including a one-sentence abstract suitable for research proposals.

---

## 🚀 Features

- 🔍 Extracts and summarizes ecological papers from PDFs
- 🧠 Retrieves relevant context using FAISS + OpenAI embeddings
- 📝 Outputs structured summaries with consistent formatting
- 💬 Includes a one-sentence summary for proposal writing
- 📉 Logs token usage and estimates OpenAI API costs
- ✅ Validates output using [Pydantic](https://docs.pydantic.dev)

---

## 📦 Installation

```bash
git clone https://github.com/dapffel/ecology_summarizer.git
cd ecology_summarizer
pip install .
