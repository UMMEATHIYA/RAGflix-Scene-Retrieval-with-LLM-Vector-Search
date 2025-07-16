# 🎬 RAGflix Pro – Movie Scene Explorer with LLMs + LangChain

RAGflix Pro is an intelligent scene retrieval and rewriting tool powered by **LangChain**, **LLMs (OpenAI)**, and **emotion-aware filtering**. It allows users to search movie scenes, analyze emotional content, and rewrite scenes in different tones — all from a single Colab notebook.

> 🔥 Built to impress recruiters by showcasing retrieval-augmented generation, prompt engineering, and interactive visualizations!

---

## 🚀 Features

- 🔍 **Scene Retrieval** using FAISS and Hugging Face embeddings
- 🎭 **Emotion Detection** using Transformers (`bert-base-uncased-emotion`)
- ✏️ **LLM-powered Scene Rewriting** (e.g., turn sad scene into comedy)
- 🧠 **LangChain Agents & Memory** to handle multi-step prompts
- 📊 **Stunning Visualizations** (radar chart, heatmap, bubble chart, timeline)
- 🧪 **Interactive QA** with Conversational Retrieval Chain
- ☁️ **Google Colab-based development** for portability and ease

---

## 🛠️ Tech Stack

- **LangChain** (`langchain`, `langchain-openai`, `langchain-community`)
- **OpenAI API** (for LLMs like GPT-4)
- **Hugging Face Transformers**
- **FAISS** (local vector store)
- **Seaborn, Plotly, Matplotlib** (for charts)
- **WordCloud, Scikit-learn** (text analysis)

---

## 🧑‍💻 Setup (Google Colab)

1. **Open the Colab notebook**: [RAGflix Pro.ipynb](link-to-notebook)
2. Install dependencies:

```bash
!pip install -U langchain langchain-community langchain-openai langchain-huggingface
!pip install openai sentence-transformers faiss-cpu tiktoken
!pip install matplotlib seaborn plotly wordcloud scikit-learn
```

## Add your OpenAI API key in the notebook:

```
import os
os.environ["OPENAI_API_KEY"] = "your-openai-key-here"
```

## Run the cells step by step to:

- Load sample scripts
- Embed scenes
- Classify emotion
- Query and rewrite scenes
- Visualize content

---

## 📊 Visualizations Include
- 🎭 Emotion Radar Chart
- 📈 Emotion Time-Series Plot
- ☁️ Scene Word Clouds
- 📦 Genre vs Emotion Heatmap
- 💬 Word Bubble Frequency
- 🧠 Scene Embedding Clusters (PCA)
---

## 🤖 Example Agent Prompt
- "Find me a sad scene and rewrite it with a romantic tone."
- LangChain agent automatically:
- Retrieves relevant scene
- Rewrites it using GPT-4
- Returns a beautifully formatted result
---

## Sample Demo

<img width="1798" height="662" alt="image" src="https://github.com/user-attachments/assets/fbe1ee97-7503-4740-b025-b0b67ede6cb4" />

---
<img width="1606" height="672" alt="image" src="https://github.com/user-attachments/assets/ffa405af-04c3-4d35-8b3e-019b602442e2" />

---

<img width="996" height="738" alt="image" src="https://github.com/user-attachments/assets/ee004385-02fe-4c18-a3a5-3bacd5347568" />

---

## 💡 Future Improvements
- 🎞️ Multimodal support (subtitles + video)
- 🎛️ UI with Gradio or Streamlit
- 💾 Save favorite scenes as journals
- 🧠 Long-term memory with LangChain + Pinecone

---

## 📬 Contact
- Umme Athiya – Applied AI/ML Engineer
- 📍 Chicago, IL
- 🔗 GitHub | LinkedIn
- 📫 Email: uathiya4@gmail.com
