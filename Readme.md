# 📖 RAG-based Q&A Chatbot (Streamlit + LangChain + Ollama)

This project is a **Retrieval-Augmented Generation (RAG)** chatbot built using:
- [LangChain](https://www.langchain.com/)  
- [Ollama](https://ollama.com/) for embeddings & LLMs  
- [FAISS](https://github.com/facebookresearch/faiss) for vector search  
- [Streamlit](https://streamlit.io/) for a simple web UI  

The app lets you:
1. Upload any **`.txt` file**.  
2. Automatically **chunk, embed, and store** the content.  
3. Ask **natural language questions** from the uploaded file.  
4. Get answers powered by **Mistral (LLM)** + **nomic-embed-text (Embeddings)** via Ollama.  

---

## 📂 Project Structure
RAG_APP/
│── app.py # Streamlit web app
│── model.ipynb # Notebook for building and testing RAG pipeline
│── data.txt # Sample text file
│── vectorstore.pkl # Pickled FAISS vectorstore (optional)
│── README.md # Project documentation



---

## ⚙️ Requirements
- Python 3.9+  
- [Ollama](https://ollama.com/) installed and running locally  
- Install the required Python libraries:  

```bash
pip install streamlit langchain langchain-community langchain-text-splitters faiss-cpu

📥 Clone this Repository
git clone https://github.com/your-username/rag-streamlit-app.git
cd rag-streamlit-app

▶️ Run the Streamlit App
Start the app with:
    streamlit run app.py

Then:
Upload a .txt file.
Type your question.
Get AI-powered answers 🎉

🧠 Models Used
Embeddings: nomic-embed-text:latest (via Ollama)
LLM: mistral:latest (via Ollama)
Make sure these are available in Ollama:
            ollama pull nomic-embed-text
            ollama pull mistral

🤝 Contributing
Feel free to fork this repo, create a branch, and submit pull requests 🚀

📜 License
This project is licensed under the MIT License.
