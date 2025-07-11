# 📄 Skill Matrix Filler App

An intelligent document automation tool that **extracts structured data from resumes** and **dynamically fills DOCX templates** — including complex Skill/Qualification matrices — using **local LLMs (via Ollama)** and **ChromaDB-powered RAG**.

---

## 🚀 Features

* ✅ Upload and process resumes in `.pdf`, `.docx`, or `.doc` formats
* ✅ Analyze template structure using LLM to infer expected fields
* ✅ Extract structured resume data (skills, experience, certifications)
* ✅ Dynamically fill any resume-based template using extracted data
* ✅ Special support for **Skill/Qualification Matrix** templates
* ✅ Preview filled templates as HTML and download as `.docx`
* ✅ Ask questions about the resume/template using a simple chat interface
* ✅ All data is embedded and queried using **ChromaDB**
* ✅ Runs fully **offline** using Ollama + Mistral (no OpenAI API needed)

---

## 🧠 How It Works – 10 Steps

1. **Upload Resume and Template:**
   Users upload a resume and a DOCX template via the web interface.

2. **Save & Track Session:**
   Files are saved with timestamps, and a unique `session_id` is generated.

3. **Parse Resume to Structured JSON:**
   Text is extracted and sent to a local LLM to output fields like name, experience, education, etc.

4. **Analyze Template Schema:**
   A second LLM analyzes the empty template to understand its layout, sections, and repeatable blocks.

5. **Embed Content with ChromaDB:**
   All text content and parsed metadata are stored as vector embeddings in ChromaDB for RAG-based retrieval.

6. **Display Extracted Data:**
   The parsed resume data is shown in JSON form in the UI for user verification.

7. **Trigger Template Filling:**
   The backend uses the resume data + template schema to build LLM instructions for dynamic field filling.

8. **Fill the Template Dynamically:**
   With `python-docx`, the app fills in the fields, paragraphs, and tables using LLM-generated content.

9. **Save and Serve Filled DOCX:**
   The filled document is saved under `filled_docs/` and made downloadable via a backend endpoint.

10. **Preview + Q\&A:**
    Users can preview the DOCX as HTML and ask questions via a mini chat powered by the LLM and ChromaDB.

---

## 📂 Project Structure

```
Skill_Matrix_App/
├── app.py                         # Main Flask backend
├── core/
│   ├── llama_runner.py            # Handles Ollama LLM prompts
│   ├── resume_parser.py           # Extracts structured data from resume
│   ├── template_analyzer.py       # Infers structure of blank templates
│   ├── template_processor.py      # Fills templates dynamically
│   └── chroma_service.py          # Handles embedding/indexing in ChromaDB
├── utils/
│   └── file_utils.py              # File reading, cleaning, chunking
├── static/
│   └── index.html                 # Frontend UI
├── uploads/                       # Uploaded files
├── filled_docs/                   # Output files
└── app.log                        # Runtime logs
```

---

## 🛠️ Requirements

* Python 3.10+
* [Ollama](https://ollama.com) installed and running locally
* Model pulled: `mistral:7b-instruct-v0.3-q4_K_M`
* `pip install -r requirements.txt` (typical: `flask`, `sentence-transformers`, `chromadb`, `python-docx`, `mammoth`, `fitz`, `ollama`, etc.)

---

## 🧪 Running the App

```bash
# Start the Ollama server first
ollama serve

# Ensure model is pulled
ollama pull mistral:7b-instruct-v0.3-q4_K_M

# Then run the Flask app
python app.py

# Open in browser
http://localhost:8008
```

---

## ✨ Live Demo Features

* Upload `.pdf`, `.docx`, or `.doc` resumes
* Upload empty or semi-filled templates
* Click **"Upload & Ingest"** to parse + embed
* Click **"Fill Template"** to get the final DOCX
* Preview as HTML and download
* Ask questions like:

  * “What certifications does the candidate hold?”
  * “List all companies worked at in the last 5 years”

---