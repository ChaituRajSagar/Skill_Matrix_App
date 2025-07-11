import os
import uuid
import logging
import json
from datetime import datetime
import time
import mammoth
from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Import core modules and utilities
from core.llama_runner import run_llama_prompt
from core.resume_parser import parse_resume_to_structured_data
from core.template_processor import fill_skill_matrix_template, fill_dynamic_template
from core.template_analyzer import analyze_template_structure
from utils.file_utils import extract_text_from_file, chunk_text
# has_documents_for_session is removed as it's no longer needed
from core.chroma_service import add_documents_to_chroma, query_chroma, clear_collection

# --- Centralized Logging Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE_PATH = os.path.join(BASE_DIR, "app.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Prevents adding handlers multiple times in debug mode
if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
# END Centralized Logging ---


app = Flask(__name__)
app.json.sort_keys = False
CORS(app)

# Define paths relative to the Skill_Matrix_App directory
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
FILLED_DOCS_FOLDER = os.path.join(BASE_DIR, "filled_docs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FILLED_DOCS_FOLDER, exist_ok=True)
logger.info("UPLOAD_FOLDER '%s' and FILLED_DOCS_FOLDER '%s' ensured to exist.", UPLOAD_FOLDER, FILLED_DOCS_FOLDER)

# Define a single ChromaDB collection name for all documents related to a session
CHROMA_COLLECTION_NAME = "resume_template_docs_ollama"

@app.route("/")
def serve_index():
    """Serves the index.html file from the configured static folder."""
    return send_file(os.path.join(app.static_folder, 'index.html'))


@app.route("/upload_and_ingest", methods=["POST"])
def upload_and_ingest_route():
    start_total_time = time.perf_counter()
    logger.info("Received request to /upload_and_ingest endpoint.")

    resume_path_temp = None
    template_path_temp = None
    session_id = str(uuid.uuid4())
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        if "resume" not in request.files or "template" not in request.files:
            logger.error("Missing 'resume' or 'template' file in request.")
            return jsonify({"error": "Missing resume or template file."}), 400

        resume_file = request.files["resume"]
        template_file = request.files["template"]

        resume_filename_original = secure_filename(resume_file.filename)
        template_filename_original = secure_filename(template_file.filename)

        resume_base_name, resume_ext = os.path.splitext(resume_filename_original)
        template_base_name, template_ext = os.path.splitext(template_filename_original)

        resume_unique_filename = f"resume_{resume_base_name}_{timestamp}{resume_ext}"
        template_unique_filename = f"template_{template_base_name}_{timestamp}{template_ext}"

        resume_path_temp = os.path.join(UPLOAD_FOLDER, resume_unique_filename)
        template_path_temp = os.path.join(UPLOAD_FOLDER, template_unique_filename)

        start_save_time = time.perf_counter()
        resume_file.save(resume_path_temp)
        template_file.save(template_path_temp)
        end_save_time = time.perf_counter()
        logger.info("Saved temp resume to %s and temp template to %s for session %s (Duration: %.4f s)", resume_path_temp, template_path_temp, session_id, end_save_time - start_save_time)

        # Step 1a: Parse Resume and Extract Structured Data using LLM
        start_parse_resume_time = time.perf_counter()
        logger.info("Parsing resume to structured data using LLM for session %s: %s", session_id, resume_path_temp)
        resume_data_result = parse_resume_to_structured_data(resume_path_temp)
        end_parse_resume_time = time.perf_counter()
        logger.info("Resume parsing completed (Duration: %.4f s)", end_parse_resume_time - start_parse_resume_time)

        if "error" in resume_data_result:
            logger.error("Resume parsing failed for session %s: %s", session_id, resume_data_result['error'])
            return jsonify({"error": f"Resume parsing failed: {resume_data_result['error']}"}), 500

        start_add_structured_time = time.perf_counter()
        add_documents_to_chroma(
            CHROMA_COLLECTION_NAME,
            documents=[json.dumps(resume_data_result)],
            metadatas=[{
                "source_type": "structured_resume_data",
                "session_id": session_id,
                "resume_filename": resume_filename_original,
                "timestamp": os.path.getmtime(resume_path_temp)
            }],
            ids=[f"structured_resume_data_{session_id}"]
        )
        end_add_structured_time = time.perf_counter()
        logger.info("Successfully extracted and stored structured resume data in ChromaDB for session %s (Duration: %.4f s).", session_id, end_add_structured_time - start_add_structured_time)
        logger.debug("Extracted Resume Data (partial) for session %s: %s", session_id, json.dumps(resume_data_result, indent=2)[:500])

        # Step 1b: Analyze Template Structure using LLM
        start_analyze_template_time = time.perf_counter()
        logger.info("Analyzing template structure using LLM for session %s: %s", session_id, template_path_temp)
        template_structure_result = analyze_template_structure(template_path_temp)
        end_analyze_template_time = time.perf_counter()
        logger.info("Template structure analysis completed (Duration: %.4f s).", end_analyze_template_time - start_analyze_template_time)

        if "error" in template_structure_result:
            logger.error("Template structure analysis failed for session %s: %s", session_id, template_structure_result['error'])
            return jsonify({"error": f"Template analysis failed: {template_structure_result['error']}"}), 500
        
        start_add_template_schema_time = time.perf_counter()
        add_documents_to_chroma(
            CHROMA_COLLECTION_NAME,
            documents=[json.dumps(template_structure_result)],
            metadatas=[{
                "source_type": "template_schema",
                "session_id": session_id,
                "template_filename": template_filename_original,
                "timestamp": os.path.getmtime(template_path_temp)
            }],
            ids=[f"template_schema_{session_id}"]
        )
        end_add_template_schema_time = time.perf_counter()
        logger.info("Successfully analyzed and stored template schema in ChromaDB for session %s (Duration: %.4f s).", session_id, end_add_template_schema_time - start_add_template_schema_time)
        logger.debug("Template Schema (partial) for session %s: %s", session_id, json.dumps(template_structure_result, indent=2)[:500])

        # Step 2: Extract raw text from resume and template, chunk, embed, and store in ChromaDB (RAG INGESTION)
        start_rag_ingest_time = time.perf_counter()
        resume_raw_text = extract_text_from_file(resume_path_temp)
        resume_chunks = chunk_text(resume_raw_text)
        resume_metadatas = [
            {
                "source_type": "resume_text_chunk",
                "file_name": resume_filename_original,
                "session_id": session_id,
                "chunk_idx": i,
                "original_path": resume_path_temp,
                "template_original_path": template_path_temp
            }
            for i in range(len(resume_chunks))
        ]
        resume_chunk_ids = [f"resume_chunk_{i}_{session_id}" for i in range(len(resume_chunks))]
        add_documents_to_chroma(CHROMA_COLLECTION_NAME, resume_chunks, resume_metadatas, resume_chunk_ids)
        logger.info("Resume chunks added to ChromaDB for session %s.", session_id)

        template_raw_text = extract_text_from_file(template_path_temp)
        template_chunks = chunk_text(template_raw_text)
        template_metadatas = [
            {
                "source_type": "template_text_chunk",
                "file_name": template_filename_original,
                "session_id": session_id,
                "chunk_idx": i,
                "original_path": template_path_temp,
                "resume_original_path": resume_path_temp
            }
            for i in range(len(template_chunks))
        ]
        template_chunk_ids = [f"template_chunk_{i}_{session_id}" for i in range(len(template_chunks))]
        add_documents_to_chroma(CHROMA_COLLECTION_NAME, template_chunks, template_metadatas, template_chunk_ids)
        end_rag_ingest_time = time.perf_counter()
        logger.info("Template chunks added to ChromaDB for session %s (RAG Ingestion Duration: %.4f s).", session_id, end_rag_ingest_time - start_rag_ingest_time)


        end_total_time = time.perf_counter()
        logger.info("Total /upload_and_ingest route completed for session %s (Total Duration: %.4f s).", session_id, end_total_time - start_total_time)

        return jsonify({
            "status": "success",
            "message": "Resume parsed and documents indexed in ChromaDB for Q&A.",
            "session_id": session_id
        })

    except Exception as e:
        logger.exception("An unexpected error occurred during /upload_and_ingest endpoint execution for session %s.", session_id)
        return jsonify({"error": f"An unexpected server error occurred: {e}"}), 500

@app.route("/get_structured_resume_data", methods=["POST"])
def get_structured_resume_data_route():
    start_time = time.perf_counter()
    logger.info("Received request to /get_structured_resume_data endpoint.")
    data = request.get_json()
    session_id = data.get('session_id')

    if not session_id:
        logger.error("No 'session_id' provided for /get_structured_resume_data request.")
        return jsonify({"error": "No session ID provided."}), 400

    try:
        structured_data_results = query_chroma(
            CHROMA_COLLECTION_NAME,
            query_texts=["structured resume data"],
            n_results=1,
            where_metadata={
                "$and": [
                    {"session_id": session_id},
                    {"source_type": "structured_resume_data"}
                ]
            }
        )

        structured_resume_docs = structured_data_results.get('documents', [[]])[0]
        if not structured_resume_docs:
            logger.warning("Structured resume data not found in ChromaDB for session %s.", session_id)
            return jsonify({"error": "Structured resume data not found for this session."}), 404

        resume_data = json.loads(structured_resume_docs[0])
        end_time = time.perf_counter()
        logger.info("Successfully retrieved structured resume data for session %s (Duration: %.4f s).", session_id, end_time - start_time)
        return jsonify({
            "status": "success",
            "structured_resume_data": resume_data
        })

    except Exception as e:
        logger.exception("Error retrieving structured resume data from ChromaDB for UI display for session %s.", session_id)
        return jsonify({"error": f"Error retrieving structured resume data: {e}"}), 500


# @app.route("/fill_template", methods=["POST"])
# def fill_template_route():
#     start_total_time = time.perf_counter()
#     logger.info("Received request to /fill_template endpoint.")
#     data = request.get_json()
#     session_id = data.get('session_id')

#     if not session_id:
#         logger.error("No 'session_id' provided for /fill_template request.")
#         return jsonify({"error": "No session ID provided. Please upload and ingest documents first."}), 400

#     try:
#         # Retrieve Structured Resume Data
#         structured_data_results = query_chroma(
#             CHROMA_COLLECTION_NAME,
#             query_texts=["structured resume data"],
#             n_results=1,
#             where_metadata={"session_id": session_id, "source_type": "structured_resume_data"}
#         )
#         structured_resume_docs = structured_data_results.get('documents', [[]])[0]
#         if not structured_resume_docs:
#             return jsonify({"error": "Structured resume data not found for this session. Please re-upload."}), 400
#         structured_resume_data = json.loads(structured_resume_docs[0])

#         # Retrieve Template Schema
#         template_schema_results = query_chroma(
#             CHROMA_COLLECTION_NAME,
#             query_texts=["template schema"],
#             n_results=1,
#             where_metadata={"session_id": session_id, "source_type": "template_schema"}
#         )
#         template_schema_docs = template_schema_results.get('documents', [[]])[0]
#         template_schema = json.loads(template_schema_docs[0]) if template_schema_docs else {}

#         # Retrieve Template File Info
#         template_info_results = query_chroma(
#             CHROMA_COLLECTION_NAME,
#             query_texts=["template file path"],
#             n_results=1,
#             where_metadata={"session_id": session_id, "source_type": "template_text_chunk"}
#         )
#         template_metadatas = template_info_results.get('metadatas', [[]])[0]
#         if not template_metadatas:
#             return jsonify({"error": "Template file information not found for this session. Please re-upload."}), 400

#         template_path_temp = template_metadatas[0]['original_path']
#         template_filename_original = template_metadatas[0]['file_name']
#         resume_original_path = template_metadatas[0]['resume_original_path']
#         base_resume_name = os.path.splitext(os.path.basename(resume_original_path))[0]

#     except Exception as e:
#         logger.exception("Error retrieving session data from ChromaDB for session %s.", session_id)
#         return jsonify({"error": f"An unexpected server error occurred while retrieving data: {e}"}), 500

#     try:
#         if "skill_matrix" in template_filename_original.lower() or "qualification_matrix" in template_filename_original.lower():
#             logger.info("Detected Skill/Qualification Matrix template. Calling specific filler for session %s.", session_id)
#             output_filename = f"Skill_Matrix_template_{base_resume_name}_{session_id[:8]}.docx"
#             filler_function = fill_skill_matrix_template
#         else:
#             logger.warning("No specific filler for template '%s'. Attempting dynamic/generic fill with schema.", template_filename_original)
#             output_filename = f"filled_{os.path.splitext(template_filename_original)[0]}_{base_resume_name}_{session_id[:8]}.docx"
#             # Use a lambda to pass the extra template_schema argument
#             filler_function = lambda t, r, o: fill_dynamic_template(t, r, o, template_schema)
        
#         output_path = os.path.join(FILLED_DOCS_FOLDER, output_filename)
#         fill_result = filler_function(template_path_temp, structured_resume_data, output_path)

#         if "error" in fill_result:
#             logger.error("Template filling failed for session %s: %s", session_id, fill_result['error'])
#             return jsonify({"error": f"Template filling failed: {fill_result['error']}"}), 500

#         end_total_time = time.perf_counter()
#         logger.info("Total /fill_template route completed for session %s (Total Duration: %.4f s).", session_id, end_total_time - start_total_time)

#         return jsonify({
#             "status": "success",
#             "message": "Resume template filled successfully.",
#             "filled_file_name": output_filename,
#             "download_url": f"/download_filled_resume/{output_filename}"
#         })

#     except Exception as e:
#         logger.exception("An unexpected error occurred during template filling for session %s.", session_id)
#         return jsonify({"error": f"An unexpected server error occurred during template filling: {e}"}), 500

# Replace the existing fill_template_route function in app.py with this one.

@app.route("/fill_template", methods=["POST"])
def fill_template_route():
    start_total_time = time.perf_counter()
    logger.info("Received request to /fill_template endpoint.")
    data = request.get_json()
    session_id = data.get('session_id')

    if not session_id:
        logger.error("No 'session_id' provided for /fill_template request.")
        return jsonify({"error": "No session ID provided. Please upload and ingest documents first."}), 400

    try:
        # --- FIX APPLIED HERE ---
        # Retrieve Structured Resume Data with the correct $and filter
        structured_data_results = query_chroma(
            CHROMA_COLLECTION_NAME,
            query_texts=["structured resume data"],
            n_results=1,
            where_metadata={
                "$and": [
                    {"session_id": session_id},
                    {"source_type": "structured_resume_data"}
                ]
            }
        )
        structured_resume_docs = structured_data_results.get('documents', [[]])[0]
        if not structured_resume_docs:
            return jsonify({"error": "Structured resume data not found for this session. Please re-upload."}), 400
        structured_resume_data = json.loads(structured_resume_docs[0])

        # --- FIX APPLIED HERE ---
        # Retrieve Template Schema with the correct $and filter
        template_schema_results = query_chroma(
            CHROMA_COLLECTION_NAME,
            query_texts=["template schema"],
            n_results=1,
            where_metadata={
                "$and": [
                    {"session_id": session_id},
                    {"source_type": "template_schema"}
                ]
            }
        )
        template_schema_docs = template_schema_results.get('documents', [[]])[0]
        template_schema = json.loads(template_schema_docs[0]) if template_schema_docs else {}

        # --- FIX APPLIED HERE ---
        # Retrieve Template File Info with the correct $and filter
        template_info_results = query_chroma(
            CHROMA_COLLECTION_NAME,
            query_texts=["template file path"],
            n_results=1,
            where_metadata={
                "$and": [
                    {"session_id": session_id},
                    {"source_type": "template_text_chunk"}
                ]
            }
        )
        template_metadatas = template_info_results.get('metadatas', [[]])[0]
        if not template_metadatas:
            return jsonify({"error": "Template file information not found for this session. Please re-upload."}), 400

        template_path_temp = template_metadatas[0]['original_path']
        template_filename_original = template_metadatas[0]['file_name']
        resume_original_path = template_metadatas[0]['resume_original_path']
        base_resume_name = os.path.splitext(os.path.basename(resume_original_path))[0]

    except Exception as e:
        logger.exception("Error retrieving session data from ChromaDB for session %s.", session_id)
        return jsonify({"error": f"An unexpected server error occurred while retrieving data: {e}"}), 500

    try:
        if "skill_matrix" in template_filename_original.lower() or "qualification_matrix" in template_filename_original.lower():
            logger.info("Detected Skill/Qualification Matrix template. Calling specific filler for session %s.", session_id)
            output_filename = f"Skill_Matrix_template_{base_resume_name}_{session_id[:8]}.docx"
            filler_function = fill_skill_matrix_template
        else:
            logger.warning("No specific filler for template '%s'. Attempting dynamic/generic fill with schema.", template_filename_original)
            output_filename = f"filled_{os.path.splitext(template_filename_original)[0]}_{base_resume_name}_{session_id[:8]}.docx"
            filler_function = lambda t, r, o: fill_dynamic_template(t, r, o, template_schema)
        
        output_path = os.path.join(FILLED_DOCS_FOLDER, output_filename)
        fill_result = filler_function(template_path_temp, structured_resume_data, output_path)

        if "error" in fill_result:
            logger.error("Template filling failed for session %s: %s", session_id, fill_result['error'])
            return jsonify({"error": f"Template filling failed: {fill_result['error']}"}), 500

        end_total_time = time.perf_counter()
        logger.info("Total /fill_template route completed for session %s (Total Duration: %.4f s).", session_id, end_total_time - start_total_time)

        return jsonify({
            "status": "success",
            "message": "Resume template filled successfully.",
            "filled_file_name": output_filename,
            "download_url": f"/download_filled_resume/{output_filename}"
        })

    except Exception as e:
        logger.exception("An unexpected error occurred during template filling for session %s.", session_id)
        return jsonify({"error": f"An unexpected server error occurred during template filling: {e}"}), 500


# @app.route("/ask", methods=["POST"])
# def ask_llm_route():
#     data = request.get_json()
#     question = data.get('question')
#     session_id = data.get('session_id')

#     if not question or not session_id:
#         return jsonify({"error": "Question and session ID are required."}), 400

#     try:
#         # Retrieve structured data to add to context
#         structured_resume_data_from_db = {}
#         structured_data_results = query_chroma(
#             CHROMA_COLLECTION_NAME,
#             query_texts=["structured resume data"],
#             n_results=1,
#             where_metadata={"session_id": session_id, "source_type": "structured_resume_data"}
#         )
#         if structured_data_results.get('documents', [[]])[0]:
#             structured_resume_data_from_db = json.loads(structured_data_results['documents'][0][0])
        
#         # Retrieve relevant text chunks from RAG
#         retrieved_results = query_chroma(
#             CHROMA_COLLECTION_NAME,
#             query_texts=[question],
#             n_results=5,
#             where_metadata={"session_id": session_id}
#         )
#         retrieved_chunks = retrieved_results.get('documents', [[]])[0]
        
#         context_for_llm = "--- Relevant Information from Documents ---\n"
#         for chunk in retrieved_chunks:
#             context_for_llm += f"Content: {chunk}\n\n"
        
#         context_for_llm += f"--- Structured Resume Data ---\n{json.dumps(structured_resume_data_from_db, indent=2)}\n"

#         llm_prompt = f"""
#         You are an intelligent assistant. Based *ONLY* on the provided context, answer the following question fully and accurately.
#         If the context includes a document title like "Skill-Qualification Matrix", you may infer its common purpose if not explicitly stated, but only if it's a very common document type.
#         Adhere strictly to the information given in the "Relevant Information from Documents" and "Structured Resume Data" sections.
#         DO NOT use any outside knowledge, make assumptions, or invent details beyond common knowledge of document types.
#         If the answer is not explicitly present or directly inferable from the given context, you MUST state: "I cannot answer this question based on the provided documents."
#         Be concise and direct in your answer.

#         User Question: {question}

#         Answer:
#         """
#         llm_response = run_llama_prompt(llm_prompt, context="", max_new_tokens=500, temperature=0.1)
        
#         return jsonify({"answer": llm_response})
#     except Exception as e:
#         logger.exception("Error during LLM Q&A for session %s.", session_id)
#         return jsonify({"error": f"Error interacting with LLM for Q&A: {e}"}), 500


@app.route("/ask", methods=["POST"])
def ask_llm_route():
    data = request.get_json()
    question = data.get('question')
    session_id = data.get('session_id')

    if not question or not session_id:
        return jsonify({"error": "Question and session ID are required."}), 400

    try:
        # Retrieve structured data to add to context
        structured_resume_data_from_db = {}
        
        # --- FIX APPLIED HERE ---
        # Added the $and operator to correctly filter by two conditions
        structured_data_results = query_chroma(
            CHROMA_COLLECTION_NAME,
            query_texts=["structured resume data"],
            n_results=1,
            where_metadata={
                "$and": [
                    {"session_id": session_id},
                    {"source_type": "structured_resume_data"}
                ]
            }
        )
        if structured_data_results.get('documents', [[]])[0]:
            structured_resume_data_from_db = json.loads(structured_data_results['documents'][0][0])
        
        # Retrieve relevant text chunks from RAG
        # This query is fine as it only has one filter condition
        retrieved_results = query_chroma(
            CHROMA_COLLECTION_NAME,
            query_texts=[question],
            n_results=5,
            where_metadata={"session_id": session_id}
        )
        retrieved_chunks = retrieved_results.get('documents', [[]])[0]

        if not retrieved_chunks and not structured_resume_data_from_db:
            return jsonify({"answer": "I couldn't find relevant information to answer that question."})

        context_for_llm = "--- Relevant Information from Documents ---\n"
        for chunk in retrieved_chunks:
            context_for_llm += f"Content: {chunk}\n\n"
        
        context_for_llm += f"--- Structured Resume Data ---\n{json.dumps(structured_resume_data_from_db, indent=2)}\n"

        llm_prompt = f"""
        You are a document analysis assistant. Your task is to answer the user's question based *ONLY* on the provided context below.

        **Strict Rules:**
        1.  You are forbidden from using any external knowledge.
        You are forbidden from using any external knowledge.
        2.  You must base your answer exclusively on the text within the "--- Relevant Information from Documents ---", "--- Structured Resume Data ---", and "--- Template Schema ---" sections.
        3.  If the provided context does not contain the information needed to answer the question, you MUST respond with the exact phrase: "The provided documents do not contain sufficient information to answer this question." Do not try to guess or apologize.

        User Question: {question}

        Answer:
        """
        llm_response = run_llama_prompt(llm_prompt, context="", max_new_tokens=500, temperature=0.1)
        
        return jsonify({"answer": llm_response})

    except Exception as e:
        logger.exception("Error during LLM Q&A for session %s.", session_id)
        return jsonify({"error": f"Error interacting with LLM for Q&A: {e}"}), 500

@app.route("/preview_filled_doc", methods=["POST"])
def preview_filled_doc():
    data = request.get_json()
    filename = data.get("filename")

    if not filename:
        return jsonify({"error": "No filename provided"}), 400

    filepath = os.path.join(FILLED_DOCS_FOLDER, secure_filename(filename))
    if not os.path.exists(filepath):
        return jsonify({"error": "Filled document not found"}), 404

    try:
        with open(filepath, "rb") as docx_file:
            result = mammoth.convert_to_html(docx_file)
            return jsonify({"html": result.value})
    except Exception as e:
        logger.exception("Failed to convert DOCX to HTML for preview.")
        return jsonify({"error": str(e)}), 500

@app.route('/download_filled_resume/<filename>', methods=['GET'])
def download_filled_resume(filename):
    """Endpoint to download the generated filled resume."""
    file_path = os.path.join(FILLED_DOCS_FOLDER, secure_filename(filename))
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True, download_name=os.path.basename(filename),
                         mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
    logger.warning("File not found for download: %s", file_path)
    return jsonify({"error": "File not found."}), 404


if __name__ == "__main__":
    app.run(debug=True, port=8008)