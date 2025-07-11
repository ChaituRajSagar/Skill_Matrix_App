#core/template_analyzer.py
import logging
import json
import re
import time

from core.llama_runner import run_llama_prompt
from utils.file_utils import extract_text_from_file

logger = logging.getLogger(__name__)

# Define a maximum context size for the LLM during template analysis
# Similar to resume parsing, to fit within LLM context window
MAX_TEMPLATE_CONTEXT_CHARS = 6000 

def analyze_template_structure(template_path: str) -> dict:
    """
    Analyzes a blank template file to infer its structure, sections, and expected fields
    using an LLM, returning a structured schema.
    """
    start_total_time = time.perf_counter()
    logger.info("Attempting to analyze template structure for file: '%s'", template_path)
    
    template_raw_text = ""
    try:
        start_extract_time = time.perf_counter()
        template_raw_text = extract_text_from_file(template_path)
        end_extract_time = time.perf_counter()
        logger.info("Successfully extracted raw text from template. Length: %d (Duration: %.4f s)", len(template_raw_text), end_extract_time - start_extract_time)
        if not template_raw_text:
            logger.error("No text extracted from template file: %s", template_path)
            return {"error": "Could not extract text from the template file for analysis."}
    except Exception as e:
        logger.error("Failed to extract text from template file '%s': %s", template_path, e, exc_info=True)
        return {"error": f"Failed to read template file for analysis: {e}"}

    truncated_template_text = template_raw_text
    if len(template_raw_text) > MAX_TEMPLATE_CONTEXT_CHARS:
        truncated_template_text = template_raw_text[:MAX_TEMPLATE_CONTEXT_CHARS]
        logger.warning(f"Resume text truncated from {len(template_raw_text)} to {MAX_TEMPLATE_CONTEXT_CHARS} characters for LLM context. Information at the end of the resume might be missed.")
        truncated_template_text += "\n\n[TEMPLATE TEXT TRUNCATED DUE TO LENGTH]"

    # Prompt for template structure analysis
    prompt = f"""
    You are an expert document structure analyzer. Your task is to analyze the provided EMPTY TEMPLATE TEXT.
    Infer the logical sections, expected fields, and structural patterns (like tables or repeating blocks) within this template.
    Your output MUST be a single, valid JSON object representing this inferred structure.
    
    For each identified section or field, provide a descriptive name and indicate its expected type (e.g., "string", "list of strings", "object", "list of objects").
    If you identify tables, describe their columns and indicate if rows are repeatable.
    If sections or fields are clearly meant for personal data (name, contact), experience, education, or certifications, classify them accordingly.
    
    STRICT GUIDELINES:
    1.  **Strictly from Context:** Infer structure ONLY from the provided EMPTY TEMPLATE TEXT. Do NOT use outside knowledge, hallucinate, or invent elements not implied by the template's layout or headings.
    2.  **Valid JSON:** Your entire response MUST be a valid JSON object.
    3.  **No Explanations:** Do NOT include any conversational text or markdown outside the JSON block.
    4.  **Repeatable Blocks:** If a section (like a job experience, education, or certification) appears to be a repeating block, indicate this with a boolean flag (e.g., `"is_repeatable": true`) and describe the structure of a *single instance* of that block.

    Example JSON Schema for Template Structure (adjust as appropriate for the template):
    ```json
    {{
      "template_name": "string",
      "sections": [
        {{
          "section_name": "string",
          "description": "string",
          "type": "string", // e.g., "paragraph", "list", "table"
          "fields": [
            {{"field_name": "string", "expected_type": "string"}}, // e.g., "Full Name": "string"
            // ... more fields
          ],
          "table_structure": {{ // If type is "table"
            "columns": ["string"], // e.g., ["JOB TITLE/ROLE", "START/END DATE", "RESPONSIBILITIES"]
            "is_repeatable": "boolean", // e.g., true for multiple job experiences
            "example_row_fields": {{}} // Example of a single row's fields
          }},
          "is_repeatable": "boolean" // For entire sections like Education or Certifications
        }}
        // ... more sections
      ]
    }}
    ```
    
    Analyze the provided EMPTY TEMPLATE TEXT and generate its structured schema as a JSON object.
    
    EMPTY TEMPLATE TEXT:
    ---
    {truncated_template_text}
    ---
    
    Your JSON output:
    ```json
    """

    # MODIFICATION: json_str is now initialized outside the try block
    # so it can be used in the except block for logging.
    json_str = ""
    try:
        start_llm_time = time.perf_counter()
        llm_response = run_llama_prompt(prompt, context=truncated_template_text, max_new_tokens=2100, temperature=0.1) 
        end_llm_time = time.perf_counter()
        logger.info("LLM responded for template structure analysis (Duration: %.4f s). Length: %d", end_llm_time - start_llm_time, len(llm_response))
        
        # --- MODIFICATION START ---
        # The incorrect error log that fired on every run has been removed from here.
        # The overly complex JSON extraction logic has been replaced with a simpler, robust version.
        
        start_json_parse_time = time.perf_counter()
        
        # 1. Try to find JSON inside markdown
        json_match = re.search(r'```json\n(.*)\n```', llm_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # 2. If no markdown, find the first '{' and last '}' as a fallback
            logger.warning("JSON markdown block not found. Attempting to extract JSON using brace matching as a fallback.")
            first_brace = llm_response.find('{')
            last_brace = llm_response.rfind('}')
            if first_brace != -1 and last_brace != -1:
                json_str = llm_response[first_brace : last_brace + 1].strip()
            else:
                # 3. If no braces found, use the whole response and let it fail in the parser
                json_str = llm_response.strip()
        # --- MODIFICATION END ---


        template_schema = json.loads(json_str)
        end_json_parse_time = time.perf_counter()
        logger.info("Successfully parsed template schema from LLM response (Duration: %.4f s).", end_json_parse_time - start_json_parse_time)
        
        end_total_time = time.perf_counter()
        logger.info("Total analyze_template_structure completed (Total Duration: %.4f s).", end_total_time - start_total_time)
        return template_schema
    except json.JSONDecodeError as e:
        # MODIFICATION: The raw response is now logged here, ONLY when a JSON error actually occurs.
        logger.error(f"JSON Decode Error during template structure analysis: {e}")
        logger.error(f"Malformed JSON from LLM that caused the error:\n{json_str}")
        return {"error": f"Failed to parse template structure from LLM: {e}. Raw LLM output: {json_str}"}
    except Exception as e:
        logger.error(f"Error calling LLM for template structure analysis: {e}", exc_info=True)
        return {"error": f"LLM processing error during template structure analysis: {e}"}