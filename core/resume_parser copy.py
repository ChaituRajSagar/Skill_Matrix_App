import logging
import json
import re
from docx import Document
from utils.file_utils import extract_text_from_file
from core.llama_runner import run_llama_prompt
import time
from collections import OrderedDict  # ✅ Required for preserving field order

logger = logging.getLogger(__name__)

MAX_RESUME_CONTEXT_CHARS = 6000

def parse_resume_to_structured_data(file_path: str) -> dict:
    start_total_time = time.perf_counter()
    logger.info("Attempting to parse resume file '%s' to structured data.", file_path)
    resume_raw_text = ""
    try:
        start_extract_time = time.perf_counter()
        resume_raw_text = extract_text_from_file(file_path)
        end_extract_time = time.perf_counter()
        logger.info("Successfully extracted raw text from resume. Length: %d (Duration: %.4f s)", len(resume_raw_text), end_extract_time - start_extract_time)
        if not resume_raw_text:
            logger.error("No text extracted from resume file: %s", file_path)
            return {"error": "Could not extract text from the resume file."}
    except Exception as e:
        logger.error("Failed to extract text from resume file '%s': %s", file_path, e, exc_info=True)
        return {"error": f"Failed to read resume file: {e}"}

    truncated_resume_text = resume_raw_text
    if len(resume_raw_text) > MAX_RESUME_CONTEXT_CHARS:
        truncated_resume_text = resume_raw_text[:MAX_RESUME_CONTEXT_CHARS]
        logger.warning(f"Resume text truncated from {len(resume_raw_text)} to {MAX_RESUME_CONTEXT_CHARS} characters for LLM context. Information at the end of the resume might be missed.")
        truncated_resume_text += "\n\n[RESUME TEXT TRUNCATED DUE TO LENGTH - IMPORTANT: INFORMATION AT THE END MAY BE MISSING]"

    extracted_name = ""
    first_few_lines = "\n".join(resume_raw_text.splitlines()[:5])

    name_extraction_prompt = f"""
    From the following text, extract the full name and return it in a JSON object with a single key "Full Name".
    The name is typically at the very beginning. If no name is clearly identifiable, respond with {{"Full Name": "[NAME_NOT_FOUND]"}}.
    Text:
    ---
    {first_few_lines}
    ---
    Your JSON output:
    """

    start_name_llm_time = time.perf_counter()
    name_llm_response = run_llama_prompt(name_extraction_prompt, context=first_few_lines, max_new_tokens=50, temperature=0.0)
    end_name_llm_time = time.perf_counter()
    logger.info("LLM responded for name extraction (Duration: %.4f s). Response: '%s'", end_name_llm_time - start_name_llm_time, name_llm_response)

    try:
        json_match = re.search(r'```json\n(.*)\n```', name_llm_response, re.DOTALL)
        if json_match:
            name_json_str = json_match.group(1).strip()
        else:
            name_json_str = name_llm_response.strip()

        name_data = json.loads(name_json_str)
        candidate_name = name_data.get("Full Name", "").strip()
        if candidate_name and candidate_name.lower() != "[name_not_found]":
            extracted_name = candidate_name
            logger.info("Successfully extracted name: '%s'", extracted_name)
    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning(f"Could not parse name from LLM response ('{name_llm_response}'). Will rely on main parsing. Error: {e}")
        extracted_name = ""

    prompt = f"""
    #You are an expert resume parser assistant. Your primary goal is to extract detailed information from the provided resume text.
    You are an expert resume parser assistant. Your primary goal is to extract detailed information from the provided resume text.
    You MUST return the data as a single, strictly valid JSON object suitable for parsing using Python's json.loads(). Do not include any conversational text, explanations, or markdown outside the JSON block.
    Do NOT include comments, trailing commas, or malformed structures. Each key must be properly quoted, and every list item must be comma-separated.
    Truncate extremely lengthy fields (especially professional_summary) to a short paragraph. Format multi-line text like summary using \\n line breaks for readability.
    If a field is not found or is not applicable in the resume, its corresponding value in the JSON should be an empty string ("") for single values, or an empty list ([]) for lists.
    Ensure all other extracted data is directly and accurately from the resume.

    Strictly adhere to the following JSON schema for your output.
    ```json
    {{
      "full_name": "{extracted_name}",
      "email": "string",
      "phone_number": "string",
      "linkedin_profile": "string",
      "github_profile": "string",
      "personal_website": "string",
      "professional_summary": "string (3 to 5 short lines, use \\n for new lines, avoid long paragraphs and punctuation issues)",
      "skills": ["string"],
      "technologies": ["string"],
      "core_competencies": ["string"],
      "experience": [
        {{
          "title": "string",
          "company": "string",
          "location": "string",
          "duration": "string",
          "responsibilities": ["string"]
        }}
      ],
      "education": [
        {{
          "degree": "string",
          "university": "string",
          "location": "string",
          "year_obtained": "string"
        }}
      ],
      "awards": ["string"],
      "certifications": ["string"],
      "languages": ["string"]
    }}
    ```

    Resume Text to Parse:
    ---
    {truncated_resume_text}
    ---

    Your JSON output:
    ```json
    """

    try:
        start_llm_time = time.perf_counter()
        llm_response = run_llama_prompt(prompt, context=truncated_resume_text, max_new_tokens=2100, temperature=0.1)
        end_llm_time = time.perf_counter()
        logger.info("LLM responded for main resume data extraction (Duration: %.4f s).", end_llm_time - start_llm_time)

        start_json_parse_time = time.perf_counter()
        json_match = re.search(r'```json\n(.*)\n```', llm_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_str = llm_response.strip()

        try:
            # ✅ CLEAN UP BAD FORMATTING FROM LLM OUTPUT
            cleaned_json_str = json_str

            # Remove trailing commas before closing brackets
            cleaned_json_str = re.sub(r',\s*([\]}])', r'\1', cleaned_json_str)

            # Normalize smart quotes to normal quotes
            cleaned_json_str = cleaned_json_str.replace("“", "\"").replace("”", "\"").replace("’", "'")
            cleaned_json_str = cleaned_json_str.replace("\u201c", "\"").replace("\u201d", "\"").replace("\u2019", "'")

            # Remove markdown fences if any
            cleaned_json_str = re.sub(r'^```json\s*', '', cleaned_json_str.strip())
            cleaned_json_str = re.sub(r'\s*```$', '', cleaned_json_str.strip())

            resume_data = json.loads(cleaned_json_str)

            # Truncate overly long summaries
            if isinstance(resume_data.get("professional_summary"), str):
                resume_data["professional_summary"] = resume_data["professional_summary"][:2000]

            for field in ["email", "phone_number", "linkedin_profile", "github_profile", "personal_website"]:
                if field in resume_data and isinstance(resume_data[field], str):
                    resume_data[field] = resume_data[field].strip()

            if extracted_name and not resume_data.get("full_name"):
                resume_data["full_name"] = extracted_name
                logger.info("Overriding final 'full_name' with pre-extracted name: '%s'", extracted_name)

            desired_order = [
                "full_name", "email", "phone_number", "linkedin_profile", "github_profile", "personal_website",
                "professional_summary", "skills", "technologies", "core_competencies",
                "experience", "education", "awards", "certifications", "languages"
            ]
            ordered_resume_data = OrderedDict()
            for key in desired_order:
                if key in resume_data:
                    ordered_resume_data[key] = resume_data[key]
            for key in resume_data:
                if key not in ordered_resume_data:
                    ordered_resume_data[key] = resume_data[key]

            logger.info("Successfully parsed structured resume data.")
            return ordered_resume_data

        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error during resume data extraction: {e}")
            logger.error(f"Malformed LLM JSON (after cleaning):\n{cleaned_json_str[:1500]}")
            return {"error": f"Failed to parse structured resume data from LLM: {e}. Raw LLM output: {cleaned_json_str[:500]}..."}
    except Exception as e:
        logger.error(f"Error calling LLM for resume data extraction: {e}", exc_info=True)
        return {"error": f"LLM processing error during resume data extraction: {e}"}