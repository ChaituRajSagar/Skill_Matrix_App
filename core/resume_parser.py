import logging
import json
import re
import time
from collections import OrderedDict

from utils.file_utils import extract_text_from_file
from core.llama_runner import run_llama_prompt

logger = logging.getLogger(__name__)

MAX_RESUME_CONTEXT_CHARS = 6000

def validate_json_structure(resume_data: dict) -> bool:
    """
    Validates that the final assembled JSON object has the correct keys and value types.
    """
    required_keys = {
        "full_name": str,
        "email": str,
        "phone_number": str,
        "linkedin_profile": str,
        "github_profile": str,
        "personal_website": str,
        "professional_summary": str,
        "skills": list,
        "technologies": list,
        "core_competencies": list,
        "experience": list,
        "education": list,
        "awards": list,
        "certifications": list,
        "languages": list
    }

    for key, expected_type in required_keys.items():
        if key not in resume_data:
            logger.warning(f"Missing key in final JSON: {key}")
            # This check is important, but the main function now ensures all keys exist.
            # We'll keep it as a safeguard.
            return False
        if not isinstance(resume_data[key], expected_type):
            logger.warning(f"Invalid type for key '{key}': expected {expected_type}, got {type(resume_data[key])}")
            return False
    return True

def parse_resume_to_structured_data(file_path: str) -> dict:
    """
    Parses a resume by breaking the task into sequential, smaller LLM calls for each section.
    This is the most robust method for models that struggle with large, single-shot JSON generation.
    """
    start_total_time = time.perf_counter()
    logger.info("Starting ROBUST sequential resume parsing for file '%s'.", file_path)

    # 1. Extract Text from the file
    try:
        resume_raw_text = extract_text_from_file(file_path)
        if not resume_raw_text:
            return {"error": "Could not extract text from the resume file."}
    except Exception as e:
        logger.error(f"Failed to read resume file: {e}", exc_info=True)
        return {"error": f"Failed to read resume file: {e}"}

    truncated_resume_text = resume_raw_text[:MAX_RESUME_CONTEXT_CHARS]
    if len(resume_raw_text) > len(truncated_resume_text):
        logger.warning(f"Resume text truncated to {MAX_RESUME_CONTEXT_CHARS} characters.")

    # This will hold the final, assembled data in the correct order
    final_resume_data = OrderedDict()

    # Define the sections to parse sequentially
    # Each entry has: keys for the final dict, the prompt for the LLM
    parsing_steps = [
        (
            ["full_name", "email", "phone_number", "linkedin_profile", "github_profile", "personal_website"],
            """From the resume text, extract contact information. Respond with a valid JSON object containing these keys: "full_name", "email", "phone_number", "linkedin_profile", "github_profile", "personal_website". Use empty strings "" if a value is not found. Output JSON only."""
        ),
        (
            ["professional_summary"],
            """From the resume text, extract the professional summary. Respond with a valid JSON object with one key: "professional_summary". The value should be a single string, truncated to 5 clear lines using \\n. Output JSON only."""
        ),
        (
            ["skills", "technologies", "core_competencies"],
            """From the resume text, extract skills, technologies, and core competencies. Respond with a valid JSON object with three keys: "skills", "technologies", "core_competencies". Each key's value must be a list of strings. Output JSON only."""
        ),
        (
            ["experience"],
            """From the resume text, extract all work experience entries. Respond with a valid JSON object with one key: "experience". The value must be a list of objects. Each object must have these keys: "title", "company", "location", "duration", and "responsibilities" (which must be a list of strings). Output JSON only."""
        ),
        (
            ["education"],
            """From the resume text, extract all education entries. Respond with a valid JSON object with one key: "education". The value must be a list of objects. Each object must have these keys: "degree", "university", "location", "year_obtained". Output JSON only."""
        ),
        (
            ["awards", "certifications", "languages"],
            """From the resume text, extract all awards, certifications, and languages. Respond with a valid JSON object with three keys: "awards", "certifications", "languages". Each key's value must be a list of strings. Output JSON only."""
        )
    ]

    # 2. Sequentially Parse Each Section
    for keys, prompt in parsing_steps:
        logger.info(f"Parsing section(s): {', '.join(keys)}...")
        json_str = ""
        try:
            llm_response = run_llama_prompt(prompt, context=truncated_resume_text, max_new_tokens=2000, temperature=0.05)
            
            # Extract JSON from markdown
            json_match = re.search(r'```json\n(.*)\n```', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else: # Fallback for non-markdown responses
                first_brace = llm_response.find('{')
                last_brace = llm_response.rfind('}')
                if first_brace != -1 and last_brace != -1:
                    json_str = llm_response[first_brace : last_brace + 1].strip()
                else:
                    json_str = llm_response.strip()

            # Clean the JSON string to fix common LLM errors
            cleaned_json_str = re.sub(r',\s*([\]}])', r'\1', json_str)  # Fix trailing commas
            cleaned_json_str = cleaned_json_str.replace("“", "\"").replace("”", "\"").replace("’", "'")  # Fix smart quotes
            cleaned_json_str = cleaned_json_str.replace("\u201c", "\"").replace("\u201d", "\"").replace("\u2019", "'")
            
            # Parse the cleaned string
            section_data = json.loads(cleaned_json_str)

            # Add the parsed data to our final dictionary
            if isinstance(section_data, dict):
                final_resume_data.update(section_data)

        except json.JSONDecodeError as e:
            logger.error(f"Could not parse JSON for section '{keys[0]}'. Error: {e}. Raw Response: {json_str[:500]}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while parsing section '{keys[0]}': {e}", exc_info=True)

    # 3. Final Validation and Cleanup
    desired_order = [
        "full_name", "email", "phone_number", "linkedin_profile", "github_profile", "personal_website",
        "professional_summary", "skills", "technologies", "core_competencies",
        "experience", "education", "awards", "certifications", "languages"
    ]
    
    # Ensure all keys exist with default values and are in the correct order
    ordered_resume_data = OrderedDict()
    for key in desired_order:
        value = final_resume_data.get(key)
        if value is None:
            # If a section failed to parse, provide a default empty value
            if key in ["experience", "education", "skills", "technologies", "core_competencies", "awards", "certifications", "languages"]:
                ordered_resume_data[key] = []
            else:
                ordered_resume_data[key] = ""
        else:
            ordered_resume_data[key] = value

    end_total_time = time.perf_counter()
    logger.info("Total sequential resume parsing completed in %.4f s.", end_total_time - start_total_time)

    # Final validation of the assembled object
    if not validate_json_structure(ordered_resume_data):
        logger.error("The final assembled JSON has an invalid structure.")
        return {"error": "The final assembled JSON has an invalid structure."}

    return ordered_resume_data