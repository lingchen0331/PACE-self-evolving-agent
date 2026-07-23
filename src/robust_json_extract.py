"""Robust JSON extraction utility for solving MMLU-style tasks with malformed output."""
import re

def robust_json_extract(text_dict_raw):
    """
    Safely extract dict with fallback parsing
    
    Args:
        text_dict_raw: Either a dict OR a JSON string from LLM output
    
    Returns:
        dict with 'reasoning', 'answer', 'response' keys (fallback OK)
    """
    result = {}
    
    # Handle non-dict input (string/parse error)
    if not isinstance(text_dict_raw, dict):
        raw_str = str(text_dict_raw)[:500] if text_dict_raw else ""
        result["error"] = "Non-JSON output, parsing from raw text"
        result["reasoning"] = raw_str if raw_str != "NO reasoning IN DICTIONARY" else ""
        result["answer"] = "ERROR"
        result["response"] = raw_str
        return result
    
    # Extract reasoning/key (handle duplicates if model messed up)
    reasoning_keys = [k for k in text_dict_raw.keys() if k.lower().replace(" ", "_") == "reasoning"]
    if reasoning_keys:
        rkey = reasoning_keys[0]
        result["reasoning"] = str(text_dict_raw.get(rkey, "")).strip()
    else:
        # Fall back to combined keys
        all_reason = "".join([text_dict_raw.get(k, "") for k in text_dict_raw if "reason" in k.lower()])
        result["reasoning"] = all_reason.strip()
    
    # Extract answer
    answer_text = ""
    if "answer" in text_dict_raw:
        answer_text = str(text_dict_raw["answer"]).strip()
    elif any("answer" in k.lower() and k != "answer" for k in text_dict_raw):
        for k in text_dict_raw:
            if "answer" in k.lower() and k != "answer":
                answer_text = str(text_dict_raw[k]).strip()
                break
    
    # Clean answer - extract letter A/B/C/D
    answer_text = re.sub(r"[\W]", "", answer_text).strip().upper()
    a_match = re.match(r"[ABCD]", answer_text)
    result["answer"] = a_match.group() if a_match else (answer_text if answer_text else "UNKNOWN")
    
    # Extract response  
    if "response" in text_dict_raw:
        result["response"] = str(text_dict_raw["response"]).strip()
    elif any("response" in k.lower() and k != "response" for k in text_dict_raw):
        for k in text_dict_raw:
            if "response" in k.lower() and k != "response":
                result["response"] = str(text_dict_raw[k]).strip()
                break
    else:
        result["response"] = answer_text
    
    return result
