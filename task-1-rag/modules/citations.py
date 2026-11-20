"""Check citation syntax and bounds, not factual entailment."""
import re


def validate_answer(answer, source_count, not_found):
    answer = answer.strip()
    if not answer:
        raise ValueError("The model returned an empty answer. Please retry.")
    if answer == not_found:
        return answer
    # Preserve explicit passage IDs, discarding invented line precision.
    # Unknown citation bodies are deliberately left for rejection below.
    answer = re.sub(
        r"【\s*(\d+)\s*(?:†\s*L\d+(?:\s*[-–]\s*L?\d+)?)?\s*】",
        lambda match: f"[{match.group(1)}]",
        answer,
    )
    if any(symbol in answer for symbol in ("【", "】", "†")):
        raise ValueError("Unsupported citation format. Please retry the question.")
    references = re.findall(r"\[(\d+)\]", answer)
    if not references:
        raise ValueError("The answer has no passage citations. Please retry.")
    if any(not 1 <= int(reference) <= source_count for reference in references):
        raise ValueError("The answer cites a passage that was not retrieved. Please retry.")
    return answer
