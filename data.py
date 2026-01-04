"""
Data loading utilities for test-time compute experiments.
"""
import json


def load_jsonl_dataset(path: str) -> list[dict]:
    """
    Load a JSONL dataset file.
    
    Each line should be a JSON object with at least: "id", "problem", "answer"
    
    Args:
        path: Path to the JSONL file
        
    Returns:
        List of problem dicts
    """
    items: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


# Backward compatibility alias
load_frontier_math = load_jsonl_dataset
