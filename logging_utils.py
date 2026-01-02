import json


def write_jsonl(path: str, rows: list[dict]):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")




