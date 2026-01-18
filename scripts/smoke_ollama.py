import json
from pathlib import Path
import yaml

from src.classification.llm_client import LLMClient

def load_yaml(path: str) -> dict:
    p = Path(path)
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}

if __name__ == "__main__":
    cfg = load_yaml("configs/llm.yml")
    client = LLMClient(config=cfg, profile="classification_baseline")

    text = "Die Stadt fördert Balkonkraftwerke mit einem Zuschuss von 100 Euro pro Haushalt."
    res = client.extract_batch(texts=[text], titles=["Balkon-PV Förderung"], urls=[None], scope="TEST_CITY")

    print(json.dumps(res[0], ensure_ascii=False, indent=2))
    assert isinstance(res[0], dict)
    assert "confidence_score" in res[0]
