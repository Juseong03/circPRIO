"""Download the study's frozen SpliceBERT checkpoint (not tracked by Git)."""
from pathlib import Path
from huggingface_hub import snapshot_download
ROOT = Path(__file__).resolve().parents[1]
snapshot_download('multimolecule/splicebert', revision='8e3f4e5088b3bbf76876d8b931078d9f03caffdd',
                  local_dir=str(ROOT / 'artifacts/models/splicebert'), allow_patterns=['*.json', '*.safetensors', '*.txt'])
