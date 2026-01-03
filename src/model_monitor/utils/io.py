import yaml
from pathlib import Path
from typing import Union

def load_yaml(path: Union[str, Path]):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
