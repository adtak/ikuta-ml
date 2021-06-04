import pandas as pd
from csv import QUOTE_ALL
from pathlib import Path
from typing import Any, List


def write_csv_from_list(
    data: List[Any],  # mainly dataclass
    dir_path: Path,
    file_name: str,
    mode: str = 'x'
) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / file_name

    df = pd.DataFrame(data)
    df.to_csv(
        file_path,
        mode=mode,
        encoding='utf-8',
        header=False if mode == 'a' else True,
        index=False,
        quoting=QUOTE_ALL,
    )

    return file_path


def read_csv(
    dir_path: Path,
    file_name: str,
) -> pd.DataFrame:
    file_path = dir_path / file_name
    return pd.read_csv(file_path)
