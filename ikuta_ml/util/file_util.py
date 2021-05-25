from csv import QUOTE_ALL
from pandas import DataFrame
from pathlib import Path
from typing import Any, List


def write_csv_from_list(
    data: List[Any],  # mainly dataclass
    dir_path: Path,
    file_name: str,
) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / file_name

    df = DataFrame(data)
    df.to_csv(
        file_path,
        encoding='utf-8',
        header=True,
        index=False,
        quoting=QUOTE_ALL,
    )

    return file_path
