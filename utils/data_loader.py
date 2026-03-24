"""
DataLoader: Handles CSV, Excel, JSON, Parquet, and SQLite sources.
"""

from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".parquet", ".db", ".sqlite"}


class DataLoader:
    """Load tabular data from common file formats into a DataFrame."""

    def load(self, path: Path) -> pd.DataFrame:
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
            )
        loader = getattr(self, f"_load{ext.replace('.', '_')}", None)
        if loader is None:
            raise NotImplementedError(f"No loader implemented for '{ext}'")
        return loader(path)

    def _load_csv(self, path: Path) -> pd.DataFrame:
        with open(path, "r", errors="ignore") as f:
            sample = f.read(2048)
        sep = ";" if sample.count(";") > sample.count(",") else ","
        return pd.read_csv(path, sep=sep, low_memory=False)

    def _load_xlsx(self, path: Path) -> pd.DataFrame:
        return pd.read_excel(path)

    def _load_xls(self, path: Path) -> pd.DataFrame:
        return pd.read_excel(path)

    def _load_json(self, path: Path) -> pd.DataFrame:
        return pd.read_json(path)

    def _load_parquet(self, path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)

    def _load_db(self, path: Path) -> pd.DataFrame:
        return self._load_sqlite(path)

    def _load_sqlite(self, path: Path) -> pd.DataFrame:
        import sqlite3

        con = sqlite3.connect(path)
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", con)
        if tables.empty:
            raise ValueError("No tables found in SQLite database.")
        table_name = tables.iloc[0]["name"]
        df = pd.read_sql(f"SELECT * FROM '{table_name}'", con)
        con.close()
        print(f"[DataLoader] Loaded table '{table_name}' from SQLite.")
        return df


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Convenience helper used by API inference endpoint."""
    return DataLoader().load(Path(path))
