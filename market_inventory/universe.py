import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from .text_utils import normalize_text


@dataclass(frozen=True)
class CoinUniverse:
    symbols: set[str]
    name_to_symbol: dict[str, str]

    @classmethod
    def from_json(cls, path: str | Path = "coins_universe.json") -> Self:
        with Path(path).open("r", encoding="utf-8") as handle:
            entries = json.load(handle)
        symbols = set()
        name_to_symbol: dict[str, str] = {}
        for entry in entries:
            sym = (entry.get("symbol") or "").strip().lower()
            name = (entry.get("name") or "").strip().lower()
            if sym:
                symbols.add(sym)
            if name and sym:
                name_to_symbol[name] = sym
        return cls(symbols=symbols, name_to_symbol=name_to_symbol)


@dataclass(frozen=True)
class ProjectUniverse:
    """
    Deterministic mapping of known project identifiers -> canonical labels.

    Example JSON input entries:
      {"key": "metamask", "label": "MetaMask", "type": "project"}
      {"key": "usd.ai", "label": "USD.AI", "type": "project"}

    Notes:
    - We only use "key" and "label". Other fields are ignored (but allowed).
    - Matching is exact on normalized tokens/phrases (no fuzzy matching).
    """

    key_to_label: dict[str, str]

    @classmethod
    def from_json(cls, path: str | Path = "projects_universe.json") -> Self:
        """
        Load a ProjectUniverse from a JSON file containing a list of objects.

        Each object should contain:
          - key: str (normalized matching key; we normalize again defensively)
          - label: str (canonical label to output)
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            entries = json.load(handle)

        if not isinstance(entries, list):
            raise ValueError(f"{path} must contain a JSON list of objects")

        key_to_label: dict[str, str] = {}
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"{path}[{i}] must be an object, got {type(entry).__name__}"
                )

            raw_key = entry.get("key")
            raw_label = entry.get("label")

            if not isinstance(raw_key, str) or not raw_key.strip():
                continue
            if not isinstance(raw_label, str) or not raw_label.strip():
                continue

            key = normalize_text(raw_key)
            label = raw_label.strip()

            if not key:
                continue

            key_to_label[key] = label

        return cls(key_to_label=key_to_label)

    def match(self, text: str, max_phrase_len: int = 4) -> str | None:
        """
        Attempt to match `text` against known project keys.

        Strategy (deterministic):
        1) Exact token match
        2) Exact phrase match for 2..max_phrase_len tokens (longest-first)

        Returns:
          canonical label if matched, else None.
        """
        query = normalize_text(text)
        if not query:
            return None

        tokens = query.split()

        for token in tokens:
            label = self.key_to_label.get(token)
            if label is not None:
                return label

        max_len = min(max_phrase_len, len(tokens))
        for n in range(max_len, 1, -1):
            for i in range(len(tokens) - n + 1):
                phrase = " ".join(tokens[i : i + n])
                label = self.key_to_label.get(phrase)
                if label is not None:
                    return label

        return None
