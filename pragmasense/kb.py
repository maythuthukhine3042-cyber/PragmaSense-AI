from __future__ import annotations
"""Bilingual pragmatic knowledge base (symbolic memory).

Loads data/knowledge_base.json and provides:
  - Thai particle extraction (used by the perception tier)
  - Myanmar marker detection (used by the critic)
  - the rule that turns a set of markers into one register level
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_KB_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"

_THAI_PUNCT = " \t\n.,!?…\"'()"


@dataclass
class Marker:
    text: str
    level: float | None
    functions: list
    gender: str | None = None


@dataclass
class RegisterProfile:
    """Summary of the pragmatic markers found in one utterance."""
    markers: list = field(default_factory=list)
    level: float = 2.0
    functions: set = field(default_factory=set)
    gender: str = "unknown"

    def marker_texts(self):
        return [m.text for m in self.markers]


class KnowledgeBase:
    def __init__(self, path=DEFAULT_KB_PATH):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        self.raw = raw
        self.default_level = raw["default_level"]
        self.threshold = raw["threshold"]
        self.suggestions = raw["suggestions"]
        self.vocatives = raw["thai_vocatives"]

        # Thai: longest markers first so that e.g. "ได้แล้ว" is tried before shorter ones
        self.thai = sorted(raw["thai"], key=lambda e: -len(e["marker"]))
        self.thai_suffix = [e for e in self.thai if e["position"] == "suffix"]
        self.thai_prefix = [e for e in self.thai if e["position"] == "prefix"]
        self.thai_anywhere = [e for e in self.thai if e["position"] == "anywhere"]

        end = raw["myanmar_end"]
        self.myanmar = []
        for e in raw["myanmar"]:
            pat = e["pattern"].replace("END", end)
            self.myanmar.append((re.compile(pat), e))

    # ------------------------------------------------------------------ Thai
    def _suffix_chain(self, chunk):
        """Peel sentence-final particles off the end of one space-separated chunk.
        e.g. 'ขอยืมปากกาหน่อยนะคะ' -> ['หน่อย', 'นะ', 'คะ']"""
        found = []
        rest = chunk.strip(_THAI_PUNCT)
        # drop a trailing vocative such as 'นะลูก' -> 'นะ'
        for voc in self.vocatives:
            if rest.endswith(voc) and len(rest) > len(voc):
                rest = rest[: -len(voc)]
                break
        changed = True
        while changed and rest:
            changed = False
            for e in self.thai_suffix:
                if rest.endswith(e["marker"]):
                    found.append(e)
                    rest = rest[: -len(e["marker"])]
                    changed = True
                    break
        return list(reversed(found)), rest

    def analyse_thai(self, text):
        markers = []
        propositional = []
        for chunk in text.split():
            for e in self.thai_prefix:
                if chunk.startswith(e["marker"]):
                    markers.append(e)
            for e in self.thai_anywhere:
                if e["marker"] in chunk:
                    markers.append(e)
            chain, rest = self._suffix_chain(chunk)
            markers.extend(chain)
            if rest:
                propositional.append(rest)
        profile = self._profile([Marker(e["marker"], e["level"], e["functions"], e.get("gender"))
                                 for e in markers])
        return profile, " ".join(propositional)

    # --------------------------------------------------------------- Myanmar
    def analyse_myanmar(self, text):
        text = text.strip()
        markers = []
        for rx, e in self.myanmar:
            for _ in rx.finditer(text):
                markers.append(Marker(e["marker"], e["level"], e["functions"], e.get("gender")))
        return self._profile(markers)

    # ----------------------------------------------------------- combination
    def _profile(self, markers):
        """Combine markers into one register level.
        Rule: a rough/familiar marker overrides everything (one 'ကွာ' makes the
        whole utterance rough), so if any familiarity marker is present we take
        the minimum level; otherwise the most polite marker wins (maximum)."""
        levels = [m.level for m in markers if m.level is not None]
        funcs = set(f for m in markers for f in m.functions)
        if not levels:
            level = self.default_level
        elif "familiarity" in funcs:
            level = min(levels)
        else:
            level = max(levels)
        genders = {m.gender for m in markers if m.gender}
        gender = genders.pop() if len(genders) == 1 else ("mixed" if genders else "unknown")
        return RegisterProfile(markers=markers, level=level, functions=funcs, gender=gender)

    def suggest(self, function, gender="unknown"):
        s = self.suggestions[function]
        if isinstance(s, dict):
            return s.get(gender, s["unknown"])
        return s
