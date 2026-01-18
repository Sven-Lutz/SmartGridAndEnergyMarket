# src/classification/schema.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .label_schemas import (
    PolicyArea,
    InstrumentType,
    TargetSector,
    ClimateDimension,
    GovernanceType,
    FundingAvailable,
    MeasureLabels,
)


# ---------------------------------------------------------
# Policy-Candidates (Zwischenprodukt nach extract-policies)
# ---------------------------------------------------------


@dataclass
class PolicyCandidateRecord:
    """
    Repräsentiert einen Eintrag in policy_candidates.ndjson.
    """

    candidate_id: str
    document_id: str
    municipality_id: Optional[str]
    municipality: Optional[str]
    title: str
    snippet: str
    url: Optional[str]

    # Metadaten aus der Heuristik
    keyword_hit: bool
    confidence_score: float

    # Label-Bündel (alle optional)
    labels: MeasureLabels

    # ---------------------------------------
    # Fabrikmethoden ↔ Dict (NDJSON/JSON)
    # ---------------------------------------

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyCandidateRecord":
        """
        Baut einen PolicyCandidateRecord aus einem Dict, wie es aus
        policy_candidates.ndjson eingelesen wird.
        Erwartet, dass Label-Felder als Strings bzw. Listen von Strings
        vorliegen (kompatibel zu MeasureLabels.from_dict_str).
        """
        labels = MeasureLabels.from_dict_str(data)

        return cls(
            candidate_id=data["candidate_id"],
            document_id=data["document_id"],
            municipality_id=data.get("municipality_id"),
            municipality=data.get("municipality"),
            title=data.get("title", ""),
            snippet=data.get("snippet", ""),
            url=data.get("url"),
            keyword_hit=bool(data.get("keyword_hit", False)),
            confidence_score=float(data.get("confidence_score", 0.0)),
            labels=labels,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialisiert den Record zurück in ein Dict, das mit der aktuellen
        policy_candidates.ndjson-Struktur kompatibel ist.
        Enums werden dabei zu Strings (value) konvertiert.
        """
        base: Dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "document_id": self.document_id,
            "municipality_id": self.municipality_id,
            "municipality": self.municipality,
            "title": self.title,
            "snippet": self.snippet,
            "url": self.url,
            "keyword_hit": self.keyword_hit,
            "confidence_score": self.confidence_score,
        }

        base.update(self.labels.to_dict_str())
        return base

    def to_debug_dict(self) -> Dict[str, Any]:
        """
        Dict mit Enum-Namen (statt Values) – hilfreich für Logging.
        """
        debug = {
            "candidate_id": self.candidate_id,
            "document_id": self.document_id,
            "municipality_id": self.municipality_id,
            "municipality": self.municipality,
            "title": self.title,
            "snippet": self.snippet,
            "url": self.url,
            "keyword_hit": self.keyword_hit,
            "confidence_score": self.confidence_score,
        }
        debug.update(self.labels.to_debug_dict())
        return debug


PolicyCandidateList = List[PolicyCandidateRecord]


# ---------------------------------------------------------
# Measures (Output der Measure-Extraktion)
# ---------------------------------------------------------


@dataclass
class MeasureRecord:
    """
    Repräsentiert eine konkrete Maßnahme, wie sie später im
    policy-knowledgebase-Dataset landen soll.

    Aktuell: 1:1 aus einem PolicyCandidateRecord erzeugt (keine
    weitere Aufspaltung). Später kann hier Feingranularität hinzukommen.
    """

    measure_id: str
    candidate_id: str
    document_id: str
    municipality_id: Optional[str]
    municipality: Optional[str]
    title: str          # Kurzbezeichnung der Maßnahme (heute: Dokumenttitel)
    description: str    # Beschreibung / Snippet
    url: Optional[str]

    labels: MeasureLabels

    @classmethod
    def from_candidate(
        cls,
        candidate: PolicyCandidateRecord,
        idx_within_candidate: int = 0,
    ) -> "MeasureRecord":
        """
        Baut eine Measure aus einem PolicyCandidateRecord.

        Aktuell wird pro Candidate genau eine Measure erzeugt.
        Der Parameter idx_within_candidate ist vorbereitet für den Fall,
        dass wir später einen Candidate in mehrere Maßnahmen aufsplitten.
        """
        measure_id = f"{candidate.candidate_id}::m{idx_within_candidate}"

        return cls(
            measure_id=measure_id,
            candidate_id=candidate.candidate_id,
            document_id=candidate.document_id,
            municipality_id=candidate.municipality_id,
            municipality=candidate.municipality,
            title=candidate.title,
            description=candidate.snippet,
            url=candidate.url,
            labels=candidate.labels,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MeasureRecord":
        """
        Falls wir Measures wieder aus NDJSON einlesen wollen.
        """
        labels = MeasureLabels.from_dict_str(data)

        return cls(
            measure_id=data["measure_id"],
            candidate_id=data["candidate_id"],
            document_id=data["document_id"],
            municipality_id=data.get("municipality_id"),
            municipality=data.get("municipality"),
            title=data.get("title", ""),
            description=data.get("description", ""),
            url=data.get("url"),
            labels=labels,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialisiert die Measure in ein flaches Dict mit String-Labels.
        """
        base: Dict[str, Any] = {
            "measure_id": self.measure_id,
            "candidate_id": self.candidate_id,
            "document_id": self.document_id,
            "municipality_id": self.municipality_id,
            "municipality": self.municipality,
            "title": self.title,
            "description": self.description,
            "url": self.url,
        }
        base.update(self.labels.to_dict_str())
        return base

    def to_debug_dict(self) -> Dict[str, Any]:
        debug = {
            "measure_id": self.measure_id,
            "candidate_id": self.candidate_id,
            "document_id": self.document_id,
            "municipality_id": self.municipality_id,
            "municipality": self.municipality,
            "title": self.title,
            "description": self.description,
            "url": self.url,
        }
        debug.update(self.labels.to_debug_dict())
        return debug


MeasureList = List[MeasureRecord]
