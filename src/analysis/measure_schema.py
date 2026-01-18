from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MeasureAnalysisRow:
    # Identifikatoren / Scope
    scope: str
    country_code: str
    state_code: str
    municipality_id: str
    municipality_name: str
    measure_local_id: int          # Laufende ID pro Dokument/Municipality
    measure_uid: str               # z.B. de_by::nuernberg::42::0

    # Dokument-/Crawl-Verknüpfung
    document_id: str
    source_url: str
    crawl_timestamp: Optional[str]
    depth: Optional[int]
    referrer_url: Optional[str]
    http_status: Optional[int]
    mime_type: Optional[str]

    # Textinhalte
    document_title: str
    measure_title: str
    text_excerpt: str

    # Taxonomie-Felder
    policy_area: str
    instrument_type: str
    target_sectors: List[str]
    climate_dimension: str
    governance_type: str
    funding_available: bool

    # Label-Infos / Versionen
    label_source: str
    confidence_score: float
    extraction_version: str
    classification_version: str
    keyword_hit: bool
