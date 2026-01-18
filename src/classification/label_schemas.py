# src/classification/label_schemas.py
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Type, TypeVar


# ------------------------------------------------------------
# ENUM-DEFINITIONEN
# ------------------------------------------------------------

class PolicyArea(Enum):
    ENERGY_BUILDINGS = "energy_buildings"
    MOBILITY_TRANSPORT = "mobility_transport"
    LAND_USE_URBAN_PLANNING = "land_use_urban_planning"
    WASTE_CIRCULAR_ECONOMY = "waste_circular_economy"
    INDUSTRY_COMMERCE_SERVICES = "industry_commerce_services"
    AGRICULTURE_FORESTRY = "agriculture_forestry"
    CLIMATE_ADAPTATION_RESILIENCE = "climate_adaptation_resilience"
    CROSS_SECTOR_GOVERNANCE = "cross_sector_governance"
    OTHER = "other"


class InstrumentType(Enum):
    REGULATION_STANDARD = "regulation_standard"
    ECONOMIC_INCENTIVE = "economic_incentive"
    INFORMATION_EDUCATION = "information_education"
    PLANNING_STRATEGY = "planning_strategy"
    INFRASTRUCTURE_INVESTMENT = "infrastructure_investment"
    PUBLIC_PROCUREMENT_OPERATIONS = "public_procurement_operations"
    VOLUNTARY_AGREEMENT_PARTNERSHIP = "voluntary_agreement_partnership"
    OTHER = "other"


class TargetSector(Enum):
    HOUSEHOLDS_RESIDENTS = "households_residents"
    BUSINESSES_INDUSTRY = "businesses_industry"
    TRANSPORT_OPERATORS = "transport_operators"
    PUBLIC_SECTOR_MUNICIPAL = "public_sector_municipal"
    PUBLIC_SECTOR_OTHER = "public_sector_other"
    AGRICULTURE_FORESTRY = "agriculture_forestry"
    CIVIL_SOCIETY_NGOS = "civil_society_ngos"
    MULTIPLE_GENERAL_PUBLIC = "multiple_general_public"


class ClimateDimension(Enum):
    MITIGATION = "mitigation"
    ADAPTATION = "adaptation"
    BOTH = "both"
    NOT_DIRECTLY_CLIMATE = "not_directly_climate"


class GovernanceType(Enum):
    OWN_OPERATIONS = "own_operations"
    REGULATOR = "regulator"
    ENABLER_SUPPORTER = "enabler_supporter"
    COORDINATOR_NETWORKER = "coordinator_networker"
    PARTICIPATION_COPRODUCTION = "participation_coproduction"
    IMPLEMENTATION_HIGHER_LEVEL = "implementation_higher_level"
    OTHER = "other"


class FundingAvailable(Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


# ------------------------------------------------------------
# GENERISCHE UTILITY-FUNKTIONEN
# ------------------------------------------------------------

E = TypeVar("E", bound=Enum)


def normalize_label_string(value: str) -> str:
    """
    Normiert ein Label zu einem maschinenlesbaren Schlüssel, der den Enum-Values ähnelt:

      "Energy & Buildings"  -> "energy_buildings"
      "ENERGY-BUILDINGS"    -> "energy_buildings"
      " energy / buildings" -> "energy_buildings"
    """
    value = value.strip().lower()
    # Alles, was kein Buchstabe/Ziffer ist, durch '_' ersetzen
    value = re.sub(r"[^a-z0-9]+", "_", value)
    # Führende/abschließende '_' entfernen
    value = value.strip("_")
    return value


def enum_choices(enum_cls: Type[E]) -> List[str]:
    """Liste aller möglichen .value-Strings eines Enums."""
    return [e.value for e in enum_cls]


def parse_enum(
    enum_cls: Type[E],
    raw: Optional[str],
    default: Optional[E] = None,
) -> Optional[E]:
    """
    Wandelt einen String robust in ein Enum um.

    - None oder leerer String -> default
    - Case-insensitive
    - toleriert Leerzeichen, '-', '/', '&', etc. über normalize_label_string
    - Matcht gegen .value der Enum-Mitglieder

    Beispiel:
      parse_enum(PolicyArea, "Energy & Buildings") -> PolicyArea.ENERGY_BUILDINGS
      parse_enum(FundingAvailable, "TRUE") -> FundingAvailable.TRUE
    """
    if raw is None:
        return default

    raw = raw.strip()
    if not raw:
        return default

    # 1) exakter Match gegen .value
    for member in enum_cls:
        if raw == member.value:
            return member

    # 2) normalisierte Form
    norm = normalize_label_string(raw)
    for member in enum_cls:
        if norm == member.value:
            return member

    # 3) manchmal wird mit Namen statt value gearbeitet
    for member in enum_cls:
        if norm == normalize_label_string(member.name):
            return member

    return default


def enum_to_str(value: Optional[Enum]) -> Optional[str]:
    """Gibt value.value zurück (oder None, falls value None ist)."""
    if value is None:
        return None
    return value.value


def parse_enum_list(
    enum_cls: Type[E],
    raw_values: Iterable[Optional[str]],
) -> List[E]:
    """
    Konvertiert eine Liste von Strings nach Enum-Liste.
    Nicht parsebare/None-Werte werden still verworfen.

    Sinnvoll für:
      - TargetSector (Multi-Label)
      - CSV-Import mit Komma-getrennten Werten etc.
    """
    result: List[E] = []
    for raw in raw_values:
        member = parse_enum(enum_cls, raw, default=None)
        if member is not None and member not in result:
            result.append(member)
    return result


def parse_comma_separated_enum_list(
    enum_cls: Type[E],
    raw: Optional[str],
) -> List[E]:
    """
    Spezialfall: 'households_residents,businesses_industry'
    oder 'Households & Residents; Businesses / Industry'
    """
    if raw is None:
        return []

    # Split an Komma, Semikolon etc.
    parts = re.split(r"[;,]+", raw)
    return parse_enum_list(enum_cls, parts)


# ------------------------------------------------------------
# DATACLASS FÜR GEBÜNDELTE LABELS
# ------------------------------------------------------------

@dataclass
class MeasureLabels:
    """
    Zentrales Label-Bündel für ein Policy-Maßnahmen-Objekt.

    Alle Felder sind optional; Target-Sektoren sind Multi-Label.
    """
    policy_area: Optional[PolicyArea] = None
    instrument_type: Optional[InstrumentType] = None
    target_sectors: List[TargetSector] = field(default_factory=list)
    climate_dimension: Optional[ClimateDimension] = None
    governance_type: Optional[GovernanceType] = None
    funding_available: Optional[FundingAvailable] = None

    # ----------------------------
    # Serialisierung ↔ Dict/JSON
    # ----------------------------

    def to_dict_str(self) -> Dict[str, Any]:
        """
        Serialisiert die Labels in ein Dict mit String-Werten,
        passend für JSON/CSV/NDJSON.
        """
        return {
            "policy_area": enum_to_str(self.policy_area),
            "instrument_type": enum_to_str(self.instrument_type),
            "target_sectors": [s.value for s in self.target_sectors],
            "climate_dimension": enum_to_str(self.climate_dimension),
            "governance_type": enum_to_str(self.governance_type),
            "funding_available": enum_to_str(self.funding_available),
        }

    @classmethod
    def from_dict_str(cls, data: Dict[str, Any]) -> "MeasureLabels":
        """
        Baut ein MeasureLabels-Objekt aus einem Dict mit String-Werten.
        """
        return cls(
            policy_area=parse_enum(PolicyArea, data.get("policy_area")),
            instrument_type=parse_enum(InstrumentType, data.get("instrument_type")),
            target_sectors=parse_enum_list(
                TargetSector,
                data.get("target_sectors") or [],
            ),
            climate_dimension=parse_enum(ClimateDimension, data.get("climate_dimension")),
            governance_type=parse_enum(GovernanceType, data.get("governance_type")),
            funding_available=parse_enum(FundingAvailable, data.get("funding_available")),
        )

    def to_debug_dict(self) -> Dict[str, Any]:
        """
        Hilfsfunktion für Logging / Debugging:
        Gibt Enum-Namen statt Values aus.
        """
        def name_or_none(v: Optional[Enum]) -> Optional[str]:
            return None if v is None else v.name

        return {
            "policy_area": name_or_none(self.policy_area),
            "instrument_type": name_or_none(self.instrument_type),
            "target_sectors": [s.name for s in self.target_sectors],
            "climate_dimension": name_or_none(self.climate_dimension),
            "governance_type": name_or_none(self.governance_type),
            "funding_available": name_or_none(self.funding_available),
        }

__all__ = [
    "PolicyArea",
    "InstrumentType",
    "TargetSector",
    "ClimateDimension",
    "GovernanceType",
    "FundingAvailable",
    "MeasureLabels",
    "normalize_label_string",
    "enum_choices",
    "parse_enum",
    "enum_to_str",
    "parse_enum_list",
    "parse_comma_separated_enum_list",
]
