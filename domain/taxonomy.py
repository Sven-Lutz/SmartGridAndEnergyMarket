# src/domain/taxonomy.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, TypedDict


# -------------------------------------------------------------------
# Grundidee:
# - Canonical IDs (klein, snake_case) für Speicherung/Evaluation
# - Deutsche Labels & kurze Beschreibung + Beispiel-Keywords
# - Entspricht Antrag: Policyfeld, Instrument, Sektor, Klimadimension, Ebene, Förderung
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Typ-Aliase (für bessere Typisierung)
# -------------------------------------------------------------------

PolicyAreaId = Literal[
    "buildings",
    "energy_supply",
    "transport_mobility",
    "industry_commerce",
    "households",
    "agriculture_land_use",
    "cross_sectoral",
    "climate_adaptation",
]

InstrumentTypeId = Literal[
    "financial_incentive",
    "regulation",
    "strategy",
    "administrative",
    "advisory_service",
    "information_outreach",
    "reporting_monitoring",
]

JurisdictionLevelId = Literal[
    "eu",
    "federal",
    "state",
    "local",
]

TargetSectorId = Literal[
    "private_households",
    "enterprises",
    "public_admin",
    "ngos_civil_society",
    "energy_suppliers",
    "agriculture",
    "transport_users",
    "other",
]

ClimateDimensionId = Literal[
    "mitigation",
    "adaptation",
    "cross_cutting",
    "unspecified",
]

GovernanceTypeId = Literal[
    "legally_binding",
    "political_commitment",
    "administrative_directive",
    "voluntary",
    "incentive_based",
    "unspecified",
]


# -------------------------------------------------------------------
# Dataklassen mit Meta-Infos
# -------------------------------------------------------------------

@dataclass(frozen=True)
class TaxonomyEntry:
    id: str
    label_de: str
    label_en: str
    description: str
    examples: Sequence[str]


# -------------------------------------------------------------------
# POLICY AREA
# -------------------------------------------------------------------

POLICY_AREAS: Dict[PolicyAreaId, TaxonomyEntry] = {
    "buildings": TaxonomyEntry(
        id="buildings",
        label_de="Gebäude / Wärme",
        label_en="Buildings and heating",
        description=(
            "Gebäudesanierung, Heizungstausch, Wärmenetze, Dämmung, "
            "Energieeffizienz in Wohn- und Nichtwohngebäuden."
        ),
        examples=[
            "Heizungstausch",
            "Gebäudesanierung",
            "kommunales Wärmenetz",
            "Wärmeplanung (gebäudebezogen)",
        ],
    ),
    "energy_supply": TaxonomyEntry(
        id="energy_supply",
        label_de="Energieversorgung",
        label_en="Energy supply",
        description=(
            "Erzeugung und Verteilung von Strom und Wärme, erneuerbare Energien, "
            "Netze und Versorgungsinfrastruktur."
        ),
        examples=[
            "Photovoltaik-Ausbau",
            "Windenergie",
            "Fernwärme",
            "Ökostromprodukte",
        ],
    ),
    "transport_mobility": TaxonomyEntry(
        id="transport_mobility",
        label_de="Verkehr und Mobilität",
        label_en="Transport and mobility",
        description=(
            "ÖPNV, Radverkehr, E-Mobilität, Verkehrsplanung mit Klimabezug."
        ),
        examples=[
            "Ausbau Radwegenetz",
            "Parkraumbewirtschaftung mit Klimaziel",
            "Förderung ÖPNV",
        ],
    ),
    "industry_commerce": TaxonomyEntry(
        id="industry_commerce",
        label_de="Wirtschaft / Industrie",
        label_en="Industry and commerce",
        description=(
            "Klimaschutzmaßnahmen und Förderungen für Unternehmen, Gewerbe und Industrie."
        ),
        examples=[
            "Energieeffizienz in Unternehmen",
            "Investitionszuschüsse für Effizienzmaßnahmen",
        ],
    ),
    "households": TaxonomyEntry(
        id="households",
        label_de="Private Haushalte",
        label_en="Households",
        description=(
            "Maßnahmen und Förderungen, die sich an private Haushalte und Bürger:innen richten."
        ),
        examples=[
            "Förderprogramm für private Sanierungen",
            "Beratung für Bürger:innen",
        ],
    ),
    "agriculture_land_use": TaxonomyEntry(
        id="agriculture_land_use",
        label_de="Landnutzung / Landwirtschaft",
        label_en="Agriculture and land use",
        description="Maßnahmen im Bereich Landwirtschaft, Wald, Boden, Flächen.",
        examples=[
            "Moorschutz",
            "klimafreundliche Landwirtschaft",
        ],
    ),
    "cross_sectoral": TaxonomyEntry(
        id="cross_sectoral",
        label_de="Querschnitt / Klimastrategie",
        label_en="Cross-sectoral / climate strategy",
        description=(
            "Klimaschutzkonzepte, integrierte Strategien, Querschnittsprogramme, "
            "kommunale Wärmeplanung mit sektorübergreifendem Charakter."
        ),
        examples=[
            "Integriertes Klimaschutzkonzept",
            "Kommunales Energie- und Klimakonzept",
            "kommunale Wärmeplanung (gesamtstädtisch)",
        ],
    ),
    "climate_adaptation": TaxonomyEntry(
        id="climate_adaptation",
        label_de="Klimaanpassung",
        label_en="Climate adaptation",
        description=(
            "Maßnahmen zur Anpassung an Klimafolgen: Hitze, Starkregen, Hochwasser, "
            "Biodiversität, Wassermanagement."
        ),
        examples=[
            "Hitzeaktionsplan",
            "Starkregenvorsorge",
            "Schwammstadt-Maßnahmen",
        ],
    ),
}


# -------------------------------------------------------------------
# INSTRUMENT TYPE
# (Paper-zentral: Förderinstrumente + andere Policy-Instrumente)
# -------------------------------------------------------------------

INSTRUMENT_TYPES: Dict[InstrumentTypeId, TaxonomyEntry] = {
    "financial_incentive": TaxonomyEntry(
        id="financial_incentive",
        label_de="Finanzieller Anreiz / Förderung",
        label_en="Financial incentive / subsidy",
        description=(
            "Zuschüsse, Kredite, Steuervergünstigungen, finanzielle Zuwendungen "
            "oder Förderprogramme zur Unterstützung von Klimamaßnahmen."
        ),
        examples=[
            "Zuschussprogramm",
            "Förderprogramm",
            "Zuwendung",
            "Kredit mit Klima-Fokus",
        ],
    ),
    "regulation": TaxonomyEntry(
        id="regulation",
        label_de="Regulierung / Verpflichtung",
        label_en="Regulation",
        description=(
            "Rechtlich verbindliche Vorgaben: Satzungen, Pflichten, Verbote, "
            "Bauordnungen, verbindliche Sanierungs- oder Effizienzanforderungen."
        ),
        examples=[
            "Solarpflicht auf Dächern",
            "lokale Klimaschutzsatzung",
        ],
    ),
    "strategy": TaxonomyEntry(
        id="strategy",
        label_de="Strategie / Konzept",
        label_en="Strategy / plan",
        description=(
            "Übergeordnete Pläne, Strategien, Konzepte mit Maßnahmenlisten "
            "und Zielsetzungen (mit oder ohne Rechtsbindung)."
        ),
        examples=[
            "Klimaschutzkonzept",
            "Klimaanpassungsstrategie",
            "kommunaler Energie- und Klimaplan",
        ],
    ),
    "administrative": TaxonomyEntry(
        id="administrative",
        label_de="Verwaltungs- / Verfahrensregel",
        label_en="Administrative / procedural",
        description=(
            "Verwaltungsvorschriften, Richtlinien, Prozesse, Zuständigkeitsregeln, "
            "die die Umsetzung von Klimapolitik steuern."
        ),
        examples=[
            "Förderrichtlinie",
            "Verwaltungsvorschrift",
            "Antragsverfahren",
        ],
    ),
    "advisory_service": TaxonomyEntry(
        id="advisory_service",
        label_de="Beratung / Services",
        label_en="Advisory services",
        description=(
            "Energieberatung, Klimaschutzberatung, Unterstützungsangebote mit "
            "direktem Bezug zu Klimaschutz oder -anpassung."
        ),
        examples=[
            "Energieberatungsstelle",
            "Klimaschutzberatung für Bürger:innen",
        ],
    ),
    "information_outreach": TaxonomyEntry(
        id="information_outreach",
        label_de="Information / Öffentlichkeitsarbeit",
        label_en="Information and outreach",
        description=(
            "Informationskampagnen, Bewusstseinsbildung, nicht-individuelle "
            "Kommunikation (Webseiten, Infobroschüren)."
        ),
        examples=[
            "Kampagne zum Energiesparen",
            "Informationsseite zu Klimaschutzangeboten",
        ],
    ),
    "reporting_monitoring": TaxonomyEntry(
        id="reporting_monitoring",
        label_de="Bericht / Monitoring",
        label_en="Reporting and monitoring",
        description=(
            "Berichte, Monitoring-Systeme, Evaluationsdokumente zu Klimazielen "
            "und -maßnahmen."
        ),
        examples=[
            "Klimaschutzbericht",
            "CO₂-Monitoring",
        ],
    ),
}


# -------------------------------------------------------------------
# Aliase für Instrumententypen (Kompatibilität zu Baseline-Regeln / Antrag)
# z.B. 'subsidy' -> 'financial_incentive'
# -------------------------------------------------------------------

INSTRUMENT_TYPE_ALIASES: Dict[str, InstrumentTypeId] = {
    "subsidy": "financial_incentive",
    "förderung": "financial_incentive",
    "foerderung": "financial_incentive",
    "standard": "regulation",
    "regulation": "regulation",
    "other": "information_outreach",  # Fallback, falls Baseline 'other' liefert
}


def normalize_instrument_type(raw: Optional[str]) -> Optional[InstrumentTypeId]:
    """Bringt frei erzeugte Labels (LLM, Heuristiken) auf unsere Canonical IDs."""
    if raw is None:
        return None
    r = raw.strip().lower()
    if r in INSTRUMENT_TYPES:
        return r  # type: ignore[return-value]
    if r in INSTRUMENT_TYPE_ALIASES:
        return INSTRUMENT_TYPE_ALIASES[r]
    return None


# -------------------------------------------------------------------
# JURISDICTION LEVEL
# -------------------------------------------------------------------

JURISDICTION_LEVELS: Dict[JurisdictionLevelId, TaxonomyEntry] = {
    "eu": TaxonomyEntry(
        id="eu",
        label_de="Europäische Ebene",
        label_en="European Union",
        description="EU-Recht, EU-Programme, EU-Strategien.",
        examples=[
            "EU-Klimagesetz",
            "LIFE-Programm",
            "Fit-for-55-Maßnahmen",
        ],
    ),
    "federal": TaxonomyEntry(
        id="federal",
        label_de="Bundesebene",
        label_en="Federal (Germany)",
        description="Bundesgesetze, Bundesförderprogramme, nationale Strategien.",
        examples=[
            "Klimaschutzgesetz des Bundes",
            "BAFA-Förderprogramm",
            "KfW-Programm",
        ],
    ),
    "state": TaxonomyEntry(
        id="state",
        label_de="Landesebene",
        label_en="State (Bundesland)",
        description="Landesgesetze, Landesprogramme (z. B. Bayern).",
        examples=[
            "BayKlimaG",
            "Landesförderprogramm Bayern",
        ],
    ),
    "local": TaxonomyEntry(
        id="local",
        label_de="Kommunale Ebene",
        label_en="Local / municipal",
        description="Maßnahmen von Städten, Gemeinden, Kreisen.",
        examples=[
            "kommunales Klimaschutzkonzept",
            "kommunaler Wärmeplan",
            "Stadtwerke-Förderprogramm",
        ],
    ),
}


# -------------------------------------------------------------------
# TARGET SECTOR
# -------------------------------------------------------------------

TARGET_SECTORS: Dict[TargetSectorId, TaxonomyEntry] = {
    "private_households": TaxonomyEntry(
        id="private_households",
        label_de="Private Haushalte",
        label_en="Private households",
        description="Maßnahmen, die sich an private Haushalte/Bürger:innen richten.",
        examples=["Zuschuss für private Gebäudesanierung"],
    ),
    "enterprises": TaxonomyEntry(
        id="enterprises",
        label_de="Unternehmen / Gewerbe",
        label_en="Enterprises",
        description="Maßnahmen für Unternehmen, Gewerbe und Industrie.",
        examples=["Energieeffizienzprogramm für KMU"],
    ),
    "public_admin": TaxonomyEntry(
        id="public_admin",
        label_de="Öffentliche Verwaltung",
        label_en="Public administration",
        description="Maßnahmen, die interne Verwaltung oder öffentliche Gebäude betreffen.",
        examples=["Klimaneutrale Stadtverwaltung bis 2040"],
    ),
    "ngos_civil_society": TaxonomyEntry(
        id="ngos_civil_society",
        label_de="Zivilgesellschaft / NGOs",
        label_en="NGOs / civil society",
        description="Programme für Vereine, Initiativen, Zivilgesellschaft.",
        examples=["Fördertopf für Umweltvereine"],
    ),
    "energy_suppliers": TaxonomyEntry(
        id="energy_suppliers",
        label_de="Energieversorger",
        label_en="Energy suppliers",
        description="Maßnahmen, die sich direkt an Energieversorger oder Netzbetreiber richten.",
        examples=["Vorgaben für Stadtwerke"],
    ),
    "agriculture": TaxonomyEntry(
        id="agriculture",
        label_de="Landwirtschaft",
        label_en="Agriculture",
        description="Maßnahmen für landwirtschaftliche Betriebe.",
        examples=["Förderung für humusaufbauende Bewirtschaftung"],
    ),
    "transport_users": TaxonomyEntry(
        id="transport_users",
        label_de="Verkehrsteilnehmer",
        label_en="Transport users",
        description="Maßnahmen für Nutzer:innen von Verkehrssystemen.",
        examples=["ÖPNV-Rabatte", "Jobticket-Programme"],
    ),
    "other": TaxonomyEntry(
        id="other",
        label_de="Sonstige",
        label_en="Other / unspecified",
        description="Sonstige Zielgruppen oder nicht eindeutig zuordenbare Maßnahmen.",
        examples=[],
    ),
}


# -------------------------------------------------------------------
# CLIMATE DIMENSION
# -------------------------------------------------------------------

CLIMATE_DIMENSIONS: Dict[ClimateDimensionId, TaxonomyEntry] = {
    "mitigation": TaxonomyEntry(
        id="mitigation",
        label_de="Minderung (Mitigation)",
        label_en="Mitigation",
        description="Maßnahmen zur Reduktion von Treibhausgasemissionen.",
        examples=["Energieeffizienz", "Erneuerbare Energien"],
    ),
    "adaptation": TaxonomyEntry(
        id="adaptation",
        label_de="Anpassung (Adaptation)",
        label_en="Adaptation",
        description="Maßnahmen zur Anpassung an Klimafolgen.",
        examples=["Hitzeaktionsplan", "Starkregenvorsorge"],
    ),
    "cross_cutting": TaxonomyEntry(
        id="cross_cutting",
        label_de="Querschnitt (Mitigation & Adaptation)",
        label_en="Cross-cutting",
        description="Maßnahmen mit sowohl Minderung- als auch Anpassungsbezug.",
        examples=["integriertes Klima- und Anpassungskonzept"],
    ),
    "unspecified": TaxonomyEntry(
        id="unspecified",
        label_de="Nicht spezifiziert",
        label_en="Unspecified",
        description="Klimabezug vorhanden, aber keine klare Zuordnung möglich.",
        examples=[],
    ),
}


# -------------------------------------------------------------------
# GOVERNANCE TYPE
# -------------------------------------------------------------------

GOVERNANCE_TYPES: Dict[GovernanceTypeId, TaxonomyEntry] = {
    "legally_binding": TaxonomyEntry(
        id="legally_binding",
        label_de="Rechtlich verbindlich",
        label_en="Legally binding",
        description="Gesetze, Satzungen, formale Rechtsakte mit Bindungswirkung.",
        examples=["kommunale Satzung", "Landesgesetz"],
    ),
    "political_commitment": TaxonomyEntry(
        id="political_commitment",
        label_de="Politische Verpflichtung",
        label_en="Political commitment",
        description="Beschlüsse, Zielsetzungen ohne volle Rechtsverbindlichkeit.",
        examples=["Ratsbeschluss zur Klimaneutralität"],
    ),
    "administrative_directive": TaxonomyEntry(
        id="administrative_directive",
        label_de="Verwaltungsanweisung",
        label_en="Administrative directive",
        description="Interne Vorgaben der Verwaltung, Richtlinien, Dienstanweisungen.",
        examples=["Dienstanweisung zur Beschaffung klimafreundlicher Produkte"],
    ),
    "voluntary": TaxonomyEntry(
        id="voluntary",
        label_de="Freiwillige Maßnahme",
        label_en="Voluntary measure",
        description="Freiwillige Initiativen ohne formale Bindung.",
        examples=["freiwilliger Klimapakt mit Unternehmen"],
    ),
    "incentive_based": TaxonomyEntry(
        id="incentive_based",
        label_de="Anreizbasiert",
        label_en="Incentive-based",
        description="Maßnahmen, die primär über Anreize (Förderung, Preise) wirken.",
        examples=["Zuschussprogramm für Wärmepumpen"],
    ),
    "unspecified": TaxonomyEntry(
        id="unspecified",
        label_de="Nicht spezifiziert",
        label_en="Unspecified",
        description="Governance-Charakter nicht erkennbar.",
        examples=[],
    ),
}


# -------------------------------------------------------------------
# Zusammenfassung als strukturierter Output-Typ
# (z.B. als Zielschema für LLM-Labeling)
# -------------------------------------------------------------------

class PolicyLabels(TypedDict, total=False):
    policy_area: PolicyAreaId
    instrument_type: InstrumentTypeId
    jurisdiction_level: JurisdictionLevelId
    target_sectors: List[TargetSectorId]
    climate_dimension: ClimateDimensionId
    governance_type: GovernanceTypeId

    funding_available: bool
    funding_amount: Optional[str]
    funding_admin_entity: Optional[str]
    funding_eligibility: Optional[str]
    funding_target_group: Optional[str]
    funding_deadline: Optional[str]
    funding_procedure: Optional[str]


# Kleine Helfer für Validierung, z.B. in eval-measures oder LLM-Postprocessing

def is_valid_policy_area(x: Optional[str]) -> bool:
    return x in POLICY_AREAS  # type: ignore[return-value]


def is_valid_instrument_type(x: Optional[str]) -> bool:
    return x in INSTRUMENT_TYPES  # type: ignore[return-value]


def is_valid_target_sector(x: Optional[str]) -> bool:
    return x in TARGET_SECTORS  # type: ignore[return-value]


def is_valid_climate_dimension(x: Optional[str]) -> bool:
    return x in CLIMATE_DIMENSIONS  # type: ignore[return-value]


def is_valid_governance_type(x: Optional[str]) -> bool:
    return x in GOVERNANCE_TYPES  # type: ignore[return-value]
