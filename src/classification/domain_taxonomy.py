# src/classification/domain_taxonomy.py
from __future__ import annotations

"""
Zentrale (vereinfachte) Taxonomie-Definition, nah am Antrag.

Die eigentlichen Regeln stecken in baseline_rules.py.
Hier halten wir nur die zulässigen Kategorien gebündelt.
"""

# grobe Policy-Areas (Sektoren / Politikfelder)
POLICY_AREAS: list[str] = [
    "cross_sectoral",      # integrierte Konzepte / Strategien
    "buildings",
    "energy_supply",
    "mobility",
    "industry",
    "waste",
    "land_use",
    "adaptation",
    "other",
]

# Instrument-Typen
INSTRUMENT_TYPES: list[str] = [
    "strategy_plan",           # Konzepte, Pläne, Leitbilder
    "regulation_standard",     # Satzungen, rechtliche Vorgaben
    "investment_infrastructure",
    "subsidy_funding_program", # Förderprogramme, Zuschüsse
    "information_advice",      # Beratung, Kampagnen
    "public_procurement",
    "other",
]

# Zielsektoren (Liste im Datensatz)
TARGET_SECTORS: list[str] = [
    "municipal_buildings",
    "private_buildings",
    "transport",
    "industry",
    "households",
    "cross_sectoral",
    "other",
]

# Klimadimension
CLIMATE_DIMENSIONS: list[str] = [
    "mitigation",
    "adaptation",
    "both",
    "unspecified",
]

# Governance-Typ (für Paper-Analyse)
GOVERNANCE_TYPES: list[str] = [
    "local_municipality",
    "regional_state",
    "national",
    "eu",
    "multi_level",
]
