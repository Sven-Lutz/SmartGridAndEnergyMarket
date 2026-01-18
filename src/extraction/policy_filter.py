# src/extraction/policy_filter.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Optional


# ------------------------------------------------------------
# Konfigurationsstrukturen
# ------------------------------------------------------------

@dataclass
class PolicyFilterConfig:
    strong_terms: Sequence[str]
    supporting_terms: Sequence[str]
    min_strong_hits: int = 2
    min_total_hits: int = 3
    min_length: int = 500  # minimale Textlänge
    negative_title_terms: Sequence[str] = ()
    negative_url_fragments: Sequence[str] = ()

    # NEU: spezifische Förder-Signale
    funding_terms: Sequence[str] = ()
    require_funding_term: bool = False


    # neue, feinere Kategorien
    measure_terms: Sequence[str] = ()
    admin_terms: Sequence[str] = ()
    framework_terms: Sequence[str] = ()

    navigation_title_terms: Sequence[str] = ()
    imprint_title_terms: Sequence[str] = ()
    news_title_terms: Sequence[str] = ()

    navigation_url_fragments: Sequence[str] = ()
    imprint_url_fragments: Sequence[str] = ()
    news_url_fragments: Sequence[str] = ()


@dataclass
class DocumentClassification:
    """
    Ergebnis der Dokumentklassifikation.

    doc_type:
      - policy_measure
      - policy_framework
      - policy_admin
      - info_navigation
      - legal_imprint
      - news_article
      - noise
    """
    doc_type: str
    is_policy_relevant: bool
    reason: str

    strong_hits: int
    supporting_hits: int
    measure_hits: int
    admin_hits: int
    framework_hits: int


# ------------------------------------------------------------
# Default-Konfiguration (Heuristiken nah am Antrag)
# ------------------------------------------------------------

DEFAULT_CONFIG = PolicyFilterConfig(
    strong_terms=[
        "klima",
        "klimaschutz",
        "klimaneutral",
        "klimaneutralität",
        "klimaplan",
        "klimaanpassung",
        "treibhausgas",
        "co2",
        "wärmeplan",
        "wärmeplanung",
    ],
    supporting_terms=[
        "energie",
        "heizung",
        "wärme",
        "solar",
        "photovoltaik",
        "pv-",
        "fernwärme",
        "dämmung",
        "sanierung",
        "emission",
        "förderprogramm",
        "förderung",
        "fördermittel",
        "zuschuss",
        "zuwendung",
        "bezuschusst",
        "nachhaltig",
    ],
    negative_title_terms=[
        "datenschutzerklärung",
        "datenschutz",
        "impressum",
        "barrierefreiheit",
        "erklärung zur barrierefreiheit",
        "leichte sprache",
        "gebärdensprache",
        "hilfe",
        "onlineantrag",
        "meine onlineanträge",
        "kontakt",
        "karriere",
        "stellenportal",
        "jobs",
        "newsletter",
        "wetter",
        "stadtplan",
        "notruf",
    ],
    negative_url_fragments=[
        "facebook.com/",
        "youtube.com/",
        "instagram.com/",
    ],

    # Förder-Signale: behalten wir, aber erzwingen sie NICHT mehr hart
    funding_terms=[
        "förderprogramm",
        "förderprogramme",
        "förderung",
        "förderungen",
        "fördermittel",
        "zuschuss",
        "zuschüsse",
        "zuwendung",
        "zuwendungen",
        "bezuschusst",
        "bezuschussung",
        "förderantrag",
        "antragstellung",
        "richtlinie",
        "förderrichtlinie",
    ],
    require_funding_term=False,  # WICHTIG: erstmal wieder auf False setzen

    # Navigation / Meta-Seiten (Titel)
    navigation_title_terms=[
        "startseite",
        "stadt & rathaus",
        "stadt und rathaus",
        "stadtverwaltung",
        "ämter und dienststellen",
        "terminvereinbarung",
        "virtuelles amt",
        "verwaltung und stadtpolitik",
        "stadtfinanzen",
        "gemeinderat",
        "bürgermeisterinnen und bürgermeister",
        "verkehr und mobilität",
        "bekanntmachungen",
        # NEU:
        "serviceleistungen",
        "bürgerdienste",
        "veranstaltungskalender",
        "veranstaltungen",
        "eventkalender",
        "kalender",
        "team sauberes karlsruhe",
        "sperrmüll",
        "weiße ware",
        "strom für münchen",
        "strom, erdgas/gas, fernwärme, wasser",
        "günstiger ökstrom",
        "strom für ihre wohnung",

    ],

    # Navigation / Meta-Seiten (URL-Fragmente)
    navigation_url_fragments=[
        "/rathaus",
        "/stadt-rathaus",
        "/stadtverwaltung",
        "/verwaltung",
        "/aemter",
        "/aemter-und-dienststellen",
        "/stadtfinanzen",
        "/gemeinderat",
        "/verkehr",
        "/mobilitaet",
        # NEU – typisch für städtische Versorgerseiten:
        "/m-strom",
        "/strom",
        "/erdgas",
        "/fernwaerme",
        "/fernwärme",
        "/energie",
    ],


    # Presse / News
    news_title_terms=[
        "pressemitteilung",
        "presseinformation",
        "presse-info",
        "aktuell",
        "neuigkeiten",
        "nachrichten",
        "news",
    ],
    news_url_fragments=[
        "/presse",
        "/news",
        "/aktuell",
        "/aktuelles",
        "/nachrichten",
    ],

    # vorerst leer lassen – wir nutzen die Typen nur grob,
    # nicht für harte Filter
    measure_terms=[],
    admin_terms=[],
    framework_terms=[],

    imprint_title_terms=[],
    imprint_url_fragments=[],
)


# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------

def _count_hits(text_lower: str, terms: Iterable[str]) -> int:
    return sum(text_lower.count(t) for t in terms)


# ------------------------------------------------------------
# Hauptfunktion: Dokumentklassifikation
# ------------------------------------------------------------

def classify_document(
    text: str,
    title: Optional[str],
    url: Optional[str],
    cfg: PolicyFilterConfig = DEFAULT_CONFIG,
) -> DocumentClassification:
    """
    Klassifiziert ein Dokument in einen groben Typ.

    Rückgabe:
      DocumentClassification mit doc_type und is_policy_relevant.
    """
    if not text:
        return DocumentClassification(
            doc_type="noise",
            is_policy_relevant=False,
            reason="empty_text",
            strong_hits=0,
            supporting_hits=0,
            measure_hits=0,
            admin_hits=0,
            framework_hits=0,
        )

    tl = text.lower()
    title_l = (title or "").lower()
    url_l = (url or "").lower()

    # (0) sehr kurze Texte → typischerweise Navigation/Teaser/Rauschen
    if len(tl) < cfg.min_length:
        return DocumentClassification(
            doc_type="noise",
            is_policy_relevant=False,
            reason="too_short",
            strong_hits=0,
            supporting_hits=0,
            measure_hits=0,
            admin_hits=0,
            framework_hits=0,
        )

    # (1) Imprint / rechtliche Seiten (optional, über eigene Listen steuerbar)
    if any(term in title_l for term in cfg.imprint_title_terms) or any(
        frag in url_l for frag in cfg.imprint_url_fragments
    ):
        return DocumentClassification(
            doc_type="legal_imprint",
            is_policy_relevant=False,
            reason="imprint_or_dataprotection",
            strong_hits=0,
            supporting_hits=0,
            measure_hits=0,
            admin_hits=0,
            framework_hits=0,
        )

    # (2) Navigations-/Übersichtsseiten
    if any(term in title_l for term in cfg.navigation_title_terms) or any(
        frag in url_l for frag in cfg.navigation_url_fragments
    ):
        return DocumentClassification(
            doc_type="info_navigation",
            is_policy_relevant=False,
            reason="navigation_page",
            strong_hits=0,
            supporting_hits=0,
            measure_hits=0,
            admin_hits=0,
            framework_hits=0,
        )

    # (3) Presse-/News-Seiten
    if any(term in title_l for term in cfg.news_title_terms) or any(
        frag in url_l for frag in cfg.news_url_fragments
    ):
        return DocumentClassification(
            doc_type="news_article",
            is_policy_relevant=False,
            reason="news_or_press",
            strong_hits=0,
            supporting_hits=0,
            measure_hits=0,
            admin_hits=0,
            framework_hits=0,
        )

    # (4) alte Negativlisten (Datenschutz, Karriere, Social Media, ...)
    if any(bad in title_l for bad in cfg.negative_title_terms):
        return DocumentClassification(
            doc_type="noise",
            is_policy_relevant=False,
            reason="negative_title_terms",
            strong_hits=0,
            supporting_hits=0,
            measure_hits=0,
            admin_hits=0,
            framework_hits=0,
        )

    if any(bad in url_l for bad in cfg.negative_url_fragments):
        return DocumentClassification(
            doc_type="noise",
            is_policy_relevant=False,
            reason="negative_url_fragment",
            strong_hits=0,
            supporting_hits=0,
            measure_hits=0,
            admin_hits=0,
            framework_hits=0,
        )

    # (5) Klimasignal im Text
    strong_hits = _count_hits(tl, cfg.strong_terms)
    supporting_hits = _count_hits(tl, cfg.supporting_terms)
    total_hits = strong_hits + supporting_hits

    if strong_hits < cfg.min_strong_hits or total_hits < cfg.min_total_hits:
        return DocumentClassification(
            doc_type="noise",
            is_policy_relevant=False,
            reason="too_weak_climate_signal",
            strong_hits=strong_hits,
            supporting_hits=supporting_hits,
            measure_hits=0,
            admin_hits=0,
            framework_hits=0,
        )

    # (6) Förder-Signal (für unseren Förder-Fokus)
    funding_hit = any(ft in tl for ft in cfg.funding_terms) or any(
        ft in title_l for ft in cfg.funding_terms
    )

    if cfg.require_funding_term and not funding_hit:
        return DocumentClassification(
            doc_type="noise",
            is_policy_relevant=False,
            reason="no_funding_signal",
            strong_hits=strong_hits,
            supporting_hits=supporting_hits,
            measure_hits=0,
            admin_hits=0,
            framework_hits=0,
        )

    # (7) Policy-Untertypen (aktuell optional, da measure/admin/framework_terms leer sein können)
    measure_hits = _count_hits(tl, cfg.measure_terms) if cfg.measure_terms else 0
    admin_hits = _count_hits(tl, cfg.admin_terms) if cfg.admin_terms else 0
    framework_hits = _count_hits(tl, cfg.framework_terms) if cfg.framework_terms else 0

    title_has_measure = any(t in title_l for t in cfg.measure_terms)
    title_has_admin = any(t in title_l for t in cfg.admin_terms)
    title_has_framework = any(t in title_l for t in cfg.framework_terms)

    measure_score = measure_hits + (2 if title_has_measure else 0)
    admin_score = admin_hits + (2 if title_has_admin else 0)
    framework_score = framework_hits + (2 if title_has_framework else 0)

    # Falls wir (noch) keine spezifischen Terms gesetzt haben:
    if measure_score == 0 and admin_score == 0 and framework_score == 0:
        return DocumentClassification(
            doc_type="policy_measure",
            is_policy_relevant=True,
            reason="climate_plus_funding_signal",
            strong_hits=strong_hits,
            supporting_hits=supporting_hits,
            measure_hits=measure_hits,
            admin_hits=admin_hits,
            framework_hits=framework_hits,
        )

    # 1) Klare Förder-/Programmseite
    if measure_score >= max(admin_score, framework_score, 1):
        return DocumentClassification(
            doc_type="policy_measure",
            is_policy_relevant=True,
            reason="measure_terms_dominate",
            strong_hits=strong_hits,
            supporting_hits=supporting_hits,
            measure_hits=measure_hits,
            admin_hits=admin_hits,
            framework_hits=framework_hits,
        )

    # 2) Verwaltungs-/Satzungsseite
    if admin_score >= max(measure_score, framework_score, 1):
        return DocumentClassification(
            doc_type="policy_admin",
            is_policy_relevant=True,
            reason="admin_terms_dominate",
            strong_hits=strong_hits,
            supporting_hits=supporting_hits,
            measure_hits=measure_hits,
            admin_hits=admin_hits,
            framework_hits=framework_hits,
        )

    # 3) Konzept-/Strategieseite
    if framework_score > 0:
        return DocumentClassification(
            doc_type="policy_framework",
            is_policy_relevant=True,
            reason="framework_terms_present",
            strong_hits=strong_hits,
            supporting_hits=supporting_hits,
            measure_hits=measure_hits,
            admin_hits=admin_hits,
            framework_hits=framework_hits,
        )

    # 4) Fallback
    return DocumentClassification(
        doc_type="noise",
        is_policy_relevant=False,
        reason="climate_but_no_policy_signal",
        strong_hits=strong_hits,
        supporting_hits=supporting_hits,
        measure_hits=measure_hits,
        admin_hits=admin_hits,
        framework_hits=framework_hits,
    )

# ------------------------------------------------------------
# Abwärtskompatibler Wrapper (wird in alter Pipeline genutzt)
# ------------------------------------------------------------

def is_policy_relevant(
    text: str,
    title: Optional[str],
    url: Optional[str],
    cfg: PolicyFilterConfig = DEFAULT_CONFIG,
) -> bool:
    """
    Abwärtskompatibler Wrapper, der nur zurückgibt, ob das Dokument
    für die Policy-Pipeline relevant ist.

    Ein Dokument ist "policy-relevant", wenn es zu einer der zentralen
    Klassen gehört:

      - policy_measure
      - policy_framework
      - policy_admin
    """
    classification = classify_document(text=text, title=title, url=url, cfg=cfg)
    return classification.is_policy_relevant

