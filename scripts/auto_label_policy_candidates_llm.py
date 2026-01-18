#!/usr/bin/env python
import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Projekt-Root bestimmen (analog zu anderen Scripts)
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# .env explizit aus dem Projekt-Root laden
load_dotenv(PROJECT_ROOT / ".env")

from openai import OpenAI

logger = logging.getLogger(__name__)

# Die Label-Spalten, die vom LLM gefüllt werden sollen
LABEL_COLUMNS: List[str] = [
    "is_policy_relevant",
    "policy_scope",
    "instrument_type",
    "sector",
    "target_group",
    "implementation_status",
    "policy_strength",
    "digitalization_level",
    "governance_level_reference",
    "is_electromobility_related",
    "doc_type",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Labelt policy_candidates_for_labeling.csv automatisiert mit einem LLM "
            "und schreibt policy_candidates_labeled.csv."
        )
    )
    parser.add_argument(
        "--scope",
        required=True,
        help="Scope (z.B. 'de_by'). Erwartet Daten unter data/<scope>/labeling/.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Optionaler Daten-Root. Default: PROJECT_ROOT / 'data'.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Optional: expliziter Pfad zur policy_candidates_for_labeling.csv. "
            "Default: <data_root>/<scope>/labeling/policy_candidates_for_labeling.csv"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional: expliziter Pfad zur policy_candidates_labeled.csv. "
            "Default: <data_root>/<scope>/labeling/policy_candidates_labeled.csv"
        ),
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI-Modell für das Labeling (Default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help=(
            "Optional: maximal zu labelnde Zeilen (0 = alle). "
            "Hilfreich zum Testen."
        ),
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.3,
        help="Schlafzeit (Sekunden) zwischen API-Calls, um Rate Limits zu schonen.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose Logging (DEBUG-Level).",
    )
    return parser.parse_args()


def build_prompt(row: Dict[str, str]) -> str:
    """
    Baut den Prompt für eine Zeile/ ein Dokument.
    Erwartet mind. Felder: municipality, document_id, url, title, text.
    """
    municipality = row.get("municipality", "")
    document_id = row.get("document_id", "")
    url = row.get("url", "")
    title = row.get("title", "")
    text = row.get("text", "")

    # Hinweis: text wurde bereits im Export-Skript auf max_text_len gekürzt.
    # Hier verwenden wir ihn einfach komplett.
    prompt = f"""
Du bist ein Experte für kommunale Klimapolitik und annotierst Dokumente deutscher Städte.

Du bekommst ein einzelnes Dokument (Website oder PDF-Textauszug) mit Metadaten
und sollst folgende Labels gemäß einem festen Schema vergeben.

ARBEITE WIRKLICH STRIKT NACH DEN DEFINITIONEN UND GIB NUR DIE ZULÄSSIGEN LABELWERTE AUS.

LABEL-SCHEMA (alle Werte als Strings):

1. is_policy_relevant
   - Bedeutet: Ist das Dokument inhaltlich relevant für kommunale Klima-/Energie-/
     Nachhaltigkeitspolitik, also Strategien, Konzepte, Programme, Maßnahmen,
     Förderungen, Planung, Berichte o.ä.?
   - Zulässige Werte:
     - "yes"       : eindeutig policy-relevant
     - "no"        : klar nicht policy-relevant
     - "uncertain" : unklar / gemischt / nicht eindeutig

2. policy_scope
   - Einordnung der Reichweite des Inhalts.
   - Zulässige Werte:
     - "city_strategy"        : gesamtstädtische Strategien/Konzepte/Pläne
     - "city_sectoral_plan"   : sektorale Pläne/Strategien (z.B. Verkehr, Wärme, Anpassung)
     - "project_or_measure"   : konkrete Projekte, Maßnahmen, Programme
     - "advisory_or_information" : reine Informations- oder Beratungsangebote
     - "other"
     - "unknown"

3. instrument_type
   - Typ des klima-/energiepolitischen Instruments.
   - Zulässige Werte:
     - "regulation"                : verbindliche Regeln, Satzungen, Vorgaben
     - "strategic_plan"            : Strategien, Konzepte, Aktionspläne
     - "subsidy_or_funding"        : Förderprogramme, Zuschüsse, finanzielle Anreize
     - "information_or_campaign"   : Informationsangebote, Kampagnen, Beratung
     - "infrastructure_investment" : Investitionen in Infrastruktur
     - "agreement_or_partnership"  : freiwillige Vereinbarungen, Netzwerke, Partnerschaften
     - "administrative_procedure"  : interne Verwaltungsprozesse, Berichtspflichten, Organisation
     - "other"
     - "unknown"

4. sector
   - Dominanter inhaltlicher Sektorbezug.
   - Zulässige Werte:
     - "cross_sector"
     - "buildings"
     - "energy"
     - "transport"
     - "land_use"
     - "waste"
     - "industry"
     - "adaptation"
     - "other"
     - "unknown"

5. target_group
   - Hauptadressaten des Dokuments.
   - Zulässige Werte:
     - "municipal_admin"           : interne Verwaltung, Stadtrat, Gremien
     - "citizens"                  : allgemeine Bevölkerung / Haushalte
     - "businesses"                : Unternehmen, Gewerbe, Industrie
     - "civil_society"             : Vereine, Initiativen, NGOs
     - "other_public_authorities"  : andere Behörden/Verwaltungen
     - "multiple"                  : mehrere dieser Gruppen
     - "unknown"

6. implementation_status
   - Stand der Umsetzung.
   - Zulässige Werte:
     - "idea_or_vision"
     - "in_planning"
     - "adopted"
     - "under_implementation"
     - "completed"
     - "unknown"

7. policy_strength
   - Einschätzung der Verbindlichkeit / Intensität der Maßnahmen.
   - Zulässige Werte:
     - "weak"    : hauptsächlich weiche Maßnahmen (Information, Beratung, Appelle)
     - "medium"  : Mischung aus weicheren und verbindlicheren Elementen
     - "strong"  : klare, verbindliche, ambitionierte Maßnahmen, große Investitionen
     - "unknown"

8. digitalization_level
   - Grad der Digitalisierung des beschriebenen Prozesses/Angebots.
   - Zulässige Werte:
     - "offline_or_analog"       : nur analoge Verfahren, kein Online-Angebot
     - "info_only"               : reine Informationsseite ohne Formularlogik
     - "downloadable_pdf_form"   : PDF-Formulare zum Ausdrucken/Zurücksenden
     - "online_form_or_portal"   : Online-Formular oder eService-Portal
     - "mixed"                   : Kombination mehrerer der obigen Formen
     - "unknown"

9. governance_level_reference
   - Auf welche politische Ebene(n) beziehen sich Programme/Maßnahmen?
   - Zulässige Werte:
     - "municipal_only"
     - "state_level"      : Bundesland
     - "federal_level"    : Bund
     - "eu_level"         : EU
     - "multiple"
     - "unknown"

10. is_electromobility_related
    - Bezieht sich das Dokument (wesentlich) auf Elektromobilität?
    - Zulässige Werte:
      - "yes"
      - "no"
      - "uncertain"

11. doc_type
    - Grobe Dokumentart.
    - Zulässige Werte:
      - "website_page"
      - "pdf_report_or_study"
      - "council_minutes_or_resolution"
      - "statute_or_regulation_document"
      - "application_form"
      - "flyer_or_info_brochure"
      - "other"
      - "unknown"

12. notes
    - Kurze Erläuterung (max. 400 Zeichen), warum du diese Label vergeben hast.
    - Frei formulierter Text, Deutsch oder Englisch erlaubt.

WICHTIG:
- Wenn das Dokument klar NICHT policy-relevant ist, setze is_policy_relevant="no"
  und fülle die anderen Felder möglichst konsistent (oder "unknown").
- Antworte NUR mit einem JSON-Objekt mit GENAU diesen Schlüsseln:
  {LABEL_COLUMNS}
- Keine zusätzliche Erklärung oder Fließtext außerhalb des JSON.

JETZT DAS KONKRETE DOKUMENT:

Metadaten:
- municipality: {municipality}
- document_id: {document_id}
- url: {url}
- title: {title}

Textauszug (evtl. gekürzt):
{text}
"""
    return prompt.strip()


def call_llm(client: OpenAI, model: str, prompt: str) -> Dict[str, Any]:
    """
    Ruft das Chat-Completions-API mit JSON-Mode auf und gibt das geparste JSON zurück.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert annotator for municipal climate policy "
                    "documents. Always respond ONLY with a single JSON object."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    content = resp.choices[0].message.content
    if content is None:
        # Defensive: sollte bei response_format=json_object eigentlich nicht passieren
        raise ValueError("LLM response content is None.")

    try:
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("LLM response is not a JSON object.")
        return data
    except json.JSONDecodeError as exc:
        logger.error("JSON-Parsing des LLM-Outputs fehlgeschlagen: %s", exc)
        logger.debug("Roh-Output des LLM: %r", content)
        raise


def auto_label_csv(
    input_csv: Path,
    output_csv: Path,
    model: str,
    max_rows: int = 0,
    sleep_sec: float = 0.3,
) -> None:
    """
    Liest policy_candidates_for_labeling.csv, annotiert jede Zeile via LLM
    und schreibt policy_candidates_labeled.csv.
    """
    logger.info("Lese Input-CSV: %s", input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input-CSV nicht gefunden: {input_csv}")

    client = OpenAI()

    with input_csv.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)

    total = len(rows)
    if max_rows > 0:
        total = min(total, max_rows)
        rows = rows[:total]

    logger.info("Insgesamt %d Zeilen zu labeln (max_rows=%d).", total, max_rows)

    # Alle Spalten beibehalten, plus Label-Spalten (falls noch nicht vorhanden)
    fieldnames = list(reader.fieldnames or [])
    for col in LABEL_COLUMNS:
        if col not in fieldnames:
            fieldnames.append(col)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Schreibe gelabelte Daten nach: %s", output_csv)

    with output_csv.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(rows, start=1):
            doc_id = f"{row.get('municipality', '')}::{row.get('document_id', '')}"
            logger.info("Labeling %d/%d (%s)", idx, total, doc_id)

            # Falls Label schon gesetzt (z.B. beim Resume), kannst du hier skippen:
            if row.get("is_policy_relevant", "").strip():
                logger.debug(
                    "Zeile %d (%s) bereits gelabelt, überspringe.", idx, doc_id
                )
                writer.writerow(row)
                continue

            prompt = build_prompt(row)

            # Einfache Retry-Logik
            for attempt in range(3):
                try:
                    labels = call_llm(client, model=model, prompt=prompt)
                    break
                except Exception as exc:
                    logger.warning(
                        "Fehler beim LLM-Call für %s (Versuch %d/3): %s",
                        doc_id,
                        attempt + 1,
                        exc,
                    )
                    time.sleep(1.5)
            else:
                logger.error(
                    "LLM-Call für %s nach 3 Versuchen gescheitert, "
                    "Label-Spalten bleiben leer.",
                    doc_id,
                )
                labels = {}

            # Labels in die Zeile schreiben, fehlende Keys defensiv auf 'unknown'/''
            for col in LABEL_COLUMNS:
                val = labels.get(col, None)
                if val is None:
                    if col == "notes":
                        row[col] = ""
                    elif col == "is_policy_relevant":
                        row[col] = "uncertain"
                    else:
                        row[col] = "unknown"
                else:
                    row[col] = str(val)

            writer.writerow(row)
            if sleep_sec > 0:
                time.sleep(sleep_sec)

    logger.info(
        "Auto-Labeling abgeschlossen. %d Zeilen nach %s geschrieben.",
        total,
        output_csv,
    )


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    data_root = (
        Path(args.data_root).resolve()
        if args.data_root is not None
        else (PROJECT_ROOT / "data")
    )

    scope_root = data_root / args.scope

    input_csv = (
        Path(args.input).resolve()
        if args.input is not None
        else scope_root / "labeling" / "policy_candidates_for_labeling.csv"
    )
    output_csv = (
        Path(args.output).resolve()
        if args.output is not None
        else scope_root / "labeling" / "policy_candidates_labeled.csv"
    )

    logger.info(
        "Starte Auto-Labeling mit scope='%s', input='%s', output='%s', model='%s'",
        args.scope,
        input_csv,
        output_csv,
        args.model,
    )

    auto_label_csv(
        input_csv=input_csv,
        output_csv=output_csv,
        model=args.model,
        max_rows=args.max_rows,
        sleep_sec=args.sleep,
    )


if __name__ == "__main__":
    main()
