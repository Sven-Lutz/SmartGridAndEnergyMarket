from __future__ import annotations

from typing import Literal

DigitalizationLevel = Literal["ONLINE_FORM", "PDF_DOWNLOAD", "CONTACT_ONLY", "UNKNOWN"]


def infer_digitalization_level(
    has_form: bool,
    has_pdf_download: bool,
    has_contact_email: bool,
) -> DigitalizationLevel:
    if has_form:
        return "ONLINE_FORM"
    if has_pdf_download:
        return "PDF_DOWNLOAD"
    if has_contact_email:
        return "CONTACT_ONLY"
    return "UNKNOWN"
