"""
Prompt registry: stable IDs, text, versioning.
No assumptions about providers.
"""

from authinfra.generation.schema import PromptEntry

REGISTRY_VERSION = "v1"

_PROMPTS: list[PromptEntry] = [
    {"id": "1", "version": REGISTRY_VERSION, "text": "Rewrite the following CRS report in the same register and structure. Preserve section headings and order. Keep total length within ±5 percent of the source. Do not add new facts. Do not use bold, italics, or any formatting. Use lists only if the original includes lists. Output only the revised report."},
    {"id": "2", "version": REGISTRY_VERSION, "text": "Rewrite the following CRS section in the same register and structure. Preserve the heading and numbering. Keep length within ±5 percent of the source. Do not add facts. Do not use bold, italics, or any formatting. Use lists only if the original includes lists. Output only the revised section."},
    {"id": "3", "version": REGISTRY_VERSION, "text": "Paraphrase the text below while preserving tone and structure. Keep length within ±5 percent. Do not introduce new information. Do not use bold, italics, or any formatting. Use lists only if the original includes lists. Output only the text."},
    {"id": "4", "version": REGISTRY_VERSION, "text": "Re-express the report below in fresh wording, matching its structure and headings. Keep length within ±5 percent. No new facts. Do not use bold, italics, or any formatting. Use lists only if the original includes lists. Output only the report text."},
    {"id": "5", "version": REGISTRY_VERSION, "text": "Rewrite this section in the same style and order of ideas. Length within ±5 percent. No added facts. Do not use bold, italics, or any formatting. Use lists only if the original includes lists. Output only the section."},
    {"id": "6", "version": REGISTRY_VERSION, "text": "Produce a reformulation of the following content, keeping headings, order, and tone. Length ±5 percent. No new content. Do not use bold, italics, or any formatting. Use lists only if the original includes lists. Output only the text."},
    {"id": "7", "version": REGISTRY_VERSION, "text": "Restate the following CRS material. Keep headings and structure, and stay within ±5 percent length. Do not add or remove facts. Do not use bold, italics, or any formatting. Use lists only if the original includes lists. Output only the text."},
    {"id": "8", "version": REGISTRY_VERSION, "text": "Rewrite the passage below with the same structure and voice. Keep headings. Length within ±5 percent. Do not add facts. Do not use bold, italics, or any formatting. Use lists only if the original includes lists. Output only the passage."},
    {"id": "9", "version": REGISTRY_VERSION, "text": "Reframe the content in equivalent language while preserving the original sections and order. Stay within ±5 percent in length. No new details. Do not use bold, italics, or any formatting. Use lists only if the original includes lists. Output only the text."},
    {"id": "10", "version": REGISTRY_VERSION, "text": "Rewrite the text below. Keep the structure and headings identical. Keep overall length within ±5 percent. Do not introduce new facts. Do not use bold, italics, or any formatting. Use lists only if the original includes lists. Output only the text."},
]


def get_registry_version() -> str:
    return REGISTRY_VERSION


def get_prompt(prompt_id: str) -> PromptEntry | None:
    """Return prompt by id, or None if not found."""
    for p in _PROMPTS:
        if p["id"] == prompt_id:
            return p
    return None


def list_prompt_ids() -> list[str]:
    """Return all prompt IDs in registry order."""
    return [p["id"] for p in _PROMPTS]
