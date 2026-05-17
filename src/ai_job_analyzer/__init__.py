"""AI Job Analyzer.

Agentic workflow that scrapes job postings for a given job title and uses
LLMs to extract the key tech stack and core competencies.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ai-job-analyzer")
except PackageNotFoundError:  # pragma: no cover - during local dev before install
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
