"""Load causal_graph.json once and build the indexes every stage needs."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GRAPH = REPO_ROOT / "causal_graph.json"


class Graph:
    def __init__(self, path: Path = DEFAULT_GRAPH):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self.raw = raw
        self.nodes: dict[str, dict] = {n["id"]: n for n in raw["nodes"]}
        self.edges: dict[str, dict] = {e["id"]: e for e in raw["edges"]}
        self.docs: dict[str, dict] = {d["document_id"]: d for d in raw["source_documents"]}

        self.edges_by_subject: dict[str, list[str]] = defaultdict(list)
        self.edges_by_object: dict[str, list[str]] = defaultdict(list)
        for e in raw["edges"]:
            self.edges_by_subject[e["subject"]].append(e["id"])
            self.edges_by_object[e["object"]].append(e["id"])

        # IUCN-style ecosystem labels actually present on edges, e.g.
        # "temperate_subhumid_grasslands". Read from the data, not hardcoded,
        # so a different graph brings its own vocabulary.
        self.contexts: set[str] = {
            e["ecosystem_context"] for e in raw["edges"] if e.get("ecosystem_context")
        }

    def node_text(self, node_id: str) -> str:
        """Searchable text for a node. `embedding_text` already joins name,
        entity_term, measured_attribute, qualifier and applied_to labels."""
        n = self.nodes[node_id]
        return n.get("embedding_text") or " ".join(
            str(n.get(k) or "") for k in ("name", "entity_term", "measured_attribute")
        )
