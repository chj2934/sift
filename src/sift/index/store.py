"""LanceDB-backed hybrid search over note chunks.

We run a vector query and a full-text (BM25) query separately and fuse them with
Reciprocal Rank Fusion. Doing the fusion ourselves (rather than leaning on
LanceDB's built-in ``query_type="hybrid"``) keeps behaviour stable across
LanceDB versions and makes it unit-testable without a GPU or network.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path

import lancedb
import pyarrow as pa

from sift.config import get_settings
from sift.index import embed as _embed

TABLE = "notes"
RRF_K = 60


def _schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),  # {note_id}::{chunk_index}
            pa.field("note_id", pa.string()),
            pa.field("slug", pa.string()),
            pa.field("type", pa.string()),
            pa.field("title", pa.string()),
            pa.field("heading", pa.string()),
            pa.field("text", pa.string()),  # excerpt shown to the user
            pa.field("search_text", pa.string()),  # title + heading + text, for BM25
            pa.field("chunk_index", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("source", pa.string()),
            pa.field("url", pa.string()),
            pa.field("cwe_str", pa.string()),  # space-joined, for LIKE filters
            pa.field("tags_str", pa.string()),
            pa.field("severity", pa.string()),
            pa.field("program", pa.string()),
            pa.field("path", pa.string()),
            pa.field("mtime", pa.float64()),
        ]
    )


@dataclass
class ChunkRow:
    note_id: str
    slug: str
    type: str
    title: str
    heading: str
    text: str
    chunk_index: int
    vector: list[float]
    source: str = ""
    url: str = ""
    cwe: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    severity: str = ""
    program: str = ""
    path: str = ""
    mtime: float = 0.0

    def to_record(self) -> dict:
        parts = [self.title, self.heading, self.text, " ".join(self.tags), " ".join(self.cwe)]
        return {
            "id": f"{self.note_id}::{self.chunk_index}",
            "note_id": self.note_id,
            "slug": self.slug,
            "type": self.type,
            "title": self.title,
            "heading": self.heading,
            "text": self.text,
            "search_text": "\n".join(p for p in parts if p),
            "chunk_index": self.chunk_index,
            "vector": self.vector,
            "source": self.source,
            "url": self.url,
            "cwe_str": " ".join(self.cwe),
            "tags_str": " ".join(self.tags),
            "severity": self.severity or "",
            "program": self.program or "",
            "path": self.path,
            "mtime": self.mtime,
        }


@dataclass
class Hit:
    note_id: str
    slug: str
    type: str
    title: str
    url: str
    source: str
    severity: str
    program: str
    path: str
    score: float
    excerpt: str
    heading: str
    matched_chunks: int


class Store:
    def __init__(self, db_path: Path | None = None, dim: int | None = None):
        s = get_settings()
        self.db_path = Path(db_path) if db_path else s.resolved_db()
        self.dim = dim or s.effective_embed_dim()
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self.db_path))

    # ---- lifecycle ----------------------------------------------------
    def _table_names(self) -> list[str]:
        try:
            return list(self._db.list_tables().tables)  # lancedb >= 0.30
        except AttributeError:
            return list(self._db.table_names())

    def table(self):
        if TABLE in self._table_names():
            tbl = self._db.open_table(TABLE)
            self._check_dim(tbl)
            return tbl
        return self._db.create_table(TABLE, schema=_schema(self.dim), exist_ok=True)

    def _check_dim(self, tbl) -> None:
        try:
            field = tbl.schema.field("vector")
            existing = field.type.list_size
        except Exception:  # noqa: BLE001
            return
        if existing and existing != self.dim:
            raise RuntimeError(
                f"index was built with {existing}-dim vectors but the current model "
                f"produces {self.dim}-dim. Run `sift reindex --force` after changing SIFT_EMBED_MODEL."
            )

    def drop(self) -> None:
        if TABLE in self._table_names():
            self._db.drop_table(TABLE)

    def ensure_fts(self) -> None:
        tbl = self.table()
        try:
            from lancedb.index import FTS

            tbl.create_index("search_text", config=FTS(), replace=True)
        except (TypeError, ImportError):
            tbl.create_fts_index("search_text", replace=True)  # older lancedb
        except Exception as exc:  # noqa: BLE001
            print(f"  ! FTS index build failed ({exc}); keyword search degraded")

    def count(self) -> int:
        try:
            return self.table().count_rows()
        except Exception:  # noqa: BLE001
            return 0

    # ---- writes -----------------------------------------------------------
    def delete_note(self, note_id: str) -> None:
        safe = note_id.replace("'", "''")
        with contextlib.suppress(Exception):
            self.table().delete(f"note_id = '{safe}'")

    def add_chunks(self, rows: list[ChunkRow]) -> None:
        if not rows:
            return
        self.table().add([r.to_record() for r in rows])

    def upsert_note(self, rows: list[ChunkRow]) -> None:
        if not rows:
            return
        self.delete_note(rows[0].note_id)
        self.add_chunks(rows)

    # ---- search ---------------------------------------------------------
    @staticmethod
    def _where(filters: dict | None) -> str | None:
        if not filters:
            return None
        clauses: list[str] = []
        if t := filters.get("type"):
            clauses.append(f"type = '{str(t).replace(chr(39), '')}'")
        if p := filters.get("program"):
            clauses.append(f"program = '{str(p).replace(chr(39), '')}'")
        if c := filters.get("cwe"):
            c = str(c).upper().replace("'", "")
            if not c.startswith("CWE-") and c.isdigit():
                c = f"CWE-{c}"
            clauses.append(f"cwe_str LIKE '%{c}%'")
        if sev := filters.get("severity"):
            clauses.append(f"severity = '{str(sev).lower().replace(chr(39), '')}'")
        return " AND ".join(clauses) if clauses else None

    def _run(self, make_query, k: int, where: str | None, *, label: str) -> list[dict]:
        last: Exception | None = None
        for attempt in ("prefilter", "postfilter", "nofilter"):
            try:
                q = make_query().limit(k)
                if where and attempt == "prefilter":
                    q = q.where(where, prefilter=True)
                elif where and attempt == "postfilter":
                    q = q.where(where)
                return q.to_list()
            except Exception as exc:  # noqa: BLE001
                last = exc
        print(f"  ! {label} search failed: {last}")
        return []

    def _vector_hits(self, vec: list[float], k: int, where: str | None) -> list[dict]:
        return self._run(lambda: self.table().search(vec), k, where, label="vector")

    def _fts_hits(self, query: str, k: int, where: str | None) -> list[dict]:
        return self._run(
            lambda: self.table().search(query, query_type="fts", fts_columns="search_text"),
            k,
            where,
            label="fts",
        )

    def search(
        self,
        query: str,
        *,
        k: int = 8,
        filters: dict | None = None,
        pool: int = 40,
    ) -> list[Hit]:
        where = self._where(filters)
        vec = _embed.get_embedder().embed_query(query)
        vhits = self._vector_hits(vec, pool, where)
        fhits = self._fts_hits(query, pool, where)

        # Reciprocal Rank Fusion over chunk ids.
        scores: dict[str, float] = {}
        chunk_by_id: dict[str, dict] = {}
        for hits in (vhits, fhits):
            for rank, row in enumerate(hits):
                cid = row["id"]
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
                chunk_by_id.setdefault(cid, row)

        # Collapse chunks -> notes.
        notes: dict[str, Hit] = {}
        for cid, sc in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
            row = chunk_by_id[cid]
            nid = row["note_id"]
            if nid in notes:
                notes[nid].matched_chunks += 1
                notes[nid].score += sc * 0.25  # diminishing credit for extra chunks
                continue
            notes[nid] = Hit(
                note_id=nid,
                slug=row.get("slug", ""),
                type=row.get("type", ""),
                title=row.get("title", ""),
                url=row.get("url", "") or "",
                source=row.get("source", "") or "",
                severity=row.get("severity", "") or "",
                program=row.get("program", "") or "",
                path=row.get("path", "") or "",
                score=sc,
                excerpt=(row.get("text", "") or "")[:600],
                heading=row.get("heading", "") or "",
                matched_chunks=1,
            )

        ranked = sorted(notes.values(), key=lambda h: h.score, reverse=True)
        return ranked[:k]
