"""Link-graph expansion - the Obsidian-like part.

Given a set of high-scoring notes, pull in their 1-hop ``[[wikilink]]`` /
``links:`` neighbours so the model sees connected context (e.g. a report links
the technique it used and the CVE it resembles).

The graph is one slim record per note file (`LinkRec`: identity, title, path and
outgoing links, never the body), derived from the vault catalog's rows and rebuilt
only when the catalog changed. The catalog holds exactly the files iter_notes reads:
a note the graph skipped could never be expanded to, and one it held that iter_notes
skips (a template, a trashed copy) would be expanded to but could never be fetched.
`build_link_index` explains the cache; `_record` lists the keys a note can be looked
up by.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slugify import slugify

from sift.vault.notes import walk_note_entries

log = logging.getLogger(__name__)

# --- keys -------------------------------------------------------------------------

# Note slugs carry a source prefix from their id ("tech-", "research-", "top10-"),
# but a `[[wikilink]]` is written from the human-readable title and almost never
# includes it. Every cross-link in the first batch of technique notes was broken this
# way - 0 of 5 resolved - so the link graph returned nothing. Register each note under
# its prefix-stripped slug too, without ever shadowing a real one.
_ID_PREFIXES = (
    "tech",
    "research",
    "top10",
    "writeup",
    "idea",
    "repo",
    "cve",
    "targ",
    "find",
    "writ",
)


def _aliases(slug: str) -> list[str]:
    for p in _ID_PREFIXES:
        if slug.startswith(p + "-"):
            return [slug.removeprefix(p + "-")]
    return []


# Slugs were cut to 80 characters, so distinct notes shared one (16 real pairs). Every
# index row written before the next `reindex --force`, and every link copied from an
# old remember()/capture_idea() result, still carries that cut form, so it stays a key.
_LEGACY_SLUG_LEN = 80


def _cap(slug: str) -> str:
    """``slugify(text, max_length=80)`` given ``slugify(text)``, without slugifying twice.

    python-slugify truncates last, as ``text[:max_length].strip("-")``;
    tests/test_graph_resolution.py pins the equivalence.
    """
    return slug[:_LEGACY_SLUG_LEN].strip("-")


# The keys a note answers to, strongest first. A key resolves to whichever note claims
# it at the strongest tier - so an alias never shadows a real slug and an exact id
# beats everything - and notes tied at that tier are all kept (LinkIndex.ambiguous).
_TIER_ID = 0  # meta.id verbatim: what a search hit carries as note_id
_TIER_SLUG = 1  # Note.slug: what expand() emits and get_note takes
_TIER_LEGACY = 2  # the old 80-char slug, and the id slug at full length
# The filename outranks the title because that is how Obsidian resolves [[Title]]:
# a second note with the same title gets `Title (2).md`, which sorts first (' ' < '.'),
# so with one shared tier the "(2)" sibling took every [[Title]] link.
_TIER_STEM = 3
_TIER_TITLE = 4
_TIER_STRIPPED = 5  # a slug without its source prefix, see _aliases


@dataclass(frozen=True, slots=True)
class LinkRec:
    """One note's slice of the graph: identity, title, path and outgoing links.

    No body. The graph used to hold every parsed Note - 37M characters, ~129 MB on the
    real vault - only to answer link lookups.
    """

    note_id: str
    slug: str
    path: Path
    title: str
    type: str
    url: str
    links: tuple[str, ...]  # Note.all_links(): slugs from links: and [[wikilinks]]
    lookup_keys: tuple[tuple[int, str], ...] = ()  # (tier, key), strongest first

    @property
    def id(self) -> str:
        return self.note_id

    @property
    def meta(self) -> LinkRec:
        """Read like a Note: ``rec.meta.title`` / ``.type`` / ``.url`` / ``.id``."""
        return self

    @property
    def aliases(self) -> tuple[str, ...]:
        """Every lookup key except the id and the slug."""
        return tuple(key for tier, key in self.lookup_keys if tier > _TIER_SLUG)

    def all_links(self) -> list[str]:
        return list(self.links)


def _record(row: Any) -> LinkRec:
    """The graph's record of one vault catalog row (`sift.vault.catalog.CatalogRow`:
    id, slug, path, title, type, url and links, the outgoing links as slugs)."""
    note_id = row.id
    title = row.title or ""
    path = Path(row.path)
    slug = row.slug
    id_slug = slugify(note_id)
    title_slug = slugify(title)
    # The filename is the title for nearly every note; slugify is ~25 us a call.
    stem_slug = title_slug if path.stem == title else slugify(path.stem)
    # Note.slug before slugs went injective, whatever form Note.slug has now.
    legacy = _cap(id_slug) or _cap(title_slug)
    ranked = [
        (_TIER_ID, note_id),
        (_TIER_SLUG, slug),
        (_TIER_LEGACY, legacy),
        (_TIER_LEGACY, id_slug),
        # Links are slugified at full length, so long filenames and titles need their
        # full-length key; the cut form stays for links written from cut slugs.
        (_TIER_STEM, stem_slug),
        (_TIER_STEM, _cap(stem_slug)),
        (_TIER_TITLE, title_slug),
        (_TIER_TITLE, _cap(title_slug)),
        *((_TIER_STRIPPED, a) for s in (slug, legacy, id_slug) for a in _aliases(s)),
    ]
    keys: dict[str, int] = {}
    for tier, key in ranked:  # strongest first, so a key keeps its best tier
        if key:
            keys.setdefault(key, tier)
    return LinkRec(
        note_id=note_id,
        slug=slug,
        path=path,
        title=title,
        type=str(row.type),
        url=row.url or "",
        links=tuple(row.links),
        lookup_keys=tuple((tier, key) for key, tier in keys.items()),
    )


class LinkIndex(dict[str, LinkRec]):
    """key -> LinkRec, plus every key more than one note claims.

    To existing callers it is a plain dict: ``idx[key]`` is the claimant first in path
    order, the note get_note's scan returns too. ``ambiguous`` maps each contested key
    to all of its claimants in path order, so nothing is silently shadowed: expand()
    follows every one, and a resolver that must not guess can refuse instead.

    Shared between callers and threads: treat it as read-only.
    """

    __slots__ = ("ambiguous",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ambiguous: dict[str, tuple[LinkRec, ...]] = {}


def lookup(link_index: Mapping[str, Any], key: str) -> tuple[Any, ...]:
    """Every note ``key`` resolves to: none, one, or all claimants of a contested key."""
    contested = getattr(link_index, "ambiguous", None)
    if contested:
        many = contested.get(key)
        if many:
            return many
    one = link_index.get(key)
    return () if one is None else (one,)


def _assemble(recs: Iterable[LinkRec]) -> LinkIndex:
    """Resolve every key to its strongest claimant(s). ``recs`` must be in path order."""
    best: dict[str, tuple[int, LinkRec]] = {}
    tied: dict[str, list[LinkRec]] = {}
    for rec in recs:
        for tier, key in rec.lookup_keys:
            held = best.get(key)
            if held is None:
                best[key] = (tier, rec)
            elif tier < held[0]:
                best[key] = (tier, rec)
                tied.pop(key, None)
            elif tier == held[0]:
                if key in tied:
                    tied[key].append(rec)
                else:
                    tied[key] = [held[1], rec]
    index = LinkIndex({key: rec for key, (_tier, rec) in best.items()})
    index.ambiguous = {key: tuple(claimants) for key, claimants in tied.items()}
    return index


# --- the vault catalog ----------------------------------------------------------------

# The graph is derived from the vault catalog (`sift.vault.catalog`): one row per note
# file with its id, slug, title, type, url and outgoing links, validated by a stat walk
# and persisted under the DB dir. The graph used to keep its own per-file parse cache
# beside it - a second walk per expanding search, and a second full parse of the vault
# (~5-9 s over 13,856 notes) in every new MCP server process, where the catalog starts
# from its cache file. The walk rules, the racy-file re-read and the once-per-file
# warning for an unreadable note are the catalog's.


def vault_fingerprint(vault: Path) -> str:
    """A stat-only digest of every note file's (path, mtime_ns, size).

    It changes on any add, delete, edit or rename, over the same files the catalog
    walks (0-byte files left out). Stable across processes. A same-size rewrite inside
    one timestamp tick is invisible to any stat signal; the catalog re-reads recently
    modified files to catch it.
    """
    h = hashlib.blake2b(digest_size=16)
    count = 0
    root = Path(vault)
    if root.is_dir():
        for path, st in walk_note_entries(root):
            if st.st_size:
                count += 1
                line = f"{path}\0{st.st_mtime_ns}\0{st.st_size}\n"
                h.update(line.encode("utf-8", "surrogatepass"))
    return f"{count}-{h.hexdigest()}"


# The old private name, for callers that compared fingerprints.
_fingerprint = vault_fingerprint


# --- the cache ----------------------------------------------------------------------


@dataclass(slots=True)
class _VaultCache:
    catalog: Any = None  # the VaultCatalog the index was built from
    generation: int = -1  # its generation then
    # id(row) -> (row, record). A catalog row is an immutable object replaced whenever
    # its file changes, so an unchanged row reuses its record (slugify is ~25 us a call).
    records: dict[int, tuple[Any, LinkRec]] = field(default_factory=dict)
    index: LinkIndex | None = None


_LOCK = threading.Lock()
_STATE: dict[str, _VaultCache] = {}


def _records(
    rows: Iterable[Any], prev: Mapping[int, tuple[Any, LinkRec]]
) -> tuple[list[LinkRec], dict[int, tuple[Any, LinkRec]]]:
    out: list[LinkRec] = []
    kept: dict[int, tuple[Any, LinkRec]] = {}
    for row in rows:
        held = prev.get(id(row))
        rec = held[1] if held is not None and held[0] is row else _record(row)
        kept[id(row)] = (row, rec)
        out.append(rec)
    return out, kept


def build_link_index(vault: Path, *, use_cache: bool = True) -> LinkIndex:
    """Map every key a note in ``vault`` answers to onto that note's LinkRec.

    Each call brings the vault catalog up to date (a stat walk that re-reads only new,
    changed or just-modified files), so a write costs one parse and is visible to the
    very next call. The same object comes back for as long as the catalog did not
    change; concurrent callers wait for one refresh and share it. ``use_cache=False``
    parses everything into a fresh object and leaves every cache alone.
    """
    vault = Path(vault)
    if not vault.is_dir():
        return LinkIndex()
    from sift.vault.catalog import VaultCatalog, get_catalog

    if not use_cache:
        scratch = VaultCatalog(vault, None)  # no cache file: parse everything
        scratch.refresh()
        return _assemble(_record(row) for row in scratch.rows())
    with _LOCK:
        cat = get_catalog(vault)
        generation = cat.ensure_fresh()  # before reading rows: a later change rebuilds
        cache = _STATE.get(str(vault))
        if cache is None:
            cache = _STATE[str(vault)] = _VaultCache()
        if cache.index is not None and cache.catalog is cat and cache.generation == generation:
            return cache.index
        prev = cache.records if cache.catalog is cat else {}
        recs, cache.records = _records(cat.rows(), prev)
        cache.index = _assemble(recs)
        cache.catalog, cache.generation = cat, generation
        return cache.index


def clear_link_index_cache() -> None:
    """Drop every cached graph (the vault catalog is left alone). Tests use this;
    production revalidates through the catalog on every call."""
    with _LOCK:
        _STATE.clear()


def warm_link_index(vault: Path) -> threading.Thread:
    """Build the graph on a daemon thread, e.g. when the MCP server starts.

    The cold build (the catalog's first parse, then the records) then does not land on
    the first expanding search, which waits on the lock and shares the result.
    Failures are logged, never raised.
    """

    def run() -> None:
        try:
            build_link_index(vault)
        except Exception:  # noqa: BLE001 - a warm-up must never take the server down
            log.exception("link graph warm-up failed for %s", vault)

    thread = threading.Thread(target=run, name="sift-link-graph-warmup", daemon=True)
    thread.start()
    return thread


# --- expansion ----------------------------------------------------------------------


def _neighbours(seeds: Sequence[str], link_index: Mapping[str, Any], hops: int) -> Iterator[Any]:
    """Notes reachable from ``seeds`` in up to ``hops`` link steps, nearest first."""
    seed_keys = set(seeds)
    seen: set[int] = set()
    frontier: list[Any] = []
    for key in seeds:
        for rec in lookup(link_index, key):
            if id(rec) not in seen:
                seen.add(id(rec))
                frontier.append(rec)
    for _ in range(hops):
        nxt: list[Any] = []
        for rec in frontier:
            for key in rec.all_links():
                for target in lookup(link_index, key):
                    if id(target) in seen or target.slug in seed_keys:
                        continue
                    seen.add(id(target))
                    nxt.append(target)
                    yield target
        frontier = nxt


def expand_records(
    seeds: Sequence[str],
    link_index: Mapping[str, Any],
    *,
    hops: int = 1,
    limit: int = 12,
) -> list[Any]:
    """Like expand(), but returns the neighbour records themselves, capped at ``limit``.

    Exact where a list of slugs cannot be: two notes that share a slug are two records
    here. Seeding with note ids (a search hit's ``note_id``) makes the start exact too.
    """
    if limit <= 0:
        return []
    out: list[Any] = []
    for rec in _neighbours(seeds, link_index, hops):
        out.append(rec)
        if len(out) >= limit:
            break
    return out


def expand(
    seeds: list[str],
    link_index: Mapping[str, Any],
    *,
    hops: int = 1,
    limit: int = 12,
) -> list[str]:
    """Return neighbour slugs (excluding seeds), nearest first, capped at ``limit``.

    Seeds may be slugs, any alias, or note ids. A wikilink may name an alias
    ("cookie-sandwich") rather than the note's real slug ("tech-cookie-sandwich"); the
    canonical slug is what comes back, so callers can feed it straight into get_note.
    A key several notes claim yields all of them.
    """
    if limit <= 0:
        return []
    out: list[str] = []
    emitted: set[str] = set()
    for rec in _neighbours(seeds, link_index, hops):
        if rec.slug in emitted:
            continue
        emitted.add(rec.slug)
        out.append(rec.slug)
        if len(out) >= limit:
            break
    return out
