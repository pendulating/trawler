"""CLI for the Gutenberg corpus cache.

Subcommands:
    refresh-catalog       Snapshot Gutendex metadata.
    select top-k          Pick the K most-downloaded books.
    select top-authors    Pick top-K authors then top-N of each.
    materialize           Build chunks.parquet from a saved selection.

Examples:
    python -m dagspaces.common.gutenberg.cli refresh-catalog --max-pages 32
    python -m dagspaces.common.gutenberg.cli select top-k --k 50 --out /tmp/sel.yaml
    python -m dagspaces.common.gutenberg.cli select top-authors --k-authors 10 --n 5 --out /tmp/sel.yaml
    python -m dagspaces.common.gutenberg.cli materialize --selection /tmp/sel.yaml \\
        --chunk-size 6000 --overlap 1000 --out /tmp/chunks.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

from . import catalog as catalog_mod
from . import materialize as materialize_mod
from . import select as select_mod
from .paths import cache_root, selections_dir


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _selection_to_yaml(selection: select_mod.Selection) -> dict:
    return {
        "strategy": selection.strategy,
        "params": selection.params,
        "books": [asdict(b) for b in selection.books],
        "author_rankings": selection.author_rankings,
    }


def _selection_from_yaml(data: dict) -> select_mod.Selection:
    books = [
        select_mod.BookRef(
            gutenberg_id=str(b["gutenberg_id"]),
            title=b["title"],
            authors=tuple(b.get("authors") or []),
            download_count=int(b.get("download_count") or 0),
            languages=tuple(b.get("languages") or []),
        )
        for b in data.get("books") or []
    ]
    return select_mod.Selection(
        strategy=data.get("strategy", "ids"),
        params=data.get("params") or {},
        books=books,
        author_rankings=data.get("author_rankings") or [],
    )


def cmd_refresh_catalog(args: argparse.Namespace) -> int:
    out = catalog_mod.refresh_catalog(
        languages=tuple(args.languages),
        max_pages=args.max_pages,
        max_age_days=args.max_age_days,
        force=args.force,
    )
    print(f"catalog at {out}")
    return 0


def _resolve_out(args_out: str | None, default_name: str) -> Path:
    if args_out:
        return Path(args_out)
    return selections_dir() / default_name


def cmd_select_top_k(args: argparse.Namespace) -> int:
    sel = select_mod.top_k_by_popularity(
        k=args.k,
        languages=tuple(args.languages),
        min_downloads=args.min_downloads,
        only_fiction=args.only_fiction,
    )
    out = _resolve_out(args.out, f"top_k_{args.k}_{'_'.join(args.languages)}.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(_selection_to_yaml(sel), sort_keys=False))
    print(f"selected {len(sel.books)} books -> {out}")
    for b in sel.books[:5]:
        print(f"  {b.gutenberg_id:>7}  {b.download_count:>7}  {b.title[:70]}")
    return 0


def cmd_select_top_authors(args: argparse.Namespace) -> int:
    sel = select_mod.top_k_authors_n_books(
        k_authors=args.k_authors,
        n_per_author=args.n,
        languages=tuple(args.languages),
        min_downloads=args.min_downloads,
        only_fiction=args.only_fiction,
    )
    out = _resolve_out(
        args.out,
        f"top_authors_{args.k_authors}x{args.n}_{'_'.join(args.languages)}.yaml",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(_selection_to_yaml(sel), sort_keys=False))
    print(f"selected {len(sel.books)} books across {len(sel.author_rankings)} authors -> {out}")
    for r in sel.author_rankings[:5]:
        print(f"  {r['author'][:50]:<50} {r['summed_downloads']:>10} downloads")
    return 0


def cmd_materialize(args: argparse.Namespace) -> int:
    data = yaml.safe_load(Path(args.selection).read_text())
    selection = _selection_from_yaml(data)

    summaries = None
    if args.summaries_json:
        raw = json.loads(Path(args.summaries_json).read_text())
        summaries = {str(gid): info.get("summary", "") for gid, info in raw.items()}

    summary = materialize_mod.materialize_dataset(
        selection=selection,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        out_path=Path(args.out),
        book_summaries=summaries,
    )
    sidecar = Path(args.out).with_suffix(".manifest.json")
    sidecar.write_text(json.dumps(summary, indent=2))
    print(f"wrote {summary['out_path']} ({summary['rows']} rows)")
    print(f"  cached={summary['books_cached']}  "
          f"fetch_failed={len(summary['books_fetch_failed'])}  "
          f"failed={len(summary['books_failed'])}")
    print(f"  manifest -> {sidecar}")
    if summary["books_fetch_failed"]:
        print("  fetch_failed ids:", summary["books_fetch_failed"][:10])
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dagspaces.common.gutenberg.cli")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    rc = sub.add_parser("refresh-catalog", help="snapshot Gutendex metadata")
    rc.add_argument("--languages", nargs="+", default=["en"])
    rc.add_argument("--max-pages", type=int, default=catalog_mod.DEFAULT_MAX_PAGES)
    rc.add_argument("--max-age-days", type=float, default=30.0)
    rc.add_argument("--force", action="store_true")
    rc.set_defaults(func=cmd_refresh_catalog)

    sl = sub.add_parser("select", help="produce a selection yaml")
    sl_sub = sl.add_subparsers(dest="strategy", required=True)

    tk = sl_sub.add_parser("top-k", help="K most-downloaded books")
    tk.add_argument("--k", type=int, required=True)
    tk.add_argument("--languages", nargs="+", default=["en"])
    tk.add_argument("--min-downloads", type=int, default=0)
    tk.add_argument("--only-fiction", action="store_true",
                    help="Restrict to fiction novels (drops textbooks, reference works, "
                         "history, philosophy, etc.). Always drops audiobooks.")
    tk.add_argument("--out", default=None)
    tk.set_defaults(func=cmd_select_top_k)

    ta = sl_sub.add_parser("top-authors", help="top-K authors x N books each")
    ta.add_argument("--k-authors", type=int, required=True)
    ta.add_argument("--n", type=int, required=True, help="books per author")
    ta.add_argument("--languages", nargs="+", default=["en"])
    ta.add_argument("--min-downloads", type=int, default=0)
    ta.add_argument("--only-fiction", action="store_true",
                    help="Restrict to fiction novels. Always drops audiobooks.")
    ta.add_argument("--out", default=None)
    ta.set_defaults(func=cmd_select_top_authors)

    mt = sub.add_parser("materialize", help="build chunks.parquet from a selection")
    mt.add_argument("--selection", required=True)
    mt.add_argument("--chunk-size", type=int, default=6000)
    mt.add_argument("--overlap", type=int, default=1000)
    mt.add_argument("--out", required=True)
    mt.add_argument("--summaries-json", default=None,
                    help="optional JSON {gid: {summary: str}} for book_summary column")
    mt.set_defaults(func=cmd_materialize)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    print(f"# cache_root={cache_root()}")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
