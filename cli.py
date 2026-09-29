"""Command line: `trustlens demo --out reports/`."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from . import __version__
from .demos import ALL
from .report import _CSS, model_card_md, render_html


def run_demo(out: Path) -> list:
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for key, fn in ALL.items():
        r = fn()
        (out / f"{key}.html").write_text(render_html(r), encoding="utf-8")
        (out / f"{key}_model_card.md").write_text(model_card_md(r), encoding="utf-8")
        results.append((key, r))
        print(f"[{key}] trust={r.trust_score:.0f}  {r.verdict[0]}")
    (out / "summary.json").write_text(json.dumps({k: r.to_dict() for k, r in results}, indent=2, default=float), encoding="utf-8")
    cards = "".join(f'<a class="card" style="display:block;text-decoration:none;color:inherit" href="{k}.html"><b>{html.escape(r.name)}</b>'
                    f'<div class="sub">{html.escape(r.verdict[0])}</div><h1 style="color:{r.verdict[2]}">{r.trust_score:.0f}<small>/100</small></h1></a>'
                    for k, r in results)
    (out / "index.html").write_text(f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                                    f'<title>TrustLens</title><style>{_CSS}</style></head><body><div class="w"><h1>TrustLens</h1>'
                                    f'<p class="sub">Trust &amp; evaluation layer for human-governed AI · demo audits</p><div class="grid">{cards}</div></div></body></html>', encoding="utf-8")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="trustlens", description="Trust & evaluation layer for human-governed AI.")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="run the three demo audits and write HTML reports + model cards")
    d.add_argument("--out", default="reports", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "demo":
        run_demo(args.out)
        print(f"Open {args.out}/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
