"""Command-line interface for itslive-cloudfree."""

from __future__ import annotations

import argparse
import sys

from . import search
from .results import to_csv, to_geojson, to_dataframe


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="itslive-cloudfree",
        description=(
            "Identify cloud-free Landsat/Sentinel-2 scenes over a geographic "
            "area using ItsLive feature-tracking success as a proxy."
        ),
    )
    p.add_argument(
        "--bbox",
        required=True,
        metavar="minlon,minlat,maxlon,maxlat",
        help="Bounding box in WGS-84 degrees, comma-separated.",
    )
    p.add_argument("--start-date", required=True, metavar="YYYY-MM-DD")
    p.add_argument("--end-date", required=True, metavar="YYYY-MM-DD")
    p.add_argument(
        "--sensor",
        choices=["landsat", "sentinel2", "both"],
        default="both",
        help="Restrict to a sensor family (default: both).",
    )
    p.add_argument(
        "--max-dt-days",
        type=int,
        default=None,
        metavar="N",
        help="Exclude pairs with more than N days between acquisitions.",
    )
    p.add_argument(
        "--min-valid-pixels",
        type=float,
        default=None,
        metavar="PCT",
        help="Exclude pairs with fewer than PCT%% valid pixels.",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=20,
        metavar="N",
        help="Show only the top N results (default: 20).",
    )
    p.add_argument(
        "--max-granules",
        type=int,
        default=None,
        metavar="N",
        help="Cap total granules fetched (for quick tests).",
    )
    p.add_argument(
        "--output",
        choices=["table", "csv", "geojson"],
        default="table",
        help="Output format (default: table).",
    )
    p.add_argument(
        "--out-file",
        metavar="PATH",
        default=None,
        help="Write output to this file (default: stdout).",
    )
    p.add_argument(
        "--enrich-cloud-cover",
        action="store_true",
        help="Fetch official eo:cloud_cover from source STAC catalogs.",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    # Parse bbox
    try:
        bbox = tuple(float(x) for x in args.bbox.split(","))
        assert len(bbox) == 4
    except Exception:
        print("ERROR: --bbox must be four comma-separated floats.", file=sys.stderr)
        sys.exit(1)

    results = search(
        bbox=bbox,
        start_date=args.start_date,
        end_date=args.end_date,
        max_dt_days=args.max_dt_days,
        min_valid_pixels=args.min_valid_pixels,
        max_granules=args.max_granules,
    )

    if not results:
        print("No optical granules found for the given parameters.", file=sys.stderr)
        sys.exit(0)

    # Sensor filter post-hoc (simpler than threading it through search())
    if args.sensor == "landsat":
        results = [r for r in results if r.platform.startswith("L")]
    elif args.sensor == "sentinel2":
        results = [r for r in results if r.platform.startswith("S2")]

    # Optional enrichment
    if args.enrich_cloud_cover:
        from .enrich import enrich_cloud_cover
        results = enrich_cloud_cover(results)

    results = results[: args.top_n]

    # Output
    if args.output == "geojson":
        text = to_geojson(results)
    elif args.output == "csv":
        if args.out_file:
            to_csv(results, args.out_file)
            print(f"Wrote {len(results)} rows to {args.out_file}")
            return
        else:
            import io
            buf = io.StringIO()
            to_dataframe(results).to_csv(buf, index=False)
            text = buf.getvalue()
    else:  # table
        df = to_dataframe(results)
        cols = ["scene_id", "platform", "acquisition_date", "path_row",
                "pair_count", "mean_valid_pixels", "cloud_free_score"]
        if "eo_cloud_cover" in df.columns and df["eo_cloud_cover"].notna().any():
            cols.append("eo_cloud_cover")
        text = df[cols].to_string(index=False)

    if args.out_file:
        with open(args.out_file, "w") as fh:
            fh.write(text)
        print(f"Wrote output to {args.out_file}")
    else:
        print(text)


if __name__ == "__main__":
    main()
