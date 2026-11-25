"""Command-line interface for the shutter demo material.

The web player is the graphical entry point (see webapp/server.py); this CLI
only supports it with authoring and verification commands:

    python -m scripts.shutter_cli author --name dial_demo [--param value ...]
    python -m scripts.shutter_cli rebuild outputs/shutter_demo/assets/dial_demo
    python -m scripts.shutter_cli list
    python -m scripts.shutter_cli measure

Every render goes through the real exposure integrators; there is no blur
filter path and no fixed canned image is ever served.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shutter_blur.assets import (  # noqa: E402
    AssetSpec,
    create_asset,
    load_asset,
    rebuild_asset,
)
from scripts.measure_exposure_vs_readout import run as run_measure  # noqa: E402

ASSET_ROOT = ROOT / "outputs" / "shutter_demo" / "assets"
MEASURE_DIR = ROOT / "outputs" / "shutter_demo"


def _edit_spec(args) -> AssetSpec:
    spec = AssetSpec(name=args.name)
    editable = {
        "height": int, "width": int, "frame_index": int,
        "gt_samples": int, "trajectory_samples": int, "omega": float,
        "strip_drift": float, "strip_osc_amp": float,
        "strip_osc_freq": float, "global_angle": float,
        "rolling_angle": float, "rolling_readout_frac": float,
        "short_angle": float, "num_samples": int,
    }
    for key, cast in editable.items():
        val = getattr(args, key)
        if val is not None:
            setattr(spec, key, cast(val))
    if args.rolling_profile is not None:
        spec.rolling_profile = args.rolling_profile
    return spec


def cmd_author(args) -> int:
    spec = _edit_spec(args)
    manifest = create_asset(ASSET_ROOT, spec)
    print(json.dumps({
        "asset": spec.name,
        "dir": str(ASSET_ROOT / spec.name),
        "files": sorted((ASSET_ROOT / spec.name).rglob("*.*"))
        and [str(p.relative_to(ASSET_ROOT / spec.name))
             for p in sorted((ASSET_ROOT / spec.name).rglob("*.*"))],
        "variants": list(manifest["kernels"].keys()),
    }, indent=2))
    report = rebuild_asset(ASSET_ROOT / spec.name,
                           tolerance=args.tolerance)
    print("rebuild check:", "OK" if report["ok"] else "FAILED")
    for label, item in report["variants"].items():
        print(f"  {label}: rmse={item['rmse_vs_reference']:.6f} "
              f"within={item['within_tolerance']}")
    return 0 if report["ok"] else 1


def cmd_rebuild(args) -> int:
    report = rebuild_asset(args.asset_dir, tolerance=args.tolerance)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def cmd_list(args) -> int:
    if not ASSET_ROOT.exists():
        print("no assets under", ASSET_ROOT)
        return 0
    for manifest_path in sorted(ASSET_ROOT.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        bundle = load_asset(manifest_path.parent)
        traj = bundle["trajectory"]
        print(f"- {manifest['spec']['name']}  ({manifest_path.parent})")
        print(f"    variants: {', '.join(manifest['kernels'])}")
        print(f"    trajectory samples: {len(traj['time'])} "
              f"spokes: {traj['spoke_world_angle'].shape[0]}")
    return 0


def cmd_measure(args) -> int:
    report = run_measure(MEASURE_DIR)
    print(json.dumps(report["measured_decision"], indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shutter demo asset CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_author = sub.add_parser("author", help="create a demo asset bundle")
    p_author.add_argument("--name", default="dial_demo")
    p_author.add_argument("--rolling-profile",
                          choices=("box", "trapezoid", "gaussian"))
    p_author.add_argument("--tolerance", type=float, default=5e-3)
    for key in ("height", "width", "frame_index", "gt_samples",
                "trajectory_samples", "num_samples"):
        p_author.add_argument(f"--{key}", type=int)
    for key in ("omega", "strip_drift", "strip_osc_amp", "strip_osc_freq",
                "global_angle", "rolling_angle", "rolling_readout_frac",
                "short_angle"):
        p_author.add_argument(f"--{key}", type=float)
    p_author.set_defaults(func=cmd_author)

    p_rebuild = sub.add_parser(
        "rebuild", help="re-render an asset and verify against its reference")
    p_rebuild.add_argument("asset_dir")
    p_rebuild.add_argument("--tolerance", type=float, default=5e-3)
    p_rebuild.set_defaults(func=cmd_rebuild)

    p_list = sub.add_parser("list", help="list available demo assets")
    p_list.set_defaults(func=cmd_list)

    p_measure = sub.add_parser(
        "measure", help="run the exposure-vs-readout empirical comparison")
    p_measure.set_defaults(func=cmd_measure)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
