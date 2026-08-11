from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TrainingRun:
    name: str
    source: str
    model: str
    seeds: tuple[int, ...]
    include_routing_flags: bool = False
    lbbb_auxiliary_weight: float = 0.05


def _run(command: list[str], dry_run: bool) -> None:
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one confirmation seed is required")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the pre-specified corrected IECBES 2026 experiment matrix."
    )
    parser.add_argument("--data-dir", default="data/ptbxl")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--backend",
        choices=["custom", "neurokit", "ecgdeli", "auto"],
        default="custom",
    )
    parser.add_argument("--ecgdeli-fiducials")
    parser.add_argument("--confirmation-seeds", type=_parse_seeds, default=(42, 2026, 815))
    parser.add_argument("--sensitivity-seed", type=int, default=42)
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--resume-extraction", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    if not args.dry_run and not (data_dir / "records500").is_dir():
        raise FileNotFoundError(
            f"Missing {data_dir / 'records500'}; obtain the official PTB-XL 500 Hz waveforms before running."
        )
    if args.backend == "ecgdeli" and not args.ecgdeli_fiducials:
        parser.error("--backend ecgdeli requires --ecgdeli-fiducials")

    protocol_dir = Path("data/iecbes2026")
    artifacts_dir = Path("artifacts/iecbes2026_corrected")
    sources = ("scp", "either", "raw")
    if not args.skip_extraction:
        for source in sources:
            command = [
                sys.executable,
                "scripts/extract_morphology.py",
                "--data-dir",
                args.data_dir,
                "--backend",
                args.backend,
                "--lbbb-source",
                source,
                "--out",
                str(protocol_dir / f"morphology_{source}.csv"),
                "--beats-out",
                str(protocol_dir / "morphology_beats_scp.csv") if source == "scp" else "",
                "--label-out",
                str(protocol_dir / f"labels_{source}.csv"),
            ]
            if args.ecgdeli_fiducials:
                command.extend(["--ecgdeli-fiducials", args.ecgdeli_fiducials])
            if args.resume_extraction:
                command.append("--resume")
            _run(command, args.dry_run)

    confirm = args.confirmation_seeds
    sensitivity = (int(args.sensitivity_seed),)
    runs = (
        TrainingRun("primary_scp_fusion", "scp", "morphology_fusion", confirm),
        TrainingRun(
            "waveform_only_seresnet",
            "scp",
            "seresnet",
            confirm,
            lbbb_auxiliary_weight=0.0,
        ),
        TrainingRun(
            "routing_flags_ablation",
            "scp",
            "morphology_fusion",
            confirm,
            include_routing_flags=True,
        ),
        TrainingRun(
            "no_lbbb_aux_ablation",
            "scp",
            "morphology_fusion",
            sensitivity,
            lbbb_auxiliary_weight=0.0,
        ),
        TrainingRun("either_lbbb_sensitivity", "either", "morphology_fusion", sensitivity),
        TrainingRun("raw_lbbb_sensitivity", "raw", "morphology_fusion", sensitivity),
    )
    for run in runs:
        for seed in run.seeds:
            command = [
                sys.executable,
                "scripts/train.py",
                "--config",
                "configs/default.yaml",
                "--manifest",
                str(protocol_dir / f"labels_{run.source}.csv"),
                "--model",
                run.model,
                "--device",
                args.device,
                "--seed",
                str(seed),
                "--lbbb-auxiliary-weight",
                str(run.lbbb_auxiliary_weight),
                "--out-dir",
                str(artifacts_dir / run.name / f"seed_{seed}"),
            ]
            command.append(
                "--include-label-routing-flags"
                if run.include_routing_flags
                else "--no-include-label-routing-flags"
            )
            _run(command, args.dry_run)


if __name__ == "__main__":
    main()
