from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from miclass3.experiments import DEFAULT_REQUIRED_METRICS, build_validation_leaderboard, metric_floor


def deep_update(base: dict, overrides: Mapping[str, object]) -> dict:
    """Merge a candidate override without mutating the base configuration."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_update(dict(merged[key]), value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def command_for_trial(
    config_path: Path,
    out_dir: Path,
    device: str,
    model: str | None,
    max_records: int | None,
    skip_test: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--config",
        str(config_path),
        "--out-dir",
        str(out_dir),
        "--device",
        device,
    ]
    if model:
        command.extend(["--model", model])
    if max_records is not None:
        command.extend(["--max-records", str(max_records)])
    if skip_test:
        command.append("--skip-test")
    return command


def run_trial(command: list[str], out_dir: Path) -> tuple[dict[str, object] | None, str | None]:
    """Run one train command and return its validation metrics or a log summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
        return None, f"exit={completed.returncode}: {' | '.join(tail)}"
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.is_file():
        return None, "training exited successfully but metrics.json was not written"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    validation = payload.get("results", {}).get("val")
    if not isinstance(validation, dict):
        return None, "training did not write validation metrics"
    return validation, None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only Miclass3 hyperparameter search with an explicit all-metric quality floor."
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Base training configuration.")
    parser.add_argument("--tuning-config", default="configs/tuning.yaml", help="Candidate override configuration.")
    parser.add_argument("--out-dir", default="artifacts/tuning", help="Directory for candidate runs and leaderboard.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--model", choices=["morphology_fusion", "inceptiontime", "seresnet"])
    parser.add_argument("--max-records", type=int, help="Optional smoke-test limit; never use it for final selection.")
    parser.add_argument("--max-trials", type=int, help="Run only the first N candidates.")
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="After validation-only selection, retrain the selected candidate once and evaluate fold 10.",
    )
    args = parser.parse_args()

    base_path = ROOT / args.config
    tuning_path = ROOT / args.tuning_config
    out_root = ROOT / args.out_dir
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    tuning = yaml.safe_load(tuning_path.read_text(encoding="utf-8")) or {}
    acceptance = tuning.get("acceptance", {}) or {}
    target = float(acceptance.get("target", 0.85))
    required_metrics = tuple(acceptance.get("required_metrics", DEFAULT_REQUIRED_METRICS))
    candidates = list(tuning.get("candidates", []))
    if args.max_trials is not None:
        candidates = candidates[: max(0, args.max_trials)]
    if not candidates:
        raise ValueError("No candidates selected. Check configs/tuning.yaml or --max-trials.")

    configs_dir = out_root / "configs"
    trials: list[dict[str, object]] = []
    trial_paths: dict[str, tuple[Path, str | None]] = {}
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, Mapping) or "name" not in candidate:
            raise ValueError("Each tuning candidate must be a mapping with a unique name")
        name = str(candidate["name"])
        overrides = candidate.get("overrides", {}) or {}
        if not isinstance(overrides, Mapping):
            raise ValueError(f"Candidate {name!r} overrides must be a mapping")
        config = deep_update(base_config, overrides)
        config_path = configs_dir / f"{index:02d}_{name}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        model = str(candidate.get("model")) if candidate.get("model") else args.model
        trial_dir = out_root / "trials" / f"{index:02d}_{name}"
        print(f"[{index}/{len(candidates)}] validation-only trial: {name}")
        metrics, error = run_trial(
            command_for_trial(config_path, trial_dir, args.device, model, args.max_records, skip_test=True),
            trial_dir,
        )
        trials.append({"trial": name, "metrics": metrics or {}, "error": error})
        trial_paths[name] = (config_path, model)

    leaderboard = build_validation_leaderboard(trials, required_metrics, target)
    out_root.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(out_root / "validation_leaderboard.csv", index=False)
    if leaderboard.empty or not bool(leaderboard.iloc[0]["passes_target"]):
        selection = {
            "target": target,
            "required_metrics": list(required_metrics),
            "selected_trial": None,
            "reason": "No validation-only candidate reached the all-metric target; fold 10 was not evaluated.",
        }
        (out_root / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
        print(selection["reason"])
        return

    selected_name = str(leaderboard.iloc[0]["trial"])
    selection = {
        "target": target,
        "required_metrics": list(required_metrics),
        "selected_trial": selected_name,
        "validation_metric_floor": float(leaderboard.iloc[0]["validation_metric_floor"]),
        "finalized_on_test": False,
    }
    if args.finalize:
        selected_config, selected_model = trial_paths[selected_name]
        final_dir = out_root / "final" / selected_name
        print(f"Finalizing selected candidate on fold 10: {selected_name}")
        test_metrics, error = run_trial(
            command_for_trial(selected_config, final_dir, args.device, selected_model, args.max_records, skip_test=False),
            final_dir,
        )
        # run_trial intentionally returns validation metrics. Read the test
        # artifact here only after the validation winner is frozen.
        if error:
            selection["finalization_error"] = error
        else:
            payload = json.loads((final_dir / "metrics.json").read_text(encoding="utf-8"))
            test = payload.get("results", {}).get("test", {})
            test_floor = metric_floor(test, required_metrics)
            selection.update(
                finalized_on_test=True,
                test_metrics=test,
                test_metric_floor=test_floor,
                test_passes_target=bool(test_floor >= target),
            )
    (out_root / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
