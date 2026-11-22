import argparse
from pathlib import Path
import subprocess

from config.config import DEFAULT_CONFIG, Config
from data.abide_loader import load_abide_data
from data.data_loader import load_brain_connectivity_data
from training.trainer import run_cross_validation
from utils.logger import create_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GLGAN cross-validation experiments.")
    parser.add_argument("--dataset", choices=("paired", "abide"), default="paired")
    parser.add_argument("--abide-manifest", type=Path, help="CSV with subject_id, site, label, timeseries.")
    parser.add_argument("--abide-template", type=Path, help="Population-average AAL-90 DTI matrix as .npy.")
    parser.add_argument("--data-dir", default=DEFAULT_CONFIG.data_dir)
    parser.add_argument("--fmri-file", default=DEFAULT_CONFIG.fmri_file)
    parser.add_argument("--dti-file", default=DEFAULT_CONFIG.dti_file)
    parser.add_argument("--fmri-key", default=DEFAULT_CONFIG.fmri_key)
    parser.add_argument("--dti-key", default=DEFAULT_CONFIG.dti_key)
    parser.add_argument("--label-key", default=DEFAULT_CONFIG.label_key)
    parser.add_argument("--fmri-id-key", default=DEFAULT_CONFIG.fmri_id_key)
    parser.add_argument("--dti-id-key", default=DEFAULT_CONFIG.dti_id_key)
    parser.add_argument("--label-id-key", default=DEFAULT_CONFIG.label_id_key)
    parser.add_argument("--connectivity-mode", choices=("static", "window_absmean"), default=DEFAULT_CONFIG.connectivity_mode)
    parser.add_argument("--fmri-time-series", action="store_true", help="Interpret square fMRI arrays as time series.")
    parser.add_argument("--coupling-mode", choices=("unidirectional", "bidirectional"), default=DEFAULT_CONFIG.coupling_mode)
    parser.add_argument("--contrastive-mode", choices=("roi_infonce", "class_conditional"), default=DEFAULT_CONFIG.contrastive_mode)
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG.output_dir)
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG.weight_decay)
    parser.add_argument("--k-folds", type=int, default=DEFAULT_CONFIG.k_folds)
    parser.add_argument("--n-repeats", type=int, help="Defaults to 10 for paired data, 1 for ABIDE.")
    parser.add_argument("--repeat-split-mode", choices=("resplit", "fixed"), default=DEFAULT_CONFIG.repeat_split_mode)
    parser.add_argument("--aggregate-unit", choices=("run", "fold"), help="Defaults to run for paired data, fold for ABIDE.")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.random_seed)
    parser.add_argument("--n-regions", type=int, default=DEFAULT_CONFIG.n_regions)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_CONFIG.hidden_dim)
    parser.add_argument("--class-values", nargs=2, type=int, metavar=("NEGATIVE", "POSITIVE"))
    parser.add_argument("--negative-classes", nargs="+", type=int, metavar="LABEL")
    parser.add_argument("--positive-classes", nargs="+", type=int, metavar="LABEL")
    parser.add_argument("--weighted-pagerank", action="store_true", help="Use DTI strengths for PageRank transitions.")
    parser.add_argument("--no-checkpoints", action="store_true", help="Disable fold checkpoint writing.")
    parser.add_argument("--disable-local", action="store_true", help="Ablation: disable the local GAT branch.")
    parser.add_argument("--disable-global", action="store_true", help="Ablation: disable the global PageRank branch.")
    parser.add_argument("--disable-pagerank", action="store_true", help="Ablation: remove PageRank weighting.")
    parser.add_argument("--disable-self-attention", action="store_true", help="Ablation: remove self-attention blocks.")
    args = parser.parse_args()
    if (args.negative_classes is None) != (args.positive_classes is None):
        parser.error("--negative-classes and --positive-classes must be supplied together")
    if args.class_values is not None and args.negative_classes is not None:
        parser.error("--class-values cannot be combined with the class group options")
    if args.dataset == "abide":
        if not args.abide_manifest or not args.abide_template:
            parser.error("--dataset abide requires --abide-manifest and --abide-template")
        if args.n_regions != 90 or args.connectivity_mode != "static" or args.fmri_time_series:
            parser.error("ABIDE uses static Pearson connectivity on AAL-90 time series")
        if args.class_values is not None or args.negative_classes is not None:
            parser.error("ABIDE uses the fixed control vs. ASD task")
    elif args.abide_manifest or args.abide_template:
        parser.error("--abide-manifest and --abide-template require --dataset abide")
    return args


def build_config(args: argparse.Namespace) -> Config:
    return DEFAULT_CONFIG.with_updates(
        data_dir=args.data_dir,
        fmri_file=args.fmri_file,
        dti_file=args.dti_file,
        fmri_key=args.fmri_key,
        dti_key=args.dti_key,
        label_key=args.label_key,
        fmri_id_key=args.fmri_id_key,
        dti_id_key=args.dti_id_key,
        label_id_key=args.label_id_key,
        connectivity_mode=args.connectivity_mode,
        fmri_time_series=args.fmri_time_series,
        coupling_mode=args.coupling_mode,
        contrastive_mode=args.contrastive_mode,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        k_folds=args.k_folds,
        n_repeats=(1 if args.dataset == "abide" else DEFAULT_CONFIG.n_repeats)
        if args.n_repeats is None else args.n_repeats,
        repeat_split_mode=args.repeat_split_mode,
        aggregate_unit=args.aggregate_unit or ("fold" if args.dataset == "abide" else DEFAULT_CONFIG.aggregate_unit),
        random_seed=args.seed,
        n_regions=args.n_regions,
        hidden_dim=args.hidden_dim,
        class_values=args.class_values,
        negative_classes=args.negative_classes,
        positive_classes=args.positive_classes,
        pagerank_weighted=args.weighted_pagerank,
        save_checkpoints=not args.no_checkpoints,
        use_local_branch=not args.disable_local,
        use_global_branch=not args.disable_global,
        use_pagerank=not args.disable_pagerank,
        use_self_attention=not args.disable_self_attention,
    )


def _git_commit() -> str:
    source_dir = Path(__file__).resolve().parent
    if not (source_dir / ".git").exists():
        return "unavailable"

    git = ["git", "-C", str(source_dir)]
    try:
        commit = subprocess.check_output(
            [*git, "rev-parse", "--verify", "HEAD"], text=True, stderr=subprocess.DEVNULL, timeout=5
        ).strip()
        changes = subprocess.check_output(
            [*git, "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return f"{commit} (local changes)" if changes else commit


def main() -> None:
    args = parse_args()
    config = build_config(args)
    logger = create_logger("GLGAN", config.output_path / "training.log")

    logger.info("Starting GLGAN experiment (dataset=%s)", args.dataset)
    logger.info("Git commit=%s", _git_commit())
    logger.info("Experiment config=%r", config)
    logger.info("PyTorch implementation; Adam lr=%s, weight_decay=%s", config.learning_rate, config.weight_decay)
    logger.info("Epochs=%s, batch_size=%s, k_folds=%s", config.epochs, config.batch_size, config.k_folds)
    logger.info(
        "Repeats=%s, split mode=%s, aggregation unit=%s, base seed=%s",
        config.n_repeats, config.repeat_split_mode, config.aggregate_unit, config.random_seed,
    )
    logger.info("PageRank: enabled=%s, weighted=%s, alpha=%s",
                config.use_pagerank, config.pagerank_weighted, config.pagerank_alpha)

    if args.dataset == "abide":
        data = load_abide_data(args.abide_manifest, args.abide_template)
    else:
        data = load_brain_connectivity_data(config)
    result = run_cross_validation(data.fmri, data.dti, data.labels, config, subject_ids=data.subject_ids)

    for metric, value in result.aggregate_metrics.items():
        logger.info("%s: %.6f", metric, value)


if __name__ == "__main__":
    main()
