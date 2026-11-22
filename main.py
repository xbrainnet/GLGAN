import argparse
from pathlib import Path

from config.config import DEFAULT_CONFIG, Config
from data.data_loader import load_brain_connectivity_data
from training.trainer import run_cross_validation
from utils.logger import create_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GLGAN cross-validation experiments.")
    parser.add_argument("--data-dir", default=DEFAULT_CONFIG.data_dir)
    parser.add_argument("--fmri-file", default=DEFAULT_CONFIG.fmri_file)
    parser.add_argument("--dti-file", default=DEFAULT_CONFIG.dti_file)
    parser.add_argument("--fmri-key", default=DEFAULT_CONFIG.fmri_key)
    parser.add_argument("--dti-key", default=DEFAULT_CONFIG.dti_key)
    parser.add_argument("--label-key", default=DEFAULT_CONFIG.label_key)
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG.output_dir)
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG.weight_decay)
    parser.add_argument("--k-folds", type=int, default=DEFAULT_CONFIG.k_folds)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.random_seed)
    parser.add_argument("--n-regions", type=int, default=DEFAULT_CONFIG.n_regions)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_CONFIG.hidden_dim)
    parser.add_argument("--class-values", nargs=2, type=int, default=None)
    parser.add_argument("--no-checkpoints", action="store_true", help="Disable fold checkpoint writing.")
    parser.add_argument("--disable-local", action="store_true", help="Ablation: disable the local GAT branch.")
    parser.add_argument("--disable-global", action="store_true", help="Ablation: disable the global PageRank branch.")
    parser.add_argument("--disable-pagerank", action="store_true", help="Ablation: remove PageRank weighting.")
    parser.add_argument("--disable-self-attention", action="store_true", help="Ablation: remove self-attention blocks.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    return DEFAULT_CONFIG.with_updates(
        data_dir=args.data_dir,
        fmri_file=args.fmri_file,
        dti_file=args.dti_file,
        fmri_key=args.fmri_key,
        dti_key=args.dti_key,
        label_key=args.label_key,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        k_folds=args.k_folds,
        random_seed=args.seed,
        n_regions=args.n_regions,
        hidden_dim=args.hidden_dim,
        class_values=args.class_values,
        save_checkpoints=not args.no_checkpoints,
        use_local_branch=not args.disable_local,
        use_global_branch=not args.disable_global,
        use_pagerank=not args.disable_pagerank,
        use_self_attention=not args.disable_self_attention,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    output_dir = Path(config.output_dir)
    logger = create_logger("GLGAN", output_dir / "training.log")

    logger.info("Starting GLGAN experiment")
    logger.info("PyTorch implementation; Adam lr=%s, weight_decay=%s", config.learning_rate, config.weight_decay)
    logger.info("Epochs=%s, batch_size=%s, k_folds=%s", config.epochs, config.batch_size, config.k_folds)

    data = load_brain_connectivity_data(config)
    result = run_cross_validation(data.fmri, data.dti, data.labels, config)

    for metric, value in result.aggregate_metrics.items():
        logger.info("%s: %.6f", metric, value)


if __name__ == "__main__":
    main()
