import argparse
import os
import sys


def results_mcnlm(out_dir=None, seed=None, deterministic=False, show=True):
    from mcnlm.plots import resolve_out_dir, resolve_plot_spec, run_plot

    out_dir = resolve_out_dir(out_dir)
    for name in ["mcnlm1", "mcnlm2", "mcnlm3", "mc_matches_1", "mc_matches_2"]:
        run_plot(resolve_plot_spec(name), out_dir, seed, deterministic, show)


def results_naive_nlm(out_dir=None, seed=None, deterministic=False, show=True):
    from mcnlm.plots import resolve_out_dir, resolve_plot_spec, run_plot

    out_dir = resolve_out_dir(out_dir)
    for name in ["nlm_denoise1", "nlm_denoise2"]:
        run_plot(resolve_plot_spec(name), out_dir, seed, deterministic, show)


def mc_convergence_results(out_dir=None, seed=None, deterministic=False, show=True):
    from mcnlm.plots import resolve_out_dir, resolve_plot_spec, run_plot

    out_dir = resolve_out_dir(out_dir)
    for name in ["convergence1", "convergence2"]:
        run_plot(resolve_plot_spec(name), out_dir, seed, deterministic, show)


def noise_comparison_results(out_dir=None, seed=None, deterministic=False, show=True):
    from mcnlm.plots import resolve_out_dir, resolve_plot_spec, run_plot

    out_dir = resolve_out_dir(out_dir)
    for name in ["noise_comparison_visual", "noise_comparison_visual2"]:
        run_plot(resolve_plot_spec(name), out_dir, seed, deterministic, show)


def hashed_nlm_result(out_dir=None, seed=None, deterministic=False, show=True):
    from mcnlm.plots import resolve_out_dir, resolve_plot_spec, run_plot

    out_dir = resolve_out_dir(out_dir)
    run_plot(resolve_plot_spec("hashednlm"), out_dir, seed, deterministic, show)


def standard_comparison_hashed_nlm(img_path, output_path, sigma=17, num_features=4, beta=0.88, seed=None, show=True):
    from mcnlm.hashnlm_viz import standard_comparison_hashed_nlm as _fn

    return _fn(
        img_path,
        output_path,
        sigma=sigma,
        num_features=num_features,
        beta=beta,
        seed=seed,
        show=show,
    )


def standard_comparison_hashed_nlm_zoomed(
    img_path,
    output_path,
    sigma=17,
    num_features=4,
    beta=0.88,
    zoom_size=64,
    zoom_center=None,
    seed=None,
    show=True,
):
    from mcnlm.hashnlm_viz import standard_comparison_hashed_nlm_zoomed as _fn

    return _fn(
        img_path,
        output_path,
        sigma=sigma,
        num_features=num_features,
        beta=beta,
        zoom_size=zoom_size,
        zoom_center=zoom_center,
        seed=seed,
        show=show,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate reproducible plots for docs/res."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available plots and outputs",
    )
    parser.add_argument(
        "--plot",
        action="append",
        dest="plots",
        help="Plot name or output filename (repeatable)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all plots",
    )
    parser.add_argument(
        "--out-dir",
        help="Output directory (default: docs/res)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override default seed for all plots",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic single-threaded MCNLM",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.show:
        os.environ.setdefault("MPLBACKEND", "Agg")

    from mcnlm.plots import list_plot_specs, resolve_out_dir, resolve_plot_spec, run_plots

    if args.list:
        for spec in list_plot_specs():
            outputs = ", ".join(spec.outputs)
            print(f"{spec.name}: {outputs} | {spec.description} | seed={spec.default_seed}")
        return 0

    if not args.all and not args.plots:
        parser.print_help()
        return 0

    out_dir = resolve_out_dir(args.out_dir)

    specs = []
    if args.all:
        specs = list(list_plot_specs())

    if args.plots:
        for name in args.plots:
            try:
                specs.append(resolve_plot_spec(name))
            except KeyError as exc:
                print(str(exc), file=sys.stderr)
                return 2

    unique = {}
    for spec in specs:
        unique[spec.name] = spec
    specs = list(unique.values())

    run_plots(
        specs,
        out_dir,
        seed_override=args.seed,
        deterministic=args.deterministic,
        show=args.show,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
