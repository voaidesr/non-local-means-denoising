from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


@dataclass(frozen=True)
class PlotContext:
    out_dir: Path
    seed: int | None
    deterministic: bool
    show: bool


@dataclass(frozen=True)
class PlotSpec:
    name: str
    outputs: tuple[str, ...]
    description: str
    default_seed: int
    build: Callable[[PlotContext], None]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_out_dir() -> Path:
    return repo_root() / "docs" / "res"


def resolve_out_dir(out_dir: str | Path | None) -> Path:
    if out_dir is None:
        return default_out_dir()
    return Path(out_dir)


def _plot_mcnlm1(ctx: PlotContext) -> None:
    from mcnlm.utils import show_mcnlm_result_zoomed

    show_mcnlm_result_zoomed(
        str(repo_root() / "mcnlm" / "imgs" / "clock.tiff"),
        probs=[0.3, 0.5, 0.8],
        zoom=(120, 100, 64, 64),
        output_path=str(ctx.out_dir / "mcnlm1.pdf"),
        seed=ctx.seed,
        deterministic=ctx.deterministic,
        show=ctx.show,
    )


def _plot_mcnlm2(ctx: PlotContext) -> None:
    from mcnlm.utils import show_mcnlm_result_zoomed

    show_mcnlm_result_zoomed(
        str(repo_root() / "mcnlm" / "imgs" / "man.tiff"),
        probs=[0.3, 0.5, 0.8],
        zoom=(440, 600, 64, 64),
        output_path=str(ctx.out_dir / "mcnlm2.pdf"),
        seed=ctx.seed,
        deterministic=ctx.deterministic,
        show=ctx.show,
    )


def _plot_mcnlm3(ctx: PlotContext) -> None:
    from mcnlm.utils import show_mcnlm_result_zoomed

    show_mcnlm_result_zoomed(
        str(repo_root() / "mcnlm" / "imgs" / "land.tiff"),
        probs=[0.3, 0.5, 0.8],
        zoom=(120, 100, 64, 64),
        output_path=str(ctx.out_dir / "mcnlm3.pdf"),
        seed=ctx.seed,
        deterministic=ctx.deterministic,
        show=ctx.show,
    )


def _plot_nlm_denoise1(ctx: PlotContext) -> None:
    from mcnlm.utils import show_nlm_result_zoomed

    show_nlm_result_zoomed(
        str(repo_root() / "mcnlm" / "imgs" / "land.tiff"),
        zoom=(120, 100, 64, 64),
        output_path=str(ctx.out_dir / "nlm_denoise1.pdf"),
        seed=ctx.seed,
        show=ctx.show,
    )


def _plot_nlm_denoise2(ctx: PlotContext) -> None:
    from mcnlm.utils import show_nlm_result_zoomed

    show_nlm_result_zoomed(
        str(repo_root() / "mcnlm" / "imgs" / "clock.tiff"),
        zoom=(120, 100, 64, 64),
        output_path=str(ctx.out_dir / "nlm_denoise2.pdf"),
        seed=ctx.seed,
        show=ctx.show,
    )


def _plot_mc_matches_1(ctx: PlotContext) -> None:
    from mcnlm.utils import show_matches

    show_matches(
        str(repo_root() / "mcnlm" / "imgs" / "clock.tiff"),
        [(150, 210)],
        str(ctx.out_dir / "mc_matches_1.pdf"),
        seed=ctx.seed,
        show=ctx.show,
    )


def _plot_mc_matches_2(ctx: PlotContext) -> None:
    from mcnlm.utils import show_matches

    show_matches(
        str(repo_root() / "mcnlm" / "imgs" / "clock.tiff"),
        [(90, 135), (170, 80)],
        str(ctx.out_dir / "mc_matches_2.pdf"),
        seed=ctx.seed,
        show=ctx.show,
    )


def _plot_robert_matches(ctx: PlotContext) -> None:
    from mcnlm.utils import show_matches

    show_matches(
        str(repo_root() / "mcnlm" / "imgs" / "clock.tiff"),
        [(100, 100), (150, 200)],
        str(ctx.out_dir / "robert_matches.pdf"),
        seed=ctx.seed,
        show=ctx.show,
    )


def _plot_convergence1(ctx: PlotContext) -> None:
    from mcnlm.mc_convergence import mc_convergence

    mc_convergence(
        image_path=str(repo_root() / "mcnlm" / "imgs" / "moon.tiff"),
        output_path1=str(ctx.out_dir / "convergence1_mse.pdf"),
        output_path2=str(ctx.out_dir / "convergence1_psnr.pdf"),
        seed=ctx.seed,
        deterministic=ctx.deterministic,
        show=ctx.show,
    )


def _plot_convergence2(ctx: PlotContext) -> None:
    from mcnlm.mc_convergence import mc_convergence

    mc_convergence(
        image_path=str(repo_root() / "mcnlm" / "imgs" / "clock.tiff"),
        output_path1=str(ctx.out_dir / "convergence2_mse.pdf"),
        output_path2=str(ctx.out_dir / "convergence2_psnr.pdf"),
        seed=ctx.seed,
        deterministic=ctx.deterministic,
        show=ctx.show,
    )


def _plot_noise_comparison_visual(ctx: PlotContext) -> None:
    from mcnlm.mc_convergence import compare_noise_estimation

    compare_noise_estimation(
        image_path=str(repo_root() / "mcnlm" / "imgs" / "moon.tiff"),
        output_path_visual=str(ctx.out_dir / "noise_comparison_visual.pdf"),
        output_path_mse=str(ctx.out_dir / "noise_comparison_mse.pdf"),
        output_path_psnr=str(ctx.out_dir / "noise_comparison_psnr.pdf"),
        seed=ctx.seed,
        deterministic=ctx.deterministic,
        show=ctx.show,
    )


def _plot_noise_comparison_visual2(ctx: PlotContext) -> None:
    from mcnlm.mc_convergence import compare_noise_estimation

    compare_noise_estimation(
        image_path=str(repo_root() / "mcnlm" / "imgs" / "clock.tiff"),
        output_path_visual=str(ctx.out_dir / "noise_comparison_visual2.pdf"),
        output_path_mse=str(ctx.out_dir / "noise_comparison_mse2.pdf"),
        output_path_psnr=str(ctx.out_dir / "noise_comparison_psnr2.pdf"),
        seed=ctx.seed,
        deterministic=ctx.deterministic,
        show=ctx.show,
    )


def _plot_window_size_mse(ctx: PlotContext) -> None:
    from mcnlm.window_size_comparison import window_size_comparison

    window_size_comparison(
        image_path=repo_root() / "mcnlm" / "imgs" / "moon.tiff",
        output_path=ctx.out_dir / "window_size_mse.pdf",
        search_radii=range(8, 51, 3),
        sampling_prob=0.5,
        resize_to=(64, 64),
        seed=ctx.seed,
        deterministic=ctx.deterministic,
    )


def _plot_hashednlm4(ctx: PlotContext) -> None:
    from mcnlm.hashnlm_viz import standard_comparison_hashed_nlm

    standard_comparison_hashed_nlm(
        str(repo_root() / "mcnlm" / "imgs" / "clock.tiff"),
        str(ctx.out_dir / "hashednlm4.pdf"),
        seed=ctx.seed,
        show=ctx.show,
        num_features=4
    )

def _plot_hashednlm6(ctx: PlotContext) -> None:
    from mcnlm.hashnlm_viz import standard_comparison_hashed_nlm

    standard_comparison_hashed_nlm(
        str(repo_root() / "mcnlm" / "imgs" / "clock.tiff"),
        str(ctx.out_dir / "hashednlm6.pdf"),
        seed=ctx.seed,
        show=ctx.show,
        num_features=6
    )

def _plot_hashednlm8(ctx: PlotContext) -> None:
    from mcnlm.hashnlm_viz import standard_comparison_hashed_nlm

    standard_comparison_hashed_nlm(
        str(repo_root() / "mcnlm" / "imgs" / "land.tiff"),
        str(ctx.out_dir / "hashednlm8.pdf"),
        seed=ctx.seed,
        show=ctx.show,
        num_features=8
    )

def _plot_hashednlm_zoomed4(ctx: PlotContext) -> None:
    from mcnlm.hashnlm_viz import standard_comparison_hashed_nlm_zoomed

    standard_comparison_hashed_nlm_zoomed(
        str(repo_root() / "mcnlm" / "imgs" / "land.tiff"),
        str(ctx.out_dir / "hashednlm_zoomed.pdf"),
        zoom_size=64,
        zoom_center=None,
        seed=ctx.seed,
        show=ctx.show,
    )

def _plot_hashednlm_zoomed6(ctx: PlotContext) -> None:
    from mcnlm.hashnlm_viz import standard_comparison_hashed_nlm_zoomed

    standard_comparison_hashed_nlm_zoomed(
        str(repo_root() / "mcnlm" / "imgs" / "land.tiff"),
        str(ctx.out_dir / "hashednlm_zoomed.pdf"),
        zoom_size=64,
        zoom_center=None,
        seed=ctx.seed,
        show=ctx.show,
        num_features=6
    )

def _plot_knn_vs_mc_spatial(ctx: PlotContext) -> None:
    from mcnlm.analysis import analyze_patch_selection

    analyze_patch_selection(
        image_path=str(repo_root() / "mcnlm" / "imgs" / "clock.tiff"),
        output_path=str(ctx.out_dir / "knn_vs_mc_spatial.pdf"),
        test_pixels_offsets=[(-30, -30), (60, -40), (0, 40)],
        patch_size=5,
        search_radius=10,
        k_neighbors=100,
        sigma=17.0,
        seed=ctx.seed,
        show=ctx.show,
    )


def _plot_methods_comparison_clock(ctx: PlotContext) -> None:
    from mcnlm.comparison import compare_all_methods

    compare_all_methods(
        image_path=str(repo_root() / "mcnlm" / "imgs" / "clock.tiff"),
        output_path=str(ctx.out_dir / "methods_comparison_clock.pdf"),
        zoom=(130, 120, 64, 64),
        sampling_prob=0.5,
        seed=ctx.seed,
        deterministic=ctx.deterministic,
        show=ctx.show,
    )


def _plot_methods_comparison_clock_2(ctx: PlotContext) -> None:
    from mcnlm.comparison import compare_all_methods

    compare_all_methods(
        image_path=str(repo_root() / "mcnlm" / "imgs" / "clock.tiff"),
        output_path=str(ctx.out_dir / "methods_comparison_clock_2.pdf"),
        zoom=(50, 150, 64, 64),
        sampling_prob=0.5,
        seed=ctx.seed,
        deterministic=ctx.deterministic,
        show=ctx.show,
    )


def _plot_knn_spatial_analysis(ctx: PlotContext) -> None:
    from mcnlm.analysis import analyze_kdtree_spatial

    analyze_kdtree_spatial(
        image_path=str(repo_root() / "mcnlm" / "imgs" / "clock.tiff"),
        output_path=str(ctx.out_dir / "knn_spatial_analysis.pdf"),
        test_pixel_offset=(0, 0),
        patch_size=5,
        search_radius=10,
        k_neighbors=100,
        sigma=17.0,
        seed=ctx.seed,
        show=ctx.show,
    )


PLOT_SPECS: tuple[PlotSpec, ...] = (
    PlotSpec(
        name="mcnlm1",
        outputs=("mcnlm1.pdf",),
        description="MCNLM zoomed results (clock)",
        default_seed=101,
        build=_plot_mcnlm1,
    ),
    PlotSpec(
        name="mcnlm2",
        outputs=("mcnlm2.pdf",),
        description="MCNLM zoomed results (man)",
        default_seed=102,
        build=_plot_mcnlm2,
    ),
    PlotSpec(
        name="mcnlm3",
        outputs=("mcnlm3.pdf",),
        description="MCNLM zoomed results (land)",
        default_seed=103,
        build=_plot_mcnlm3,
    ),
    PlotSpec(
        name="nlm_denoise1",
        outputs=("nlm_denoise1.pdf",),
        description="Naive NLM zoomed results (land)",
        default_seed=201,
        build=_plot_nlm_denoise1,
    ),
    PlotSpec(
        name="nlm_denoise2",
        outputs=("nlm_denoise2.pdf",),
        description="Naive NLM zoomed results (clock)",
        default_seed=202,
        build=_plot_nlm_denoise2,
    ),
    PlotSpec(
        name="mc_matches_1",
        outputs=("mc_matches_1.pdf",),
        description="MCNLM match visualization (single point)",
        default_seed=301,
        build=_plot_mc_matches_1,
    ),
    PlotSpec(
        name="mc_matches_2",
        outputs=("mc_matches_2.pdf",),
        description="MCNLM match visualization (multiple points)",
        default_seed=302,
        build=_plot_mc_matches_2,
    ),
    PlotSpec(
        name="robert_matches",
        outputs=("robert_matches.pdf",),
        description="MCNLM match visualization (robert points)",
        default_seed=303,
        build=_plot_robert_matches,
    ),
    PlotSpec(
        name="convergence1",
        outputs=("convergence1_mse.pdf", "convergence1_psnr.pdf"),
        description="MCNLM convergence plots (moon)",
        default_seed=401,
        build=_plot_convergence1,
    ),
    PlotSpec(
        name="convergence2",
        outputs=("convergence2_mse.pdf", "convergence2_psnr.pdf"),
        description="MCNLM convergence plots (clock)",
        default_seed=402,
        build=_plot_convergence2,
    ),
    PlotSpec(
        name="noise_comparison_visual",
        outputs=("noise_comparison_visual.pdf",),
        description="Noise estimation visual comparison (moon)",
        default_seed=501,
        build=_plot_noise_comparison_visual,
    ),
    PlotSpec(
        name="noise_comparison_visual2",
        outputs=("noise_comparison_visual2.pdf",),
        description="Noise estimation visual comparison (clock)",
        default_seed=502,
        build=_plot_noise_comparison_visual2,
    ),
    PlotSpec(
        name="window_size_mse",
        outputs=("window_size_mse.pdf",),
        description="Window size MSE comparison",
        default_seed=601,
        build=_plot_window_size_mse,
    ),
    PlotSpec(
        name="hashednlm4",
        outputs=("hashednlm.pdf",),
        description="Hashed NLM comparison",
        default_seed=701,
        build=_plot_hashednlm4,
    ),
    PlotSpec(
        name="hashednlm6",
        outputs=("hashednlm.pdf",),
        description="Hashed NLM comparison",
        default_seed=701,
        build=_plot_hashednlm6,
    ),
    PlotSpec(
        name="hashednlm8",
        outputs=("hashednlm.pdf",),
        description="Hashed NLM comparison",
        default_seed=701,
        build=_plot_hashednlm8,
    ),
    PlotSpec(
        name="hashednlm_zoomed4",
        outputs=("hashednlm_zoomed4.pdf",),
        description="Hashed NLM comparison with zoomed region",
        default_seed=702,
        build=_plot_hashednlm_zoomed4,
    ),
    PlotSpec(
        name="hashednlm_zoomed6",
        outputs=("hashednlm_zoomed6.pdf",),
        description="Hashed NLM comparison with zoomed region",
        default_seed=702,
        build=_plot_hashednlm_zoomed6,
    ),
    PlotSpec(
        name="knn_vs_mc_spatial",
        outputs=("knn_vs_mc_spatial.pdf",),
        description="k-NN locations vs local search window",
        default_seed=801,
        build=_plot_knn_vs_mc_spatial,
    ),
    PlotSpec(
        name="methods_comparison_clock",
        outputs=("methods_comparison_clock.pdf",),
        description="Compare Noisy vs MCNLM vs KD-Tree (clock)",
        default_seed=802,
        build=_plot_methods_comparison_clock,
    ),
    PlotSpec(
        name="methods_comparison_clock_2",
        outputs=("methods_comparison_clock_2.pdf",),
        description="Compare Noisy vs MCNLM vs KD-Tree (clock) zoom variant",
        default_seed=803,
        build=_plot_methods_comparison_clock_2,
    ),
    PlotSpec(
        name="knn_spatial_analysis",
        outputs=("knn_spatial_analysis.pdf",),
        description="Detailed k-NN spatial analysis plot",
        default_seed=804,
        build=_plot_knn_spatial_analysis,
    ),
)


def list_plot_specs() -> Iterable[PlotSpec]:
    return PLOT_SPECS


def _normalize_name(name: str) -> str:
    name = name.strip().lower()
    if name.endswith(".pdf"):
        name = name[:-4]
    return name


def resolve_plot_spec(name: str) -> PlotSpec:
    normalized = _normalize_name(name)
    for spec in PLOT_SPECS:
        if spec.name == normalized:
            return spec
    for spec in PLOT_SPECS:
        outputs = [_normalize_name(output) for output in spec.outputs]
        if normalized in outputs:
            return spec
    raise KeyError(f"Unknown plot '{name}'")


def run_plot(spec: PlotSpec, out_dir: Path, seed_override: int | None, deterministic: bool, show: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = seed_override if seed_override is not None else spec.default_seed
    ctx = PlotContext(out_dir=out_dir, seed=seed, deterministic=deterministic, show=show)
    spec.build(ctx)


def run_plots(
    specs: Iterable[PlotSpec],
    out_dir: Path,
    seed_override: int | None = None,
    deterministic: bool = False,
    show: bool = False,
) -> None:
    for spec in specs:
        run_plot(spec, out_dir, seed_override, deterministic, show)
