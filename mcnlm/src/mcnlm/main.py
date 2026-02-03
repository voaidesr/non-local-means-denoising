from mcnlm.mc_convergence import mc_convergence, compare_noise_estimation

from mcnlm.utils import show_mcnlm_result_zoomed, show_matches, show_nlm_result_zoomed
from mcnlm.kdtree import kdtree_nlm
import numpy as np
import sys

def results_mcnlm():
    show_mcnlm_result_zoomed(
        "imgs/land.tiff",
        probs=[0.3, 0.5, 0.8],
        zoom=(120, 100, 64, 64),
        output_path="../docs/res/mcnlm3.pdf",
    )
    
    show_mcnlm_result_zoomed(
        "imgs/clock.tiff",
        probs=[0.3, 0.5, 0.8],
        zoom=(120, 100, 64, 64),
        output_path="../docs/res/mcnlm1.pdf",
    )
    
    show_mcnlm_result_zoomed(
        "imgs/man.tiff",
        probs=[0.3, 0.5, 0.8],
        zoom=(440, 600, 64, 64),
        output_path="../docs/res/mcnlm2.pdf",
    )
    
    show_matches('imgs/clock.tiff', [(150, 210), (90, 135), (170, 80)], "../docs/res/strong_matches_p1.pdf")
    
    
def results_naive_nlm():
    show_nlm_result_zoomed(
        "imgs/land.tiff",
        zoom=(120, 100, 64, 64),
        output_path="../docs/res/nlm_denoise1.pdf",
    )
    
    show_nlm_result_zoomed(
        "imgs/clock.tiff",
        zoom=(120, 100, 64, 64),
        output_path="../docs/res/nlm_denoise2.pdf",
    )

def mc_convergence_results():
    mc_convergence(image_path='imgs/moon.tiff', output_path1='../docs/res/convergence1_mse.pdf', output_path2='../docs/res/convergence1_psnr.pdf')
    mc_convergence(image_path='imgs/clock.tiff', output_path1='../docs/res/convergence2_mse.pdf', output_path2='../docs/res/convergence2_psnr.pdf')

def noise_comparison_results():
    compare_noise_estimation(
        image_path='imgs/moon.tiff',
        output_path_visual='../docs/res/noise_comparison_visual.pdf',
        output_path_mse='../docs/res/noise_comparison_mse.pdf',
        output_path_psnr='../docs/res/noise_comparison_psnr.pdf'
    )
    
    compare_noise_estimation(
        image_path='imgs/clock.tiff',
        output_path_visual='../docs/res/noise_comparison_visual2.pdf',
        output_path_mse='../docs/res/noise_comparison_mse2.pdf',
        output_path_psnr='../docs/res/noise_comparison_psnr2.pdf'
    )

def results_kdtree_nlm():
    kdtree_nlm(
        image_path='imgs/clock.tiff',
        output_path='results/kdtree/kdtree_comparison.pdf',
    )

    kdtree_nlm(
        image_path='imgs/city.tiff',
        output_path='results/kdtree/kdtree_comparison.pdf',
    )

def main():
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'compare_noise':
            print("Running noise estimation comparison...")
            noise_comparison_results()
        elif command == 'convergence':
            print("Running MC convergence analysis...")
            mc_convergence_results()
        elif command == 'mcnlm':
            print("Running MCNLM results...")
            results_mcnlm()
        elif command == 'nlm':
            print("Running NLM results...")
            results_naive_nlm()
        elif command == 'kdtree':
            print("Running K-d tree results...")
            results_kdtree_nlm()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: compare_noise, convergence, mcnlm, nlm")
    else:
        # Default behavior
        results_naive_nlm()
        mc_convergence_results()