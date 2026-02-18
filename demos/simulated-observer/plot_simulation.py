#!/usr/bin/env python3
"""
Plot X and Delta values from the simulated observer demo.
This script runs the Go simulation and visualizes the results.
"""

import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_simulation():
    """Run the Go simulation and capture output."""
    print("Running simulation...")
    demo_dir = '.'

    result = subprocess.run(
        ["go", "run", "main.go"],
        capture_output=True,
        text=True,
        cwd=demo_dir
    )

    if result.returncode != 0:
        print(f"Error running simulation:\n{result.stderr}")
        return None

    return result.stdout


def parse_output(output):
    """Parse the simulation output to extract X and Delta values."""
    steps = []
    x_values = []
    delta_values = []

    # Pattern to match lines like: "0 : X=[...];   Delta=[...];   P=[[...]]"
    pattern = r'(\d+)\s*:\s*X=\[([\d\s.E+-]+)\]\s*;\s*Delta=\[([\d\s.E+-]+)\]'

    for line in output.split('\n'):
        match = re.search(pattern, line)
        if match:
            step = int(match.group(1))
            x_str = match.group(2)
            delta_str = match.group(3)

            # Parse the numeric values
            x_vals = [float(v) for v in x_str.split()]
            delta_vals = [float(v) for v in delta_str.split()]

            steps.append(step)
            x_values.append(x_vals)
            delta_values.append(delta_vals)

    return steps, x_values, delta_values


def plot_results(steps, x_values, delta_values):
    """Create plots for X and Delta values."""
    if not steps:
        print("No data to plot!")
        return

    steps = np.array(steps)
    x_values = np.array(x_values)
    delta_values = np.array(delta_values)

    # Determine dimensions
    num_x = x_values.shape[1] if len(x_values.shape) > 1 else 1
    num_delta = delta_values.shape[1] if len(delta_values.shape) > 1 else 1

    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Simulated Observer - Tuner State Evolution', fontsize=16, fontweight='bold')

    # Plot X values
    xLegend = ["alpha", "beta (x1E2)", "gamma (x1E4)", "delta"]
    ax1 = axes[0]
    for i in range(num_x):
        # ax1.plot(steps, x_values[:, i], marker='o', label=f'X[{i}]', linewidth=2, markersize=4)
        ax1.plot(steps, (100**i)*x_values[:, i], marker='o', label=xLegend[i], linewidth=2, markersize=4)

    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('X Values (State Vector)', fontsize=12)
    ax1.set_title('State Vector X over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Add phase boundaries
    ax1.axvline(x=20, color='red', linestyle='--', alpha=0.5, label='Phase 1→2')
    ax1.axvline(x=40, color='red', linestyle='--', alpha=0.5, label='Phase 2→3')

    # Plot Delta values
    dLegend = ["Delta TTFT", "Delta ITL"]
    ax2 = axes[1]
    for i in range(num_delta):
        # ax2.plot(steps, delta_values[:, i], marker='s', label=f'Delta[{i}]', linewidth=2, markersize=4)
        ax2.plot(steps, delta_values[:, i], marker='s', label=dLegend[i], linewidth=2, markersize=4)

    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Delta Values in msec (Innovation)', fontsize=12)
    ax2.set_title('Innovation Delta over Time', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Add phase boundaries
    ax2.axvline(x=20, color='red', linestyle='--', alpha=0.5, label='Phase 1→2')
    ax2.axvline(x=40, color='red', linestyle='--', alpha=0.5, label='Phase 2→3')

    plt.tight_layout()

    # Save the plot
    output_file = 'simulation_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    # Show the plot
    plt.show()


def main():
    """Main function to run simulation and create plots."""
    # Run the simulation
    output = run_simulation()
    if output is None:
        return

    # Parse the output
    steps, x_values, delta_values = parse_output(output)

    if not steps:
        print("No simulation data found in output!")
        return

    print(f"Parsed {len(steps)} steps of simulation data")
    print(f"X dimensions: {len(x_values[0]) if x_values else 0}")
    print(f"Delta dimensions: {len(delta_values[0]) if delta_values else 0}")

    # Create plots
    plot_results(steps, x_values, delta_values)


if __name__ == "__main__":
    main()
