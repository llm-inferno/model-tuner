# Simulation Plotting Guide

This guide explains how to use the plotting script to visualize the output from the simulated observer demo.

## Prerequisites

1. **Python 3.7 or higher**
2. **Required Python packages**: matplotlib and numpy

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install matplotlib numpy
```

## Usage

Run the plotting script from the project root directory:

```bash
python3 plot_simulation.py
```

Or make it executable and run it directly:

```bash
chmod +x plot_simulation.py
./plot_simulation.py
```

## What It Does

The script performs the following steps:

1. **Runs the simulation**: Executes `demos/simulated-observer/main.go`
2. **Parses the output**: Extracts X (state vector) and Delta (innovation) values from each step
3. **Creates visualizations**: Generates two plots:
   - **Top plot**: State vector X evolution over time
   - **Bottom plot**: Innovation Delta evolution over time
4. **Saves the plot**: Saves the visualization as `simulation_plot.png` (300 DPI)
5. **Displays the plot**: Opens an interactive window to view the results

## Understanding the Plots

### Phase Boundaries

The simulation runs in three phases (20 steps each):

- **Phase 1** (steps 0-19): Initial configuration
- **Phase 2** (steps 20-39): Modified parameters
- **Phase 3** (steps 40-59): Final configuration

Red dashed lines indicate phase transitions.

### State Vector (X)

The state vector represents the tuner's internal state parameters, which may include:

- Batch size configurations
- Token processing parameters
- System performance metrics

### Innovation (Delta)

The innovation vector represents the prediction error or "surprise" in the tuner's observations, indicating how much new information is being incorporated at each step.

## Output Files

- `simulation_plot.png`: High-resolution plot of the simulation results

## Troubleshooting

### "go: command not found"

Make sure Go is installed and in your PATH.

### "No simulation data found in output"

This could mean the Go program failed to run. Try running the simulation manually first:

```bash
go run demos/simulated-observer/main.go
```

### Module import errors

Ensure all Python dependencies are installed:

```bash
pip install -r requirements.txt
```
