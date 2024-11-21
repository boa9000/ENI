# Techno-Economic Evaluation of Combined Energy Storage Technologies in Neighborhoods

## Overview

This project explores the techno-economic feasibility of integrating photovoltaic (PV) systems with multiple energy storage technologies in residential neighborhoods. It aims to enhance energy self-sufficiency and economic viability by combining:

- **Battery Energy Storage Systems (BESS)**
- **Thermal Energy Storage Systems (TESS)**
- **Hydrogen Energy Storage Systems (HESS)**

Using a 20-year simulation model, the project evaluates the cost-effectiveness of these storage methods in conjunction with surplus PV energy production, using typical meteorological year (TMY) data and diverse household load profiles.

## Features

- **Hybrid Storage System Simulation**: Models the performance of BESS, TESS, and HESS in residential setups.
- **Particle Swarm Optimization (PSO)**: Optimizes the configuration of PV systems and storage capacities for maximum economic return.
- **Detailed Economic Analysis**: Calculates Net Present Value (NPV) for the integrated system versus grid reliance.
- **Energy Flow Management**: Implements a control algorithm to prioritize charging and discharging across different storage systems.

### Codebase
- **`pvcalc.py`**: Contains the simulation and optimization logic. Key functionalities include:
  - PV system modeling using meteorological data.
  - Rule-based control algorithms for energy distribution.
  - Particle Swarm Optimization for finding optimal storage configurations.
  
  Note: for PSO, the code is set to run on multiple cores with the concurrent library. Remove it if you want to avoid slowing down your system.


## Usage

1. **Input Data**:
   - Place your TMY data file (`Munich_weather.csv`) in the appropriate directory.
   - Ensure household load profiles and economic parameters are correctly formatted in the `eni_seminar/Data` folder.

2. **Outputs**:
   The script generates a DataFrame containing:
   - State of Charge (SOC) for all storage systems.
   - Energy flows (thermal and electrical).
   - And most importantly, the optimized storage capacities and economic outcomes.


### Research Paper
- The research paper, *"Techno-Economic Evaluation of Combined Energy Storage Technologies in Neighborhoods"*, written as part of "Energy Informatics Seminar", provides the theoretical background and methodology:
  - Data preprocessing from TMY and household profiles.
  - Detailed equations governing PV, BESS, TESS, and HESS operations.
  - Economic parameters for assessing feasibility.
You can request it from me if you are interested in reading it.
 


