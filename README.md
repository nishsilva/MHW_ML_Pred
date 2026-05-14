# MHW_IRP_Project

**Prediction of Marine Heatwaves and Their Impact on Ocean Productivity using Deep Learning**

This repository contains code and notebooks for identifying, forecasting, and analyzing Marine Heatwaves (MHWs) using satellite-derived Sea Surface Temperature (SST) and Chlorophyll-a (Chl-a) data. The project focuses on four regions in the northern Indian Ocean:

- Bay of Bengal
- Gulf of Mannar
- Laccadive Sea
- Sri Lankan Coast

Deep learning models such as LSTM, CNN, and Transformer architectures are used to forecast MHW events and study their effects on ocean productivity.

## Contents

- `Data_chl_2020/` - Region-wise Chlorophyll-a data files
- `Data_sst_2020/` - Region-wise Sea Surface Temperature data files
- `requirements.txt` - Python dependencies for the project

## Notebooks and Scripts

- `exploratory_data_analysis.ipynb` - General EDA for SST and Chl-a datasets
- `exploratory_data_analysis_script.py` - Reusable EDA helper functions
- `exploratory_data_analysis_mhw.ipynb` - EDA focused on Marine Heatwave detection
- `exploratory_data_analysis_mhw_script.py` - Scripts for MHW EDA utilities
- `marineHeatWaves.py` - Marine heatwave detection, labeling, and analysis utilities (based on Hobday et al., 2016)
- `mhw_model_CNN.ipynb` - CNN-based SST forecasting for MHWs
- `mhw_model_LSTM.ipynb` - LSTM-based SST forecasting for MHWs
- `mhw_model_Transformer.ipynb` - Transformer-based SST forecasting for MHWs
- `chl_model_CNN.ipynb` - CNN-based Chlorophyll-a prediction
- `chl_model_LSTM.ipynb` - LSTM-based Chlorophyll-a prediction
- `model_build_marineheatwaves.py` - Functions and model-building utilities for MHW forecasting
- `model_build_chlorophyll.py` - Functions and model-building utilities for Chlorophyll-a forecasting
- `mhw_chl_impact.py` - Scripts for analyzing MHW impacts on Chlorophyll-a
- `mhw_impact_on_chl_EDA.ipynb` - EDA for MHW impacts on Chl-a

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run notebooks or scripts from the project root.

## Usage

- Use the EDA notebooks to explore the datasets and understand MHW patterns.
- Run `marineHeatWaves.py` to generate MHW detection labels and summary statistics.
- Train forecasting models using the notebooks and helper scripts in `model_build_*`.
- Analyze the impact of MHW events on Chlorophyll-a using `mhw_chl_impact.py` and the related notebook.

## Notes

- Keep the `Data_chl_2020/` and `Data_sst_2020/` directories populated with the required NetCDF files before running the notebooks.
- The project uses region-averaged time series and deep learning models to support forecast experiments.

## License

This project is available for research and educational use.
