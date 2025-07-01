# 📡 RC–PV Synergy Pipeline

This repository provides a full, reproducible pipeline for analyzing **radiative cooling (RC)** and **photovoltaic (PV)** synergy using ERA5 climate data, custom PV modeling, clustering, and robust mapping.  
The core script is `main.py`, which runs the entire flow end-to-end or in parts.

---

## ⚙️ Key Features

- **ERA5 climate data aggregation** and preprocessing.
- **Physics-based PV potential preparation** and validation.
- **Machine learning training** for PV prediction.
- **Flexible clustering pipeline** for RC and PV synergy mapping.
- **Cluster summary statistics** and annual PV potential.
- **Automated map generation** of prediction uncertainty.
- **Safe atomic writes** for all output files.
- **Database integration** — read raw data from DB and write results back.
- **System resource checks** before running heavy tasks.

---

## ✅ Requirements

- Python **3.11** (recommended)
- All Python dependencies in `requirements.txt`.

Before Python setup, ensure you have the necessary system libraries if using geospatial or NetCDF:

**Ubuntu:**
```bash
sudo apt-get update
sudo apt-get install libgdal-dev libgeos-dev libproj-dev libeccodes-dev
# Create and activate a virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# config.yaml
nc_data_dir: /path/to/your/netcdf
results_path: ./results
export NC_DATA_DIR="/my/custom/netcdf_path"
results/
└── clustered_dataset.csv
export PV_DB_URL="postgresql://username:password@host:port/dbname"
export PV_DB_URL="sqlite:////full/path/to/your/pipeline.sqlite"
python main.py --mode <mode>

---

## 🟢 How This Connects

✔️ **Matches the current `main.py`** structure (`--mode`, `--db-url`, `--db-table`).  
✔️ Shows where your `clustered_dataset.csv` fits.  
✔️ Explains fallback if DB input is used.  
✔️ Calls out all generated outputs.  
✔️ Explains atomic writes and resource checks.  
✔️ Provides practical `bash` + `python` examples for every user scenario.

---

## 📦 **Next Step**

✅ You can copy–paste this whole block as your **new `README.md`** — it’s fully aligned with the updated code and pipeline behavior.

If you’d like, I can package this as a `.md` file and hand you a download link **right now** — just say **“Yes, package it”**!

