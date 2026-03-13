#!/bin/bash
# ============================================================
# F1 Race Predictor — Project Setup Script
# Run this from the root of your cloned repo:
#   bash setup_project.sh
# ============================================================

echo "🏎️  Setting up F1 Race Predictor project structure..."

# --- Create directories ---
mkdir -p data/raw
mkdir -p data/processed
mkdir -p notebooks
mkdir -p src/data
mkdir -p src/features
mkdir -p src/models
mkdir -p src/dashboard
mkdir -p models

# --- Create __init__.py files so Python treats src/ as a package ---
touch src/__init__.py
touch src/data/__init__.py
touch src/features/__init__.py
touch src/models/__init__.py
touch src/dashboard/__init__.py

# --- Create placeholder files ---
touch notebooks/01_data_exploration.ipynb
touch notebooks/02_feature_engineering.ipynb
touch notebooks/03_model_training.ipynb

touch src/features/engineer.py
touch src/models/train.py
touch src/models/predict.py
touch src/models/evaluate.py
touch src/dashboard/app.py
touch run_pipeline.py

# --- Create .env.example ---
cat > .env.example << 'EOF'
# No API keys required — OpenF1 and FastF1 are free and open
# Add any future config here
FASTF1_CACHE_DIR=data/raw/fastf1_cache
EOF

# --- Create .gitignore ---
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.pyo
venv/
.env

# Data (too large for git)
data/raw/
data/processed/

# Models
models/*.pkl
models/*.json

# Jupyter checkpoints
.ipynb_checkpoints/

# OS
.DS_Store
EOF

echo ""
echo "✅ Done! Your project structure is ready."
echo ""
echo "Next steps:"
echo "  1. Copy fetch_openf1.py and fetch_fastf1.py into src/data/"
echo "  2. Run: pip install -r requirements.txt"
echo "  3. Run: python src/data/fetch_openf1.py"
