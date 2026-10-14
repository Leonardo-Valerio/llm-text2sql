#!/usr/bin/env bash
# Uso: ./setup_linux.sh [cu126|cu128|cu124|cu118]   (padrao: cu126)
set -e
CU=${1:-cu126}
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/$CU
pip install -r requirements.txt
python check_env.py
