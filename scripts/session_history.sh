#!/usr/bin/env bash
set -u -o pipefail

cd /workspace/
git clone --recursive https://github.com/plasmax/GeometryCrafter
cd GeometryCrafter/
git fetch --all
git checkout max
git config user.name "Max Last"
git config user.email "tsalxam@gmail.com"

python -m venv .venv
source .venv/bin/activate

pip install uv
uv pip install -r requirements.txt 

# runpodctl receive 4080-viva-respond-eternal-8

history > session_history.sh
