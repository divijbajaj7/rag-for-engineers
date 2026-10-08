#!/usr/bin/env bash
# One-command setup for students.
#   Usage:  cd code && bash setup.sh
# Creates a virtualenv, installs dependencies, scaffolds .env, and fetches the
# Kubernetes docs used by M0. Safe to re-run.
set -euo pipefail

cd "$(dirname "$0")"   # the code/ directory

PY="${PYTHON:-python3}"

echo "==> Creating virtualenv (.venv)"
if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip"
python -m pip install --quiet --upgrade pip

echo "==> Installing requirements (first run downloads models later, be patient)"
pip install --quiet -r requirements.txt

if [ ! -f .env ]; then
  echo "==> Creating .env from .env.example  (now add ONE API key)"
  cp .env.example .env
else
  echo "==> .env already exists, leaving it as-is"
fi

echo "==> Downloading the Kubernetes docs for M0"
python download_data.py

cat <<'DONE'

✅ Setup complete.

Next:
  1. Open code/.env and paste ONE key:
       OPENAI_API_KEY=sk-...            (OpenAI)
       or
       OPENROUTER_API_KEY=sk-or-v1-...  (OpenRouter)

  2. Activate the venv in each new shell:
       source code/.venv/bin/activate

  3. Run the modules:
       cd code/scripts
       python m0_naive.py
       python m2_hybrid.py
       streamlit run m3_myntra_chatbot.py
DONE
