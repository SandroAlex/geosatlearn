#!/bin/bash
# Initialize chroma vector data base.

# If any of the commands in your code fails for any reason, the entire script fails.
set -o errexit

# Fail exit if one of your pipe command fails.
set -o pipefail

# Exits if any of your variables is not set.
set -o nounset

# Starts the FastAPI application in development mode.
fastapi dev --host 0.0.0.0 --port 8000 api/main.py