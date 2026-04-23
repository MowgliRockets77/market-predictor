#!/bin/bash
cd /app
exec python -m uvicorn backend.main:app --host 0.0.0.0 --port 8080
