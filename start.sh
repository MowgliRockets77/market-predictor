#!/bin/bash
cd /app
sleep 3
exec python -m uvicorn backend.main:app --host 0.0.0.0 --port 8080
