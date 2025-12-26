#!/bin/bash
# start.sh
echo "🚀 Starting Crypto Trading API..."
echo "Current directory: $(pwd)"
echo "Directory contents:"
ls -la
echo ""
echo "API directory:"
ls -la api/
echo ""
echo "Starting server..."

# Change to api directory and start
cd api
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}