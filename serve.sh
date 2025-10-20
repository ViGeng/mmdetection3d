#!/bin/bash
# Simple HTTP server to view detection results
# Usage: ./serve.sh [port]

PORT=${1:-8000}
HOSTNAME=$(hostname)

echo "========================================"
echo "Starting HTTP Server"
echo "========================================"
echo ""
echo "📊 View visualization at:"
echo "  http://${HOSTNAME}:${PORT}/detection_results.html"
echo ""
echo "💡 If hostname doesn't work, use the server's IP address instead"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python3 -m http.server $PORT
