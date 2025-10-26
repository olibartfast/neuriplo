#!/bin/bash
# Test script for server/client mode

set -e

echo "================================"
echo "Neuriplo Server/Client Test"
echo "================================"
echo ""

# Check if server and client are built
if [ ! -f "server/neuriplo_server" ]; then
    echo "Error: neuriplo_server not found. Build with -DBUILD_SERVER=ON"
    exit 1
fi

if [ ! -f "client/neuriplo_client_example" ]; then
    echo "Error: neuriplo_client_example not found. Build with -DBUILD_CLIENT=ON"
    exit 1
fi

# Check for model file
MODEL_FILE="${1:-resnet50-v2-7.onnx}"
if [ ! -f "$MODEL_FILE" ]; then
    echo "Model file not found: $MODEL_FILE"
    echo "Downloading sample ONNX model..."
    wget -q https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx
    MODEL_FILE="resnet50-v2-7.onnx"
fi

# Check for test image
TEST_IMAGE="${2:-space_shuttle.jpg}"
if [ ! -f "$TEST_IMAGE" ]; then
    echo "Test image not found: $TEST_IMAGE"
    echo "Downloading sample image..."
    wget -q https://raw.githubusercontent.com/opencv/opencv/master/samples/data/space_shuttle.jpg
    TEST_IMAGE="space_shuttle.jpg"
fi

echo "Using model: $MODEL_FILE"
echo "Using image: $TEST_IMAGE"
echo ""

# Start server in background
echo "Starting server..."
./server/neuriplo_server --model "$MODEL_FILE" --port 8080 > server.log 2>&1 &
SERVER_PID=$!

# Wait for server to start
echo "Waiting for server to start..."
sleep 3

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "Error: Server failed to start"
    cat server.log
    exit 1
fi

echo "Server started (PID: $SERVER_PID)"
echo ""

# Test health endpoint
echo "Testing health endpoint..."
HEALTH=$(curl -s http://localhost:8080/health)
if [ $? -eq 0 ]; then
    echo "✓ Health check passed"
    echo "  Response: $HEALTH"
else
    echo "✗ Health check failed"
    kill $SERVER_PID
    exit 1
fi
echo ""

# Test model info endpoint
echo "Testing model info endpoint..."
MODEL_INFO=$(curl -s http://localhost:8080/model_info)
if [ $? -eq 0 ]; then
    echo "✓ Model info retrieved"
    echo "  Response: ${MODEL_INFO:0:100}..."
else
    echo "✗ Model info failed"
    kill $SERVER_PID
    exit 1
fi
echo ""

# Test client
echo "Testing client..."
./client/neuriplo_client_example --image "$TEST_IMAGE" --server localhost --port 8080 > client.log 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Client inference successful"
    grep -E "(Inference completed|Results)" client.log | head -5
else
    echo "✗ Client inference failed"
    cat client.log
    kill $SERVER_PID
    exit 1
fi
echo ""

# Test stats endpoint
echo "Testing stats endpoint..."
STATS=$(curl -s http://localhost:8080/stats)
if [ $? -eq 0 ]; then
    echo "✓ Stats retrieved"
    echo "  Response: $STATS"
else
    echo "✗ Stats failed"
fi
echo ""

# Cleanup
echo "Stopping server..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "================================"
echo "All tests passed! ✓"
echo "================================"
echo ""
echo "Server/client mode is working correctly."
echo ""
echo "Next steps:"
echo "  - Start server: ./server/neuriplo_server --model $MODEL_FILE"
echo "  - Run client: ./client/neuriplo_client_example --image $TEST_IMAGE"
echo "  - See docs: ../docs/SERVER_CLIENT_USAGE.md"
echo ""
