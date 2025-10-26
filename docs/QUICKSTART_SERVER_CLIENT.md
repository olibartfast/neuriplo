# Quick Start: Server/Client Mode

This guide will get you up and running with Neuriplo's server/client mode in 5 minutes.

## Prerequisites

- CMake 3.10+
- C++17 compiler
- OpenCV
- glog

## Step 1: Build Server and Client

```bash
cd /path/to/neuriplo
mkdir build && cd build

# Build with ONNX Runtime backend and server/client support
cmake -DBUILD_SERVER=ON \
      -DBUILD_CLIENT=ON \
      -DDEFAULT_BACKEND=ONNX_RUNTIME \
      ..

make -j$(nproc)
```

This creates:
- `server/neuriplo_server` - The inference server
- `client/neuriplo_client_example` - Example client application

## Step 2: Start the Server

In one terminal:

```bash
cd build/server

# Download a sample ONNX model (ResNet50 from ONNX Model Zoo)
wget https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx

# Start server
./neuriplo_server --model resnet50-v2-7.onnx --port 8080
```

You should see:
```
I1026 12:00:00.000 neuriplo_server.cpp:95] Starting Neuriplo Inference Server
I1026 12:00:00.001 neuriplo_server.cpp:100] Backend initialized successfully
I1026 12:00:00.002 neuriplo_server.cpp:110] Server ready to accept connections
I1026 12:00:00.002 neuriplo_server.cpp:112] POST http://0.0.0.0:8080/infer
```

## Step 3: Test with curl

In another terminal:

```bash
# Check server health
curl http://localhost:8080/health

# Should return:
# {"status":"healthy","gpu_available":false,"model_path":"resnet50-v2-7.onnx","total_requests":0}

# Get model info
curl http://localhost:8080/model_info
```

## Step 4: Run Client Example

```bash
cd build/client

# Download a test image
wget https://raw.githubusercontent.com/opencv/opencv/master/samples/data/space_shuttle.jpg

# Run client
./neuriplo_client_example --image space_shuttle.jpg --server localhost --port 8080
```

You should see inference results:
```
I1026 12:01:00.000 neuriplo_client_example.cpp:50] Starting Neuriplo Client
I1026 12:01:00.001 InferenceClient.hpp:25] Client connected to server
I1026 12:01:00.100 neuriplo_client_example.cpp:80] Inference completed in 95ms
I1026 12:01:00.100 neuriplo_client_example.cpp:90] Results: ...
```

## Step 5: Use in Your Code

### C++ Example

```cpp
#include "InferenceBackendSetup.hpp"
#include <opencv2/opencv.hpp>

int main() {
    // Create client
    auto client = setup_inference_engine(
        InferenceMode::CLIENT,
        "localhost",  // server host
        false, 1, {}, // ignored for client
        8080          // server port
    );
    
    // Load image
    cv::Mat image = cv::imread("test.jpg");
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0/255.0, cv::Size(224, 224));
    
    // Run inference
    auto [outputs, shapes] = client->get_infer_results(blob);
    
    return 0;
}
```

Compile:
```bash
g++ -std=c++17 my_app.cpp \
    -I/path/to/neuriplo/include \
    -I/path/to/neuriplo/backends/src \
    -I/path/to/neuriplo/client/src \
    -I/path/to/neuriplo/server/src \
    -L/path/to/neuriplo/build \
    -lneuriplo -lopencv_core -lopencv_imgproc -lglog \
    -o my_app
```

### Python Client Example

```python
import requests
import cv2
import base64
import numpy as np

# Load image
image = cv2.imread("test.jpg")
image = cv2.resize(image, (224, 224))

# Encode as PNG and base64
_, buffer = cv2.imencode('.png', image)
image_b64 = base64.b64encode(buffer).decode('utf-8')

# Send request
response = requests.post(
    "http://localhost:8080/infer",
    json={"image": image_b64},
    timeout=30
)

# Parse results
result = response.json()
outputs = result['outputs'][0]['data']
print(f"Top prediction: {np.argmax(outputs)}")
print(f"Inference time: {result['inference_time_ms']}ms")
```

## Next Steps

- **Multiple clients**: Start the client from different machines pointing to the same server
- **GPU acceleration**: Add `--gpu` flag when starting the server
- **Different backends**: Rebuild with `-DDEFAULT_BACKEND=LIBTORCH` or other backends
- **Production deployment**: See [full documentation](SERVER_CLIENT_USAGE.md) for security and performance tips

## Troubleshooting

**Server won't start?**
- Check if port 8080 is free: `lsof -i :8080`
- Verify model file exists

**Client can't connect?**
- Check server is running: `curl http://localhost:8080/health`
- Try `--server 127.0.0.1` instead of `localhost`

**Need help?**
- See [full documentation](SERVER_CLIENT_USAGE.md)
- Check server logs for errors
- Open an issue on GitHub

## Comparison: OFFLINE vs CLIENT Mode

```cpp
// OFFLINE Mode (local inference)
auto engine = setup_inference_engine(
    "model.onnx",  // local model path
    true,          // use GPU
    1              // batch size
);

// CLIENT Mode (remote inference)
auto engine = setup_inference_engine(
    InferenceMode::CLIENT,
    "192.168.1.100",  // server IP
    false, 1, {},     // ignored
    8080              // port
);

// Same API for both!
auto [outputs, shapes] = engine->get_infer_results(blob);
```

Choose OFFLINE when:
- Running on the same machine as the model
- Need lowest possible latency
- No network available

Choose CLIENT when:
- Centralizing GPU resources
- Multiple applications need inference
- Client machines have limited resources
- Deploying at scale
