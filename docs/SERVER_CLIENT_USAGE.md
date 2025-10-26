# Server/Client Mode Usage Guide

## Overview

Neuriplo now supports two operation modes:

1. **OFFLINE Mode** (Default): Traditional local inference where the model runs on the same machine as your application
2. **CLIENT Mode** (New): Remote inference where a server hosts the model and clients connect via HTTP to perform inference

This architecture allows you to:
- Centralize model deployment and management
- Scale inference across multiple client applications
- Reduce client-side resource requirements
- Support heterogeneous client platforms

## Architecture

```
OFFLINE Mode:
┌─────────────────────────────────────┐
│  Your Application                   │
│  ┌────────────────────────────────┐ │
│  │ Neuriplo (Local Backend)       │ │
│  │ - ONNX / LibTorch / TensorRT   │ │
│  │ - GPU / CPU                     │ │
│  └────────────────────────────────┘ │
└─────────────────────────────────────┘

CLIENT Mode:
┌─────────────────────┐         ┌──────────────────────────────┐
│  Client Machine     │         │  Server Machine              │
│  ┌────────────────┐ │  HTTP   │  ┌─────────────────────────┐ │
│  │ Your App +     │ │ ───────▶│  │ Neuriplo Server         │ │
│  │ Neuriplo Client│ │ ◀───────│  │ - Backend (ONNX/etc)    │ │
│  └────────────────┘ │         │  │ - GPU / CPU             │ │
└─────────────────────┘         │  └─────────────────────────┘ │
                                └──────────────────────────────┘
```

## Building with Server/Client Support

### Prerequisites

- C++17 compiler
- OpenCV
- glog
- CMake 3.10+
- Dependencies will be automatically fetched:
  - [cpp-httplib](https://github.com/yhirose/cpp-httplib) - HTTP server/client
  - [nlohmann/json](https://github.com/nlohmann/json) - JSON parsing

### Build Commands

```bash
# Build with server support
cmake -DBUILD_SERVER=ON -DDEFAULT_BACKEND=ONNX_RUNTIME ..
make

# Build with client support
cmake -DBUILD_CLIENT=ON -DDEFAULT_BACKEND=ONNX_RUNTIME ..
make

# Build both server and client
cmake -DBUILD_SERVER=ON -DBUILD_CLIENT=ON -DDEFAULT_BACKEND=ONNX_RUNTIME ..
make
```

This will create:
- `neuriplo_server` - Standalone inference server executable
- `neuriplo_client_example` - Example client application
- Libraries for integration into your projects

## Server Usage

### Starting the Server

Basic usage:
```bash
./neuriplo_server --model path/to/model.onnx
```

Advanced options:
```bash
./neuriplo_server \
  --model path/to/model.onnx \
  --host 0.0.0.0 \
  --port 8080 \
  --gpu \
  --batch-size 4
```

Options:
- `--model PATH`: Path to the model file (required)
- `--host HOST`: Server bind address (default: 0.0.0.0)
- `--port PORT`: Server port (default: 8080)
- `--gpu`: Enable GPU acceleration
- `--batch-size SIZE`: Batch size for inference (default: 1)

### Server Endpoints

Once running, the server exposes the following REST API:

#### 1. Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "model_path": "/path/to/model.onnx",
  "total_requests": 42
}
```

#### 2. Model Information
```bash
GET /model_info
```

Response:
```json
{
  "inputs": [
    {
      "name": "input",
      "shape": [3, 224, 224],
      "batch_size": 1
    }
  ],
  "outputs": [
    {
      "name": "output",
      "shape": [1000],
      "batch_size": 1
    }
  ]
}
```

#### 3. Inference
```bash
POST /infer
Content-Type: application/json

{
  "image": "<base64-encoded-png>"
}
```

Response:
```json
{
  "outputs": [
    {
      "data": [0.123, 0.456, ...],
      "shape": [1000],
      "type": "float"
    }
  ],
  "inference_time_ms": 15.3,
  "total_time_ms": 18.7
}
```

#### 4. Statistics
```bash
GET /stats
```

Response:
```json
{
  "total_requests": 1000,
  "failed_requests": 5,
  "success_rate": 99.5,
  "total_inferences": 1000,
  "avg_inference_time_ms": 15.2,
  "memory_usage_mb": 512
}
```

## Client Usage

### Using the Example Client

```bash
./neuriplo_client_example \
  --image test.jpg \
  --server localhost \
  --port 8080
```

Options:
- `--image PATH`: Path to input image (required)
- `--server HOST`: Server hostname or IP (default: localhost)
- `--port PORT`: Server port (default: 8080)

### Integrating Client into Your Code

#### C++ Integration

```cpp
#include "InferenceBackendSetup.hpp"
#include <opencv2/opencv.hpp>

int main() {
    // Create client (connects to server)
    auto client = setup_inference_engine(
        InferenceMode::CLIENT,
        "192.168.1.100",  // server host
        false,            // use_gpu (ignored for client)
        1,                // batch_size (ignored for client)
        {},               // input_sizes (ignored for client)
        8080              // server port
    );
    
    // Load and preprocess image
    cv::Mat image = cv::imread("test.jpg");
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0/255.0, 
                          cv::Size(224, 224));
    
    // Run inference (automatically sent to server)
    auto [outputs, shapes] = client->get_infer_results(blob);
    
    // Process results
    for (size_t i = 0; i < outputs.size(); ++i) {
        std::cout << "Output " << i << ": " 
                  << outputs[i].size() << " elements\n";
    }
    
    return 0;
}
```

#### Switching Between OFFLINE and CLIENT Modes

```cpp
// OFFLINE mode (local inference)
auto engine = setup_inference_engine(
    InferenceMode::OFFLINE,
    "model.onnx",     // local model path
    true,             // use GPU
    1,                // batch size
    {}                // input sizes
);

// CLIENT mode (remote inference)
auto engine = setup_inference_engine(
    InferenceMode::CLIENT,
    "192.168.1.100",  // server host
    false,            // ignored for client
    1,                // ignored for client
    {},               // ignored for client
    8080              // server port
);

// Same interface for both!
auto [outputs, shapes] = engine->get_infer_results(blob);
```

## Example Workflow

### 1. Start Server on GPU Machine

```bash
# On server (e.g., 192.168.1.100 with GPU)
./neuriplo_server \
  --model resnet50.onnx \
  --host 0.0.0.0 \
  --port 8080 \
  --gpu \
  --batch-size 8
```

### 2. Run Clients from Multiple Machines

```bash
# On client machine 1
./neuriplo_client_example \
  --image photo1.jpg \
  --server 192.168.1.100 \
  --port 8080

# On client machine 2
./neuriplo_client_example \
  --image photo2.jpg \
  --server 192.168.1.100 \
  --port 8080
```

## Testing the Server

### Using curl

```bash
# Health check
curl http://localhost:8080/health

# Model info
curl http://localhost:8080/model_info

# Inference (with base64-encoded image)
curl -X POST http://localhost:8080/infer \
  -H "Content-Type: application/json" \
  -d '{
    "image": "'$(base64 -w 0 test.jpg)'"
  }'

# Statistics
curl http://localhost:8080/stats
```

### Using Python

```python
import requests
import cv2
import base64
import json

# Load image
image = cv2.imread("test.jpg")
_, buffer = cv2.imencode('.png', image)
image_base64 = base64.b64encode(buffer).decode('utf-8')

# Send inference request
response = requests.post(
    "http://localhost:8080/infer",
    json={"image": image_base64}
)

results = response.json()
print(f"Inference time: {results['inference_time_ms']}ms")
print(f"Output shape: {results['outputs'][0]['shape']}")
```

## Performance Considerations

### Network Latency
- Client mode adds network overhead (~5-50ms depending on network)
- Best for scenarios where inference time >> network latency
- Use local networks for best performance

### Image Compression
- Images are PNG-compressed and base64-encoded
- Typical overhead: ~30% for base64, but PNG compression helps
- Consider JPEG for larger images if quality loss is acceptable

### Batch Processing
- Server supports batch processing (use `--batch-size`)
- Clients can send individual requests concurrently
- Server handles multiple clients simultaneously

### GPU Utilization
- Server mode allows better GPU utilization from multiple clients
- Recommended for expensive models (e.g., large transformers, high-res detection)

## Security Considerations

**Important**: The current implementation is designed for trusted networks only.

For production deployments, consider:
- Adding authentication (API keys, OAuth)
- Using HTTPS/TLS encryption
- Implementing rate limiting
- Adding request validation
- Using a reverse proxy (nginx, traefik)

Example with nginx:
```nginx
server {
    listen 443 ssl;
    server_name inference.example.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Troubleshooting

### Server won't start
- Check if port is already in use: `lsof -i :8080`
- Verify model file exists and is readable
- Check backend is properly compiled

### Client can't connect
- Verify server is running: `curl http://server:8080/health`
- Check firewall rules
- Ensure correct host/port in client config

### Slow inference
- Monitor server stats: `curl http://server:8080/stats`
- Check GPU utilization on server
- Consider increasing batch size for throughput
- Profile network latency

### Out of memory
- Reduce batch size
- Use smaller models
- Monitor memory: `curl http://server:8080/stats`

## Future Enhancements

Planned features for future releases:
- gRPC support for better performance
- Streaming inference for video
- Model hot-swapping without restart
- Load balancing across multiple servers
- Authentication and authorization
- Prometheus metrics export
- Docker containerization

## See Also

- [Main README](../Readme.md)
- [Backend Documentation](../backends/README.md)
- [Dependency Management](DEPENDENCY_MANAGEMENT.md)
