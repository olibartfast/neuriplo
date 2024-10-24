#!/bin/bash
docker run -v $(pwd):/workspace pytorch/pytorch:latest /bin/bash -cx "pip install onnx && python export_torchvision_classifier.py"