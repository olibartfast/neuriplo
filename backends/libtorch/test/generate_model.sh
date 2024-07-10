#!/bin/bash
docker run -v $(pwd):/workspace pytorch/pytorch:latest /bin/bash -cx "python export_torchvision_classifier.py "