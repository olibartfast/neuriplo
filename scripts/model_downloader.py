#!/usr/bin/env python3
"""
Dynamic Model Downloader and Converter for InferenceEngines Testing
Downloads and converts models on-the-fly for each backend during testing.
"""

import os
import sys
import tempfile
import shutil
import subprocess
import urllib.request
import logging
from pathlib import Path
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class ModelDownloader:
    """Downloads and converts models for different backends."""
    
    def __init__(self, temp_dir=None):
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="inference_models_")
        self.downloaded_files = []
        logger.info(f"Using temporary directory: {self.temp_dir}")
    
    def cleanup(self):
        """Clean up all downloaded files and temporary directory."""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up: {e}")
    
    def download_file(self, url, filename):
        """Download a file from URL to temporary directory."""
        filepath = os.path.join(self.temp_dir, filename)
        
        if os.path.exists(filepath):
            logger.info(f"File already exists: {filename}")
            return filepath
        
        try:
            logger.info(f"Downloading {filename} from {url}")
            urllib.request.urlretrieve(url, filepath)
            self.downloaded_files.append(filepath)
            logger.info(f"Downloaded: {filename} ({os.path.getsize(filepath)} bytes)")
            return filepath
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            return None
    
    def generate_onnx_resnet18(self):
        """Generate ResNet-18 ONNX model using PyTorch."""
        output_path = os.path.join(self.temp_dir, "resnet18.onnx")
        
        if os.path.exists(output_path):
            logger.info("ONNX ResNet-18 already exists")
            return output_path
        
        try:
            import torch
            import torchvision.models as models
            
            logger.info("Generating ResNet-18 ONNX model...")
            model = models.resnet18(weights='IMAGENET1K_V1')  # Updated weights parameter
            model.eval()
            
            example_input = torch.rand(1, 3, 224, 224)
            
            torch.onnx.export(
                model,
                example_input,
                output_path,
                opset_version=12,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            logger.info(f"Generated ONNX model: {output_path}")
            return output_path
            
        except ImportError:
            logger.warning("PyTorch not available, trying to download pre-built ONNX model")
            return self.download_pretrained_onnx()
        except Exception as e:
            logger.warning(f"Failed to generate ONNX model with PyTorch: {e}")
            logger.info("Trying to download pre-built ONNX model...")
            return self.download_pretrained_onnx()
    
    def download_pretrained_onnx(self):
        """Download a pre-built ONNX ResNet-18 model."""
        # Try multiple sources for ONNX ResNet-18
        urls = [
            # ONNX Model Zoo
            "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet18-v1-7.onnx",
            "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet18-v1-7.onnx",
            # Alternative source
            "https://s3.amazonaws.com/download.pytorch.org/models/resnet18-5c106cde.pth"
        ]
        
        for url in urls:
            filename = "resnet18.onnx" if url.endswith(".onnx") else "resnet18.pth"
            result = self.download_file(url, filename)
            
            if result:
                # If we got a .pth file, try to convert it to ONNX
                if filename.endswith(".pth"):
                    return self.convert_pytorch_to_onnx(result)
                else:
                    return result
        
        # If all downloads fail, create a simple ONNX model
        logger.warning("Could not download pretrained model, creating simple test model")
        return self.create_simple_onnx_model()
    
    def convert_pytorch_to_onnx(self, pth_path):
        """Convert PyTorch .pth file to ONNX."""
        try:
            import torch
            import torchvision.models as models
            
            logger.info("Converting PyTorch model to ONNX...")
            
            # Load the model architecture
            model = models.resnet18()
            
            # Load the weights
            state_dict = torch.load(pth_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            
            # Export to ONNX
            onnx_path = os.path.join(self.temp_dir, "resnet18.onnx")
            example_input = torch.rand(1, 3, 224, 224)
            
            torch.onnx.export(
                model,
                example_input,
                onnx_path,
                opset_version=12,
                input_names=["input"],
                output_names=["output"]
            )
            
            return onnx_path
            
        except ImportError:
            logger.error("PyTorch not available for conversion")
            return None
        except Exception as e:
            logger.error(f"Failed to convert PyTorch to ONNX: {e}")
            return None
    
    def create_simple_onnx_model(self):
        """Create a simple ONNX model for testing when download fails."""
        onnx_path = os.path.join(self.temp_dir, "resnet18.onnx")
        
        # Create a minimal ONNX file manually (without onnx package dependency)
        logger.info("Creating minimal test model (dummy ONNX-like structure)...")
        
        # For testing purposes, we'll create a dummy file that the tests can load
        # This is not a real ONNX model but will allow tests to run
        dummy_content = b"""
        This is a dummy model file for testing purposes.
        Real applications should use proper ONNX models.
        Model format: ResNet-18 equivalent
        Input: [1, 3, 224, 224]
        Output: [1, 1000]
        """
        
        try:
            with open(onnx_path, 'wb') as f:
                f.write(dummy_content)
            
            logger.warning(f"Created dummy model file for testing: {onnx_path}")
            logger.warning("This is not a real ONNX model - tests may fail with actual inference")
            
            return onnx_path
            
        except Exception as e:
            logger.error(f"Failed to create dummy model: {e}")
            return None
    
    def convert_to_tensorrt(self, onnx_path):
        """Convert ONNX model to TensorRT engine."""
        if not onnx_path or not os.path.exists(onnx_path):
            logger.error("ONNX model not available for TensorRT conversion")
            return None
        
        engine_path = os.path.join(self.temp_dir, "resnet18.engine")
        
        if os.path.exists(engine_path):
            logger.info("TensorRT engine already exists")
            return engine_path
        
        # Check if trtexec is available
        if not shutil.which("trtexec"):
            logger.warning("trtexec not found, cannot convert to TensorRT")
            return None
        
        try:
            logger.info("Converting ONNX to TensorRT engine...")
            cmd = [
                "trtexec",
                f"--onnx={onnx_path}",
                f"--saveEngine={engine_path}",
                "--fp16",
                "--workspace=1024",
                "--minShapes=input:1x3x224x224",
                "--maxShapes=input:4x3x224x224",
                "--optShapes=input:1x3x224x224"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and os.path.exists(engine_path):
                logger.info(f"Generated TensorRT engine: {engine_path}")
                return engine_path
            else:
                logger.error(f"TensorRT conversion failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("TensorRT conversion timed out")
            return None
        except Exception as e:
            logger.error(f"TensorRT conversion error: {e}")
            return None
    
    def convert_to_openvino(self, onnx_path):
        """Convert ONNX model to OpenVINO IR format."""
        if not onnx_path or not os.path.exists(onnx_path):
            logger.error("ONNX model not available for OpenVINO conversion")
            return None
        
        xml_path = os.path.join(self.temp_dir, "resnet18.xml")
        bin_path = os.path.join(self.temp_dir, "resnet18.bin")
        
        if os.path.exists(xml_path) and os.path.exists(bin_path):
            logger.info("OpenVINO IR already exists")
            return xml_path
        
        # Try both ovc (new) and mo (legacy) converters
        converters = ["ovc", "mo"]
        
        for converter in converters:
            if not shutil.which(converter):
                continue
            
            try:
                logger.info(f"Converting ONNX to OpenVINO IR using {converter}...")
                
                if converter == "ovc":
                    cmd = [
                        "ovc",
                        onnx_path,
                        "--output_model", os.path.join(self.temp_dir, "resnet18"),
                        "--input_shape", "[1,3,224,224]",
                        "--compress_to_fp16"
                    ]
                else:  # mo
                    cmd = [
                        "mo",
                        "--input_model", onnx_path,
                        "--output_dir", self.temp_dir,
                        "--model_name", "resnet18",
                        "--input_shape", "[1,3,224,224]",
                        "--compress_to_fp16"
                    ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                
                if result.returncode == 0 and os.path.exists(xml_path):
                    logger.info(f"Generated OpenVINO IR: {xml_path}")
                    return xml_path
                else:
                    logger.warning(f"OpenVINO conversion with {converter} failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                logger.warning(f"OpenVINO conversion with {converter} timed out")
            except Exception as e:
                logger.warning(f"OpenVINO conversion with {converter} error: {e}")
        
        logger.error("All OpenVINO conversion attempts failed")
        return None
        
    def generate_tensorflow_savedmodel(self):
        """Generate TensorFlow SavedModel using pretrained ResNet from model zoo."""
        saved_model_dir = os.path.join(self.temp_dir, "saved_model")
        
        if os.path.exists(saved_model_dir):
            logger.info("TensorFlow SavedModel already exists")
            return saved_model_dir
        
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
            
            logger.info("Attempting to load pretrained ResNet model for TensorFlow SavedModel...")
            
            # Try TensorFlow Hub for a pretrained ResNet18 (or closest available)
            hub_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5"  # Using ResNet50 as ResNet18 is rare
            try:
                logger.info("Trying TensorFlow Hub for pretrained ResNet...")
                model = tf.keras.Sequential([
                    hub.KerasLayer(hub_url, input_shape=(224, 224, 3))
                ])
                logger.info("Loaded pretrained ResNet from TensorFlow Hub")
            except Exception as hub_e:
                logger.warning(f"Failed to load from TensorFlow Hub: {hub_e}")
                logger.info("Trying Keras Applications ResNet50...")
                
                # Try Keras Applications ResNet50
                try:
                    model = tf.keras.applications.ResNet50(
                        include_top=True,
                        weights='imagenet',
                        input_shape=(224, 224, 3),
                        classes=1000
                    )
                    logger.info("Loaded pretrained ResNet50 from Keras Applications")
                except Exception as keras_e:
                    logger.error(f"Failed to load ResNet50 from Keras Applications: {keras_e}")
                    logger.error("No pretrained model available")
                    return None
            
            # Save as TensorFlow SavedModel
            tf.keras.models.save_model(model, saved_model_dir, save_format='tf')
            
            logger.info(f"Generated TensorFlow SavedModel: {saved_model_dir}")
            return saved_model_dir
            
        except ImportError:
            logger.error("TensorFlow or TensorFlow Hub not available, cannot generate SavedModel")
            return None
        except Exception as e:
            logger.error(f"Failed to generate TensorFlow SavedModel: {e}")
            return None
            
        
    
    def convert_to_torchscript(self, onnx_path=None):
        """Generate or convert to TorchScript model."""
        torchscript_path = os.path.join(self.temp_dir, "resnet18.pt")
        
        if os.path.exists(torchscript_path):
            logger.info("TorchScript model already exists")
            return torchscript_path
        
        try:
            import torch
            import torchvision.models as models
            
            logger.info("Generating TorchScript model...")
            
            model = models.resnet18(pretrained=True)
            model.eval()
            
            # Convert to TorchScript
            example_input = torch.rand(1, 3, 224, 224)
            traced_model = torch.jit.trace(model, example_input)
            traced_model.save(torchscript_path)
            
            logger.info(f"Generated TorchScript model: {torchscript_path}")
            return torchscript_path
            
        except ImportError:
            logger.error("PyTorch not available, cannot generate TorchScript")
            return None
        except Exception as e:
            logger.error(f"Failed to generate TorchScript model: {e}")
            return None
    
    def get_model_for_backend(self, backend):
        """Get appropriate model for specified backend."""
        logger.info(f"Preparing model for backend: {backend}")
        
        backend = backend.upper()
        
        if backend == "OPENCV_DNN":
            # OpenCV DNN can use ONNX
            return self.generate_onnx_resnet18()
        
        elif backend == "ONNX_RUNTIME":
            # ONNX Runtime uses ONNX format
            return self.generate_onnx_resnet18()
        
        elif backend == "LIBTORCH":
            # LibTorch can use TorchScript or ONNX
            torchscript = self.convert_to_torchscript()
            if torchscript:
                return torchscript
            else:
                return self.generate_onnx_resnet18()
        
        elif backend == "LIBTENSORFLOW":
            # TensorFlow uses SavedModel
            return self.generate_tensorflow_savedmodel()
        
        elif backend == "TENSORRT":
            # TensorRT uses engine files (converted from ONNX)
            onnx_path = self.generate_onnx_resnet18()
            if onnx_path:
                engine_path = self.convert_to_tensorrt(onnx_path)
                return engine_path if engine_path else onnx_path
            return None
        
        elif backend == "OPENVINO":
            # OpenVINO uses IR format (converted from ONNX)
            onnx_path = self.generate_onnx_resnet18()
            if onnx_path:
                ir_path = self.convert_to_openvino(onnx_path)
                return ir_path if ir_path else onnx_path
            return None
        
        else:
            logger.error(f"Unknown backend: {backend}")
            return None

def main():
    parser = argparse.ArgumentParser(description="Download and convert models for InferenceEngines testing")
    parser.add_argument("backend", help="Backend to prepare model for")
    parser.add_argument("--output-dir", help="Output directory for model files")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files")
    
    args = parser.parse_args()
    
    downloader = ModelDownloader()
    
    try:
        model_path = downloader.get_model_for_backend(args.backend)
        
        if model_path:
            # Copy to output directory if specified
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                if os.path.isfile(model_path):
                    output_path = os.path.join(args.output_dir, os.path.basename(model_path))
                    shutil.copy2(model_path, output_path)
                    print(output_path)
                elif os.path.isdir(model_path):
                    output_path = os.path.join(args.output_dir, os.path.basename(model_path))
                    if os.path.exists(output_path):
                        shutil.rmtree(output_path)
                    shutil.copytree(model_path, output_path)
                    print(output_path)
            else:
                print(model_path)
            
            logger.info(f"Model ready for {args.backend}: {model_path}")
        else:
            logger.error(f"Failed to prepare model for {args.backend}")
            sys.exit(1)
    
    finally:
        if not args.keep_temp:
            downloader.cleanup()

if __name__ == "__main__":
    main()
