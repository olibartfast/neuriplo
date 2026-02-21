#pragma once
#include <NvInfer.h>  // for TensorRT API
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR)
        {
            std::cerr << "TensorRT [ERROR]: " << msg << std::endl;
        }
        else if (severity == Severity::kWARNING)
        {
            std::cerr << "TensorRT [WARNING]: " << msg << std::endl;
        }
        // kINFO and kVERBOSE are suppressed
    }
};