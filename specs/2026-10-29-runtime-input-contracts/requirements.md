# Runtime input datatype contracts

## Required behavior

Resolve every input datatype from authoritative runtime/model metadata, never
from the model family and never by replacing an unknown type with FP32.
TorchScript serialization can erase graph scalar types: exporters must capture
the actual typed example inputs inside the artifact, and the loader must read
that metadata automatically. AOTInductor packages use their native metadata map.
Known graph types must agree with any archive declaration. Reject unknown,
malformed, unsupported or inconsistent input contracts explicitly.

Validate exact shape times element-size bytes with overflow checking before
wrapping caller buffers. Preserve existing device selection and supported
numeric computations. Unsupported tensor kinds remain explicit errors.

MIGraphX must preserve compiled parameter/output shape datatypes, instead of
constructing every argument as float and interpreting every result as float.
Use compiled output metadata when available rather than an untyped dummy run.

## Scope and compatibility

One corrective branch covers these datatype boundaries and their exporter/test
fixtures. No new SDK dependency or release bump. Legacy untyped TorchScript and
AOTInductor artifacts need re-export with automatically captured input types;
there is no human-facing dtype selector or model-name exception. The user
explicitly requires runtime detection instead of the current FP32 assumption.
Real-weight matrix validation is a separate gate. AMD hardware validation is
now authorized on an available MI300X; Jetson runs remain deferred. CPU-reference
and metadata tests must not claim AMD GPU coverage. Record GPU results separately.

AOTInductor additionally needs compiler-recorded static input shapes/contiguous
layout checked before its first probe: upstream native input checks can be
disabled, so caller-provided shapes alone are not a safe buffer-size oracle.
Dynamic/non-contiguous AOTI input contracts remain explicitly unsupported by
this exporter; they must not silently specialize or reinterpret caller data.
