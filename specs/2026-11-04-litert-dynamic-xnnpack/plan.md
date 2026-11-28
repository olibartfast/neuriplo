# Plan

Use an explicitly owned XNNPACK delegate with default options plus subgraph reshaping, retaining the single-thread CPU default. Disable automatic default delegates to avoid duplicate delegation. Apply after input resizing and before allocation. Add fixture-backed numerical parity coverage and run existing LiteRT tests, then downstream app and serving gates.
