# Plan

For unresolved output dimensions, preallocate the byte upper bound from IExecutionContext::getMaxOutputSize and register a bounded IOutputAllocator. Its callbacks reuse the owned device buffer and record notifyShape dimensions. Read those dimensions after the existing stream synchronization in both output APIs. Static buffers retain their current path. Store callbacks with engine state for transactional reload.
