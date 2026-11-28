# Data-dependent TensorRT output buffers

LightGlue app validation requires outputs matches [M,3] and mscores [M], with M known only during execution. Preserve model values, typed outputs, static model execution, engine selection and fallback behavior. Allocate sufficient output storage and report concrete output shapes to consumers. Keep ownership and transactional engine reload behavior.
