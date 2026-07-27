import os


if os.getenv("DEEP_LEARNING_SCRATCH_USE_SCRATCH") == "1":
    from .base_scratch import Pipeline
else:
    from .base_fabric import Pipeline
