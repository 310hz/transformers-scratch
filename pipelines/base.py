import os


match os.getenv("DEEP_LEARNING_SCRATCH_BASE_TYPE"):
    case "scratch":
        from .base_scratch import Pipeline
    case "fabric":
        from .base_fabric import Pipeline
    case _:
        raise ValueError(
            "Environment variable 'DEEP_LEARNING_SCRATCH_BASE_TYPE' "
            "must be set to either 'scratch' or 'fabric'."
        )
