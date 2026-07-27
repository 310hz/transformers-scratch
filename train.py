import typer

from pipelines import get_pipeline


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
app = typer.Typer(add_completion=False, context_settings=CONTEXT_SETTINGS)


@app.command()
def main(
    src: str = typer.Argument(
        help=(
            "Path to the config (TOML) file when starting training, or "
            "the path to a checkpoint directory when resuming training."
        ),
    ),
    dpath_ckpt: str = typer.Option(
        None,
        "-c", "--dpath-ckpt",
        help="Path to the directory where checkpoints will be saved.",
    ),
    test_stream: bool = typer.Option(
        False,
        "--test-stream",
        help=(
            "Stream test data instead of downloading it. Training data "
            "is always streamed."
        ),
    ),
    accelerator: str = typer.Option(
        "auto",
        "-a", "--accelerator",
        help=(
            "Accelerator to use for training. This will be passed to "
            "Lightning Fabric."
        ),
    ),
    devices: str = typer.Option(
        "auto",
        "-d", "--devices",
        help=(
            "Number of devices to use for training. This will be "
            "passed to Lightning Fabric."
        ),
    ),
    strategy: str = typer.Option(
        "auto",
        "-s", "--strategy",
        help=(
            "Strategy to use for distributed training. This will be "
            "passed to Lightning Fabric."
        ),
    ),
    use_scratch: bool = typer.Option(
        False,
        "--use-scratch",
        help=(
            "Use the scratch implementation of the pipeline instead of "
            "the Lightning Fabric implementation."
        ),
    ),
):
    """Train a model."""
    pipeline = get_pipeline(src, use_scratch=use_scratch)
    pipeline.train(
        dpath_ckpt=dpath_ckpt,
        test_stream=test_stream,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy
    )


if __name__ == "__main__":
    app()
