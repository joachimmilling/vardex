"""The trainer for Apple silicon: LoRA with MLX, through mlx-lm, on the machine itself.

A trainer takes a folder of chat lines, train.jsonl and valid.jsonl, and a base model, and
leaves an adapter: the small set of weights LoRA learns, beside the base model's own, which
stay as they are. mlx-lm runs in a process of its own, as dbt does, so a training run that
runs out of memory takes its process down and not Vardex's, and its output is read as text.

The adapter is served with the base model by `mlx_lm.server --model BASE --adapter-path
ADAPTER`, which speaks Chat Completions; Vardex then reaches it with the `openai` provider.
On a GPU server, a trainer for another library plugs in beside this one by entry point.
"""

import importlib.util
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from vardex.config import Settings
from vardex.training import Trained, TrainError

# Small batches and few layers: a 4-bit model of a few billion parameters, its activations for
# one long prompt and LoRA's weights. On a 16 GB Mac that fits only once the GPU may use more
# than its default share of memory, about two thirds
DEFAULTS = {
    "--batch-size": "1",
    "--num-layers": "8",
    "--learning-rate": "1e-4",
    "--max-seq-length": "8192",  # Vardex's SQL prompt is several thousand tokens long
    "--steps-per-report": "10",
    "--steps-per-eval": "50",
}
# mlx-lm reports a loss as "Iter 10: Train loss 0.647" up to 0.31, and from 0.32 as a table
# row, "    10    0.647" for training and "    50    val 0.023" for validation
LOSS = re.compile(r"Iter (\d+): (Train|Val) loss ([\d.]+)")
ROW = re.compile(r"^\s*(\d+)\s+(val\s+)?(\d+\.\d+)\b")
COLOUR = re.compile(r"\x1b\[[0-9;]*m")
OUT_OF_MEMORY = "Insufficient Memory"


def losses_in(line: str) -> list[tuple[str, float]]:
    """The losses one line of mlx-lm's output reports, as ("Train" or "Val", loss)."""
    line = COLOUR.sub("", line)
    found = [(kind, float(loss)) for _, kind, loss in LOSS.findall(line)]
    if not found and (row := ROW.match(line)):
        found = [("Val" if row[2] else "Train", float(row[3]))]
    return found


class MlxTrainer:
    """LoRA on a 4-bit model with mlx-lm, which runs only on Apple silicon."""

    name = "mlx"

    def __init__(self, settings: Settings) -> None:
        if importlib.util.find_spec("mlx_lm") is None:
            raise TrainError(
                "mlx-lm is not installed. On a Mac with Apple silicon: pip install 'vardex[mlx]'."
            )
        self.settings = settings

    def command(self, base: str, data: Path, adapter: Path, iterations: int) -> list[str]:
        return [
            sys.executable, "-m", "mlx_lm", "lora",
            "--model", base,
            "--train",
            "--data", str(data),
            "--adapter-path", str(adapter),
            "--iters", str(iterations),
            # Learn the plan from the prompt, not the prompt itself: the loss counts only
            # the answer's tokens, which is what the model is being taught to write
            "--mask-prompt",
            "--grad-checkpoint",  # recompute rather than keep: slower, and it fits
            *(part for pair in DEFAULTS.items() for part in pair),
        ]  # fmt: skip

    def train(
        self,
        base: str,
        data: Path,
        adapter: Path,
        *,
        iterations: int,
        say: Callable[[str], None] = lambda line: None,
    ) -> Trained:
        """Train an adapter for the base model on the folder's train.jsonl, reporting the
        loss on valid.jsonl as it goes. Each line mlx-lm prints is passed to `say`."""
        losses: dict[str, float] = {}
        process = subprocess.Popen(
            self.command(base, data, adapter, iterations),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        output = []
        for line in process.stdout:
            output.append(line)
            say(line.rstrip())
            for kind, loss in losses_in(line):
                losses[kind] = loss
        if process.wait() != 0:
            if any(OUT_OF_MEMORY in line for line in output):
                raise TrainError(
                    "The GPU ran out of memory. Close other programs, train fewer layers, or "
                    "on a Mac let the GPU use more memory: sudo sysctl iogpu.wired_limit_mb=N, "
                    "with N about 85% of the memory in MB, until the next restart."
                )
            raise TrainError("mlx-lm stopped:\n" + "".join(output[-20:]))
        return Trained(adapter, base, iterations, losses.get("Train"), losses.get("Val"))
