"""
Run the TTSDS multilingual benchmark suite to evaluate Nepali TTS output.

Evaluates two finetuned-model inference outputs (CPU and GPU) against a
Nepali reference corpus, using multilingual benchmarks suited for Nepali.

Expected layout (relative to this repo root):
    Dataset/cpu/            -> TTS-generated audio (CPU inference)
    Dataset/gpu/            -> TTS-generated audio (GPU inference)
    Dataset/Reference_Audio -> Nepali reference speech
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
DATASET_DIR = REPO_ROOT / "Dataset"

# The package uses a src/ layout, so it must be installed (pip install -e .) or
# on sys.path. Prefer the installed package; fall back to the checked-out src/
# so the script works even before installation.
SRC_DIR = REPO_ROOT / "src"
try:
    import ttsds  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(SRC_DIR))

from ttsds import BenchmarkSuite
from ttsds.util.dataset import DirectoryDataset

# Evaluated datasets: the two finetuned-model outputs to compare.
evaluated_datasets = [
    DirectoryDataset(
        str(DATASET_DIR / "cpu"),
        name="nepali_finetuned_cpu",
    ),
    DirectoryDataset(
        str(DATASET_DIR / "gpu"),
        name="nepali_finetuned_gpu",
    ),
]

# Reference datasets: real Nepali speech, matched to the same speakers/utterances.
reference_datasets = [
    DirectoryDataset(
        str(DATASET_DIR / "Reference_Audio"),
        name="nepali_reference",
    ),
]


def main() -> None:
    suite = BenchmarkSuite(
        datasets=evaluated_datasets,
        reference_datasets=reference_datasets,
        multilingual=True,          # use mHuBERT / XLSR / mWhisper benchmarks
        device="cuda",              # set to "cpu" if no GPU is available
        n_workers=4,                # parallel distance computation
        skip_errors=True,
        write_to_file=str(REPO_ROOT / "nepali_results.csv"),
    )

    results = suite.run()
    print(results)

    print("\n=== Aggregated results ===")
    print(suite.get_aggregated_results())


if __name__ == "__main__":
    main()