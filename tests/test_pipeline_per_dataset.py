from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from main import run_pipeline_for_dataset  # noqa: E402


def _write_fakenewsnet_fixture(dataset_root: Path) -> None:
    source_dir = dataset_root / "FakeNews dataset"
    source_dir.mkdir(parents=True, exist_ok=True)

    def _rows(label_word: str, count: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "id": [f"{label_word}-{i}" for i in range(count)],
                "title": [f"{label_word} headline number {i}" for i in range(count)],
                "text": [
                    f"This is {label_word} article body text number {i} discussing a topic in detail."
                    for i in range(count)
                ],
                "publish_date": [f"2024-01-{(i % 27) + 1:02d}" for i in range(count)],
            }
        )

    _rows("fake", 20).to_csv(source_dir / "BuzzFeed_fake_news_content.csv", index=False)
    _rows("real", 20).to_csv(source_dir / "BuzzFeed_real_news_content.csv", index=False)


def test_run_pipeline_for_dataset_produces_expected_outputs(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_dir = tmp_path / "outputs" / "fakenewsnet"
    _write_fakenewsnet_fixture(dataset_root)

    metrics = run_pipeline_for_dataset(
        dataset_name="fakenewsnet",
        dataset_root=dataset_root,
        output_dir=output_dir,
        use_cuda=False,
        embedding_backend="hashing",
        text_feature_dim=16,
        epochs=1,
        early_stopping_patience=1,
    )

    assert (output_dir / "dataset_registry.json").exists()
    assert (output_dir / "dataset_registry.json").stat().st_size > 0

    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "model_state_dict.pt").exists()
    assert (output_dir / "processed_graph.pt").exists()
    assert (output_dir / "artifact_manifest.json").exists()

    for split_name in ("train_idx.npy", "val_idx.npy", "test_idx.npy"):
        assert (output_dir / "splits" / split_name).exists()

    assert "accuracy" in metrics
    assert "f1" in metrics
    assert metrics["split_sizes"]["train"] > 0


def test_run_pipeline_for_dataset_rejects_unknown_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_pipeline_for_dataset(
            dataset_name="not-a-real-dataset",
            dataset_root=tmp_path / "dataset",
            output_dir=tmp_path / "outputs",
        )
