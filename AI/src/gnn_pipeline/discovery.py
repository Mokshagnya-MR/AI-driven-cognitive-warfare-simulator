from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class DatasetRegistry:
    archive_dirs: List[str] = field(default_factory=list)
    fakenews_fake_csvs: List[str] = field(default_factory=list)
    fakenews_real_csvs: List[str] = field(default_factory=list)
    liar_tsvs: Dict[str, List[str]] = field(default_factory=dict)
    pheme_candidates: List[str] = field(default_factory=list)
    news_user_files: Dict[str, List[str]] = field(default_factory=dict)
    user_user_files: Dict[str, List[str]] = field(default_factory=dict)
    news_files: Dict[str, List[str]] = field(default_factory=dict)
    user_files: Dict[str, List[str]] = field(default_factory=dict)
    user_feature_mats: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _source_prefix(path: Path) -> str:
    name = path.name
    if "_" in name:
        return name.split("_", 1)[0]
    if name.endswith("NewsUser.txt"):
        return name[: -len("NewsUser.txt")]
    if name.endswith("UserUser.txt"):
        return name[: -len("UserUser.txt")]
    if name.endswith("News.txt"):
        return name[: -len("News.txt")]
    if name.endswith("User.txt"):
        return name[: -len("User.txt")]
    if name.endswith("UserFeature.mat"):
        return name[: -len("UserFeature.mat")]
    return path.stem


def _append_map_list(target: Dict[str, List[str]], key: str, value: str) -> None:
    if key not in target:
        target[key] = []
    target[key].append(value)


def discover_datasets(dataset_root: str | Path) -> DatasetRegistry:
    root = Path(dataset_root)
    registry = DatasetRegistry()

    archive_dirs = sorted([p for p in root.glob("archive*") if p.is_dir()])
    registry.archive_dirs = [str(p) for p in archive_dirs]

    for archive_dir in archive_dirs:
        for path in archive_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.name.endswith(":Zone.Identifier"):
                continue

            lower_name = path.name.lower()

            if lower_name.endswith("_fake_news_content.csv"):
                registry.fakenews_fake_csvs.append(str(path))
                continue
            if lower_name.endswith("_real_news_content.csv"):
                registry.fakenews_real_csvs.append(str(path))
                continue

            if lower_name in {"train.tsv", "valid.tsv", "test.tsv"}:
                split = lower_name.replace(".tsv", "")
                _append_map_list(registry.liar_tsvs, split, str(path))
                continue

            if lower_name.endswith("newsuser.txt"):
                _append_map_list(registry.news_user_files, _source_prefix(path), str(path))
                continue
            if lower_name.endswith("useruser.txt"):
                _append_map_list(registry.user_user_files, _source_prefix(path), str(path))
                continue
            if lower_name.endswith("news.txt") and not lower_name.endswith("newsuser.txt"):
                _append_map_list(registry.news_files, _source_prefix(path), str(path))
                continue
            if lower_name.endswith("user.txt") and not lower_name.endswith("useruser.txt") and not lower_name.endswith("newsuser.txt"):
                _append_map_list(registry.user_files, _source_prefix(path), str(path))
                continue
            if lower_name.endswith("userfeature.mat"):
                _append_map_list(registry.user_feature_mats, _source_prefix(path), str(path))
                continue

            if lower_name.endswith(".csv"):
                registry.pheme_candidates.append(str(path))

    registry.fakenews_fake_csvs = sorted(set(registry.fakenews_fake_csvs))
    registry.fakenews_real_csvs = sorted(set(registry.fakenews_real_csvs))
    registry.pheme_candidates = sorted(set(registry.pheme_candidates))
    registry.liar_tsvs = {k: sorted(set(v)) for k, v in registry.liar_tsvs.items()}
    registry.news_user_files = {k: sorted(set(v)) for k, v in registry.news_user_files.items()}
    registry.user_user_files = {k: sorted(set(v)) for k, v in registry.user_user_files.items()}
    registry.news_files = {k: sorted(set(v)) for k, v in registry.news_files.items()}
    registry.user_files = {k: sorted(set(v)) for k, v in registry.user_files.items()}
    registry.user_feature_mats = {k: sorted(set(v)) for k, v in registry.user_feature_mats.items()}
    return registry
