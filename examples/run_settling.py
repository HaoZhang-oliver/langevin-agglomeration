from __future__ import annotations

from pathlib import Path

import yaml

from ldagg.settling import SettlingConfig, run_settling

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config_path = ROOT / "examples" / "configs" / "settling_air_300K.yml"
    with config_path.open("r", encoding="utf-8") as fh:
        config = SettlingConfig.from_mapping(yaml.safe_load(fh))
    run_settling(config, ROOT / "outputs" / "settling")


if __name__ == "__main__":
    main()
