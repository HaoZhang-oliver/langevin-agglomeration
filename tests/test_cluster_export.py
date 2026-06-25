from __future__ import annotations

import numpy as np

from ldagg.analysis import cluster_geometry_nm_array, export_cluster_geometry_nm_csv
from ldagg.clusters import dimer_seed


def test_cluster_geometry_nm_array_and_csv_no_header(tmp_path) -> None:
    cluster = dimer_seed(0, 100.0e-9)
    arr = cluster_geometry_nm_array(cluster)

    assert arr.shape == (2, 4)
    assert np.allclose(arr[:, 3], 100.0)

    path = export_cluster_geometry_nm_csv(cluster, tmp_path / "cluster.csv")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert not lines[0].lower().startswith("x")

    loaded = np.loadtxt(path, delimiter=",")
    assert loaded.shape == (2, 4)
    assert np.allclose(loaded[:, 3], 100.0)
