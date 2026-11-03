import pytest

from fraud_features.storage import latest_version, read_table, write_table


@pytest.mark.delta
def test_each_write_is_a_new_readable_version(delta_spark, tmp_path):
    path = str(tmp_path / "features")
    v0 = write_table(delta_spark.createDataFrame([(1, "a")], "id int, v string"), path)
    v1 = write_table(delta_spark.createDataFrame([(1, "b"), (2, "c")], "id int, v string"), path)

    assert (v0, v1) == (0, 1)
    assert latest_version(delta_spark, path) == 1
    assert [r.v for r in read_table(delta_spark, path, version=0).collect()] == ["a"]
    assert read_table(delta_spark, path).count() == 2
