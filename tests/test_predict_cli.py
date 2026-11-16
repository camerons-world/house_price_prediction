import pytest

from src import predict


@pytest.fixture
def mocks(monkeypatch):
    calls = {"predict": 0, "load": 0, "train": 0}
    monkeypatch.setattr(
        predict, "predict_price", lambda *a, **k: calls.__setitem__("predict", calls["predict"] + 1)
    )
    monkeypatch.setattr(
        predict.joblib, "load", lambda *a, **k: calls.__setitem__("load", calls["load"] + 1)
    )
    monkeypatch.setattr(
        predict, "train_and_save", lambda: calls.__setitem__("train", calls["train"] + 1)
    )
    return calls


@pytest.mark.parametrize("argv", [[], ["SW1A 1AA"]])
def test_missing_positionals_is_usage_error(monkeypatch, mocks, argv):
    monkeypatch.setattr("sys.argv", ["predict", *argv])
    with pytest.raises(SystemExit) as exc:
        predict.main()
    assert exc.value.code == 2
    assert mocks["predict"] == 0
    assert mocks["load"] == 0


def test_train_alone_reaches_train_and_save(monkeypatch, mocks):
    monkeypatch.setattr("sys.argv", ["predict", "--train"])
    predict.main()
    assert mocks["train"] == 1
    assert mocks["predict"] == 0
