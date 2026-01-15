"""Tests for PlaywrightExecutor locator parsing."""


from codevid.executor.playwright import PlaywrightExecutor


class DummyPage:
    def locator(self, selector: str):
        return ("css", selector)


def test_get_locator_evals_locator_chain_for_locator_expr(monkeypatch):
    executor = PlaywrightExecutor()
    sentinel = object()

    def fake_eval(page, expr: str):
        assert expr == "locator('#form')"
        return sentinel

    monkeypatch.setattr(executor, "_eval_locator_chain", fake_eval)

    result = executor._get_locator(DummyPage(), "locator('#form')")
    assert result is sentinel


def test_get_locator_evals_locator_chain_for_get_by_role(monkeypatch):
    executor = PlaywrightExecutor()
    sentinel = object()

    def fake_eval(page, expr: str):
        assert expr.startswith("get_by_role(")
        return sentinel

    monkeypatch.setattr(executor, "_eval_locator_chain", fake_eval)

    result = executor._get_locator(DummyPage(), "get_by_role('button', name='Submit')")
    assert result is sentinel


def test_get_locator_falls_back_to_css_locator():
    executor = PlaywrightExecutor()
    assert executor._get_locator(DummyPage(), "#submit") == ("css", "#submit")
