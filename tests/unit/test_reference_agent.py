from typing import get_args

import pytest
import yaml

from callcheck.core.scenario import FieldType
from callcheck.reference_agent import BUGS, FormStore, load_reference_agent
from callcheck.reference_agent.loader import INTAKE_YAML


async def test_clean_store_takes_the_correction():
    s = FormStore(["phone"], {"phone": "phone"}, frozenset())
    await s.execute("save_field", {"name": "phone", "value": "415 555 0143"})
    await s.execute("save_field", {"name": "phone", "value": "415 555 0134"})
    assert s.snapshot() == {"fields": {"phone": "415 555 0134"}, "submitted": False}


async def test_ignore_corrections_keeps_first_value():
    s = FormStore(["phone"], {"phone": "phone"}, frozenset({"ignore_corrections"}))
    first = await s.execute("save_field", {"name": "phone", "value": "415 555 0143"})
    second = await s.execute("save_field", {"name": "phone", "value": "415 555 0134"})
    assert first == second == {"ok": True}
    assert s.snapshot()["fields"]["phone"] == "415 555 0143"


async def test_day_month_swap_only_when_day_le_12():
    s = FormStore(["dob"], {"dob": "date"}, frozenset({"day_month_swap"}))
    await s.execute("save_field", {"name": "dob", "value": "March 7 1988"})
    assert s.snapshot()["fields"]["dob"] == "1988-07-03"
    await s.execute("save_field", {"name": "dob", "value": "March 14 1988"})
    assert s.snapshot()["fields"]["dob"] == "1988-03-14"


async def test_clean_store_stores_dates_as_iso():
    s = FormStore(["dob"], {"dob": "date"}, frozenset())
    await s.execute("save_field", {"name": "dob", "value": "March 7 1988"})
    assert s.snapshot()["fields"]["dob"] == "1988-03-07"


@pytest.mark.parametrize("bugs", [frozenset(), frozenset({"day_month_swap"})])
async def test_unparseable_date_is_stored_as_given(bugs):
    s = FormStore(["dob"], {"dob": "date"}, bugs)
    await s.execute("save_field", {"name": "dob", "value": "sometime in spring"})
    assert s.snapshot()["fields"]["dob"] == "sometime in spring"


async def test_non_date_kinds_are_stored_verbatim_even_with_the_swap_bug():
    s = FormStore(["phone"], {"phone": "phone"}, frozenset({"day_month_swap"}))
    await s.execute("save_field", {"name": "phone", "value": "415 555 0143"})
    assert s.snapshot()["fields"]["phone"] == "415 555 0143"


async def test_save_field_does_not_echo_the_stored_value():
    s = FormStore(["dob"], {"dob": "date"}, frozenset({"day_month_swap"}))
    result = await s.execute("save_field", {"name": "dob", "value": "March 7 1988"})
    assert result == {"ok": True}


async def test_unknown_field_is_rejected_and_not_stored():
    s = FormStore(["phone"], {"phone": "phone"}, frozenset())
    result = await s.execute("save_field", {"name": "shoe_size", "value": "42"})
    assert result == {"ok": False, "error": "unknown field"}
    assert s.snapshot() is None


async def test_unknown_tool_is_rejected():
    s = FormStore(["phone"], {"phone": "phone"}, frozenset())
    assert await s.execute("hang_up", {}) == {"ok": False, "error": "unknown tool"}


async def test_snapshot_is_none_until_first_save_then_tracks_submission():
    s = FormStore(["phone"], {"phone": "phone"}, frozenset())
    assert s.snapshot() is None
    await s.execute("save_field", {"name": "phone", "value": "415 555 0143"})
    assert await s.execute("submit_form", {}) == {"ok": True, "submitted": True}
    assert s.snapshot() == {"fields": {"phone": "415 555 0143"}, "submitted": True}


async def test_snapshot_returns_a_copy():
    s = FormStore(["phone"], {"phone": "phone"}, frozenset())
    await s.execute("save_field", {"name": "phone", "value": "415 555 0143"})
    s.snapshot()["fields"]["phone"] = "tampered"
    assert s.snapshot()["fields"]["phone"] == "415 555 0143"


def test_loader_renders_fields():
    a = load_reference_agent(["first_name", "phone"], frozenset({"strict_order"}))
    assert "- first_name:" in a.system_prompt
    assert [t.name for t in a.tools] == ["save_field", "submit_form"]
    assert a.tools[0].input_schema["properties"]["name"]["enum"] == ["first_name", "phone"]


def test_loader_leaves_no_stray_blank_lines_or_placeholders():
    for bugs in (frozenset(), frozenset(BUGS)):
        prompt = load_reference_agent(["first_name"], bugs).system_prompt
        assert "\n\n\n" not in prompt
        assert "{" not in prompt and "}" not in prompt


def test_loader_rejects_an_unknown_field_by_name():
    with pytest.raises(KeyError, match="shoe_size"):
        load_reference_agent(["first_name", "shoe_size"])


async def test_loader_wires_the_executor_and_result_to_one_store():
    a = load_reference_agent(["phone"], frozenset())
    assert a.result() is None
    await a.tool_executor("save_field", {"name": "phone", "value": "415 555 0143"})
    assert a.result() == {"fields": {"phone": "415 555 0143"}, "submitted": False}
    assert a.voice == {"nova_sonic": "matthew", "openai_realtime": "alloy"}


def test_intake_yaml_has_twenty_fields_with_valid_kinds():
    fields = yaml.safe_load(INTAKE_YAML.read_text())["fields"]
    assert len(fields) == 20
    assert {"first_name", "date_of_birth", "phone", "consent_to_text", "visit_type"} <= set(fields)
    assert fields["date_of_birth"]["kind"] == "date" and fields["phone"]["kind"] == "phone"
    assert all(f["kind"] in get_args(FieldType) and f["question"] for f in fields.values())


def test_prompt_tells_the_agent_to_read_back_the_callers_words():
    prompt = load_reference_agent(["phone"], frozenset()).system_prompt
    assert "exactly as the caller said it" in prompt
    assert "Never read back anything the tool returned" in prompt


def _store(bugs):
    return FormStore(["first_name", "last_name", "phone"], {}, frozenset(bugs))


async def test_strict_order_drops_an_out_of_order_save_and_a_later_one():
    s = _store({"strict_order"})
    assert await s.execute("save_field", {"name": "phone", "value": "1"}) == {"ok": True}
    await s.execute("save_field", {"name": "first_name", "value": "Ann"})
    await s.execute("save_field", {"name": "last_name", "value": "Lee"})
    await s.execute("save_field", {"name": "phone", "value": "2"})
    assert s.snapshot()["fields"] == {"first_name": "Ann", "last_name": "Lee"}


async def test_strict_order_stores_in_order_saves():
    s = _store({"strict_order"})
    for n in ("first_name", "last_name", "phone"):
        await s.execute("save_field", {"name": n, "value": "x"})
    await s.execute("save_field", {"name": "first_name", "value": "y"})
    assert s.snapshot()["fields"] == {"first_name": "y", "last_name": "x", "phone": "x"}


async def test_without_strict_order_an_out_of_order_save_is_stored():
    s = _store(set())
    await s.execute("save_field", {"name": "phone", "value": "1"})
    assert s.snapshot()["fields"] == {"phone": "1"}


def test_strict_order_does_not_change_the_prompt():
    names = ["first_name", "phone"]
    assert (
        load_reference_agent(names, frozenset({"strict_order"})).system_prompt
        == load_reference_agent(names, frozenset()).system_prompt
    )
