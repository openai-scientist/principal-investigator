"""The PI service's wire models agree with the vendored PI contract from management."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from principal_investigator import contract

VENDOR = Path(__file__).parent / "vendor"
#: Contract names that this service spells differently.
RENAMED = {"PIHealth": "Health"}


def _names() -> list[str]:
    return sorted(p.stem for p in (VENDOR / "examples").glob("*.json"))


@pytest.mark.parametrize("name", _names())
def test_example_round_trips_through_service_model_and_schema(name):
    model = getattr(contract, RENAMED.get(name, name), None)
    assert model is not None, f"the service has no model for {name}"
    example = json.loads((VENDOR / "examples" / f"{name}.json").read_text(encoding="utf-8"))
    dumped = model.model_validate(example).model_dump(mode="json", by_alias=True)
    schema = json.loads((VENDOR / "schema" / f"{name}.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(dumped)


def test_contract_version_matches_vendored_readme():
    readme = (VENDOR / "README.md").read_text(encoding="utf-8")
    assert f"Contract version: **{contract.CONTRACT_VERSION}**" in readme


def test_vendor_matches_management():
    source = Path(__file__).resolve().parents[3] / "management" / "docs" / "contract"
    if not source.is_dir():
        pytest.skip("management/docs/contract is not adjacent")

    def read(path: Path) -> bytes:
        return path.read_bytes().replace(b"\r\n", b"\n")

    expected = {p.relative_to(source / "pi"): read(p) for p in (source / "pi").rglob("*.json")}
    expected[Path("README.md")] = read(source / "README.md")
    actual = {
        p.relative_to(VENDOR): read(p)
        for p in VENDOR.rglob("*")
        if p.is_file() and p.name != "SOURCE"
    }
    assert actual == expected, "re-copy management/docs/contract/pi and README, update SOURCE"
