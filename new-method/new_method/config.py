"""Configuration for the single production IR pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationConfig:
    name: str


ABLATIONS = {
    "IR": AblationConfig("IR"),
}


def resolve_ablation(value: str | AblationConfig | None) -> AblationConfig:
    if value is None:
        return ABLATIONS["IR"]
    if isinstance(value, AblationConfig):
        return value
    if value not in ABLATIONS:
        raise ValueError(f"unknown ablation {value!r}; choose one of {sorted(ABLATIONS)}")
    return ABLATIONS[value]
