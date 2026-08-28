"""Named ablations for comparable SymPlanner IR experiments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationConfig:
    name: str
    enable_ir_repair: bool = True
    enable_bidirectional_verifier: bool = True
    enable_code_repair: bool = True


ABLATIONS = {
    "IR-Codegen": AblationConfig("IR-Codegen", enable_bidirectional_verifier=False, enable_code_repair=False),
    "IR-BiVerify": AblationConfig("IR-BiVerify", enable_bidirectional_verifier=True, enable_code_repair=False),
    "IR-Full": AblationConfig("IR-Full", enable_bidirectional_verifier=True, enable_code_repair=True),
}


def resolve_ablation(value: str | AblationConfig | None) -> AblationConfig:
    if value is None:
        return ABLATIONS["IR-Full"]
    if isinstance(value, AblationConfig):
        return value
    if value not in ABLATIONS:
        raise ValueError(f"unknown ablation {value!r}; choose one of {sorted(ABLATIONS)}")
    return ABLATIONS[value]
