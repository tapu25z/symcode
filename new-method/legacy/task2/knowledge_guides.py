#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small formula/strategy guides selected before Module 4 codegen."""

from __future__ import annotations

import re
from typing import Any, Dict, List


NL_SINUSOIDAL_ENERGY_GUIDE: Dict[str, Any] = {
    "id": "nl_sinusoidal_energy",
    "title": "LC/inductor/capacitor sinusoidal energy",
    "use_when": [
        "The target asks for electric/magnetic field energy in a capacitor, inductor, coil, or LC circuit.",
        "The question gives a sinusoidal voltage/current/charge function such as U(t), u(t), I(t), i(t), or q(t).",
        "The question asks for energy at a specified time or maximum energy.",
    ],
    "rules": [
        "Capacitor electric energy: W_C = 1/2 * C * U^2.",
        "If capacitor charge q is given instead of voltage, use W_C = q^2 / (2*C), NOT 1/2*C*q^2.",
        "Inductor/coil magnetic energy: W_L = 1/2 * L * I^2.",
        "For a specified time, declare one symbol t and substitute using that exact symbol.",
        "For maximum energy of U0*sin(w*t), U0*cos(w*t), I0*sin(w*t), or I0*cos(w*t), use the amplitude directly because max(sin^2)=max(cos^2)=1.",
        "Do not use sp.Max on an unevaluated symbolic sine/cosine expression.",
    ],
    "code_pattern": [
        "For maximum capacitor energy: W_max = sp.Rational(1, 2) * C * U0**2",
        "For maximum inductor energy: W_max = sp.Rational(1, 2) * L * I0**2",
        "For energy at time t0: t = sp.symbols('t'); value = expr.subs(t, t0); W = sp.Rational(1, 2) * parameter * value**2",
        "Before printing: final_answer = float(sp.N(W))",
    ],
    "avoid": [
        "Do not leave t symbolic in the final answer.",
        "Do not use sp.Max(expr) for sinusoidal maxima.",
    ],
}

LC_STATE_GUIDE: Dict[str, Any] = {
    "id": "lc_complementary_state",
    "title": "Ideal LC complementary states",
    "use_when": [
        "The problem involves an ideal LC oscillation/circuit.",
        "The target asks for capacitor voltage/charge/electric energy or inductor current/magnetic energy at a maximum/minimum/zero state.",
    ],
    "rules": [
        "In an ideal LC circuit, total energy is conserved and alternates between capacitor electric energy and inductor magnetic energy.",
        "When current is maximum: magnetic energy is maximum, capacitor charge q=0, capacitor voltage U_C=0, and capacitor electric energy W_C=0.",
        "When capacitor voltage or charge is maximum: current i=0 and magnetic energy W_L=0.",
        "If W_L=0 at a stated time, W_C equals the total constant energy.",
    ],
    "code_pattern": [
        "If target is capacitor voltage at maximum current: final_answer = 0.0",
        "If target is magnetic energy when electric energy is maximum: final_answer = 0.0",
    ],
    "avoid": [
        "Do not leave symbolic U_C_max/I_max when the LC state implies zero.",
    ],
}

RLC_REACTANCE_SCALING_GUIDE: Dict[str, Any] = {
    "id": "rlc_reactance_frequency_scaling",
    "title": "RLC reactance scaling when frequency changes",
    "use_when": [
        "The problem gives R, X_L, X_C, source voltage, and says the frequency is multiplied/divided.",
        "The target asks for current, impedance, or power in the resistor after the frequency change.",
    ],
    "rules": [
        "For a frequency multiplier n: X_L_new = n * X_L and X_C_new = X_C / n.",
        "For a frequency divider n: X_L_new = X_L / n and X_C_new = n * X_C.",
        "Series RLC impedance magnitude: Z = sqrt(R^2 + (X_L_new - X_C_new)^2).",
        "Circuit current magnitude: I = U / Z.",
        "Power consumed by resistor R: P_R = I^2 * R.",
        "Declare every given variable before using it: R, X_L, X_C, U, and n.",
    ],
    "code_pattern": [
        "X_L_new = n * X_L",
        "X_C_new = X_C / n",
        "Z = sp.sqrt(R**2 + (X_L_new - X_C_new)**2)",
        "P_R = (U / Z)**2 * R",
    ],
    "avoid": [
        "Do not reuse old X_L/X_C after a frequency change.",
        "Do not leave variables such as X_C or R undefined.",
    ],
}

QA_KINEMATICS_GUIDE: Dict[str, Any] = {
    "id": "qa_kinematics_units",
    "title": "Kinematics equations and unit consistency",
    "use_when": [
        "The problem involves motion, speed, distance, time, acceleration, or trajectories.",
        "The problem uses common school-level constant-speed relationships.",
    ],
    "rules": [
        "PREFER using original units from raw_question (km/h, km, hours, minutes) instead of SI-converted values from normalized_problem if it keeps the equation simpler.",
        "If you choose SI (m/s, m, s): keep ALL quantities in SI consistently. Convert to target units ONLY in the final print step.",
        "NEVER mix units: e.g., do NOT mix speed in m/s with time in hours, or distance in km with speed in m/s.",
        "Conversion reminders: 1 km/h = 1/3.6 m/s. 1 hour = 3600 s. 1 km = 1000 m.",
        "For constant speed: distance = speed * time. If speed is km/h, time must be in hours and distance in km.",
        "For two vehicles moving toward each other: (v1 + v2) * t = initial_distance.",
        "If two travelers start from opposite ends of the same path and would complete the whole path in T1 and T2, then after time t their separation is abs(D - D*t/T1 - D*t/T2).",
        "For same-direction chase/closing: abs(v_fast - v_slow) * t = gap_closed.",
        "For boat/current: v_down = v_boat + v_current and v_up = v_boat - v_current.",
        "For a round trip on a river: d/v_down + d/v_up = total_time.",
        "For a go-return same distance with time difference Δt: d/v_slow = d/v_fast + Δt, with all times in the same unit.",
        "For stop/breakdown schedule problems: planned_time = D/v_plan; actual_time = traveled_time + stop_time + remaining_distance/(v_plan + speed_increase).",
        "For flight path / incline distance: hypotenuse s = v * t. Vertical height h = s * sin(theta). Horizontal distance d = s * cos(theta).",
        "For multi-answer speeds, print one boxed answer with values separated by semicolons and repeat the unit for each answer, e.g. \\boxed{v1 \\, \\mathrm{km/h}; v2 \\, \\mathrm{km/h}}.",
    ],
    "code_pattern": [
        "If original was 54 km/h and target is km/h: use 54 directly and convert minutes to hours.",
        "If solving in SI: final_answer_kmh = final_answer_ms * 3.6",
        "solutions = [sol for sol in sp.solve(equations, variables, dict=True) if all(sol[v] > 0 for v in variables)]",
    ],
    "avoid": [
        "Do not write equations like 't + 1' if t is in seconds and 1 means 1 hour.",
        "Do not convert km/h values to m/s and then divide a km distance by them.",
    ]
}

LD_COULOMB_GUIDE: Dict[str, Any] = {
    "id": "ld_coulomb_vectors",
    "title": "Coulomb Forces and Electric Field Vectors",
    "use_when": [
        "The problem involves multiple point charges, Coulomb forces, or electric fields from point charges.",
        "The problem ID prefix is LD.",
    ],
    "rules": [
        "Electric force and electric field are VECTORS. You must sum them using components (x, y) or vector geometry.",
        "For force on target charge q0 at position P from source charge qi at position ri, use signed vector form: F_i = k * q0 * qi * (P - ri) / |P - ri|^3.",
        "For electric field at P from source charge qi at ri, use signed vector form: E_i = k * qi * (P - ri) / |P - ri|^3.",
        "After summing components, report magnitude = sqrt(Fx^2 + Fy^2) unless the question explicitly asks for a signed component.",
        "For triangle distances AB, AC, BC, place A=(0,0), B=(AB,0), C=(x,y) with x=(AC^2 + AB^2 - BC^2)/(2*AB), y=sqrt(AC^2 - x^2).",
        "When charges have EQUAL magnitude and SAME sign, their fields at the midpoint CANCEL (vector sum = 0). Do NOT add their magnitudes.",
        "When equal opposite charges are on the ends and the target is the midpoint, the two force/field contributions point in the same direction; do NOT subtract them to zero.",
        "For a right triangle with sides a, b, hypotenuse c: use cos/sin from the actual triangle ratios (e.g., cos = adjacent/hypotenuse, NOT arbitrary formulas).",
        "Force = charge × Electric_field (F = q * E). Do not confuse E (V/m) with F (N).",
        "If charges are not perfectly symmetric (e.g., q1 != q2), the field/force does NOT perfectly cancel out.",
        "Use the standard Coulomb constant k = 9e9 N*m^2/C^2 in air/vacuum unless the problem explicitly gives a dielectric constant.",
    ],
    "code_pattern": [
        "r_vec = P - source_position",
        "F_vec = k * q_target * q_source * r_vec / (sp.sqrt(r_vec.dot(r_vec))**3)",
        "F_total = F1_vec + F2_vec + F3_vec",
        "F_mag = sp.sqrt(F_total.dot(F_total))",
    ],
    "avoid": [
        "Do NOT add the magnitudes of vectors directly unless they are perfectly collinear and pointing in the same direction.",
    ]
}

DT_ELECTRIC_FIELD_GUIDE: Dict[str, Any] = {
    "id": "dt_dielectric_field",
    "title": "Electric Fields, Dielectrics, and Continuous Distributions",
    "use_when": [
        "The problem involves calculating electric fields, dielectrics (epsilon), or continuous charge distributions (like a disk).",
        "The problem ID prefix is DT.",
    ],
    "rules": [
        "If the problem specifies a dielectric medium (e.g., alcohol ε=2.2, water ε=81), divide the Coulomb constant k by ε: k_eff = k / ε = 9e9 / ε.",
        "For a uniformly charged disk of radius R and surface charge density σ, the electric field on the axis at distance z is: E_z = (σ / (2*ε0)) * (1 - z / sp.sqrt(z**2 + R**2)). Do NOT use the point-charge formula.",
        "For point charges on a line, use signed vector/component form: E_i = k_eff * qi * (P - ri) / |P - ri|^3.",
        "At the midpoint between two equal same-sign charges, E=0 because fields cancel by symmetry.",
        "At the midpoint between opposite-sign charges, the fields point in the same direction; add signed components rather than subtracting magnitudes.",
        "N/C and V/m are equivalent units for electric field.",
    ],
    "code_pattern": [
        "k_eff = 9e9 / epsilon",
        "E = k_eff * abs(q) / r**2",
        "E_x = k_eff * q * (x_point - x_charge) / abs(x_point - x_charge)**3",
    ],
    "avoid": [
        "Do NOT forget to divide by the dielectric constant if one is given.",
        "Do NOT use point charge formula E = k*q/r^2 for a large disk or sheet.",
    ]
}

DDT_FARADAY_GUIDE: Dict[str, Any] = {
    "id": "ddt_faraday_induction",
    "title": "Electromagnetic Induction and Faraday's Law",
    "use_when": [
        "The problem involves magnetic flux, induced EMF, loops of wire, or Faraday's law.",
        "The problem ID prefix is DDT.",
    ],
    "rules": [
        "Faraday's Law: EMF = N * |ΔΦ_per_turn| / Δt.",
        "If flux per turn is given and current decreases to zero, treat the final flux per turn as 0 unless another final flux is stated.",
        "Do NOT multiply Φ_per_turn by N twice. If the problem gives flux through one turn, multiply by N. If it gives total flux linkage, do not multiply by N again.",
        "In LC circuits: when current is maximum, capacitor voltage = 0 (and vice versa).",
    ],
    "code_pattern": [
        "dPhi = abs(Phi2 - Phi1)",
        "EMF = N * dPhi / dt",
    ],
    "avoid": [
        "Do NOT compute ΔΦ = N*(Φ2 - N*Φ1).",
    ]
}

TD_PARALLEL_PLATE_CAPACITOR_GUIDE: Dict[str, Any] = {
    "id": "td_parallel_plate_capacitor_scaling",
    "title": "Parallel-plate capacitor scaling",
    "use_when": [
        "The problem gives an initial parallel-plate capacitance and changes plate separation and/or dielectric constant.",
        "The target asks for the new capacitance.",
    ],
    "rules": [
        "For a parallel-plate capacitor with fixed plate area: C ∝ ε_r / d.",
        "Use scaling from the known initial capacitance: C2 = C1 * (ε_r2 / ε_r1) * (d1 / d2).",
        "If the initial medium is air/vacuum and no dielectric constant is stated, use ε_r1 = 1.",
        "If target unit is pF and initial C is in pF, keep pF throughout the ratio.",
    ],
    "code_pattern": [
        "C2 = C1 * (epsilon_r2 / epsilon_r1) * (d1 / d2)",
    ],
    "avoid": [
        "Do not introduce ε0 or area A when using a ratio is enough.",
        "Do not use an undefined variable such as r2; use the given dielectric constant.",
    ],
}

CHLT_RESONANCE_GUIDE: Dict[str, Any] = {
    "id": "chlt_resonance_tolerance",
    "title": "Resonance Frequency Tolerance",
    "use_when": [
        "The problem is a Yes/No question asking if resonance occurs.",
        "The problem ID prefix is CHLT.",
    ],
    "rules": [
        "For resonance yes/no questions, use a relative tolerance (1-2%) instead of a fixed absolute tolerance.",
        "If |f - f_res| / f_res <= 0.02, answer 'Yes' (True), otherwise 'No' (False).",
    ],
    "code_pattern": [
        "f_res = 1 / (2 * sp.pi * sp.sqrt(L * C))",
        "is_resonance = abs(f - f_res) / f_res <= 0.02",
        "answer = 'Yes' if is_resonance else 'No'",
    ],
    "avoid": [
        "Do NOT use strict equality `f == f_res` or strict absolute tolerance like `abs(f - f_res) < 0.1`.",
    ]
}

def _flatten_problem_text(raw_question: str, normalized_problem: Dict[str, Any]) -> str:
    parts = [str(raw_question or "")]
    for item in normalized_problem.get("given", []) or []:
        if isinstance(item, dict):
            parts.extend(str(item.get(k, "")) for k in ("name", "symbol", "unit", "original_unit"))
    for item in normalized_problem.get("target", []) or []:
        if isinstance(item, dict):
            parts.extend(str(item.get(k, "")) for k in ("name", "symbol", "unit"))
    parts.extend(str(x) for x in normalized_problem.get("conditions", []) or [])
    return " ".join(parts).lower()


def _score_guide(problem_type: str, answer_type: str, text: str, target_prefix: str, keywords: List[str]) -> int:
    score = 0
    if problem_type.startswith(target_prefix):
        score += 5
    for kw in keywords:
        if re.search(r"\b" + re.escape(kw) + r"s?\b", text):
            score += 2
    return score


def _has_any(text: str, patterns: List[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _score_lc_state(problem_type: str, answer_type: str, text: str) -> int:
    if answer_type != "numeric_compute":
        return 0
    has_lc_context = bool(re.search(r"\blc\b", text)) or (
        "electric field energy" in text and "magnetic field energy" in text
    )
    if not has_lc_context:
        return 0
    has_complementary_state = _has_any(text, [
        r"\bcurrent\b.*\bmaximum\b",
        r"\bcurrent\b.*\bmax\b",
        r"\belectric field energy\b.*\bmaximum\b",
        r"\belectric field energy\b.*\bmax\b",
        r"\bmagnetic field energy\b.*\bmaximum\b",
        r"\bmagnetic field energy\b.*\bmax\b",
        r"\bwl\s*=\s*0\b",
        r"\bw_l\s*=\s*0\b",
    ])
    if not has_complementary_state:
        return 0
    score = 0
    if re.search(r"\blc\b", text):
        score += 3
    if _has_any(text, [r"\bcurrent\b", r"\bvoltage\b", r"\bcharge\b", r"\benergy\b"]):
        score += 2
    if _has_any(text, [r"\bmaximum\b", r"\bminimum\b", r"\bzero\b", r"\bmax\b", r"\bmin\b"]):
        score += 2
    if _has_any(text, [r"\bcapacitor\b", r"\binductor\b", r"\belectric field energy\b", r"\bmagnetic field energy\b"]):
        score += 2
    return score


def _score_rlc_reactance_scaling(problem_type: str, answer_type: str, text: str) -> int:
    if answer_type != "numeric_compute":
        return 0
    score = 0
    if re.search(r"\brlc\b", text):
        score += 2
    if _has_any(text, [r"\bxl\b", r"\bx_l\b", r"\binductive reactance\b"]):
        score += 3
    if _has_any(text, [r"\bxc\b", r"\bx_c\b", r"\bcapacitive reactance\b"]):
        score += 3
    if _has_any(text, [r"\bfrequency\b", r"\bdoubled\b", r"\btripled\b", r"\bincreased\b", r"\bdecreased\b"]):
        score += 2
    if _has_any(text, [r"\bpower\b", r"\bimpedance\b", r"\bcurrent\b"]):
        score += 2
    return score


def _score_qa_kinematics(problem_type: str, answer_type: str, text: str) -> int:
    if answer_type != "numeric_compute":
        return 0
    score = 0
    if problem_type.startswith("QA"):
        score += 3
    if _has_any(text, [r"\bkm/h\b", r"\bm/s\b", r"\bkm\b", r"\bhours?\b", r"\bminutes?\b"]):
        score += 2
    if _has_any(text, [r"\bspeed\b", r"\bvelocity\b", r"\bdistance\b", r"\btime\b", r"\btravel(?:s|ed|ing)?\b", r"\bpath\b"]):
        score += 2
    if _has_any(text, [r"\bcar\b", r"\bvehicle\b", r"\bmotorbike\b", r"\bmotorcycle\b", r"\bmotorcyclist\b", r"\bcyclist\b", r"\bmotorboat\b", r"\bboat\b", r"\bplane\b", r"\bairplane\b"]):
        score += 2
    if _has_any(text, [r"\bdownstream\b", r"\bupstream\b", r"\bcurrent\b", r"\bmeet\b", r"\bcatch(?:es)? up\b", r"\breturn\b", r"\bopposite\b", r"\bsame direction\b", r"\bapart\b", r"\bstart\b", r"\btowards?\b", r"\bfrom a to b\b"]):
        score += 2
    return score


def _score_coulomb_vectors(problem_type: str, answer_type: str, text: str) -> int:
    if answer_type != "numeric_compute":
        return 0
    score = 0
    if problem_type.startswith("LD"):
        score += 3
    has_point_charges = _has_any(text, [r"\bpoint charges?\b", r"\bcharges?\s+q[0-9]?\b", r"\bq[0-9]?\s*=", r"\btest charge\b"])
    has_force_context = _has_any(text, [r"\bforce\b", r"\bcoulomb\b", r"\bnewton\b"])
    if not (problem_type.startswith("LD") or has_point_charges or has_force_context):
        return 0
    if _has_any(text, [r"\bcharge\b", r"\bcharges\b", r"\bpoint charge\b", r"\bq[0-9]?\b"]):
        score += 3
    if has_force_context:
        score += 3
    if _has_any(text, [r"\btriangle\b", r"\bmidpoint\b", r"\bvertices\b", r"\bperpendicular\b", r"\bcenter\b"]):
        score += 2
    return score


def _score_electric_field(problem_type: str, answer_type: str, text: str) -> int:
    if answer_type != "numeric_compute":
        return 0
    if re.search(r"\belectric field energy\b", text):
        return 0
    score = 0
    if problem_type.startswith("DT"):
        score += 3
    if _has_any(text, [r"\belectric field\b", r"\bfield strength\b", r"\bv/m\b", r"\bn/c\b"]):
        score += 4
    if _has_any(text, [r"\bcharge\b", r"\bpoint charge\b", r"\bmidpoint\b"]):
        score += 2
    if _has_any(text, [r"\bdielectric\b", r"\bepsilon\b", r"\balcohol\b", r"\bwater\b"]):
        score += 2
    if _has_any(text, [r"\bdisk\b", r"\bsurface charge density\b", r"\baxis\b", r"\bz-axis\b"]):
        score += 4
    return score


def _score_faraday(problem_type: str, answer_type: str, text: str) -> int:
    if answer_type != "numeric_compute":
        return 0
    score = 0
    if problem_type.startswith("DDT"):
        score += 3
    if _has_any(text, [r"\bmagnetic flux\b", r"\bflux per turn\b", r"\bwb\b"]):
        score += 4
    if _has_any(text, [r"\bemf\b", r"\belectromotive force\b", r"\binduced\b", r"\bfaraday\b"]):
        score += 4
    if _has_any(text, [r"\bturns?\b", r"\bsolenoid\b", r"\bloop\b", r"\bcoil\b"]):
        score += 2
    return score


def _score_parallel_plate_capacitor(problem_type: str, answer_type: str, text: str) -> int:
    if answer_type != "numeric_compute":
        return 0
    has_change = _has_any(text, [
        r"\bchanged?\b", r"\breplaced?\b", r"\bnew\b", r"\binitial\b",
        r"\bfrom\b.*\bto\b", r"\bif the plate separation\b",
    ])
    if not has_change:
        return 0
    score = 0
    if problem_type.startswith("TD"):
        score += 3
    if _has_any(text, [r"\bparallel[- ]plate\b", r"\bplate separation\b", r"\bseparation\b"]):
        score += 3
    if _has_any(text, [r"\bcapacitance\b", r"\bcapacitor\b", r"\bpf\b"]):
        score += 3
    if _has_any(text, [r"\bdielectric\b", r"\bconstant\b", r"\bepsilon\b"]):
        score += 2
    return score


def _score_nl_sinusoidal_energy(problem_type: str, answer_type: str, text: str) -> int:
    if answer_type != "numeric_compute":
        return 0
    score = 0
    if problem_type == "NL":
        score += 2
    if re.search(r"\b(energy|stored energy|field energy)\b", text):
        score += 3
    if re.search(r"\b(capacitor|capacitance|inductor|inductance|coil|lc circuit)\b", text):
        score += 2
    if re.search(r"\b(u\(t\)|i\(t\)|q\(t\)|voltage function|current function|charge function|sin|cos|sine|cosine)\b", text):
        score += 3
    if re.search(r"\b(maximum|max|at time|instantaneous|t\s*=)\b", text):
        score += 2
    return score


def select_knowledge_guides(
    problem_type: str,
    answer_type: str,
    raw_question: str,
    normalized_problem: Dict[str, Any],
    max_guides: int = 2,
) -> List[Dict[str, Any]]:
    """Select a small number of relevant guides for Module 4."""
    text = _flatten_problem_text(raw_question, normalized_problem)
    candidates = [
        (_score_rlc_reactance_scaling(problem_type, answer_type, text), RLC_REACTANCE_SCALING_GUIDE),
        (_score_lc_state(problem_type, answer_type, text), LC_STATE_GUIDE),
        (_score_nl_sinusoidal_energy(problem_type, answer_type, text), NL_SINUSOIDAL_ENERGY_GUIDE),
        (_score_qa_kinematics(problem_type, answer_type, text), QA_KINEMATICS_GUIDE),
        (_score_coulomb_vectors(problem_type, answer_type, text), LD_COULOMB_GUIDE),
        (_score_electric_field(problem_type, answer_type, text), DT_ELECTRIC_FIELD_GUIDE),
        (_score_faraday(problem_type, answer_type, text), DDT_FARADAY_GUIDE),
        (_score_parallel_plate_capacitor(problem_type, answer_type, text), TD_PARALLEL_PLATE_CAPACITOR_GUIDE),
        (_score_guide(problem_type, answer_type, text, "CHLT", ["resonance", "resonate", "frequency", "lc", "circuit"]), CHLT_RESONANCE_GUIDE),
    ]
    
    selected = [
        guide for score, guide in sorted(candidates, key=lambda item: item[0], reverse=True)
        if score >= 5
    ]
    return selected[:max_guides]
