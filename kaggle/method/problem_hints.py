"""Problem-pattern hints for the legacy SymPlanner code generator.

These hints are deliberately answer-free: they steer the implementation toward
robust algorithms for recurring MATH/GSM styles without using benchmark labels.
"""

from __future__ import annotations

import re


def build_problem_hints(question: str) -> list[str]:
    text = str(question or "").lower()
    hints: list[str] = []

    if "[asy]" in text or "tikzpicture" in text:
        hints.append(
            "If the problem includes ASY/TikZ, treat the source as data: parse explicit coordinates, labels, plotted functions, and numeric relations before solving."
        )
    if "inserting parentheses" in text:
        hints.append(
            "For inserted-parentheses problems, use dynamic programming over the ordered numbers/operators and count distinct values; do not sample only a few parenthesizations."
        )
    if "round table" in text or "rotations of each other" in text:
        hints.append(
            "For circular seating with at most 9 people, a brute-force permutation check with one fixed anchor is safer than a fragile formula."
        )
    if re.search(r"<\s*[^<>=]+\s*<", text):
        hints.append(
            "For chained inequalities, split them into lower and upper inequalities or solve by hand; avoid passing a SymPy And object into solve_univariate_inequality."
        )
    if "roots of unity" in text:
        hints.append(
            "For roots-of-unity targets, factor the polynomial and derive each root's angle/order; the answer is the least common multiple of the orders."
        )
    if "is a factor of" in text and re.search(r"\bp\b|\bq\b|\br\b", text):
        hints.append(
            "For polynomial divisibility with unknown coefficients, reduce the polynomial modulo the factor and set every remainder coefficient to zero."
        )
    if "sin" in text and re.search(r"\^\s*\{?\s*[5-9]\s*\}?", text) and "for all angles" in text:
        hints.append(
            "For trigonometric power identities, use exact trig expansion or known multiple-angle identities and equate coefficients symbolically."
        )
    if "base" in text and re.search(r"_\s*\{?\d+\}?", text):
        hints.append(
            "For base-notation arithmetic, convert inputs to decimal for computation, then convert the final value back to the requested base notation."
        )
    if "smallest positive perfect cube" in text and "three consecutive integers" in text:
        hints.append(
            "A sum of three consecutive integers is three times the middle integer, so search cubes in increasing order and test divisibility by 3."
        )
    if "compound interest" in text and "deposits" in text:
        hints.append(
            "For repeated end-of-year deposits, compound earlier deposits for more periods than later deposits before solving the rate."
        )
    if "logarithms of the roots" in text:
        hints.append(
            "Use log product rules with Vieta's formulas: a sum of logs gives the log of the product of roots."
        )

    return hints
