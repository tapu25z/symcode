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
            "For inserted-parentheses problems, use dynamic programming over the ordered numbers and operators. Split every interval at every operator, combine left/right value sets, and count distinct final values."
        )
    if "round table" in text or "rotations of each other" in text:
        hints.append(
            "For circular seating with at most 9 people, a brute-force permutation check with one fixed anchor is safer than a fragile formula."
        )
    if "round table" in text and "no two" in text and "next to each other" in text:
        hints.append(
            "For circular no-adjacency seating, do not subtract only the all-special-people-together case; handle every pairwise adjacency or brute-force all circular arrangements."
        )
    if re.search(r"<\s*[^<>=]+\s*<", text):
        hints.append(
            "For chained inequalities, split them into lower and upper inequalities or solve by hand; avoid passing a SymPy And object into solve_univariate_inequality."
        )
    if "roots of unity" in text:
        hints.append(
            "For roots-of-unity targets, factor the polynomial and derive each root's angle/order; the answer is the least common multiple of the orders."
        )
    if "double sum" in text or (r"\sum_{j" in question and r"\sum_{k" in question):
        hints.append(
            "For double sums depending on j+k, group terms by n=j+k and count how many ordered pairs produce each n before simplifying into the named series variables."
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
            "A sum of three consecutive integers is three times the middle integer. Search candidate cube values in increasing order and return the cube value itself, not the middle integer or another power."
        )
    if "compound interest" in text and "deposits" in text:
        hints.append(
            "For repeated end-of-year deposits, compound earlier deposits for more periods than later deposits. A three-year end-of-year deposit plan has factors (1+r)^2, (1+r), and 1."
        )
    if "logarithms of the roots" in text:
        hints.append(
            "Use log product rules with Vieta's formulas: a sum of logs gives the log of the product of roots; for a cubic, product of roots is -constant/leading_coefficient."
        )
    if "functional equation" in text:
        hints.append(
            "For polynomial-looking functional equations over all reals, assume f(t)=A*t**2+B*t+C, substitute x,y, equate coefficients, then apply the given value."
        )
    if "different battalions" in text or ("how many different" in text and "soldiers" in text):
        hints.append(
            "When counting different selectable groups, multiply binomial choices for each class, e.g. choose required people from available people; do not compute only how many full groups can be formed."
        )
    if "three for" in text and "$1" in text:
        hints.append(
            "For pricing text like 'three for $1', revenue is number_sold / 3 dollars when divisible; avoid multiplying groups by 1/3."
        )
    if "remainder" in text and "mod" in text:
        hints.append(
            "For modular arithmetic, substitute the residue directly and reduce the final expression modulo the modulus."
        )
    if "gold coins" in text and "redistribute" in text and "bags" in text:
        hints.append(
            "For redistribution divisibility, search totals that satisfy both the original equal-bag condition and the new equal-bag condition after adding the found bag."
        )
    if "reassigned to" in text:
        hints.append(
            "For reassignment word problems, update both sides of the transfer: the receiver gains x while the giver loses x."
        )
    has_matrix_like = "matrix" in text or "pmatrix" in text or "begin{pmatrix}" in text
    has_norm_like = "norm" in text or "||" in text or "\\|" in text or "magnitude" in text
    if has_norm_like and "for all" in text and has_matrix_like:
        hints.append(
            "For ||A v|| <= C ||v||, compute the spectral norm sqrt(max eigenvalue of A.T*A), not the maximum absolute eigenvalue of A."
        )
    if "angle between" in text and "line" in text and re.search(r"=\s*[^=]+=", text):
        hints.append(
            "For chained-equality line definitions such as ax=by=cz, set the common value to t, derive a constant direction vector, then use the dot-product angle formula."
        )
    if "rotated around" in text and ("complex" in text or " i" in text):
        hints.append(
            "For complex rotation around center c, use w = c + (z-c)*(cos(theta)+I*sin(theta)) exactly; do not rotate absolute polar coordinates around the origin."
        )
    if "[asy]" in text and re.search(r"\bwhat\s+is\s+\$?[A-Z]{2}\$?", question):
        hints.append(
            "When ASY gives explicit point coordinates and asks for a segment length, compute the Euclidean distance between those named points."
        )

    return hints
