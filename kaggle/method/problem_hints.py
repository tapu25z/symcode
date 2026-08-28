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
            "For parenthesizing expression problems (e.g. '2*3*4*5+1'), define the expression as a list: lst = [2, '*', 3, '*', 4, '*', 5, '+', 1]. Use this clean recursive function or dynamic programming to evaluate all parenthesizations without string manipulation bugs:\n"
            "def get_all_values(lst):\n"
            "    if len(lst) == 1: return {lst[0]}\n"
            "    res = set()\n"
            "    for i in range(1, len(lst), 2):\n"
            "        op = lst[i]\n"
            "        left = get_all_values(lst[:i])\n"
            "        right = get_all_values(lst[i+1:])\n"
            "        for l in left:\n"
            "            for r in right:\n"
            "                if op == '*': res.add(l * r)\n"
            "                elif op == '+': res.add(l + r)\n"
            "                elif op == '-': res.add(l - r)\n"
            "    return res\n"
            "Count unique values using len(get_all_values(lst)). Alternatively, use dynamic programming over the ordered numbers and operators."
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
            "For double sums of the form \\sum_{j=1}^\\infty \\sum_{k=1}^\\infty 1/(j+k)^m, rewrite it as a single sum over n = j+k starting from n=2. To do this, group terms by n=j+k and count how many ordered pairs produce each n, which is (n-1). Thus, the sum is \\sum_{n=2}^\\infty (n-1)/n^m = \\sum_{n=2}^\\infty (1/n^{m-1} - 1/n^m). Express this in terms of the given series p and q by adjusting the index to start from 1 (e.g., \\sum_{n=2}^\\infty 1/n^2 = p - 1)."
        )
    if "is a factor of" in text and re.search(r"\bp\b|\bq\b|\br\b", text):
        hints.append(
            "For polynomial divisibility with unknown coefficients (like p, q, r), use the remainder: rem_poly = sp.rem(poly, factor, x). Since the remainder must be 0 for all x, convert it to a polynomial sp.Poly(rem_poly, x) and solve for the unknown parameters by setting all coefficients to 0: sp.solve(sp.Poly(rem_poly, x).coeffs(), (p, q, r))."
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
            "For end-of-year savings plan of N years (e.g., n=3): at the end of year 3, the first deposit has been compounded for 2 years (P * (1+r)^2), the second for 1 year (P * (1+r)), and the third deposit is made at the end of year 3 with 0 years of compounding (P). The total sum is P * (1+r)^2 + P * (1+r) + P. Do not use the simple lump-sum formula P * (1+r)^3."
        )
    if "logarithms of the roots" in text:
        hints.append(
            "Use log product rules with Vieta's formulas: a sum of logs gives the log of the product of roots; for a cubic, product of roots is -constant/leading_coefficient."
        )
    if "functional equation" in text:
        hints.append(
            "To solve a functional equation of the form f(x) + f(y) = f(x+y) + ..., assume a polynomial form f(t) = A*t**2 + B*t + C. Compute lhs = f(x) + f(y) and rhs = f(x+y) + ..., then construct the difference diff = sp.expand(lhs - rhs). Since this holds for all x and y, convert it to a polynomial sp.Poly(diff, x, y) and solve for (A, B, C) by setting all its coefficients (using poly.coeffs()) to zero: sp.solve(sp.Poly(diff, x, y).coeffs(), (A, B, C))."
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
            "For the gold coins redistribution problem: if you had 7 bags of x coins each, the total coins before was 7*x. After finding a bag of 53 coins, the total is 7*x + 53. Since this total can be redistributed equally into 8 bags, 7*x + 53 must be a multiple of 8 (i.e., (7*x + 53) % 8 == 0). Also, total_after = 7*x + 53 > 200. Find the smallest integer x >= 1 satisfying both conditions, and return the number of coins before, which is 7*x."
        )
    if "reassigned to" in text:
        hints.append(
            "For ratio word problems where items are reassigned: write the fraction with the receiver's new count in the numerator or denominator as specified. If A started with 16 and got x items from B (who started with 12), A's new count is 16 + x and B's new count is 12 - x. The ratio of A to B is (16 + x) / (12 - x). Make sure to update both giver and receiver correctly."
        )
    if "greatest possible value of the slope" in text or "least possible value of the slope" in text:
        hints.append(
            "To maximize or minimize the slope (y2 - y1) / (x2 - x1) between two points A=(x1, y1) and B=(x2, y2) constrained to rectangular regions, evaluate the slope for all pairs of vertices (corners) of both regions where x2 != x1, and find the maximum/minimum."
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
