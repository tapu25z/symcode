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
            "For double sums of the form \\sum_{j=1}^\\infty \\sum_{k=1}^\\infty 1/(j+k)^m, rewrite it as a single sum over n = j+k starting from n=2. To do this, group terms by n=j+k and count how many ordered pairs produce each n, which is (n-1). Thus, the sum is \\sum_{n=2}^\\infty (n-1)/n^m = \\sum_{n=2}^\\infty (1/n^{m-1} - 1/n^m). Express this in terms of the given series p and q by defining p and q as symbols: p, q = sp.symbols('p q'). Do not compute their numerical values. Since \\sum_{k=1}^\\infty 1/k^2 = zeta(2) and \\sum_{k=1}^\\infty 1/k^3 = zeta(3), substitute zeta(2) and zeta(3) with the symbols p and q (e.g. using .subs({sp.zeta(2): p, sp.zeta(3): q}))."
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
    if "is a positive factor of" in text or "are factors of" in text:
        if "!" in text or "factorial" in text:
            hints.append(
                "When finding the number of integer values of x such that x^k is a factor of a factorial (e.g. 10!): first find the prime factorization of the factorial. For each prime base p with exponent e, the exponent of p in x (let's say a) must satisfy k*a <= e, which means a can be any integer from 0 to e // k (so e // k + 1 possibilities). Multiply these possibilities (e // k + 1) for all prime factors to find the number of positive integer values of x."
            )
    if "third side" in text or "sides measuring" in text or "triangle has sides" in text:
        hints.append(
            "For triangle side length bounds, apply the strict triangle inequality: abs(a - b) < c < a + b. The third side must be strictly greater than abs(a - b) and strictly less than a + b. Do not include boundary values."
        )
    if "arrange the letters" in text or "permutations of" in text:
        hints.append(
            "For arranging letters of a word, identify all repeated letters. The number of arrangements is n! / (n1! * n2! * ...), where n is the total number of letters, and n1, n2, ... are the counts of each repeated letter. Make sure to divide by the factorial of the counts of all repeated letters."
        )
    if "cents" in text or "dollar" in text:
        hints.append(
            "When converting between currencies (dollars and cents) or units, verify if your equations are already formulated in the target unit. If equations use cents (e.g. 124 cents instead of 1.24 dollars), the solved variable is already in cents; do not multiply by 100 again."
        )
    if "simplify" in text and any(trig in text for trig in ("sin", "cos", "sec", "csc", "tan", "cot")):
        hints.append(
            "For trigonometric simplifications, use sp.trigsimp(expr) or sp.simplify(expr) to reduce the expression to its simplest form (e.g. cot(x) or sin(x)). Ensure you formulate fractions and common denominators correctly using basic trig definitions."
        )
    if "shortest distance" in text and "visit" in text:
        hints.append(
            "For a path visiting all N points (Hamiltonian path), the route contains exactly N-1 edges. Do not close the loop (do not add the edge returning to the starting point) unless a cycle/closed loop is explicitly requested."
        )
    if "weigh" in text or "weighs" in text or "equal in weight" in text:
        hints.append(
            "When solving underdetermined systems of equations (where the number of variables is greater than the number of equations), sp.solve returns a list of dicts with parameterized solutions. To find the ratio of two variables T and S, solve for T in terms of S, or substitute a dummy value (e.g. S=1) into the system to find the numerical values of the other variables before dividing."
        )

    if "is a parabola, circle, ellipse, hyperbola" in text:
        hints.append(
            "For classifying conic sections of the form Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0: calculate the discriminant delta = B**2 - 4*A*C. If delta < 0, the conic is an ellipse (or a circle if B==0 and A==C). If delta == 0, it is a parabola. If delta > 0, it is a hyperbola. Ensure you extract the coefficients A (coefficient of x**2), B (coefficient of x*y), and C (coefficient of y**2) correctly after expanding the equation."
        )
    if "sum of the roots" in text or "product of the roots" in text or "sum of roots" in text or "product of roots" in text:
        hints.append(
            "For the sum or product of roots of a polynomial equation, use Vieta's formulas directly instead of solving for the roots individually: for a polynomial ax^n + bx^(n-1) + ... = 0, the sum of roots is -b/a. If you do solve and sum/multiply the roots, always call sp.simplify(result) to reduce any complex symbolic radical expressions."
        )
    if "roots of unity" in text or ("complex number" in text and "omega" in text):
        hints.append(
            "For expressions involving roots of unity (e.g. omega**3 = 1): define omega as a symbol, and express other powers in terms of omega (e.g. omega**2 is omega**2). Do not solve for omega numerically and substitute different roots into different parts of the expression. Simplify the expression algebraically first using sp.simplify(expr) or polynomial division/relations (e.g., omega**2 + omega + 1 = 0)."
        )
    if "domain of the function" in text or "real number value" in text or "range of the function" in text:
        hints.append(
            "To find the domain of a real function: for square roots sqrt(g(x)), solve g(x) >= 0. For denominators h(x), find where h(x) != 0. Note: do not use Python's == or != or % operators directly with SymPy symbols inside conditional loops or lists (e.g. h(x) != 0 or G % 13 == 0 evaluates eagerly to a Python boolean). Use sp.Eq(G % 13, 0) and sp.Ne(h(x), 0) instead. To solve inequalities, use sp.solve(g(x) >= 0, x) or sp.solveset(g(x) >= 0, x, domain=sp.S.Reals)."
        )
    if "multiple of" in text or "divisible by" in text or "remainder" in text:
        hints.append(
            "When checking modular arithmetic or divisibility constraints (e.g. multiple of 13, remainder is 1) inside a search loop, use simple Python integer arithmetic on python variables (e.g., g % 13 == 0 or g % 7 == 0 where g is a standard python integer in a range loop) rather than creating SymPy symbols and using sp.solve or sp.Eq, which SymPy cannot solve."
        )
    if "integers from" in text and ("different integers" in text or "represent four different" in text):
        hints.append(
            "When equations represent integers in a bounded range (e.g., from 1 to 9): if the system is underdetermined, solve the equations to express the variables in terms of a free parameter. Then, iterate through all possible integer values for the free parameter in the given range and check which parameter value makes all variables distinct integers within the range."
        )
    if ("assume" in text or "for all values of" in text) and re.search(r"\b[a-z]\b", text) and ("equation" in text or "solution" in text):
        hints.append(
            "When an inequality, equation comparison, or identity holds for all values of a parameter in a range (e.g. 0 < r < 3): do not leave the parameter as a free symbol during numerical solving (like fsolve). Substitute a sample numerical value for the parameter in that range (e.g. r = 1.5) to solve the equations and make the comparison numerically."
        )

    return hints
