"""
Axelrod 1980 Tournament Simulation — Round-Adjusted Complexity Edition
Utility = raw average score − (complexity penalty * MOVES)
"""

import random
import itertools
import numpy as np
from collections import defaultdict

# ─────────────────────────────────────────────
# PAYOFF MATRIX & CONSTANTS
# ─────────────────────────────────────────────
PAYOFF = {
    ('C', 'C'): (3, 3),
    ('C', 'D'): (0, 5),
    ('D', 'C'): (5, 0),
    ('D', 'D'): (1, 1),
}
MOVES = 200   
RUNS  = 5     

COMPLEXITY_PENALTY_PER_ROUND = {
    "TIT FOR TAT":           1.105,
    "TIDEMAN AND CHIERUZZI": 7.200,
    "NYDEGGER":              1.415,
    "GROFMAN":               2.105,
    "SHUBIK":                2.205,
    "STEIN AND RAPOPORT":    4.000,
    "FRIEDMAN":              1.705,
    "DAVIS":                 1.805,
    "GRAASKAMP":             4.105,
    "DOWNING":               4.205,
    "FELD":                  2.205,
    "JOSS":                  2.205,
    "TULLOCK":               3.150,
    "(Name Withheld)":       4.050,
    "RANDOM":                1.500,
}

# ─────────────────────────────────────────────
# BASE CLASS
# ─────────────────────────────────────────────
class Strategy:
    """All strategies inherit from this.
    history_self / history_opp are lists of 'C'/'D' strings."""
    name = "Base"

    def reset(self):
        self.history_self = []
        self.history_opp  = []

    def move(self) -> str:
        raise NotImplementedError

    def record(self, my_move: str, opp_move: str):
        self.history_self.append(my_move)
        self.history_opp.append(opp_move)


# ─────────────────────────────────────────────
# 1. TIT FOR TAT  (Anatol Rapoport) — 1st place
# ─────────────────────────────────────────────
class TitForTat(Strategy):
    name = "TIT FOR TAT"
    def move(self):
        if not self.history_opp:
            return 'C'
        return self.history_opp[-1]


# ─────────────────────────────────────────────
# 2. TIDEMAN AND CHIERUZZI — 2nd place
# ─────────────────────────────────────────────
class TidemanChieruzzi(Strategy):
    name = "TIDEMAN AND CHIERUZZI"
    def reset(self):
        super().reset()
        self._punishment_remaining = 0
        self._punishment_level     = 0
        self._last_fresh_start     = 0
        self._fresh_start_pending  = 0
        self._opp_defect_runs      = 0
        self._in_opp_defect_run    = False

    def move(self):
        t = len(self.history_self)
        if t >= 198:
            return 'D'
        if self._fresh_start_pending > 0:
            self._fresh_start_pending -= 1
            return 'C'
        if self._punishment_remaining > 0:
            self._punishment_remaining -= 1
            return 'D'
        if not self.history_opp:
            return 'C'
        last_opp = self.history_opp[-1]
        if last_opp == 'D':
            if not self._in_opp_defect_run:
                self._in_opp_defect_run = True
        else:
            if self._in_opp_defect_run:
                self._in_opp_defect_run = False
                self._opp_defect_runs  += 1
                self._punishment_level += 1
                self._punishment_remaining = self._punishment_level
                self._maybe_fresh_start(t)
        if last_opp == 'D' and self._punishment_remaining == 0:
            self._punishment_remaining = self._punishment_level
            return 'D'
        return last_opp

    def _maybe_fresh_start(self, t):
        my_score  = sum(PAYOFF[(m, o)][0]
                        for m, o in zip(self.history_self, self.history_opp))
        opp_score = sum(PAYOFF[(m, o)][1]
                        for m, o in zip(self.history_self, self.history_opp))
        moves_remaining = MOVES - t
        defects  = self.history_opp.count('D')
        n = len(self.history_opp)
        expected = n / 2
        std      = (n * 0.25) ** 0.5
        stat_ok  = abs(defects - expected) >= 3 * std if std > 0 else False
        cond1 = my_score >= opp_score + 10
        cond2 = not self._in_opp_defect_run
        cond3 = (t - self._last_fresh_start) >= 20
        cond4 = moves_remaining >= 10
        cond5 = stat_ok
        if cond1 and cond2 and cond3 and cond4 and cond5:
            self._last_fresh_start     = t
            self._punishment_level     = 0
            self._punishment_remaining = 0
            self._fresh_start_pending  = 2


# ─────────────────────────────────────────────
# 3. NYDEGGER — 3rd place
# ─────────────────────────────────────────────
class Nydegger(Strategy):
    name = "NYDEGGER"
    DEFECT_SET = {1,6,7,17,22,23,26,29,30,31,33,38,39,45,49,54,55,58,61}

    def move(self):
        t = len(self.history_self)
        if t == 0:
            return 'C'
        if t == 1:
            if self.history_self[0]=='C' and self.history_opp[0]=='D':
                return 'D'
            return 'C'
        if t == 2:
            if self.history_self[0]=='C' and self.history_opp[0]=='D' \
               and self.history_self[1]=='D':
                return 'D'
            return self.history_opp[-1]
        A = 0
        weights = [16, 4, 1]
        for i, w in enumerate(weights):
            idx = t - 3 + i
            opp_d  = 2 if self.history_opp[idx]  == 'D' else 0
            self_d = 1 if self.history_self[idx] == 'D' else 0
            A += w * (opp_d + self_d)
        return 'D' if A in self.DEFECT_SET else 'C'


# ─────────────────────────────────────────────
# 4. GROFMAN — 4th place
# ─────────────────────────────────────────────
class Grofman(Strategy):
    name = "GROFMAN"
    def move(self):
        if not self.history_self:
            return 'C'
        if self.history_self[-1] != self.history_opp[-1]:
            return 'C' if random.random() < 2/7 else 'D'
        return 'C'


# ─────────────────────────────────────────────
# 5. SHUBIK — 5th place
# ─────────────────────────────────────────────
class Shubik(Strategy):
    name = "SHUBIK"
    def reset(self):
        super().reset()
        self._punishment_count = 0
        self._punishment_level = 0

    def move(self):
        if not self.history_opp:
            return 'C'
        if self._punishment_count > 0:
            self._punishment_count -= 1
            return 'D'
        if self.history_opp[-1] == 'D':
            self._punishment_level += 1
            self._punishment_count  = self._punishment_level - 1
            return 'D'
        return 'C'


# ─────────────────────────────────────────────
# 6. STEIN AND RAPOPORT — 6th place
# ─────────────────────────────────────────────
class SteinRapoport(Strategy):
    name = "STEIN AND RAPOPORT"
    def reset(self):
        super().reset()
        self._opponent_random = False

    def _chi_square_random(self):
        n = len(self.history_opp)
        if n < 10:
            return False
        cc = cd = dc = dd = 0
        for i in range(len(self.history_opp)-1):
            prev = self.history_opp[i]
            curr = self.history_opp[i+1]
            if   prev=='C' and curr=='C': cc += 1
            elif prev=='C' and curr=='D': cd += 1
            elif prev=='D' and curr=='C': dc += 1
            else: dd += 1
        total_after_c = cc + cd
        total_after_d = dc + dd
        if total_after_c == 0 or total_after_d == 0:
            return False
        chi2 = 0
        for obs, total in [(cc, total_after_c),(cd, total_after_c),
                           (dc, total_after_d),(dd, total_after_d)]:
            exp = total * 0.5
            if exp > 0:
                chi2 += (obs - exp)**2 / exp
        return chi2 < 3.84

    def move(self):
        t = len(self.history_self)
        if t < 4:
            return 'C'
        if t >= 198:
            return 'D'
        if t % 15 == 0:
            self._opponent_random = self._chi_square_random()
        if self._opponent_random:
            return 'D'
        return self.history_opp[-1]


# ─────────────────────────────────────────────
# 7. FRIEDMAN — 7th place
# ─────────────────────────────────────────────
class Friedman(Strategy):
    name = "FRIEDMAN"
    def reset(self):
        super().reset()
        self._triggered = False

    def move(self):
        if not self.history_opp:
            return 'C'
        if self.history_opp[-1] == 'D':
            self._triggered = True
        return 'D' if self._triggered else 'C'


# ─────────────────────────────────────────────
# 8. DAVIS — 8th place
# ─────────────────────────────────────────────
class Davis(Strategy):
    name = "DAVIS"
    def reset(self):
        super().reset()
        self._triggered = False

    def move(self):
        t = len(self.history_self)
        if t < 10:
            return 'C'
        if self.history_opp[-1] == 'D':
            self._triggered = True
        return 'D' if self._triggered else 'C'


# ─────────────────────────────────────────────
# 9. GRAASKAMP — 9th place
# ─────────────────────────────────────────────
class Graaskamp(Strategy):
    name = "GRAASKAMP"
    def reset(self):
        super().reset()
        self._mode      = 'tft_initial'
        self._defect_cd = None

    def move(self):
        t = len(self.history_self)
        if t < 50:
            return 'C' if t == 0 else self.history_opp[-1]
        if t == 50:
            return 'D'
        if t < 56:
            return self.history_opp[-1]
        if t == 56:
            self._classify()
        return self._act(t)

    def _classify(self):
        is_tft  = all(self.history_opp[i] == self.history_self[i-1]
                      for i in range(1, len(self.history_self)))
        is_twin = self.history_opp == self.history_self
        my_score = sum(PAYOFF[(m,o)][0]
                       for m,o in zip(self.history_self, self.history_opp))
        low_score = my_score < 2 * len(self.history_self)
        if is_twin or is_tft:
            self._mode = 'cooperative'
        elif low_score:
            self._mode = 'always_defect'
        else:
            self._mode = 'tft_occasional'
            self._defect_cd = random.randint(5, 15)

    def _act(self, t):
        if self._mode == 'cooperative':
            return self.history_opp[-1]
        if self._mode == 'always_defect':
            return 'D'
        if self._defect_cd is not None:
            self._defect_cd -= 1
            if self._defect_cd <= 0:
                self._defect_cd = random.randint(5, 15)
                return 'D'
        return self.history_opp[-1]


# ─────────────────────────────────────────────
# 10. DOWNING — 10th place
# ─────────────────────────────────────────────
class Downing(Strategy):
    name = "DOWNING"
    def reset(self):
        super().reset()
        self._after_c = [0, 0]
        self._after_d = [0, 0]

    def move(self):
        t = len(self.history_self)
        if t > 0:
            last_my  = self.history_self[-1]
            last_opp = self.history_opp[-1]
            opp_coop = 1 if last_opp == 'C' else 0
            if last_my == 'C':
                self._after_c[0] += opp_coop
                self._after_c[1] += 1
            else:
                self._after_d[0] += opp_coop
                self._after_d[1] += 1
        p_c = self._after_c[0]/self._after_c[1] if self._after_c[1] > 0 else 0.5
        p_d = self._after_d[0]/self._after_d[1] if self._after_d[1] > 0 else 0.5
        val_c = 3 * p_c
        val_d = 5 * p_d + 1 * (1 - p_d)
        return 'C' if val_c >= val_d else 'D'


# ─────────────────────────────────────────────
# 11. FELD — 11th place
# ─────────────────────────────────────────────
class Feld(Strategy):
    name = "FELD"
    def move(self):
        t = len(self.history_self)
        if not self.history_opp:
            return 'C'
        if self.history_opp[-1] == 'D':
            return 'D'
        p = 1.0 - 0.5 * (t / MOVES)
        return 'C' if random.random() < p else 'D'


# ─────────────────────────────────────────────
# 12. JOSS — 12th place
# ─────────────────────────────────────────────
class Joss(Strategy):
    name = "JOSS"
    def move(self):
        if not self.history_opp:
            return 'C'
        if self.history_opp[-1] == 'D':
            return 'D'
        return 'D' if random.random() < 0.10 else 'C'


# ─────────────────────────────────────────────
# 13. TULLOCK — 13th place
# ─────────────────────────────────────────────
class Tullock(Strategy):
    name = "TULLOCK"
    def move(self):
        t = len(self.history_self)
        if t < 11:
            return 'C'
        last10   = self.history_opp[-10:]
        opp_rate = last10.count('C') / 10
        my_rate  = max(0.0, opp_rate - 0.10)
        return 'C' if random.random() < my_rate else 'D'


# ─────────────────────────────────────────────
# 14. NAME WITHHELD — 14th place
# ─────────────────────────────────────────────
class NameWithheld(Strategy):
    name = "(Name Withheld)"
    def reset(self):
        super().reset()
        self._p = 0.30

    def move(self):
        t = len(self.history_self)
        if t > 0 and t % 10 == 0:
            self._update(t)
        return 'C' if random.random() < self._p else 'D'

    def _update(self, t):
        last10_opp = self.history_opp[-10:]
        opp_rate   = last10_opp.count('C') / 10
        if abs(opp_rate - 0.5) < 0.15:
            self._p = 0.5
        elif opp_rate > 0.7:
            self._p = min(self._p + 0.05, 0.95)
        elif opp_rate < 0.3:
            self._p = max(self._p - 0.05, 0.05)
        if t >= 130:
            my_score  = sum(PAYOFF[(m,o)][0]
                            for m,o in zip(self.history_self, self.history_opp))
            opp_score = sum(PAYOFF[(m,o)][1]
                            for m,o in zip(self.history_self, self.history_opp))
            if my_score < opp_score:
                self._p = max(self._p - 0.05, 0.05)


# ─────────────────────────────────────────────
# 15. RANDOM — 15th place
# ─────────────────────────────────────────────
class RandomStrategy(Strategy):
    name = "RANDOM"
    def move(self):
        return 'C' if random.random() < 0.5 else 'D'


# ─────────────────────────────────────────────
# GAME ENGINE
# ─────────────────────────────────────────────
def play_game(s1: Strategy, s2: Strategy, moves: int = MOVES):
    """Play one game between two strategies. Returns (score1, score2)."""
    s1.reset()
    s2.reset()
    total1 = total2 = 0
    for _ in range(moves):
        m1 = s1.move()
        m2 = s2.move()
        p1, p2 = PAYOFF[(m1, m2)]
        total1 += p1
        total2 += p2
        s1.record(m1, m2)
        s2.record(m2, m1)
    return total1, total2


# ─────────────────────────────────────────────
# TOURNAMENT
# ─────────────────────────────────────────────
def run_tournament(strategies, runs: int = RUNS):
    """
    Round-robin: each pair plays `runs` times.
    Returns dict: name → average raw score per game.
    """
    n = len(strategies)
    cumulative = defaultdict(float)
    game_count  = defaultdict(int)

    pairs      = list(itertools.combinations(range(n), 2))
    twin_pairs = [(i, i) for i in range(n)]
    all_pairs  = pairs + twin_pairs

    for run in range(runs):
        for i, j in all_pairs:
            s1 = strategies[i]
            s2 = strategies[j]
            if i == j:
                s1c = type(s1)()
                s2c = type(s2)()
                sc1, sc2 = play_game(s1c, s2c)
                cumulative[s1.name] += sc1
                game_count[s1.name] += 1
            else:
                sc1, sc2 = play_game(s1, s2)
                cumulative[s1.name] += sc1
                cumulative[s2.name] += sc2
                game_count[s1.name] += 1
                game_count[s2.name] += 1

    averages = {name: cumulative[name] / game_count[name]
                for name in cumulative}
    return averages


# ─────────────────────────────────────────────
# PAIRWISE SCORE TABLE
# ─────────────────────────────────────────────
def pairwise_scores(strategies, runs: int = RUNS):
    """Returns matrix[i][j] = average score of strategy i when playing j."""
    n = len(strategies)
    scores = np.zeros((n, n))
    counts = np.zeros((n, n))
    for run in range(runs):
        for i in range(n):
            for j in range(n):
                if i == j:
                    s1 = type(strategies[i])()
                    s2 = type(strategies[j])()
                else:
                    s1 = strategies[i]
                    s2 = strategies[j]
                sc1, _ = play_game(s1, s2)
                scores[i][j] += sc1
                counts[i][j] += 1
    return scores / counts


# ─────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────
AXELROD_ORIGINAL = {
    "TIT FOR TAT":            504,
    "TIDEMAN AND CHIERUZZI":  500,
    "NYDEGGER":               486,
    "GROFMAN":                482,
    "SHUBIK":                 481,
    "STEIN AND RAPOPORT":     478,
    "FRIEDMAN":               473,
    "DAVIS":                  472,
    "GRAASKAMP":              401,
    "DOWNING":                391,
    "FELD":                   328,
    "JOSS":                   304,
    "TULLOCK":                301,
    "(Name Withheld)":        282,
    "RANDOM":                 276,
}


# ─────────────────────────────────────────────
# RESULTS CALCULATION
# ─────────────────────────────────────────────
def print_results(averages):
    # Penalty is now scaled by the number of moves (rounds)
    utilities = {
        name: score - (COMPLEXITY_PENALTY_PER_ROUND[name] * MOVES)
        for name, score in averages.items()
    }

    # Build a lookup: original rank from Axelrod's 1980 results
    AXELROD_ORIGINAL = {
        "TIT FOR TAT": 504, "TIDEMAN AND CHIERUZZI": 500, "NYDEGGER": 486,
        "GROFMAN": 482, "SHUBIK": 481, "STEIN AND RAPOPORT": 478,
        "FRIEDMAN": 473, "DAVIS": 472, "GRAASKAMP": 401, "DOWNING": 391,
        "FELD": 328, "JOSS": 304, "TULLOCK": 301, "(Name Withheld)": 282, "RANDOM": 276
    }
    orig_rank = {name: i+1 for i, name in 
                 enumerate(sorted(AXELROD_ORIGINAL, key=lambda n: -AXELROD_ORIGINAL[n]))}

    adj_ranked = sorted(utilities.items(), key=lambda x: -x[1])

    print("\n" + "="*105)
    print(f"COMPLEXITY-ADJUSTED UTILITY (Raw Score - [Penalty * {MOVES} Rounds])")
    print("="*105)
    print(f"{'RANK':<6} {'STRATEGY':<28} {'RAW SCORE':>10} {'PEN/RND':>9} {'TOT PENALTY':>12} {'UTILITY':>12} {'ORIG'}")
    print("-"*105)

    for adj_rank, (name, util) in enumerate(adj_ranked, 1):
        raw      = averages[name]
        pen_rnd  = COMPLEXITY_PENALTY_PER_ROUND[name]
        total_pen = pen_rnd * MOVES
        o_rank   = orig_rank.get(name, '?')
        
        print(f"{adj_rank:<6} {name:<28} {raw:>10.1f} {pen_rnd:>9.3f} {total_pen:>12.1f} {util:>12.1f} {o_rank:>4}")
    print("="*105)

    # ── Summary statistics ─────────────────────────────────────────
    sim_scores  = [averages[n] for n in AXELROD_ORIGINAL]
    orig_scores = list(AXELROD_ORIGINAL.values())
    corr = np.corrcoef(sim_scores, orig_scores)[0,1]

    nice = {"TIT FOR TAT","TIDEMAN AND CHIERUZZI","NYDEGGER","GROFMAN",
            "SHUBIK","STEIN AND RAPOPORT","FRIEDMAN","DAVIS"}

    nice_util   = np.mean([utilities[n] for n in nice])
    nonice_util = np.mean([utilities[n] for n in utilities if n not in nice])

    print(f"\nRank correlation (raw simulated vs original): {corr:.4f}")
    print(f"\nComplexity-adjusted averages:")
    print(f"  Nice strategies     : {nice_util:.3f}")
    print(f"  Non-nice strategies : {nonice_util:.3f}")
    print(f"  Nice advantage      : {nice_util - nonice_util:.3f} utility points")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    strategies = [
        TitForTat(), TidemanChieruzzi(), Nydegger(), Grofman(),
        Shubik(), SteinRapoport(), Friedman(), Davis(),
        Graaskamp(), Downing(), Feld(), Joss(),
        Tullock(), NameWithheld(), RandomStrategy(),
    ]

    print("Running Axelrod 1980 Tournament Simulation (Complexity-Adjusted)...")
    print(f"  {len(strategies)} strategies | {MOVES} moves/game | {RUNS} runs\n")

    averages = run_tournament(strategies, runs=RUNS)
    print_results(averages)

    # ── Pairwise matrix ────────────────────────────────────────────
    print("\n\nPAIRWISE AVERAGE SCORES (row vs column, raw):")
    names  = [s.name for s in strategies]
    matrix = pairwise_scores(strategies, runs=RUNS)
    header = f"{'':28}" + "".join(f"{n[:6]:>8}" for n in names)
    print(header)
    for i, row in enumerate(matrix):
        line = f"{names[i]:<28}" + "".join(f"{v:>8.0f}" for v in row)
        print(line)
        