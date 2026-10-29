from __future__ import annotations

from functools import lru_cache
from itertools import combinations_with_replacement
import argparse
import hashlib
import random
import time
from pathlib import Path


EVENTS = ("U", "N", "T")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            x = f.read(1024 * 1024)
            if not x:
                break
            h.update(x)
    return h.hexdigest()


def canon(state):
    return tuple(sorted(int(x) for x in state))


def dec(x):
    return max(0, int(x) - 1)


def dec_coordinate(state, j):
    a = list(state)
    a[j] = dec(a[j])
    return canon(a)


def unknown_step(state):
    return canon(
        dec(x)
        for x in state
    )


def decode_sequence(code, horizon):
    out = []
    x = int(code)

    for _ in range(horizon):
        out.append(
            EVENTS[
                x % 3
            ]
        )
        x //= 3

    return "".join(out)


def weak_top_dominates(x, y):
    """
    Retrieval-capacity dominance candidate.

    Sort descending and require every largest-m prefix sum
    of x to be >= the corresponding largest-m sum of y.

    This is used only as a proof-scaffold audit here.
    """

    a = sorted(
        x,
        reverse=True,
    )

    b = sorted(
        y,
        reverse=True,
    )

    if len(a) != len(b):
        return False

    sx = 0
    sy = 0

    for u, v in zip(a, b):
        sx += int(u)
        sy += int(v)

        if sx < sy:
            return False

    return True


def run_config(K, B, H):
    """
    Exhaustively test:

      every canonical slack state
      every U/N/T sequence of length <= H

    Greedy is implemented independently from OPT.
    """

    states = list(
        combinations_with_replacement(
            range(B + 1),
            K,
        )
    )

    # --------------------------------------------------------------
    # Offline exact optimum.
    #
    # Sequence is represented by:
    #   code    : base-3 digits
    #   horizon : remaining number of events
    #
    # Least significant digit is the NEXT event.
    # --------------------------------------------------------------

    @lru_cache(maxsize=None)
    def OPT(state, code, horizon):

        if horizon == 0:
            return 0

        event = int(
            code % 3
        )

        rest = int(
            code // 3
        )

        next_h = int(
            horizon - 1
        )

        # U: forced global erosion.
        if event == 0:

            ns = unknown_step(
                state
            )

            return OPT(
                ns,
                rest,
                next_h,
            )

        best = -1

        # Equal-valued coordinates produce identical canonical
        # successor states. Skip exact duplicates only.
        seen_values = set()

        for j, slack in enumerate(
            state
        ):

            if slack in seen_values:
                continue

            seen_values.add(
                slack
            )

            ns = dec_coordinate(
                state,
                j,
            )

            if event == 1:
                # N
                reward = 0

            else:
                # T
                reward = int(
                    slack > 0
                )

            cand = (
                reward
                +
                OPT(
                    ns,
                    rest,
                    next_h,
                )
            )

            if cand > best:
                best = cand

        assert best >= 0

        return int(best)

    # --------------------------------------------------------------
    # Candidate greedy.
    # --------------------------------------------------------------

    @lru_cache(maxsize=None)
    def GREEDY(state, code, horizon):

        if horizon == 0:
            return 0

        event = int(
            code % 3
        )

        rest = int(
            code // 3
        )

        next_h = int(
            horizon - 1
        )

        if event == 0:

            ns = unknown_step(
                state
            )

            return GREEDY(
                ns,
                rest,
                next_h,
            )

        if event == 1:

            # N:
            # consume the smallest slack.
            #
            # If state[0] == 0, this intentionally keeps
            # dumping non-targets into an already dead leaf.
            j = 0

            ns = dec_coordinate(
                state,
                j,
            )

            return GREEDY(
                ns,
                rest,
                next_h,
            )

        # T:
        # consume the smallest POSITIVE slack.
        positive_j = None

        for j, slack in enumerate(
            state
        ):

            if slack > 0:
                positive_j = j
                break

        if positive_j is None:

            # No leaf can repair current target.
            # Assign to a dead leaf.
            ns = state

            return GREEDY(
                ns,
                rest,
                next_h,
            )

        ns = dec_coordinate(
            state,
            positive_j,
        )

        return (
            1
            +
            GREEDY(
                ns,
                rest,
                next_h,
            )
        )

    # --------------------------------------------------------------
    # Structural transition lemmas.
    # These are not the theorem proof.
    # They test the intended majorization proof scaffold.
    # --------------------------------------------------------------

    local_transition_checks = 0

    for state in states:

        # N greedy successor should top-dominate any other
        # one-coordinate decrement.
        greedy_n = dec_coordinate(
            state,
            0,
        )

        for j in range(K):

            alt = dec_coordinate(
                state,
                j,
            )

            local_transition_checks += 1

            if not weak_top_dominates(
                greedy_n,
                alt,
            ):

                return {
                    "status":
                        "COUNTEREXAMPLE_LOCAL_N",

                    "K":
                        K,

                    "B":
                        B,

                    "state":
                        state,

                    "alternative_index":
                        j,

                    "greedy_successor":
                        greedy_n,

                    "alternative_successor":
                        alt,
                }

        # T:
        # among reward-preserving positive choices, decrementing
        # smallest positive slack should dominate alternatives.
        positive = [
            j
            for j, x in enumerate(state)
            if x > 0
        ]

        if positive:

            gj = positive[0]

            greedy_t = (
                dec_coordinate(
                    state,
                    gj,
                )
            )

            for j in positive:

                alt = dec_coordinate(
                    state,
                    j,
                )

                local_transition_checks += 1

                if not weak_top_dominates(
                    greedy_t,
                    alt,
                ):

                    return {
                        "status":
                            "COUNTEREXAMPLE_LOCAL_T",

                        "K":
                            K,

                        "B":
                            B,

                        "state":
                            state,

                        "greedy_index":
                            gj,

                        "alternative_index":
                            j,

                        "greedy_successor":
                            greedy_t,

                        "alternative_successor":
                            alt,
                    }

    # --------------------------------------------------------------
    # UNKNOWN-preservation of candidate dominance relation.
    # --------------------------------------------------------------

    dominance_pairs = 0

    for x in states:

        for y in states:

            if not weak_top_dominates(
                x,
                y,
            ):
                continue

            dominance_pairs += 1

            ux = unknown_step(
                x
            )

            uy = unknown_step(
                y
            )

            if not weak_top_dominates(
                ux,
                uy,
            ):

                return {
                    "status":
                        "COUNTEREXAMPLE_U_DOMINANCE",

                    "K":
                        K,

                    "B":
                        B,

                    "x":
                        x,

                    "y":
                        y,

                    "U_x":
                        ux,

                    "U_y":
                        uy,
                }

    # --------------------------------------------------------------
    # FULL exact greedy-vs-offline parity.
    # --------------------------------------------------------------

    cases = 0

    for horizon in range(
        H + 1
    ):

        num_sequences = (
            3 ** horizon
        )

        for code in range(
            num_sequences
        ):

            for state in states:

                cases += 1

                g = GREEDY(
                    state,
                    code,
                    horizon,
                )

                o = OPT(
                    state,
                    code,
                    horizon,
                )

                if g != o:

                    return {
                        "status":
                            "COUNTEREXAMPLE_GREEDY",

                        "K":
                            K,

                        "B":
                            B,

                        "state":
                            state,

                        "horizon":
                            horizon,

                        "sequence":
                            decode_sequence(
                                code,
                                horizon,
                            ),

                        "sequence_code":
                            int(code),

                        "greedy_reward":
                            int(g),

                        "optimal_reward":
                            int(o),

                        "cases_before_failure":
                            int(cases),
                    }

    return {
        "status":
            "PASS",

        "K":
            K,

        "B":
            B,

        "H":
            H,

        "canonical_states":
            len(states),

        "exact_cases":
            int(cases),

        "local_transition_checks":
            int(
                local_transition_checks
            ),

        "dominance_pairs":
            int(
                dominance_pairs
            ),

        "OPT_cache":
            str(
                OPT.cache_info()
            ),

        "GREEDY_cache":
            str(
                GREEDY.cache_info()
            ),
    }


def random_stress(
    cases,
    max_k,
    max_b,
    max_horizon,
    seed,
):
    """
    Independent random stress beyond the exhaustive cube.

    Each instance receives its own exact suffix DP so no
    heuristic or real-query implementation is reused.
    """

    rng = random.Random(
        seed
    )

    tested = 0

    for case_id in range(
        cases
    ):

        K = rng.randint(
            1,
            max_k,
        )

        B = rng.randint(
            1,
            max_b,
        )

        horizon = rng.randint(
            0,
            max_horizon,
        )

        state = canon(
            rng.randint(
                0,
                B,
            )
            for _ in range(K)
        )

        seq = tuple(
            rng.randrange(3)
            for _ in range(
                horizon
            )
        )

        @lru_cache(maxsize=None)
        def opt_suffix(i, s):

            if i == horizon:
                return 0

            event = seq[i]

            if event == 0:

                return opt_suffix(
                    i + 1,
                    unknown_step(s),
                )

            best = -1

            seen_values = set()

            for j, slack in enumerate(
                s
            ):

                if slack in seen_values:
                    continue

                seen_values.add(
                    slack
                )

                ns = dec_coordinate(
                    s,
                    j,
                )

                reward = (
                    1
                    if (
                        event == 2
                        and
                        slack > 0
                    )
                    else
                    0
                )

                best = max(
                    best,
                    reward
                    +
                    opt_suffix(
                        i + 1,
                        ns,
                    ),
                )

            return int(best)

        def greedy_suffix():

            s = state
            reward = 0

            for event in seq:

                if event == 0:

                    s = unknown_step(
                        s
                    )

                elif event == 1:

                    s = dec_coordinate(
                        s,
                        0,
                    )

                else:

                    positive_j = None

                    for j, slack in enumerate(
                        s
                    ):

                        if slack > 0:
                            positive_j = j
                            break

                    if positive_j is not None:

                        reward += 1

                        s = dec_coordinate(
                            s,
                            positive_j,
                        )

            return int(reward)

        g = greedy_suffix()

        o = opt_suffix(
            0,
            state,
        )

        tested += 1

        if g != o:

            return {
                "status":
                    "COUNTEREXAMPLE_RANDOM",

                "case_id":
                    int(case_id),

                "K":
                    K,

                "B":
                    B,

                "state":
                    state,

                "sequence":
                    "".join(
                        EVENTS[x]
                        for x in seq
                    ),

                "greedy_reward":
                    g,

                "optimal_reward":
                    o,

                "tested":
                    tested,
            }

    return {
        "status":
            "PASS",

        "tested":
            tested,

        "seed":
            seed,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max-b",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max-horizon",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--random-cases",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--random-max-k",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--random-max-b",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--random-max-horizon",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20261005,
    )

    args = parser.parse_args()

    print("=" * 120)
    print(
        "RANK-PREFIX GREEDY THEOREM COUNTEREXAMPLE AUDIT v0"
    )
    print(
        "INDEPENDENT ABSTRACT SLACK MODEL / EXACT OFFLINE DP"
    )
    print("=" * 120)

    print(
        "max K / B / horizon =",
        args.max_k,
        args.max_b,
        args.max_horizon,
    )

    print(
        "random cases =",
        args.random_cases,
    )

    total_exact_cases = 0
    total_states = 0
    total_local_checks = 0
    total_dom_pairs = 0

    t0 = time.perf_counter()

    for K in range(
        1,
        args.max_k + 1
    ):

        for B in range(
            1,
            args.max_b + 1
        ):

            q0 = time.perf_counter()

            out = run_config(
                K,
                B,
                args.max_horizon,
            )

            elapsed = (
                time.perf_counter()
                -
                q0
            )

            print(
                f"K={K} B={B} "
                f"status={out['status']} "
                f"time={elapsed:.3f}s"
            )

            if out["status"] != "PASS":

                print()
                print("=" * 120)
                print("COUNTEREXAMPLE FOUND")
                print("=" * 120)

                for k, v in out.items():
                    print(
                        f"{k:<32} = {v}"
                    )

                print()
                print(
                    "CANDIDATE THEOREM = REJECT"
                )

                print(
                    "SCRIPT SHA256 =",
                    sha256_file(
                        Path(__file__)
                    ),
                )

                return

            total_exact_cases += int(
                out[
                    "exact_cases"
                ]
            )

            total_states += int(
                out[
                    "canonical_states"
                ]
            )

            total_local_checks += int(
                out[
                    "local_transition_checks"
                ]
            )

            total_dom_pairs += int(
                out[
                    "dominance_pairs"
                ]
            )

    print()
    print(
        "ABSTRACT EXHAUSTIVE PARITY = PASS"
    )

    print(
        "total canonical states =",
        total_states,
    )

    print(
        "total exact state-sequence cases =",
        total_exact_cases,
    )

    print(
        "local dominance transition checks =",
        total_local_checks,
    )

    print(
        "UNKNOWN dominance pairs checked =",
        total_dom_pairs,
    )

    print()

    stress = random_stress(
        args.random_cases,
        args.random_max_k,
        args.random_max_b,
        args.random_max_horizon,
        args.seed,
    )

    print(
        "RANDOM STRESS STATUS =",
        stress["status"],
    )

    for k, v in stress.items():

        if k == "status":
            continue

        print(
            f"{k:<32} = {v}"
        )

    if stress["status"] != "PASS":

        print()
        print(
            "CANDIDATE THEOREM = REJECT"
        )

        print(
            "SCRIPT SHA256 =",
            sha256_file(
                Path(__file__)
            ),
        )

        return

    elapsed = (
        time.perf_counter()
        -
        t0
    )

    print()
    print("=" * 120)
    print(
        "RANK-PREFIX GREEDY CANDIDATE THEOREM AUDIT = PASS"
    )
    print("=" * 120)

    print(
        "IMPORTANT:"
    )

    print(
        "PASS is computational evidence only."
    )

    print(
        "It is NOT a mathematical proof."
    )

    print(
        "The next gate is a formal dominance/exchange proof."
    )

    print(
        "runtime seconds =",
        elapsed,
    )

    print(
        "SCRIPT SHA256 =",
        sha256_file(
            Path(__file__)
        ),
    )

    print("=" * 120)


if __name__ == "__main__":
    main()
