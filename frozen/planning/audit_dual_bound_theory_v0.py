from __future__ import annotations

from functools import lru_cache
from itertools import product
from pathlib import Path
import hashlib
import time

UNKNOWN = -1
ABSENT = 0
PRESENT = 1


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            x = f.read(1024 * 1024)
            if not x:
                break
            h.update(x)
    return h.hexdigest()


def popcount(x):
    return int(x.bit_count())


def prefix_mask(t):
    return (1 << (t + 1)) - 1


# ================================================================
# Rank repair STOP value.
# ================================================================

def stop_value(mask, target_mask, n, B):
    out = 0

    for t in range(n):
        bit = 1 << t

        if not (target_mask & bit):
            continue

        if not (mask & bit):
            continue

        rank = popcount(
            mask
            &
            prefix_mask(t)
        )

        if rank <= B:
            out += 1

    return int(out)


# ================================================================
# Exact legal predicate-policy optimum V(M,d,L).
#
# This is deliberately independent of Work2 solver code.
# ================================================================

def exact_policy_value(
    facts,
    valid,
    target_mask,
    B,
    depth,
    leaves,
):

    n = len(valid)
    P = len(facts[0])

    full_mask = (1 << n) - 1

    @lru_cache(maxsize=None)
    def V(mask, d, L):

        base = stop_value(
            mask,
            target_mask,
            n,
            B,
        )

        if (
            d <= 0
            or
            L <= 1
            or
            popcount(mask) <= B
        ):
            return int(base)

        best = int(base)

        for p in range(P):

            has0 = False
            has1 = False

            for i in range(n):
                bit = 1 << i

                if not (mask & bit):
                    continue

                if not valid[i]:
                    continue

                x = facts[i][p]

                if x == ABSENT:
                    has0 = True
                elif x == PRESENT:
                    has1 = True

            # Frozen legality semantics:
            # both definite answers must occur among valid survivors.
            if not (has0 and has1):
                continue

            m0 = 0
            m1 = 0

            for i in range(n):
                bit = 1 << i

                if not (mask & bit):
                    continue

                x = facts[i][p]

                if (
                    x == ABSENT
                    or
                    x == UNKNOWN
                ):
                    m0 |= bit

                if (
                    x == PRESENT
                    or
                    x == UNKNOWN
                ):
                    m1 |= bit

            assert m0 != mask
            assert m1 != mask

            for l0 in range(1, L):

                l1 = L - l0

                cand = (
                    V(
                        m0,
                        d - 1,
                        l0,
                    )
                    +
                    V(
                        m1,
                        d - 1,
                        l1,
                    )
                )

                if cand > best:
                    best = int(cand)

        return int(best)

    return V(
        full_mask,
        depth,
        leaves,
    )


# ================================================================
# RPC greedy capacity C_L.
#
# Candidate types:
# U = UNKNOWN invalid candidate -> consumes slack in ALL leaves.
# N = valid non-target -> arbitrary one-leaf assignment.
# T = valid target -> reward iff assigned leaf has positive slack.
# ================================================================

def rpc_capacity_greedy(
    valid,
    target_mask,
    B,
    leaves,
):

    n = len(valid)

    # Always maintained in nondecreasing order.
    slack = [B] * leaves

    reward = 0

    for i in range(n):

        bit = 1 << i

        # UNKNOWN survives every branch.
        if not valid[i]:

            slack = [
                max(0, s - 1)
                for s in slack
            ]

            continue

        is_target = bool(
            target_mask & bit
        )

        if is_target:

            chosen = None

            # Smallest positive slack.
            for j, s in enumerate(slack):
                if s > 0:
                    chosen = j
                    break

            if chosen is not None:
                reward += 1
                slack[chosen] -= 1
                slack.sort()

            continue

        # Non-target:
        # put it in the smallest-slack leaf.
        if slack[0] > 0:
            slack[0] -= 1
            slack.sort()

    return int(reward)


# ================================================================
# Independent brute-force arbitrary-leaf relaxation.
#
# This checks RPC greedy optimality directly on tiny instances.
# ================================================================

def rpc_capacity_bruteforce(
    valid,
    target_mask,
    B,
    leaves,
):

    n = len(valid)

    @lru_cache(maxsize=None)
    def F(i, slack):

        slack = tuple(slack)

        if i >= n:
            return 0

        bit = 1 << i

        # UNKNOWN is forced into every leaf.
        if not valid[i]:

            nxt = tuple(
                max(0, s - 1)
                for s in slack
            )

            return F(
                i + 1,
                nxt,
            )

        is_target = bool(
            target_mask & bit
        )

        best = -1

        # Valid item may be assigned to ANY one leaf.
        for j in range(leaves):

            nxt = list(slack)

            gain = 0

            if nxt[j] > 0:

                if is_target:
                    gain = 1

                nxt[j] -= 1

            # Canonicalize leaves:
            # arbitrary leaves are exchangeable.
            nxt.sort()

            cand = (
                gain
                +
                F(
                    i + 1,
                    tuple(nxt),
                )
            )

            if cand > best:
                best = cand

        return int(best)

    return F(
        0,
        tuple([B] * leaves),
    )


# ================================================================
# DQ finite-depth reachability upper bound Q_{d,L}.
# ================================================================

def dq_bound(
    facts,
    valid,
    target_mask,
    B,
    depth,
    leaves,
):

    n = len(valid)
    P = len(facts[0])

    k = min(
        int(depth),
        int(leaves) - 1,
    )

    feasible = 0

    for t in range(n):

        bit_t = 1 << t

        if not (target_mask & bit_t):
            continue

        # Full-state current rank.
        rank = t + 1

        if rank <= B:
            feasible += 1
            continue

        delta = rank - B

        sigmas = []

        for p in range(P):

            ans_t = facts[t][p]

            # Targets are always valid and hence definite.
            assert ans_t in (
                ABSENT,
                PRESENT,
            )

            opposite = 1 - ans_t

            sigma = 0

            for i in range(t + 1):

                # UNKNOWN cannot be separated.
                if not valid[i]:
                    continue

                if facts[i][p] == opposite:
                    sigma += 1

            sigmas.append(
                sigma
            )

        sigmas.sort(
            reverse=True
        )

        optimistic = sum(
            sigmas[:k]
        )

        if optimistic >= delta:
            feasible += 1

    return int(feasible)


# ================================================================
# Explicit incomparability constructions.
# ================================================================

def construction_checks():

    # ------------------------------------------------------------
    # A: RPC strictly tighter.
    #
    # B=1,L=2,d=1
    #
    # t1 = 00
    # t2 = 10
    # t3 = 01
    #
    # DQ:
    # t1 already repairable.
    # t2 can separate t1 using p0.
    # t3 can separate t1,t2 using p1.
    # Q=3.
    #
    # But two B=1 leaves can reward at most two targets:
    # C=2.
    # ------------------------------------------------------------

    facts_A = (
        (0, 0),
        (1, 0),
        (0, 1),
    )

    valid_A = (
        True,
        True,
        True,
    )

    target_A = 0b111

    C_A = rpc_capacity_greedy(
        valid_A,
        target_A,
        B=1,
        leaves=2,
    )

    Q_A = dq_bound(
        facts_A,
        valid_A,
        target_A,
        B=1,
        depth=1,
        leaves=2,
    )

    V_A = exact_policy_value(
        facts_A,
        valid_A,
        target_A,
        B=1,
        depth=1,
        leaves=2,
    )

    assert C_A == 2
    assert Q_A == 3
    assert C_A < Q_A
    assert V_A <= C_A
    assert V_A <= Q_A

    # ------------------------------------------------------------
    # B: DQ strictly tighter.
    #
    # Order:
    # t1, t2, n
    #
    # p:
    #  0   0   1
    #
    # Predicate is globally legal because n has value 1.
    # But n comes after t2 and cannot separate t2 from blocker t1.
    #
    # Arbitrary two-leaf capacity can separate t1/t2:
    # C=2.
    #
    # Finite-depth predicate reachability:
    # t2 cannot remove t1:
    # Q=1.
    # ------------------------------------------------------------

    facts_B = (
        (0,),
        (0,),
        (1,),
    )

    valid_B = (
        True,
        True,
        True,
    )

    target_B = 0b011

    C_B = rpc_capacity_greedy(
        valid_B,
        target_B,
        B=1,
        leaves=2,
    )

    Q_B = dq_bound(
        facts_B,
        valid_B,
        target_B,
        B=1,
        depth=1,
        leaves=2,
    )

    V_B = exact_policy_value(
        facts_B,
        valid_B,
        target_B,
        B=1,
        depth=1,
        leaves=2,
    )

    assert C_B == 2
    assert Q_B == 1
    assert Q_B < C_B
    assert V_B <= C_B
    assert V_B <= Q_B

    return {
        "C_lt_Q": {
            "V": V_A,
            "C": C_A,
            "Q": Q_A,
        },
        "Q_lt_C": {
            "V": V_B,
            "C": C_B,
            "Q": Q_B,
        },
    }


# ================================================================
# Exhaustive tiny audit.
#
# Candidate type under P predicates:
#
# type 0:
#   invalid, all UNKNOWN
#
# type 1..2^P:
#   valid, binary predicate vector
#
# Every target subset of valid candidates is enumerated.
# ================================================================

def exhaustive_audit():

    checked_instances = 0
    checked_policy_states = 0

    rpc_exact_checks = 0

    violations_rpc_exact = 0
    violations_V_le_C = 0
    violations_V_le_Q = 0
    violations_V_le_min = 0

    strict_C_lt_Q = 0
    strict_Q_lt_C = 0
    equal_C_Q = 0

    max_C_minus_Q = 0
    max_Q_minus_C = 0

    # Enough to exhaustively verify the theorem mechanics
    # without depending on Work2 data.
    for n in range(1, 5):

        P = 2

        candidate_types = list(
            range(
                1 + (1 << P)
            )
        )

        for types in product(
            candidate_types,
            repeat=n,
        ):

            valid = []
            facts = []

            for typ in types:

                if typ == 0:

                    valid.append(
                        False
                    )

                    facts.append(
                        tuple(
                            [UNKNOWN] * P
                        )
                    )

                else:

                    valid.append(
                        True
                    )

                    code = typ - 1

                    facts.append(
                        tuple(
                            (
                                code >> p
                            )
                            &
                            1
                            for p in range(P)
                        )
                    )

            valid = tuple(valid)
            facts = tuple(facts)

            valid_ids = [
                i
                for i, x in enumerate(valid)
                if x
            ]

            # All target subsets of valid candidates.
            for target_choice in product(
                [0, 1],
                repeat=len(valid_ids),
            ):

                target_mask = 0

                for z, i in zip(
                    target_choice,
                    valid_ids,
                ):

                    if z:
                        target_mask |= (
                            1 << i
                        )

                # Empty target set is mathematically valid,
                # but uninformative for this audit.
                if target_mask == 0:
                    continue

                checked_instances += 1

                for B in range(
                    1,
                    min(2, n) + 1,
                ):

                    # First independently verify RPC greedy
                    # against brute-force arbitrary assignment.
                    for leaves in range(
                        1,
                        4,
                    ):

                        C = rpc_capacity_greedy(
                            valid,
                            target_mask,
                            B,
                            leaves,
                        )

                        C_star = rpc_capacity_bruteforce(
                            valid,
                            target_mask,
                            B,
                            leaves,
                        )

                        rpc_exact_checks += 1

                        if C != C_star:

                            violations_rpc_exact += 1

                            raise AssertionError(
                                (
                                    "RPC exactness violation",
                                    n,
                                    facts,
                                    valid,
                                    target_mask,
                                    B,
                                    leaves,
                                    C,
                                    C_star,
                                )
                            )

                    for depth in range(
                        0,
                        3,
                    ):

                        for leaves in range(
                            1,
                            4,
                        ):

                            V = exact_policy_value(
                                facts,
                                valid,
                                target_mask,
                                B,
                                depth,
                                leaves,
                            )

                            C = rpc_capacity_greedy(
                                valid,
                                target_mask,
                                B,
                                leaves,
                            )

                            Q = dq_bound(
                                facts,
                                valid,
                                target_mask,
                                B,
                                depth,
                                leaves,
                            )

                            U = min(
                                C,
                                Q,
                            )

                            checked_policy_states += 1

                            if V > C:

                                violations_V_le_C += 1

                                raise AssertionError(
                                    (
                                        "V>C",
                                        n,
                                        facts,
                                        valid,
                                        target_mask,
                                        B,
                                        depth,
                                        leaves,
                                        V,
                                        C,
                                        Q,
                                    )
                                )

                            if V > Q:

                                violations_V_le_Q += 1

                                raise AssertionError(
                                    (
                                        "V>Q",
                                        n,
                                        facts,
                                        valid,
                                        target_mask,
                                        B,
                                        depth,
                                        leaves,
                                        V,
                                        C,
                                        Q,
                                    )
                                )

                            if V > U:

                                violations_V_le_min += 1

                                raise AssertionError(
                                    (
                                        "V>min(C,Q)",
                                        n,
                                        facts,
                                        valid,
                                        target_mask,
                                        B,
                                        depth,
                                        leaves,
                                        V,
                                        C,
                                        Q,
                                    )
                                )

                            if C < Q:

                                strict_C_lt_Q += 1

                                max_Q_minus_C = max(
                                    max_Q_minus_C,
                                    Q - C,
                                )

                            elif Q < C:

                                strict_Q_lt_C += 1

                                max_C_minus_Q = max(
                                    max_C_minus_Q,
                                    C - Q,
                                )

                            else:

                                equal_C_Q += 1

    return {
        "checked_instances":
            checked_instances,

        "rpc_exact_checks":
            rpc_exact_checks,

        "checked_policy_states":
            checked_policy_states,

        "violations_rpc_exact":
            violations_rpc_exact,

        "violations_V_le_C":
            violations_V_le_C,

        "violations_V_le_Q":
            violations_V_le_Q,

        "violations_V_le_min":
            violations_V_le_min,

        "strict_C_lt_Q":
            strict_C_lt_Q,

        "strict_Q_lt_C":
            strict_Q_lt_C,

        "equal_C_Q":
            equal_C_Q,

        "max_Q_minus_C":
            max_Q_minus_C,

        "max_C_minus_Q":
            max_C_minus_Q,
    }


def main():

    print("=" * 120)
    print("DUAL CAPACITY-REACHABILITY THEORY AUDIT v0")
    print("INDEPENDENT TINY EXACT SOLVER / NO WORK2 DATA / NO RPC-DQ IMPLEMENTATION REUSE")
    print("=" * 120)

    constructions = construction_checks()

    print(
        "C<Q construction =",
        constructions["C_lt_Q"],
    )

    print(
        "Q<C construction =",
        constructions["Q_lt_C"],
    )

    print(
        "EXPLICIT INCOMPARABILITY CONSTRUCTIONS = PASS"
    )

    t0 = time.perf_counter()

    out = exhaustive_audit()

    runtime = (
        time.perf_counter()
        -
        t0
    )

    print()
    print("=" * 120)
    print("EXHAUSTIVE TINY THEORY SUMMARY")
    print("=" * 120)

    for k, v in out.items():
        print(
            f"{k} = {v}"
        )

    print(
        "runtime_seconds =",
        runtime,
    )

    assert (
        out["violations_rpc_exact"]
        ==
        0
    )

    assert (
        out["violations_V_le_C"]
        ==
        0
    )

    assert (
        out["violations_V_le_Q"]
        ==
        0
    )

    assert (
        out["violations_V_le_min"]
        ==
        0
    )

    assert (
        out["strict_C_lt_Q"]
        >
        0
    )

    assert (
        out["strict_Q_lt_C"]
        >
        0
    )

    print()
    print(
        "RPC GREEDY == BRUTE-FORCE CAPACITY = PASS"
    )

    print(
        "V <= C ADMISSIBILITY = PASS"
    )

    print(
        "V <= Q ADMISSIBILITY = PASS"
    )

    print(
        "V <= min(C,Q) ADMISSIBILITY = PASS"
    )

    print(
        "TWO-DIRECTION INCOMPARABILITY = PASS"
    )

    here = Path(__file__)

    print(
        "SCRIPT SHA256 =",
        sha256_file(here),
    )

    print("=" * 120)


if __name__ == "__main__":
    main()
