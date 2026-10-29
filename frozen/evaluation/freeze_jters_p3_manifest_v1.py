#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import numpy as np


ROOT = Path("/root/autodl-tmp/work2-algorithm-innovation")

FROZEN = ROOT / "fhex_v1/frozen_v2"

PROTOCOL = (
    FROZEN
    / "jters_v2_frozen_protocol.json"
)

FREEZE_MANIFEST = (
    FROZEN
    / "jters_v2_freeze_manifest.json"
)

OUT_DIR = (
    ROOT
    / "fhex_v1/prospective/p3_jters_v2"
)

OUT_NPZ = (
    OUT_DIR
    / "jters_p3_manifest_v1.npz"
)

OUT_JSON = (
    OUT_DIR
    / "jters_p3_manifest_v1.json"
)


EXPECTED_PROTOCOL_SHA = (
    "ce22ddf88da8a1c15727346d2679248d"
    "102eb11a65a54b641c99095394b95601"
)

EXPECTED_FREEZE_MANIFEST_SHA = (
    "7ec59b91d1803688144b914cf1727054"
    "e1a85099255594acf32c0470bc5d77be"
)


TRAIN_N = 412178
P3_N = 6000
N_SHARDS = 5


ROW_KEYS = [
    "source_row_indices",
    "query_row_indices",
    "query_rows",
    "source_rows",
    "train_rows",
    "manifest_rows",
    "sampled_rows",
    "rows",
]


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            b = f.read(1 << 20)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def extract_rows(path):

    z = np.load(
        path,
        allow_pickle=True,
    )

    found = []

    for key in ROW_KEYS:

        if key not in z.files:
            continue

        x = np.asarray(
            z[key]
        )

        if x.ndim != 1:
            continue

        if not np.issubdtype(
            x.dtype,
            np.integer,
        ):
            continue

        x = x.astype(
            np.int64
        )

        if len(x) == 0:
            continue

        # Only accept arrays that look like Python TRAIN rows.
        if (
            x.min() < 0
            or
            x.max() >= TRAIN_N
        ):
            continue

        found.append(
            (
                key,
                x,
            )
        )

    return found


def main():

    print("=" * 130)
    print("JTE-RS v2 P3 MANIFEST FREEZE v1")
    print("=" * 130)

    print("NO retrieval is computed.")
    print("NO repair result is computed.")
    print("NO P1/P2 outcome is opened.")
    print("Only query identities are used for overlap exclusion.")
    print()

    # --------------------------------------------------------
    # Frozen-method integrity.
    # --------------------------------------------------------

    protocol_sha = sha256_file(
        PROTOCOL
    )

    freeze_sha = sha256_file(
        FREEZE_MANIFEST
    )

    assert (
        protocol_sha
        ==
        EXPECTED_PROTOCOL_SHA
    )

    assert (
        freeze_sha
        ==
        EXPECTED_FREEZE_MANIFEST_SHA
    )

    print(
        "Frozen v2 protocol SHA PASS =",
        protocol_sha,
    )

    print(
        "Frozen v2 manifest SHA PASS =",
        freeze_sha,
    )

    # --------------------------------------------------------
    # Never silently regenerate P3.
    # --------------------------------------------------------

    if (
        OUT_NPZ.exists()
        or
        OUT_JSON.exists()
    ):

        raise RuntimeError(
            "P3 manifest already exists. "
            "Refusing to resample prospective P3."
        )

    # --------------------------------------------------------
    # Candidate exclusion artifacts.
    #
    # We inspect identities only. No metrics/outcomes.
    # --------------------------------------------------------

    candidates = []

    exact = [
        ROOT
        / "frozen_artifacts/"
          "work2_repairability_frontier_v1/"
          "benchmark_train/"
          "python_train_retrievals_v1.npz",

        ROOT
        / "fhex_v1/sandbox/"
          "python_trainonly_qwen3_top500_v1.npz",

        ROOT
        / "fhex_v1/prospective/"
          "rcps_p2_cross_retriever_manifest_v1.npz",
    ]

    for p in exact:

        if p.exists():
            candidates.append(p)

    # Discover P1/P2/fresh identity manifests without
    # assuming an exact historical filename.
    prospective = (
        ROOT
        / "fhex_v1/prospective"
    )

    if prospective.exists():

        for p in prospective.rglob(
            "*.npz"
        ):

            low = p.name.lower()

            if any(
                token in low
                for token in [
                    "p1",
                    "p2",
                    "manifest",
                    "fresh",
                    "prospective",
                ]
            ):

                candidates.append(p)

    # Development/fresh manifests elsewhere.
    for base in [
        ROOT / "fhex_v1",
        ROOT / "frozen_artifacts",
    ]:

        if not base.exists():
            continue

        for p in base.rglob(
            "*.npz"
        ):

            low = p.name.lower()

            if (
                "manifest"
                in low
                or
                "fresh"
                in low
            ):

                candidates.append(p)

    # Exact-path P2 location from frozen protocol history.
    p2_exact = (
        ROOT
        / "fhex_v1/prospective/"
          "rcps_p2_cross_retriever_manifest_v1.npz"
    )

    if p2_exact.exists():
        candidates.append(
            p2_exact
        )

    # Deduplicate file paths.
    candidates = sorted(
        set(
            p.resolve()
            for p in candidates
            if p.exists()
        )
    )

    exclusion_sets = []
    exclusion_audit = []

    for p in candidates:

        try:

            rows_found = extract_rows(
                p
            )

        except Exception as e:

            exclusion_audit.append({
                "path": str(p),
                "status": (
                    "unreadable: "
                    + repr(e)
                ),
            })

            continue

        if not rows_found:

            exclusion_audit.append({
                "path": str(p),
                "status":
                    "no recognized TRAIN-row identity key",
            })

            continue

        for key, rows in rows_found:

            exclusion_sets.append(
                rows
            )

            exclusion_audit.append({
                "path": str(p),
                "key": key,
                "rows": int(
                    len(rows)
                ),
                "unique_rows": int(
                    len(
                        np.unique(
                            rows
                        )
                    )
                ),
                "status":
                    "excluded",
            })

    if not exclusion_sets:

        raise RuntimeError(
            "No development/prospective TRAIN identities "
            "could be recovered. Refusing to create P3."
        )

    excluded = np.unique(
        np.concatenate(
            exclusion_sets
        )
    )

    assert (
        excluded.min()
        >=
        0
    )

    assert (
        excluded.max()
        <
        TRAIN_N
    )

    # --------------------------------------------------------
    # Remaining untouched TRAIN pool.
    # --------------------------------------------------------

    all_rows = np.arange(
        TRAIN_N,
        dtype=np.int64,
    )

    eligible_mask = np.ones(
        TRAIN_N,
        dtype=bool,
    )

    eligible_mask[
        excluded
    ] = False

    eligible = all_rows[
        eligible_mask
    ]

    if len(
        eligible
    ) < P3_N:

        raise RuntimeError(
            f"Only {len(eligible)} eligible rows remain."
        )

    # --------------------------------------------------------
    # Deterministic prospective seed derived only from the
    # PRE-P3 frozen protocol SHA.
    # --------------------------------------------------------

    seed = (
        int(
            protocol_sha[
                :16
            ],
            16,
        )
        %
        (
            2**63
            -
            1
        )
    )

    rng = np.random.default_rng(
        seed
    )

    selected = rng.choice(
        eligible,
        size=P3_N,
        replace=False,
    ).astype(
        np.int64
    )

    # Keep randomized order; shards are assigned from this
    # frozen prospective order, not corpus row ordering.
    shard_ids = (
        np.arange(
            P3_N,
            dtype=np.int16,
        )
        %
        N_SHARDS
    )

    assert (
        len(
            np.unique(
                selected
            )
        )
        ==
        P3_N
    )

    assert (
        np.intersect1d(
            selected,
            excluded,
        ).size
        ==
        0
    )

    shard_counts = {
        str(i):
            int(
                np.sum(
                    shard_ids
                    ==
                    i
                )
            )
        for i in range(
            N_SHARDS
        )
    }

    assert all(
        v
        ==
        P3_N // N_SHARDS
        for v in shard_counts.values()
    )

    # --------------------------------------------------------
    # Freeze manifest.
    # --------------------------------------------------------

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUT_NPZ,

        format=np.asarray(
            "jters_p3_manifest_v1"
        ),

        source_row_indices=
            selected,

        shard_ids=
            shard_ids,

        seed=np.asarray(
            seed,
            dtype=np.int64,
        ),

        train_row_count=np.asarray(
            TRAIN_N,
            dtype=np.int64,
        ),

        protocol_sha256=np.asarray(
            protocol_sha,
        ),

        frozen_v2_manifest_sha256=np.asarray(
            freeze_sha,
        ),
    )

    npz_sha = sha256_file(
        OUT_NPZ
    )

    metadata = {
        "name":
            "JTE-RS v2 P3 Prospective Manifest v1",

        "status":
            "FROZEN BEFORE P3 RETRIEVAL",

        "scope":
            "Python CodeSearchNet TRAIN",

        "p3_query_count":
            P3_N,

        "shards":
            shard_counts,

        "seed":
            int(seed),

        "seed_source":
            (
                "first 16 hex digits of frozen "
                "JTE-RS v2 protocol SHA256"
            ),

        "frozen_v2_protocol_sha256":
            protocol_sha,

        "frozen_v2_manifest_sha256":
            freeze_sha,

        "excluded_unique_train_rows":
            int(
                len(
                    excluded
                )
            ),

        "eligible_train_rows_before_sampling":
            int(
                len(
                    eligible
                )
            ),

        "selected_vs_excluded_overlap":
            int(
                np.intersect1d(
                    selected,
                    excluded,
                ).size
            ),

        "manifest_npz_sha256":
            npz_sha,

        "exclusion_identity_sources":
            exclusion_audit,

        "prospective_rules": [
            "No retrieval outcome was inspected before manifest freeze.",
            "No repair outcome was inspected before manifest freeze.",
            "P1 outcomes were not accessed.",
            "P2 outcomes were not accessed.",
            "Only query identities from prior consumed/frozen artifacts were used for exclusion.",
            "P3 rows are sampled without replacement.",
            "P3 has exactly five frozen shards of 1200 queries each.",
            "No resampling is permitted after this manifest is created.",
            "Frozen TECS/APQC/RBJS may not be modified based on P3.",
        ],
    }

    OUT_JSON.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    json_sha = sha256_file(
        OUT_JSON
    )

    print()
    print("=" * 130)
    print("P3 MANIFEST FREEZE COMPLETE")
    print("=" * 130)

    print(
        "excluded unique TRAIN rows =",
        len(
            excluded
        ),
    )

    print(
        "eligible untouched TRAIN rows =",
        len(
            eligible
        ),
    )

    print(
        "P3 selected rows =",
        len(
            selected
        ),
    )

    print(
        "P3 vs all exclusions overlap =",
        np.intersect1d(
            selected,
            excluded,
        ).size,
    )

    print(
        "shard counts =",
        shard_counts,
    )

    print(
        "deterministic seed =",
        seed,
    )

    print()
    print(
        "manifest =",
        OUT_NPZ,
    )

    print(
        "manifest SHA256 =",
        npz_sha,
    )

    print(
        "metadata SHA256 =",
        json_sha,
    )

    print()
    print(
        "STATUS = P3 IDENTITIES FROZEN BEFORE RETRIEVAL"
    )

    print(
        "P1 OUTCOMES NOT ACCESSED"
    )

    print(
        "P2 OUTCOMES NOT ACCESSED"
    )

    print(
        "P3 RETRIEVAL NOT ACCESSED"
    )


if __name__ == "__main__":
    main()
