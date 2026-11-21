"""
Protocol constants -- Journal study, Protocol V1.3 (data stage).

Single source of truth for paths, seeds and build criteria. Every
script in this folder imports from here; nothing is hard-coded
elsewhere. Experiment constants (doses, arms, salts, epoch grid) are
added to this file in the experiment stage, before the protocol hash.

ADC (ADC 2026 paper) is used ONLY as a reference: the numbers in
ADC_REFERENCE below are values published or recorded by the ADC study,
copied here so the new code can be checked against them. No ADC code
or ADC file is imported or executed.

Override the root for testing with the environment variable JV_ROOT.
"""

from __future__ import annotations

import os
from pathlib import Path

PROTOCOL_VERSION = "V1.3"

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
ROOT = Path(os.environ.get("JV_ROOT", r"D:\CheckPoint"))

# Raw/prepared domain folders (read-only input):
#   <SOURCE_ROOT>/<Category>/selected_interactions.csv
#   <SOURCE_ROOT>/<Category>/cantor_mappings/{user,item}_mapping.csv
#   <SOURCE_ROOT>/<Category>/cantor_resnet50_features.pt
#   <SOURCE_ROOT>/<Category>/image_manifest.json, images/, downloads/
SOURCE_ROOT = ROOT / "CANTOR_online"

PROJECT_ROOT = ROOT / "journal_v1"
CODE_DIR = PROJECT_ROOT / "code"
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"

# Optional reference files from ADC (read only, hash-checked before use).
ADC_REF_DIR = ROOT / "ADC_reference"

# ------------------------------------------------------------------
# Domains
# ------------------------------------------------------------------
DOMAINS = {
    "VG": "Video_Games",
    "CSJ": "Clothing_Shoes_and_Jewelry",
    "SO": "Sports_and_Outdoors",
    "TG": "Toys_and_Games",
}
# Domains built from raw files by build_core.py (VG and CSJ already exist).
NEW_DOMAINS = ("SO", "TG")
# Fixed domain index (Protocol V1.3 section 3.1); enters every RNG seed.
DOMAIN_INDEX = {"VG": 1, "CSJ": 2, "SO": 3, "TG": 4}

# ------------------------------------------------------------------
# Core build rule (same rule as ADC, re-implemented)
# ------------------------------------------------------------------
TARGET_USERS = 3_000
TARGET_ITEMS = 5_000
ITEM_CANDIDATE_MULTIPLIER = 2
MIN_USER_INTERACTIONS = 3
MIN_ITEM_INTERACTIONS = 2

# Feasibility criterion (decision F, V1.3). Fixed before any model output.
FEASIBLE_NUM_ITEMS = 5_000
FEASIBLE_MIN_USERS = 2_900
FEASIBLE_MAX_USERS = 3_000

FEATURE_DIM = 2_048
FEATURE_BACKBONE = "resnet50"
RESNET_WEIGHTS_NAME = "ResNet50_Weights.DEFAULT"
UNIT_NORM_ATOL = 1e-3

# Official Amazon Reviews 2023 sources, tried in order. A file placed
# manually in <Category>/downloads/ with the expected name is used as is.
RATINGS_URL_TEMPLATES = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
    "benchmark/5core/rating_only/{cat}.csv.gz",
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
    "resolve/main/benchmark/5core/rating_only/{cat}.csv",
)
META_URL_TEMPLATES = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
    "raw/meta_categories/meta_{cat}.jsonl.gz",
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
    "resolve/main/raw/meta_categories/meta_{cat}.jsonl",
)
RATINGS_FILE_NAME = "{cat}.csv.gz"       # local name (gz or plain is detected)
META_FILE_NAME = "meta_{cat}.jsonl.gz"   # local name (gz or plain is detected)

IMAGE_DOWNLOAD_WORKERS = 12
IMAGE_DOWNLOAD_TIMEOUT = 25
IMAGE_DOWNLOAD_RETRIES = 3
IMAGE_JPEG_QUALITY = 95
RESNET_BATCH_SIZE = 64

# ------------------------------------------------------------------
# Splits
# ------------------------------------------------------------------
DATA_SEEDS = (237, 337, 437, 537)
ADC_DATA_SEED = 137                 # re-derived in memory for the ADC check only
SPLIT_SEED = 7
COLD_FRACTION = 0.20
COLD_TEST_FRACTION = 0.30
NUM_PSEUDO_FOLDS = 3
PV_FOLD = 0
EXPECTED_NUM_COLD = 1_000
EXPECTED_NUM_DEV = 700
EXPECTED_NUM_TEST = 300

# Split RNG rule (DEVIATIONS C8). Item indices are popularity ranks in every
# domain, so a cold pool drawn from default_rng(data_seed) alone selects the
# SAME ranks in all four domains. Journal splits therefore seed by domain:
#   cold pool + fold jitter : default_rng([domain_index, data_seed])
#   cold DEV/TEST partition : default_rng([domain_index, data_seed, SPLIT_SEED])
# The ADC rule (default_rng(data_seed), default_rng(7)) is used ONLY to
# re-derive split 137 in memory for the ADC reproduction check (G1b).
SPLIT_RNG_RULE = "C8_domain_seeded"
ADC_SPLIT_RNG_RULE = "ADC_legacy"

# Cross-domain independence check (C8): overlap of cold-DEV item indices
# between two domains with the same data_seed. Under independent draws the
# overlap is hypergeometric, mean 700*700/5000 = 98, sd ~ 8.5.
CROSS_DOMAIN_DEV_OVERLAP_RANGE = (60, 140)

# Arrays written to the DEV-only artifact (whitelist; nothing else).
DEV_ARRAY_KEYS = (
    "train_pairs",          # strict training positives (PV fold removed)
    "pv_pairs",             # pseudo-validation positives (fold 0)
    "pv_items",             # pseudo-validation candidate catalog (fold 0 items)
    "warm_items",
    "negative_pool",        # warm items minus PV items
    "warm_val_pairs",       # 2nd most recent warm interaction per user
    "item_train_degree",    # degree in train_pairs + pv_pairs (cold items = 0)
    "cold_dev_items",
    "cold_dev_pairs",
)
TEST_ARRAY_KEYS = ("cold_test_items", "cold_test_pairs")

# ------------------------------------------------------------------
# Evaluation / chance floor (G8)
# ------------------------------------------------------------------
EVAL_K = 10
BATCH_SIZE = 2_048                  # only used here to report batches/epoch
CHANCE_MC_REPLICATES = 200
CHANCE_MC_SEED = 20261028

# Feature-pipeline consistency check (informational)
FEATURE_CHECK_ITEMS = 64
FEATURE_CHECK_SEED = 20261028
FEATURE_CHECK_MIN_COSINE = 0.99

# ------------------------------------------------------------------
# ADC reference values (copied from ADC records; used only to compare)
# ------------------------------------------------------------------
ADC_REFERENCE = {
    "VG": {
        "num_users": 2_999,
        "num_items": 5_000,
        "num_interactions": 88_267,
        # sha256(str(torch dtype) + str(shape) + bytes), ADC replay manifest
        "features_sha256_torch": (
            "58d39c69dc6852a9f6f0360e61a68efa"
            "461b9da3ee8a010f3b196696241141ac"
        ),
        "split137": {
            "cold_dev_items_sha256_torch": (
                "6eb031c6805bdd68b7c77f3b7a4f7132"
                "5ebdae001ba6a390d4c6dd7febb7424d"
            ),
            "counts": {
                "cantor_train_pairs": 42_636,
                "cantor_validation_pairs": 21_441,
                "pv_active_items": 1_333,
                "cold_dev_pairs": 12_492,
                "cold_test_pairs": 5_700,
                "batches_per_epoch": 21,
            },
            # Optional: ADC frozen_137/manifest.json copied to this path.
            # Used only if its file SHA-256 equals the value ADC recorded.
            "optional_manifest_path": (
                "Video_Games/frozen_137/manifest.json"
            ),
            "optional_manifest_sha256": (
                "6fafc21acbc44a210ae92b82a2e506d7"
                "7e12dc7a7a35a93c4c93b26f9c64541f"
            ),
        },
    },
    "CSJ": {
        "num_users": 3_000,
        "num_items": 5_000,
        "num_interactions": 73_836,
        "features_sha256_torch": None,   # not recorded by ADC
        "split137": {
            # 16-hex fingerprints: sha256(str(numpy dtype)+str(shape)+bytes)[:16]
            "fingerprints16": {
                "cantor_train_pairs": "e927c6f913e01349",
                "cantor_validation_pairs": "5abeaa6bab9e7206",
                "cold_dev_items": "c051457642c9fe29",
                "cold_dev_pairs": "870072bbaca4a810",
                "cold_test_items": "455feaed612d3ad8",
                "cold_test_pairs": "4cd10a539deefac7",
                "train_pairs": "827a6316eaba51f4",
                "validation_pairs": "aeaf637f95c4a608",
                "warm_items": "2296c7c404c23978",
                "warm_test_pairs": "1e1114bacba2d63e",
            },
            "counts": {
                "cantor_train_pairs": 34_839,
                "cantor_validation_pairs": 17_483,
                "pv_active_items": 1_334,
                "cold_dev_pairs": 10_336,
                "cold_test_pairs": 5_178,
                "batches_per_epoch": 18,
            },
        },
    },
}


# ==================================================================
# Experiment constants (Protocol V1.3 sections 4-7; DEVIATIONS C1-C8)
# ==================================================================
RUNS_DIR = PROJECT_ROOT / "runs"
STATES_DIR = PROJECT_ROOT / "states"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

TRAINING_SEEDS = (42, 52, 62, 72, 82)          # labels r
EPOCHS = 120
HORIZONS = (40, 80, 120)
PRIMARY_HORIZON = 120
K_MAX = 685
BATCH = 2_048
GRID_EPOCHS = tuple(range(1, 21)) + tuple(range(30, 121, 10))   # 30 states
GRID_ARMS = ("A0", "A3", "A5", "A7")
NUM_NEG_CANDIDATES = 16
PLAN_SALT, NEG_SALT, CR_SALT, PC_SALT = 101, 202, 303, 404
PC_FRACTIONS = {"A8": 0.10, "A9": 0.30, "A10": 0.60}
CSV_DECIMALS = 8


def effective_seed(tag: str, data_seed: int, r: int) -> int:
    """Protocol V1.3 section 4.1: 1,000,000*domain_index + 1,000*data_seed + r."""
    return 1_000_000 * DOMAIN_INDEX[tag.upper()] + 1_000 * int(data_seed) + int(r)


# Arm table (section 4). nominal_q is the arm label; k is the locked slot count;
# effective dose = k / BATCH (DEVIATIONS C1).
ARMS = {
    "A0":  {"backbone": "A", "k": 0,   "nominal_q": 0.0,    "fill": "none", "id": "train_only", "tier": 1},
    "A1":  {"backbone": "A", "k": 205, "nominal_q": 0.10,   "fill": "pv",   "id": "on_insert",  "tier": 1},
    "A2":  {"backbone": "A", "k": 410, "nominal_q": 0.20,   "fill": "pv",   "id": "on_insert",  "tier": 1},
    "A3":  {"backbone": "A", "k": 685, "nominal_q": 0.3346, "fill": "pv",   "id": "on_insert",  "tier": 1},
    "A4":  {"backbone": "A", "k": 410, "nominal_q": 0.20,   "fill": "cr",   "id": "train_only", "tier": 1},
    "A5":  {"backbone": "A", "k": 685, "nominal_q": 0.3346, "fill": "cr",   "id": "train_only", "tier": 1},
    "A6":  {"backbone": "A", "k": 410, "nominal_q": 0.20,   "fill": "pv",   "id": "train_only", "tier": 1},
    "A7":  {"backbone": "A", "k": 685, "nominal_q": 0.3346, "fill": "pv",   "id": "train_only", "tier": 1},
    "B0":  {"backbone": "B", "k": 0,   "nominal_q": 0.0,    "fill": "none", "id": "none",       "tier": 1},
    "B2":  {"backbone": "B", "k": 410, "nominal_q": 0.20,   "fill": "pv",   "id": "none",       "tier": 1},
    "B3":  {"backbone": "B", "k": 685, "nominal_q": 0.3346, "fill": "pv",   "id": "none",       "tier": 1},
    "A8":  {"backbone": "A", "k": 410, "nominal_q": 0.20,   "fill": "pc10", "id": "on_insert",  "tier": 2},
    "A9":  {"backbone": "A", "k": 410, "nominal_q": 0.20,   "fill": "pc30", "id": "on_insert",  "tier": 2},
    "A10": {"backbone": "A", "k": 410, "nominal_q": 0.20,   "fill": "pc60", "id": "on_insert",  "tier": 2},
    "B1":  {"backbone": "B", "k": 205, "nominal_q": 0.10,   "fill": "pv",   "id": "none",       "tier": 2},
}
TIER2_CUT_ORDER = (("TG", ("A8", "A9", "A10")), ("SO", ("A8", "A9", "A10")),
                   ("*", ("B1",)), ("CSJ", ("A8", "A9", "A10")))

# Confirmatory family (section 6), fixed sequence P1 -> P2.
CONFIRMATORY = (("P1", "A3", "A0"), ("P2", "A3", "A5"))
ALPHA = 0.05
MIN_DOMAINS_POSITIVE = 3                        # of 4 (V1.3 section 6, step 4)

# Secondary contrasts (section 6.2): name, arm_a, arm_b, rule, horizon.
SECONDARY = (
    ("dose_q010", "A1", "A0", "R1", 120),
    ("dose_q020", "A2", "A0", "R1", 120),
    ("P1_under_R2", "A3", "A0", "R2", 120),
    ("P2_under_R2", "A3", "A5", "R2", 120),
    ("id_A3_minus_A7", "A3", "A7", "R1", 120),
    ("id_A7_minus_A0", "A7", "A0", "R1", 120),
    ("id_A2_minus_A6", "A2", "A6", "R1", 120),
    ("id_A6_minus_A0", "A6", "A0", "R1", 120),
    ("cr_A2_minus_A4", "A2", "A4", "R1", 120),
    ("cr_A4_minus_A0", "A4", "A0", "R1", 120),
    ("B_B1_minus_B0", "B1", "B0", "R1", 120),
    ("B_B2_minus_B0", "B2", "B0", "R1", 120),
    ("B_B3_minus_B0", "B3", "B0", "R1", 120),
    ("cov10_A8_minus_A0", "A8", "A0", "R1", 120),
    ("cov30_A9_minus_A0", "A9", "A0", "R1", 120),
    ("cov60_A10_minus_A0", "A10", "A0", "R1", 120),
    ("cov100_A2_minus_A0", "A2", "A0", "R1", 120),
    ("horizon40_P1", "A3", "A0", "R1", 40),
    ("horizon80_P1", "A3", "A0", "R1", 80),
)
# MA5 interaction (RQ3): (a - b under R1) - (a - b under R2).
MA5_INTERACTIONS = (("ma5x_A3_A0", "A3", "A0"), ("ma5x_A2_A0", "A2", "A0"))
# A_ID = 1 - Delta_IDoff / Delta_LEAKY per domain (section 6.2).
A_ID_DEFS = (("A_ID_q3346", "A3", "A7", "A0"), ("A_ID_q020", "A2", "A6", "A0"))
SELECTION_RULES = ("R1", "R2", "MA5_full", "R1_from5", "MA3", "MA9", "MA5_centered", "ES10")

# G7 (section 6.1 + DEVIATIONS C4, C5, C7, C9)
G7_DIR = ANALYSIS_DIR / "g7"
G7_REPS = 10_000
G7_SIM_SEED = 20261029
G7_SIGMA_E = (0.001, 0.003, 0.005)
G7_SIGMA_V_RATIO = (0.0, 0.5, 1.0, 2.0)
G7_CONFIRM_RATE_MAX = 0.0544
G7_SU_ALPHAS = (0.01, 0.025, 0.05, 0.10)
G7_SU_FAMILY_ALPHA = 0.05      # C9: family-wise Monte Carlo tolerance (Bonferroni over K checks)
# Binding super-uniformity rule for the G7 verdict (DEVIATIONS C9, reviewer-approved
# 28/10/2026): exact binomial, Bonferroni over the K super-uniformity checks.
# FAIL at level alpha when count(p <= alpha) >= c, c = smallest integer with
# P(Binomial(N, alpha) >= c) <= G7_SU_FAMILY_ALPHA / K. "C4_per_check" (alpha + 2 SE)
# is computed and reported only.
G7_SU_RULE = "C9_exact_binomial"

# Model / training (section 4; locked with G7 so constants.py does not change later)
EMB_DIM = 64
MLP_HIDDEN = 128
INIT_STD = 0.05
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6
EVAL_USER_BATCH = 1_024
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
STATE_DTYPE = "float32"
# G4b (DEVIATIONS C6): the only data inputs a training process may open.
G4B_ALLOWED_DATA_FILES = ("dev.npz", "manifest_dev.json", "features.npy", "CORE_MANIFEST.json")
G4B_FORBIDDEN_NAMES = ("core.npz",)
G4B_FORBIDDEN_PARTS = ("sealed",)
