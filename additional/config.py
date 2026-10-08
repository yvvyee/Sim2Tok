"""
Single source of truth for the additional-experiment runs.

Everything that a run script would otherwise hard-code lives here: the GPU set,
the dataset roots on both machines, the image geometry, and the random seeds.

GPU policy
----------
The remote host TG-A40G carries six A100-PCIE-40GB cards.  Cards 0 and 1 are
reserved for other users and must never be touched.  Card 5 was excluded for
part of 2026-10-08 and readmitted the same day, so the study runs on cards
2, 3, 4 and 5.  ``CUDA_DEVICES`` is the
only place that set is written down, and ``cuda_env()`` is the only supported
way to build the environment for a subprocess.  Both refuse to emit a device
list that contains a reserved card, so a typo in a run script fails loudly
instead of quietly training on someone else's GPU.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# ── GPU policy ───────────────────────────────────────────────────────────────

#: Physical GPU indices this study is allowed to use.
CUDA_DEVICES: tuple[int, ...] = (2, 3, 4, 5)

#: Physical GPU indices reserved for other users.  Never schedule work here.
RESERVED_DEVICES: frozenset[int] = frozenset({0, 1})


def cuda_visible_devices(devices: tuple[int, ...] | None = None) -> str:
    """Return the ``CUDA_VISIBLE_DEVICES`` string for the allowed GPU set.

    Raises ``ValueError`` if any requested device is reserved, so a bad run
    script cannot silently occupy GPU 0 or 1.
    """
    devs = tuple(devices) if devices is not None else CUDA_DEVICES
    if not devs:
        raise ValueError("empty device list")
    bad = sorted(set(devs) & RESERVED_DEVICES)
    if bad:
        raise ValueError(
            f"GPU {bad} is reserved for other users and must not be used; "
            f"allowed devices are {list(CUDA_DEVICES)}"
        )
    return ",".join(str(d) for d in devs)


def cuda_env(devices: tuple[int, ...] | None = None,
             base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment dict for a training/eval subprocess, GPU policy applied."""
    env = dict(os.environ if base is None else base)
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices(devices)
    # Logical rank i maps to CUDA_DEVICES[i]; keep NCCL from probing the
    # reserved cards when it enumerates peers.
    env.setdefault("NCCL_P2P_DISABLE", "0")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def assert_policy_or_die(devices: tuple[int, ...] | None = None) -> None:
    """Abort unless every visible GPU is one this study may use.

    ``cuda_visible_devices`` only builds the mask; it cannot stop a process
    that was launched without it.  A command copied from a docstring without
    its ``CUDA_VISIBLE_DEVICES`` prefix would map local rank 0 straight onto
    physical GPU 0, allocate several GB there and very likely take down another
    user's job, with no error of its own.  This resolves the UUID of every
    device the process can actually see and refuses to continue if any of them
    is reserved, so the guarantee does not depend on how the process started.
    """
    import subprocess

    import torch

    allowed = tuple(devices) if devices is not None else CUDA_DEVICES
    cuda_visible_devices(allowed)          # rejects a reserved index outright
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device is available")

    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout
    physical = {}
    for line in out.strip().splitlines():
        idx, uuid = (p.strip() for p in line.split(",", 1))
        physical[int(idx)] = uuid.replace("GPU-", "")
    reserved = {physical[i]: i for i in sorted(RESERVED_DEVICES) if i in physical}
    permitted = {physical[i] for i in allowed if i in physical}

    for i in range(torch.cuda.device_count()):
        u = getattr(torch.cuda.get_device_properties(i), "uuid", None)
        if u is None:
            raise RuntimeError(
                "this torch build does not expose device UUIDs, so the GPU "
                "policy cannot be verified; refusing to run")
        u = str(u).replace("GPU-", "")
        if u in reserved:
            raise RuntimeError(
                f"local device {i} is physical GPU {reserved[u]}, which is "
                f"reserved for other users. Set CUDA_VISIBLE_DEVICES="
                f"{cuda_visible_devices(allowed)} before launching.")
        if u not in permitted:
            raise RuntimeError(
                f"local device {i} (uuid {u}) is not among the allowed cards "
                f"{list(allowed)}")


def nproc_per_node(devices: tuple[int, ...] | None = None) -> int:
    """Number of ranks for a torchrun launch over the allowed GPU set."""
    devs = tuple(devices) if devices is not None else CUDA_DEVICES
    cuda_visible_devices(devs)          # validate
    return len(devs)


# ── Paths ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Paths:
    """Dataset and output roots for one machine."""

    aivn_sds: str          #: AIVN-SDS 20260728 root (png/csv/describe_ko/advice_ko)
    smd_raw: str           #: directory holding the three SMD split archives
    smd_root: str          #: extracted SMD root (VIS_Onshore/VIS_Onboard/NIR)
    work: str              #: scratch root for converted JSON, frames, checkpoints
    results: str           #: where result JSON/logs are written


LOCAL = Paths(
    aivn_sds="/home/ywlee/SSD/dataset/AIVN-SDS/20260728",
    smd_raw="/home/ywlee/SSD/dataset/SMD_raw",
    smd_root="/home/ywlee/SSD/dataset/SMD",
    work="/home/ywlee/SSD/dataset/gaa_additional",
    results="/media/ywlee/T9/Documents/WORK/Workspace/GAA/additional_exp",
)

REMOTE = Paths(
    aivn_sds="/home/yvvyee/data/AIVN-SDS/20260728",
    smd_raw="/mnt/ssd1/yvvyee/gaa_additional/smd_raw",
    smd_root="/mnt/ssd1/yvvyee/gaa_additional/SMD",
    work="/mnt/ssd1/yvvyee/gaa_additional",
    results="/mnt/ssd1/yvvyee/gaa_additional/results",
)

#: Remote ssh target and conda environment.
REMOTE_SSH = "ssh -p 11022 yvvyee@129.254.222.148"
REMOTE_CONDA_SH = "/home/yvvyee/miniconda3/etc/profile.d/conda.sh"
REMOTE_CONDA_ENV = "eva"
REMOTE_LLM = "/home/yvvyee/data/Models/Llama-3.1-8B-Instruct"


# ── Data geometry ────────────────────────────────────────────────────────────

#: AIVN-SDS capture resolution (README: Unity3D capture, 1920x1080).
AIVN_IMAGE_SIZE: tuple[int, int] = (1920, 1080)

#: Vision encoder used for the primary experiments.
VISION_MODEL = "openai/clip-vit-large-patch14-336"

#: ViT patch-grid side for CLIP-ViT-L/14-336 (336 / 14).
FEATURE_SIZE = 24

#: Foreground coverage threshold for the geometric mask (paper Eq. 4, tau).
GEO_TAU = 0.5


# ── SMD layout ───────────────────────────────────────────────────────────────

#: Split name -> (video count, labelled frame count) as published by the authors.
SMD_SPLITS: dict[str, tuple[int, int]] = {
    "VIS_Onshore": (40, 17967),
    "VIS_Onboard": (11, 2400),
    "NIR": (30, 11286),
}

#: Google Drive file ids for the three published split archives.
SMD_DRIVE_IDS: dict[str, str] = {
    "VIS_Onshore": "1HnHyQzhzzDlYh15y9_K1mNZX3grlSDMM",
    "VIS_Onboard": "0B43_rYxEgelVb2VFaXB4cE56RW8",
    "NIR": "13wKWzHqkDQHMHjfuUWjgzwhTrnMoEE8P",
}


# ── Reproducibility ──────────────────────────────────────────────────────────

#: Seeds used for every run reported in additional_exp/.
SEEDS: tuple[int, ...] = (0, 1, 2)

#: Training corpus sizes for the scaling-curve story.
SCALING_SIZES: tuple[int, ...] = (264, 1000, 3000, 10000)


@dataclass(frozen=True)
class TrainSpec:
    """One training configuration in the additional study."""

    name: str
    variant: str = "inbox_ce"       #: geo_losses.VARIANTS member, or "none"
    lam: float = 0.7                #: lambda on the geometric term
    vision_only: bool = True        #: train the encoder alone (no SFT term)
    epochs: int = 5
    lr: float = 1e-5
    per_device_batch: int = 8
    n_train: int = 10000
    seed: int = 0
    extra: dict = field(default_factory=dict)
