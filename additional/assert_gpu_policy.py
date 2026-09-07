"""
Pre-flight check that a run can only touch the allowed GPUs.

Cards 0 and 1 on the remote host belong to other users.  ``config.cuda_env``
enforces that for code in this package, but a third-party launcher can reset
``CUDA_VISIBLE_DEVICES`` on its own: Ultralytics' ``select_device`` writes that
variable in some versions, and if it wrote logical indices over an already
restricted list the run would land on the reserved cards.

Rather than reason about a particular library version, this script resolves the
UUID of every device the process can actually see and compares it against the
UUIDs ``nvidia-smi`` reports for the allowed physical indices.  It exits
non-zero on any mismatch, so a run script can gate itself on the real mapping.

Usage in a run script::

    python -m additional.assert_gpu_policy || exit 1
    python -m additional.assert_gpu_policy --select-device 0,1,2,3   # Ultralytics

The second form additionally calls Ultralytics' ``select_device`` with the given
argument first, so the check covers what that call leaves behind.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from additional.config import CUDA_DEVICES, RESERVED_DEVICES


def physical_uuids() -> dict[int, str]:
    """``{physical index: uuid}`` from nvidia-smi, independent of the mask."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout
    table: dict[int, str] = {}
    for line in out.strip().splitlines():
        idx, uuid = (p.strip() for p in line.split(",", 1))
        table[int(idx)] = uuid.replace("GPU-", "")
    return table


def visible_uuids() -> list[str]:
    """UUIDs of the devices this process can see, in logical order."""
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("[gpu-policy] CUDA is not available")
    uuids = []
    for i in range(torch.cuda.device_count()):
        u = getattr(torch.cuda.get_device_properties(i), "uuid", None)
        if u is None:
            raise SystemExit(
                "[gpu-policy] this torch build does not expose device UUIDs, "
                "so the mapping cannot be verified")
        uuids.append(str(u).replace("GPU-", ""))
    return uuids


def parse_args():
    p = argparse.ArgumentParser("verify the GPU policy actually holds")
    p.add_argument("--select-device", default=None,
                   help="call Ultralytics select_device with this argument "
                        "before checking, to cover what it leaves behind")
    p.add_argument("--allow", default=None,
                   help="comma-separated physical indices to allow; defaults "
                        "to config.CUDA_DEVICES")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    allowed = (tuple(int(x) for x in args.allow.split(","))
               if args.allow else CUDA_DEVICES)
    bad = sorted(set(allowed) & RESERVED_DEVICES)
    if bad:
        print(f"[gpu-policy] FAIL: {bad} is reserved and cannot be allowed")
        return 1

    print(f"[gpu-policy] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    if args.select_device is not None:
        from ultralytics.utils.torch_utils import select_device
        dev = select_device(args.select_device, verbose=False)
        print(f"[gpu-policy] ultralytics select_device({args.select_device!r}) "
              f"-> {dev}; CUDA_VISIBLE_DEVICES is now "
              f"{os.environ.get('CUDA_VISIBLE_DEVICES')}")

    phys = physical_uuids()
    allowed_uuids = {phys[i] for i in allowed if i in phys}
    reserved_uuids = {phys[i]: i for i in sorted(RESERVED_DEVICES) if i in phys}

    seen = visible_uuids()
    ok = True
    for i, u in enumerate(seen):
        if u in reserved_uuids:
            print(f"[gpu-policy] FAIL: logical {i} is physical GPU "
                  f"{reserved_uuids[u]}, which is reserved")
            ok = False
        elif u not in allowed_uuids:
            print(f"[gpu-policy] FAIL: logical {i} (uuid {u}) is not one of the "
                  f"allowed cards {list(allowed)}")
            ok = False
        else:
            phys_idx = next(k for k, v in phys.items() if v == u)
            print(f"[gpu-policy] ok: logical {i} -> physical GPU {phys_idx}")

    if not seen:
        print("[gpu-policy] FAIL: no visible devices")
        ok = False

    if ok:
        print(f"[gpu-policy] PASS: {len(seen)} device(s) visible, none of the "
              f"reserved cards {sorted(RESERVED_DEVICES)}")
    else:
        print("[gpu-policy] FAIL: the run must not start")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
