"""
Rewrite absolute image paths in a GAA JSON to a new dataset root.

The evaluation and training sets from the submitted work store absolute paths
under ``/home/ywlee/dev/TANGO2_main/Field_Test/SDS/dataset``, a location that no
longer exists.  The images themselves survive under a different root, so the
sets are usable once the prefix is replaced.

Every rewritten path is checked on disk and the count of missing files is
reported.  A set that silently lost images would train or evaluate on fewer
samples than its name claims, which is the failure this guards against: the
script refuses to write the output when anything is missing unless
``--allow_missing`` is given.

Usage::

    python -m additional.rehome_paths \\
        --in_json  paper_data/sds_geopool.json \\
        --out_json data/paper_geopool_264.json \\
        --old_root /home/ywlee/dev/TANGO2_main/Field_Test/SDS/dataset \\
        --new_root /mnt/ssd1/yvvyee/gaa_additional/oceansim
"""
from __future__ import annotations

import argparse
import json
import os


def parse_args():
    p = argparse.ArgumentParser("rehome absolute image paths in a GAA JSON")
    p.add_argument("--in_json", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--old_root", required=True)
    p.add_argument("--new_root", required=True)
    p.add_argument("--allow_missing", action="store_true",
                   help="write the output even if some images are missing")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.in_json) as f:
        samples = json.load(f)

    rewritten = 0
    missing: list[str] = []
    for s in samples:
        path = s.get("image", "")
        if path.startswith(args.old_root):
            path = args.new_root + path[len(args.old_root):]
            s["image"] = path
            rewritten += 1
        if not os.path.exists(path):
            missing.append(path)

    print(f"[rehome] {len(samples)} 샘플, 경로 치환 {rewritten}, "
          f"없는 파일 {len(missing)}")
    if missing:
        for m in missing[:5]:
            print(f"  없음: {m}")
        if len(missing) > 5:
            print(f"  ... 그 외 {len(missing) - 5}개")
        if not args.allow_missing:
            raise SystemExit(
                "[rehome] 이미지가 없는 샘플이 있어 출력하지 않는다. "
                "의도한 경우에만 --allow_missing 을 쓴다.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(samples, f)
    print(f"[rehome] -> {args.out_json}")


if __name__ == "__main__":
    main()
