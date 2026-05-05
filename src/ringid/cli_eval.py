"""CLI: compute aggregate verification metrics from precomputed distances."""

from __future__ import annotations

import argparse
from pathlib import Path

from ringid.detect import roc_auc_from_distances, tpr_at_fpr_from_distances


def _load_float_lines(path: Path) -> list[float]:
    vals: list[float] = []
    raw = Path(path).read_text(encoding="utf-8")





    for line in raw.splitlines():


        stripped = line.strip()


        if not stripped:


            continue

        vals.append(float(stripped))



    return vals









def main() -> None:


    parser = argparse.ArgumentParser(description="Compute ROC-AUC / TPR from ℓ1 distance dumps")







    parser.add_argument("--watermarked-distances", required=True)





    parser.add_argument("--clean-distances", required=True)





    parser.add_argument("--target-fpr", type=float, default=0.01)













    args = parser.parse_args()









    wm = _load_float_lines(Path(args.watermarked_distances))



    ck = _load_float_lines(Path(args.clean_distances))




    auc = roc_auc_from_distances(wm, ck)



    tpp = tpr_at_fpr_from_distances(wm, ck, fpr=float(args.target_fpr))










    print({"roc_auc_signed_neg_distance_as_score": auc, f"tpr_at_fpr_{args.target_fpr}": tpp})




if __name__ == "__main__":




    main()

