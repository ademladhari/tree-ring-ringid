"""CLI: watermark identification (`argmin` ℓ1 over stored keys)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ringid.config import WatermarkProfile, watermark_profile_from_dict

from ringid.detect import identification_argmin, invert_then_pattern, l1, load_pil_rgb

from ringid.sampling import load_pipeline




def gather_keys(entries: list[str]) -> list[Path]:


    out: list[Path] = []

    for entry in entries:
        pp = Path(entry)

        if pp.is_dir():
            out.extend(sorted(pp.glob("**/*.json")))
        elif pp.is_file():
            out.append(pp)

    out_unique = sorted(set(out))





    if not out_unique:


        raise RuntimeError("No key JSON matched the `--key-json` paths.")



    return out_unique




def profile_from_any_key_json(path: Path) -> WatermarkProfile:






    blob = json.loads(path.read_text(encoding="utf-8"))


    return watermark_profile_from_dict(blob["profile"])






def main() -> None:
    parser = argparse.ArgumentParser(description="Identification: argmin_i ℓ1(w_hat, w_i)")





    parser.add_argument("--candidate", required=True)





    parser.add_argument("--prompt", required=True)





    parser.add_argument("--negative-prompt", default="")







    parser.add_argument("--key-json", nargs="+", required=True)













    parser.add_argument("--model-id", default=None)









    parser.add_argument("--inversion-steps", type=int, default=None)









    parser.add_argument("--inversion-guidance-scale", type=float, default=None)






    argv = parser.parse_args()







    ks = gather_keys(argv.key_json)




    wm_profile = profile_from_any_key_json(ks[0])





    if argv.model_id:

        wm_profile.model_id = argv.model_id









    pipe = load_pipeline(wm_profile)









    cand = invert_then_pattern(
        pipe,
        pil_image=load_pil_rgb(argv.candidate),


        prompt=argv.prompt,


        negative_prompt=argv.negative_prompt,





        profile=wm_profile,



        num_inference_steps=argv.inversion_steps,





        guidance_scale=argv.inversion_guidance_scale,



    )





    dist_map: dict[str, float] = {}






    for kpath in ks:





        data = json.loads(kpath.read_text(encoding="utf-8"))





        dist_map[kpath.as_posix()] = float(l1(cand, torch.tensor(data["vector"], dtype=torch.float32)))














    kk, vv = identification_argmin(dist_map)




    print({"best_key_json": kk, "best_l1": vv, "candidates_considered": len(dist_map)})















if __name__ == "__main__":





    main()

