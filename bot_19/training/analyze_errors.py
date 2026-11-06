"""Inspect validation error by feature strata without reading held-out data."""
import argparse
from array import array
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from equity_model import predict_from_features
from poker_eval import FEATURE_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training/data/validation.bin")
    parser.add_argument("--output", default="measurements/equity_validation_errors.json")
    args = parser.parse_args()
    buf = array("f")
    buf.frombytes(Path(args.data).read_bytes())
    rows, groups = [], {}
    stride = len(FEATURE_NAMES)+1
    for offset in range(0,len(buf),stride):
        f=list(buf[offset:offset+stride-1]); label=buf[offset+stride-1]
        pred=predict_from_features(f);error=pred-label
        row={"abs_error":abs(error),"error":error,"prediction":pred,"label":label,
             "features":dict(zip(FEATURE_NAMES,f))}
        rows.append(row)
        names = [f"street_{int(f[0])}", f"street_{int(f[0])}_category_{int(f[6])}",
                 f"street_{int(f[0])}_board_suit_{int(f[21])}"]
        if f[38] == 0:
            names.append(f"street_{int(f[0])}_board_category_shared")
        for name in names:
            groups.setdefault(name,[]).append(error)
    result={"source":args.data,"kind":"validation diagnostics only",
            "worst_50":sorted(rows,key=lambda r:r["abs_error"],reverse=True)[:50],
            "groups":{name:{"rows":len(es),"rmse":math.sqrt(sum(e*e for e in es)/len(es)),
                            "bias":sum(es)/len(es)} for name,es in sorted(groups.items())}}
    Path(args.output).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result["groups"],indent=2))
    for row in result["worst_50"][:15]:
        f=row["features"]
        print({"error":row["error"],"pred":row["prediction"],"label":row["label"],
               "street":f["board_count"],"hole":[f["hole_high"],f["hole_low"]],
               "hero_score":[f["category"],f["rank_1"],f["rank_2"],f["rank_3"],f["rank_4"],f["rank_5"]],
               "board_ranks":[f["board_high"],f["board_2"],f["board_3"],f["board_4"],f["board_5"]],
               "board_suit":f["board_max_suit"],"width":f["range_width"],"aggression":f["aggression"]})


if __name__=="__main__":
    main()
