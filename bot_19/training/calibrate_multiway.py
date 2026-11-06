"""Fit a tiny HU-to-multiway correction on synthetic tuning cases only."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poker_eval import features


def bucket(f):
    cat = int(f[6])
    if cat == 1 and not (f[3] or f[29] or f[30]):
        cat = 0  # paired board, no private pair
    cat = min(cat, 2)
    draw = f[0] < 5 and ((f[22] == 4 and f[23]) or (f[27] >= 2 and f[39]))
    return f"{int(f[0])}:{cat}:{int(bool(draw))}"


def power_eq(q, n, p):
    return q ** (n ** p)


def beta_eq(q, n, concentration):
    result = q
    for k in range(1, n):
        result *= (q * concentration + k) / (concentration + k)
    return result


def make_rows(data):
    rows = []
    for case in data["cases"]:
        f = features(case["hole"], case["board"], case["width"], case["aggression"])
        rows.append((bucket(f), case["equity"][0], list(zip(data["opponents"][1:], case["equity"][1:]))))
    return rows


def score(rows, prediction):
    es = [prediction(key, q, n)-eq for key, q, outcomes in rows for n, eq in outcomes]
    return {"rows": len(rows), "comparisons": len(es), "rmse": math.sqrt(sum(e*e for e in es)/len(es)),
            "mae": sum(abs(e) for e in es)/len(es), "bias": sum(es)/len(es)}


def fit(rows):
    candidates = [(.30+i*.025) for i in range(37)]
    concs = [.25, .5, .75, 1, 1.5, 2, 3, 4, 6, 8, 12, 20, 40, 100, 1e6]
    groups = {key: [row for row in rows if row[0] == key] for key in sorted({r[0] for r in rows})}
    street_power = {}
    street_conc = {}
    for street in ("3", "4", "5"):
        subset = [row for row in rows if row[0].startswith(street)]
        street_power[street] = min(candidates, key=lambda p: score(subset, lambda key,q,n:power_eq(q,n,p))["rmse"])
        street_conc[street] = min(concs, key=lambda c: score(subset, lambda key,q,n:beta_eq(q,n,c))["rmse"])
    powers, concentration = {}, {}
    for key, subset in groups.items():
        street = key[0]
        # Mild regularization keeps sparse categories near their street mean.
        powers[key] = min(candidates, key=lambda p: score(subset,lambda key,q,n:power_eq(q,n,p))["rmse"]**2*len(subset)*4 + .40*(p-street_power[street])**2)
        concentration[key] = min(concs, key=lambda c: score(subset,lambda key,q,n:beta_eq(q,n,c))["rmse"]**2*len(subset)*4 + .001*math.log(c/street_conc[street])**2)
    return {"street_power": street_power, "power": powers, "concentration": concentration,
            "group_sizes": {key:len(subset) for key,subset in groups.items()}}


def evaluate(rows, model):
    methods = {
        "original_085": lambda key,q,n:power_eq(q,n,.85),
        "street_power": lambda key,q,n:power_eq(q,n,model["street_power"][key[0]]),
        "bucket_power": lambda key,q,n:power_eq(q,n,model["power"].get(key,model["street_power"][key[0]])),
        "bucket_beta": lambda key,q,n:(power_eq(q,n,model["power"].get(key,model["street_power"][key[0]])) if key[0]=="5" else beta_eq(q,n,model["concentration"].get(key,8))),
    }
    return {method: {"all": score(rows, fn),
                     **{f"street_{street}":score([r for r in rows if r[0][0]==street],fn) for street in ("3","4","5")}}
            for method,fn in methods.items()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--train",default="measurements/equity_multiway_tuning_cases.json")
    parser.add_argument("--test")
    parser.add_argument("--output",default="measurements/equity_multiway_calibration.json")
    args=parser.parse_args()
    train=json.loads(Path(args.train).read_text())
    train_rows=make_rows(train)
    model=fit(train_rows)
    report={"training_seed":train["seed"],"model":model,"training_scores":evaluate(train_rows,model)}
    if args.test:
        test=json.loads(Path(args.test).read_text())
        if test["seed"]==train["seed"]:
            raise ValueError("train/test seed overlap")
        report["heldout_seed"]=test["seed"]
        report["heldout_scores"]=evaluate(make_rows(test),model)
    Path(args.output).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
