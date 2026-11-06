"""Compile trusted exported trees into ordinary Python conditional expressions.

This runs offline. Gameplay imports the generated source once, never calls
eval/exec or generates code while deciding. The generated model binds the
SHA256 of its JSON source and is accepted only with that exact source asset.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def expression(tree, node=0):
    entry = tree[node]
    if entry[0] < 0:
        return repr(entry[1])
    left = expression(tree, entry[2])
    right = expression(tree, entry[3])
    return f"({left} if f{entry[0]} < {f32(entry[1])!r} else {right})"


def export(model_path, output):
    source = Path(model_path).read_bytes()
    model = json.loads(source)
    digest = hashlib.sha256(source).hexdigest()
    lines = ['"""Generated offline by training/export_fast.py; do not edit."""',
             "from struct import pack as _pack, unpack as _unpack", f"SOURCE_SHA256 = {digest!r}",
             "SCHEMA = 1", ""]
    if model.get("derived_version"):
        lines.append("from equity_features import derived_features as _derived")
    inputs = len(model.get("input_feature_names", model["feature_names"]))
    names = ", ".join(f"f{i}" for i in range(inputs))
    for street, forest in sorted(model["streets"].items()):
        lines.extend([f"def predict_{street}(f):", f"    {names} = f",
                      "    f41, f42 = _unpack('ff', _pack('ff', f41, f42))",
                      f"    value = {forest['base']!r}"])
        if forest.get("derived"):
            derived_names = ", ".join(f"f{i}" for i in range(inputs, len(model["feature_names"])))
            lines.append(f"    {derived_names} = _derived(f)")
        for tree in forest["trees"]:
            lines.append("    value += " + expression(tree))
        lines.extend(["    return .001 if value < .001 else .999 if value > .999 else value", ""])
    lines.append("MODELS = {3: predict_3, 4: predict_4, 5: predict_5}")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"model_sha256": digest, "generated_sha256": hashlib.sha256(Path(output).read_bytes()).hexdigest(),
            "source_bytes": len(source), "generated_bytes": Path(output).stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="assets/equity_model.json")
    parser.add_argument("--output", default="assets/equity_fast.py")
    parser.add_argument("--report", default="measurements/equity_codegen.json")
    args = parser.parse_args()
    result = export(args.model, args.output)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
