import argparse
import json
import re
from collections import OrderedDict

LINE_RE = re.compile(r"^(.+?)\s+(\d+)\s+(\S+)\s+(-?[\d.]+)\s*±\s*(-?[\d.]+)\s*$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="result.txt")
    parser.add_argument("--output", default="result_mean.json")
    args = parser.parse_args()

    games = OrderedDict()
    with open(args.input, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or lineno == 1:
                continue
            m = LINE_RE.match(line)
            if m is None:
                raise ValueError(f"line {lineno} cannot be parsed: {line!r}")
            game, seed, method, mean, std = m.groups()
            games.setdefault(game, OrderedDict())[seed] = float(mean)

    result = OrderedDict()
    for game, seed_scores in games.items():
        scores = list(seed_scores.values())
        result[game] = OrderedDict(
            num_seeds=len(scores),
            mean=round(sum(scores) / len(scores), 2),
            seed_scores=seed_scores,
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    for game, r in result.items():
        print(f"{game:<16} seeds={r['num_seeds']}  mean={r['mean']}")
    print(f"\nwritten to {args.output}")


if __name__ == "__main__":
    main()
