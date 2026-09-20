import collections
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/results"
sample = pd.read_csv(ROOT / "dataset/журнал_событий_пример.csv", dtype=str, keep_default_na=False)
COLS = list(sample.columns)
sample[COLS[4]] = sample[COLS[4]].replace({"true": "t", "false": "f"})
sample = sample.set_index(COLS[0])
idset = set(sample.index)


def probe(s):
    proc = subprocess.Popen(
        [r"C:/Program Files/7-Zip/7z.exe", "x", "-so", str(ROOT / "dataset" / s["file"])],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    found = []
    print("COMPARE " + s["file"], flush=True)
    for chunk in pd.read_csv(proc.stdout, dtype=str, keep_default_na=False, chunksize=1000000):
        hit = chunk[chunk.iloc[:, 0].isin(idset)]
        if not hit.empty:
            found.append(hit)
    proc.stdout.close()
    error = proc.stderr.read()
    if proc.wait() != 0:
        raise RuntimeError(error)
    matches = pd.concat(found, ignore_index=True) if found else pd.DataFrame(columns=COLS)
    original = sample.reindex(matches[COLS[0]]).reset_index()
    diff = {c: int(matches[c].ne(original[c]).sum()) for c in COLS[1:]}
    all_equal = (matches[COLS[1:]].to_numpy() == original[COLS[1:]].to_numpy()).all(axis=1)
    examples = []
    for i in list(matches.index[~all_equal])[:4]:
        examples.append({"archive": matches.loc[i].to_dict(), "sample": original.loc[i].to_dict()})
    result = {
        "file": s["file"],
        "matched_rows": len(matches),
        "fully_matching_rows_normalized_alarm": int(all_equal.sum()),
        "different_fields": diff,
        "examples": examples,
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    results = []
    wanted = np.array(sorted(map(int, idset)), dtype=np.int64)
    for year in range(2019, 2027):
        path = OUT / f"ext-journal-{year}.json"
        while not path.exists():
            time.sleep(5)
        s = json.loads(path.read_text(encoding="utf-8"))
        arr = np.load(OUT / f"ext-journal-{year}.ids.npy", mmap_mode="r")
        pos = np.searchsorted(arr, wanted)
        hits = ((pos < len(arr)) & (arr[np.minimum(pos, len(arr) - 1)] == wanted)).sum()
        del arr
        if hits:
            results.append(probe(s))
    matches = sum(s["matched_rows"] for s in results)
    exact = sum(s["fully_matching_rows_normalized_alarm"] for s in results)
    differences = collections.Counter()
    for s in results:
        differences.update(s["different_fields"])
    changed = ", ".join(f"{k}: {v:,}".replace(",", " ") for k, v in differences.items() if v)
    text = (
        f"Проверены все поля совпавших ID, с приведением флага тревоги к t/f. "
        f"Совпавших исторических строк: {matches:,}; полных совпадений после нормализации: {exact:,}. "
        f"Различия по полям: {changed or 'не обнаружены'}. "
        "Если один ID имеет разные даты, эти строки нельзя считать независимыми событиями без проверки правил выгрузки."
    ).replace(",", " ")
    (OUT / "sample_comparison.json").write_text(
        json.dumps({"years": results, "report_text": text}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
