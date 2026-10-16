"""Downloads the public Avazu CTR dataset (official LIBSVM mirror, no login required) and slices the first 400K lines for this project."""
import bz2
import os
import urllib.request

URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.bz2"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
N_LINES = 400_000


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    bz2_path = os.path.join(RAW_DIR, "avazu-app.bz2")
    subset_path = os.path.join(RAW_DIR, "avazu_subset.svm")

    if not os.path.exists(bz2_path):
        print(f"downloading {URL} (~150MB) ...")
        urllib.request.urlretrieve(URL, bz2_path)

    print(f"extracting first {N_LINES} lines -> {subset_path}")
    with bz2.open(bz2_path, "rt") as fin, open(subset_path, "w") as fout:
        for i, line in enumerate(fin):
            if i >= N_LINES:
                break
            fout.write(line)
    print("done.")


if __name__ == "__main__":
    main()
