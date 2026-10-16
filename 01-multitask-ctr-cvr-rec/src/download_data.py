"""Downloads the MovieLens-1M public dataset (official GroupLens direct link, no login required)."""
import os
import urllib.request
import zipfile

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    zip_path = os.path.join(RAW_DIR, "ml-1m.zip")
    if not os.path.exists(zip_path):
        print(f"downloading {URL} ...")
        urllib.request.urlretrieve(URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW_DIR)
    print("done ->", os.path.join(RAW_DIR, "ml-1m"))


if __name__ == "__main__":
    main()
