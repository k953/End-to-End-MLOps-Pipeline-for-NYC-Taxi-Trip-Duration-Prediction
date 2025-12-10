# src/data/make_dataset.py
import pathlib
import pandas as pd

from src.features.feature_definitions import feature_build


def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    raw_train = home_dir / "data" / "raw" / "train.csv"
    processed_dir = home_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(raw_train)
    df_train = feature_build(df_train, tag="train")

    out_path = processed_dir / "train.csv"
    df_train.to_csv(out_path, index=False)
    print(f"Saved processed train data to {out_path}")


if __name__ == "__main__":
    main()
