# src/features/feature_definitions.py
import pathlib
import numpy as np
import pandas as pd


def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    R = 6371.0  # km
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng * 0.5) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b


def bearing_array(lat1, lng1, lat2, lng2):
    lat1r, lng1r, lat2r, lng2r = map(np.radians, (lat1, lng1, lat2, lng2))
    dlng = lng2r - lng1r
    y = np.sin(dlng) * np.cos(lat2r)
    x = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlng)
    return np.degrees(np.arctan2(y, x))


def datetime_feature_fix(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    if "store_and_fwd_flag" in df.columns:
        df["store_and_fwd_flag"] = (df["store_and_fwd_flag"] == "Y").astype(int)


def create_dist_features(df):
    lat1 = df["pickup_latitude"].values
    lng1 = df["pickup_longitude"].values
    lat2 = df["dropoff_latitude"].values
    lng2 = df["dropoff_longitude"].values

    df["distance_km"] = haversine_array(lat1, lng1, lat2, lng2)
    df["distance_manhattan_km"] = dummy_manhattan_distance(lat1, lng1, lat2, lng2)
    df["direction_deg"] = bearing_array(lat1, lng1, lat2, lng2)
    df["distance_km_log1p"] = np.log1p(df["distance_km"])


def create_datetime_features(df, ref_datetime=None):
    dt = df["pickup_datetime"]
    df["pickup_weekday"] = dt.dt.weekday
    df["pickup_hour"] = dt.dt.hour
    df["pickup_minute"] = dt.dt.minute

    if ref_datetime is None:
        ref = pd.Timestamp("1970-01-01")
    else:
        ref = pd.to_datetime(ref_datetime)

    df["pickup_dt"] = (dt - ref).dt.total_seconds().astype("int64")

    # cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["pickup_weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["pickup_weekday"] / 7)

    # direction as circular
    rad = np.radians(df["direction_deg"].fillna(0.0))
    df["dir_sin"] = np.sin(rad)
    df["dir_cos"] = np.cos(rad)


def feature_build(df: pd.DataFrame, tag: str, ref_datetime=None) -> pd.DataFrame:
    required = [
        "pickup_datetime",
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    datetime_feature_fix(df)
    create_dist_features(df)
    create_datetime_features(df, ref_datetime=ref_datetime)

    do_not_use_for_training = [
        "id",
        "pickup_datetime",
        "dropoff_datetime",
        "pickup_date",
        "pickup_datetime_group",
        "check_trip_duration",
    ]

    feature_names = [c for c in df.columns if c not in do_not_use_for_training]
    print(f"We have {len(feature_names)} features in {tag}.")
    return df[feature_names]
