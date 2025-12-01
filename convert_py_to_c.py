import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ========= CONFIGURATION =========

CSV_PATH = "ai 2020.csv"   # your CSV file

# If your CSV has a header row with column names:
HAS_HEADER = True

# Name of the label column (can be in the middle)
LABEL_COL_NAME = "Machine failure"   # <-- change this to your label column name

# EITHER:
#   1) Explicitly list feature columns you want to use:
FEATURE_COL_NAMES = [
    # "feature1", "feature2", "feature3",
]

# OR:
#   2) List columns you want to DROP (besides the label),
#      and keep all the rest as features:
DROP_COL_NAMES = [
     "UDI", "Product ID",
]

# If your CSV has *no* header row, set HAS_HEADER = False above
# and configure these instead:
LABEL_COL_INDEX = None      # e.g. 4
FEATURE_COL_INDEXES = []    # e.g. [0, 1, 2, 3]
DROP_COL_INDEXES = []       # e.g. [5, 6]

# Output header file:
OUT_PATH = "test_data.h"

# =================================


def main():
    if HAS_HEADER:
        df = pd.read_csv(CSV_PATH)
        le = LabelEncoder()
        df['Type'] = le.fit_transform(df['Type'])
        print("Columns found:", list(df.columns))

        if LABEL_COL_NAME not in df.columns:
            raise ValueError(f"Label column '{LABEL_COL_NAME}' not found in CSV")

        # Label column
        y = df[LABEL_COL_NAME].astype(np.float32)

        # Determine feature columns
        if FEATURE_COL_NAMES:
            # Explicit feature list
            for col in FEATURE_COL_NAMES:
                if col not in df.columns:
                    raise ValueError(f"Feature column '{col}' not found in CSV")
            feature_cols = FEATURE_COL_NAMES
        else:
            # Keep everything except label and DROP_COL_NAMES
            drop_set = set(DROP_COL_NAMES + [LABEL_COL_NAME])
            feature_cols = [c for c in df.columns if c not in drop_set]

        if not feature_cols:
            raise ValueError("No feature columns selected!")

        X = df[feature_cols].astype(np.float32)

    else:
        # No header row: use numeric indices
        df = pd.read_csv(CSV_PATH, header=None)

        if LABEL_COL_INDEX is None:
            raise ValueError("Set LABEL_COL_INDEX when HAS_HEADER = False")

        if FEATURE_COL_INDEXES:
            feat_idxs = FEATURE_COL_INDEXES
        else:
            drop_set = set(DROP_COL_INDEXES + [LABEL_COL_INDEX])
            # Keep all indices except the ones in drop_set
            feat_idxs = [i for i in range(df.shape[1]) if i not in drop_set]

        if not feat_idxs:
            raise ValueError("No feature columns selected!")

        y = df.iloc[:, LABEL_COL_INDEX].astype(np.float32)
        X = df.iloc[:, feat_idxs].astype(np.float32)

    # Convert to numpy arrays
    X_np = X.to_numpy()
    y_np = y.to_numpy()

    num_samples, num_features = X_np.shape
    print(f"Samples: {num_samples}, Features: {num_features}")

    # ====== WRITE C HEADER ======
    with open(OUT_PATH, "w") as f:
        f.write(f"// Auto-generated from {CSV_PATH}\n")
        f.write("#ifndef TEST_DATA_H\n")
        f.write("#define TEST_DATA_H\n\n")
        f.write(f"constexpr int kNumTestSamples = {num_samples};\n")
        f.write(f"constexpr int kNumFeatures = {num_features};\n\n")

        # test_inputs
        f.write("const float test_inputs[kNumTestSamples][kNumFeatures] = {\n")
        for row in X_np:
            vals = ", ".join(f"{float(v):.7f}f" for v in row)
            f.write(f"  {{{vals}}},\n")
        f.write("};\n\n")

        # expected_outputs
        f.write("const float expected_outputs[kNumTestSamples] = {\n")
        vals_y = ", ".join(f"{float(v):.7f}f" for v in y_np)
        f.write(f"  {vals_y},\n")
        f.write("};\n\n")

        f.write("#endif // TEST_DATA_H\n")

    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

