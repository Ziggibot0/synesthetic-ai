import pandas as pd, os
fr = r"C:\Users\skell\Desktop\embedding-vibes\data\logical-fallacy-repo\data"
for f in ["edu_all.csv", "climate_all.csv", "edu_train.csv", "edu_dev.csv", "edu_test.csv"]:
    p = os.path.join(fr, f)
    df = pd.read_csv(p)
    print(f, df.shape)
    print("  cols:", list(df.columns))
    print("  sample:", df.iloc[0].to_dict())
    if "fallacy" in [c.lower() for c in df.columns]:
        pass
    print()
