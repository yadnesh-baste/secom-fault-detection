"""Downloads the SECOM dataset (UCI, CC BY 4.0) and writes data/uci-secom.csv with a Time and Pass/Fail column."""
import io, zipfile, urllib.request, pandas as pd, os
os.makedirs("data", exist_ok=True)
try:
    raw = urllib.request.urlopen("https://archive.ics.uci.edu/static/public/179/secom.zip", timeout=30).read()
except Exception as e:  # fallback: a CSV copy of the same data on GitHub
    print("UCI download failed (", e, "), using GitHub mirror")
    urllib.request.urlretrieve("https://raw.githubusercontent.com/jurajpaterek/secom/main/data/uci-secom.csv", "data/uci-secom.csv")
    raise SystemExit("saved to data/uci-secom.csv")
z = zipfile.ZipFile(io.BytesIO(raw))
X = pd.read_csv(z.open("secom.data"), sep=" ", header=None)
lab = pd.read_csv(z.open("secom_labels.data"), sep=" ", header=None, names=["Pass/Fail", "Time"])
X.columns = [str(c) for c in X.columns]
df = pd.concat([lab[["Time"]], X, lab[["Pass/Fail"]]], axis=1)
df["Time"] = pd.to_datetime(df["Time"].str.strip('"'), format="%d/%m/%Y %H:%M:%S").dt.strftime("%Y-%m-%d %H:%M:%S")
df.to_csv("data/uci-secom.csv", index=False); print(df.shape, "saved to data/uci-secom.csv")
