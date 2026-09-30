# Data setup

Download the files from the [Kaggle Rossmann Store Sales competition](https://www.kaggle.com/competitions/rossmann-store-sales/data).

Place these files here:

```text
data/raw/train.csv
data/raw/store.csv
data/raw/test.csv       # optional
```

With the Kaggle CLI configured and the competition rules accepted, you can run:

```powershell
kaggle competitions download -c rossmann-store-sales -p data/raw
Expand-Archive -Path data/raw/rossmann-store-sales.zip -DestinationPath data/raw -Force
```

The raw and processed data directories are ignored by Git. Do not commit the competition data.

