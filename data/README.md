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

## Public mirror used for the verified run

The Kaggle API requires authenticated competition access. The results committed to this repository were reproduced from the public Rossmann files referenced by the [RPI Analytics Dojo tutorial](https://rpifall2019.analyticsdojo.com/notebooks/18-intro-timeseries/02-forcasting-rossman.html):

- `train.csv` SHA-256: `F6E4597C142D7D909A13D53B68A8E85C00B9A4C7B5FF40ADBB37D6829CC1F4CC`
- `store.csv` SHA-256: `F56BD124A2849489E6BBB5C000F5FC9640204355E316475C918AE4D089AFB344`

The validated files contain 1,017,209 daily records for 1,115 stores from January 1, 2013 through July 31, 2015.

