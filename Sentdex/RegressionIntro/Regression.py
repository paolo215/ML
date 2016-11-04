import quandl
import pandas as pd
import math

# df = data frame
df = quandl.get("WIKI/GOOGL")

# print df.head() # each column is a feature
# Simplify data as much as possible

# Get relevant dataframe
df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume",]]

df["HL_PCT"] = (df["Adj. High"] - df["Adj. Close"]) / df ["Adj. Close"] * 100.0


df["PCT_change"] = (df["Adj. Close"] - df["Adj. Open"]) / df ["Adj. Open"] * 100.0

df = df[["Adj. Close", "HL_PCT", "PCT_change", "Adj. Volume"]]


# Features are the attributes that make up the label
# Label is a prediction into the future

forecast_col = "Adj. Close"

# fill not available
# ML can't work with NaN data so replace or get rid 
# of data.
df.fillna("-99999", inplace=True)

# Regression Algorithm
# Get the number of days out (predict 10% of the df)
forecast_out = int(math.ceil(0.01*len(df)))


# Shifting col. negatively (up)
# Each row would be the Adj. Close price 10 days
# into the future.
df["label"] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
# print(df.head())
