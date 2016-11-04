import quandl
import pandas as pd


df = quandl.get("WIKI/GOOGL")

# print df.head() # each column is a feature
# Simplify data as much as possible

# Get relevant dataframe
df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume",]]

df["HL_PCT"] = (df["Adj. High"] - df["Adj. Close"]) / df ["Adj. Close"] * 100.0


df["PCT_change"] = (df["Adj. Close"] - df["Adj. Open"]) / df ["Adj. Open"] * 100.0

df = df[["Adj. Close", "HL_PCT", "PCT_change", "Adj. Volume"]]


print df.head()

# Features are the attributes that make up the label
# Label is a prediction into the future
