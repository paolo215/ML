import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("processed.cleveland.data", header=None)

AGE, SEX, CP, TRESTBPS, CHOL, FBS, RESTECG, THALACH, EXANG, OLDPEAK, SLOPE, CA, THAL, NUM = range(14)

label = NUM

print(df.head())

def analyze_age_diagnosis():
    # Findings: Age above 45 are more likely to get diagnosed
    plt.hist([df[df[label] == 0][AGE], df[df[label]==1][AGE]], bins=30, stacked=True, color=["g", "r"],\
            label=["Not diagnosed", "Diagnosed"])
    plt.legend(loc="upper right")
    plt.savefig("Age.png")
    plt.clf()


analyze_age_diagnosis()
