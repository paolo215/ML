import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("processed.cleveland.data", header=None)
df = df.replace('?', float(0.0))


AGE, SEX, CP, TRESTBPS, CHOL, FBS, RESTECG, THALACH, EXANG, OLDPEAK, SLOPE, CA, THAL, NUM = range(14)
label = NUM

df[label][df[label] == 0] = -1.0
df[label][df[label] > 1] = 1.0


def analyze_age_diagnosis():
    # Findings: Age above 45 are more likely to get diagnosed
    plt.hist([df[df[label] == 0][AGE], df[df[label]==1][AGE]], bins=30, stacked=True, color=["g", "r"],\
            label=["Not diagnosed", "Diagnosed"])
    plt.legend(loc="upper right")
    plt.ylabel("# of people")
    plt.xlabel("Age")
    plt.savefig("Age.png")
    plt.clf()


def analyze_sex_diagnosis():
    fig, ax = plt.subplots()
    # Ratio between Female and Males are not equal. However, it seems that Males
    # are more likely to get diagnosed
    ax.hist([df[df[label] == 0][SEX], df[df[label]==1][SEX]], bins=2, stacked=True, color=["g", "r"],\
            label=["Not diagnosed", "Diagnosed"])
    ax.set_xticks([0.25, 0.75])
    ax.set_xticklabels(["Female", "Male"])
    ax.set_ylabel("# of people")
    ax.set_xlabel("Age")
    ax.legend(loc="upper right")
    plt.savefig("Sex.png")
    plt.clf()
    
    
def analyze_age_female_diagnosis():
    fig, ax = plt.subplots()
    # Females age between 55 to 65 are more likely to get diagnosed
    ax.hist([df[df[SEX] == 0][df[label] == 0][AGE], df[df[SEX] == 0][df[label] == 1][AGE]], bins=30, stacked=True, color=["g", "r"], label=["Not diagnosed", "Diagnosed"])
    ax.set_ylabel("# of people")
    ax.set_xlabel("Age")
    ax.legend(loc="upper right")
    plt.savefig("Female_Age.png")
    plt.clf()


def analyze_age_male_diagnosis():
    fig, ax = plt.subplots()
    # Males age above 40 are likely to get diagnosed
    # 55 to 65 have higher likelihood
    ax.hist([df[df[SEX] == 1][df[label] == 0][AGE], df[df[SEX] == 1][df[label] == 1][AGE]], bins=30, stacked=True, color=["g", "r"], label=["Not diagnosed", "Diagnosed"])
    ax.legend(loc="upper right")
    ax.set_ylabel("# of people")
    ax.set_xlabel("Age")
    plt.savefig("Male_Age.png")
    plt.clf()
    

def analyze_age_sex_diagnosis():
    fig, ax = plt.subplots()
    non_diag = df[df[label] == 0]
    diag = df[df[label] == 1]
    plt.hist([non_diag[df[SEX] == 0][AGE], non_diag[df[SEX] == 1][AGE], diag[df[SEX] == 0][AGE], diag[df[SEX] == 1][AGE]], bins=30, stacked=True, label=["Female - Not diagnosed", "Male - Not diagnosed", "Female - Diagnosed", "Male - Diagnosed"] )
    ax.legend(loc="upper left")
    ax.set_ylabel("# of people")
    ax.set_xlabel("Age")
    plt.savefig("Age_Sex.png")
    plt.clf()


def analyze_age_sex():
    global df
    clf_rfc = RandomForestClassifier(n_estimators=30, max_depth=1, min_samples_split=2) 
    clf_svm = SVC(C=0.65, kernel="sigmoid")
    labels = np.array(df[label])
    df_features = df[[AGE, SEX]]

    trainingX, testingX, trainingY, testingY = train_test_split(df_features, labels, test_size=0.5, stratify=labels)

    clf_rfc.fit(trainingX, trainingY)
    clf_svm.fit(trainingX, trainingY)
    score_rfc = clf_rfc.score(testingX, testingY)
    score_svm = clf_svm.score(testingX, testingY)
    print("rfc", score_rfc)
    print("svm", score_svm) 

        
analyze_age_diagnosis()
analyze_sex_diagnosis()
analyze_age_female_diagnosis()
analyze_age_male_diagnosis()
analyze_age_sex_diagnosis()
analyze_age_sex()

