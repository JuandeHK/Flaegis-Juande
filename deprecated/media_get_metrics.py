import numpy as np
import pandas as pd
import os

df = pd.read_csv(os.getcwd() + "/resultados.csv")
df = df["Accuracy"]
df_mean = df[-5:]
df_mean = np.mean(df_mean)

df = pd.read_csv(os.getcwd() + "/resultados.csv")
df = df["Detector"]
df_det = df[-5:]
df_det = np.mean(df_det)

#print(df)
df = pd.read_csv(os.getcwd() + "/resultados.csv")
new = pd.DataFrame({"Accuracy": df_mean, "Detector": df_det}, index=[0])
df = pd.concat([df, new], ignore_index=True)
df.to_csv(os.getcwd() + "/resultados.csv", index=False)