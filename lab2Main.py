import pandas as pd
import matplotlib.pyplot as plt 

df = pd.read_csv("turkish-se-SP500vsMSCI.csv")

coordinates = []
coordinates = df.columns

row = df[coordinates[0]].tolist()
col = df[coordinates[1]].tolist()

plt.scatter(row, col) 
plt.show() 
