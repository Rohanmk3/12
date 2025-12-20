import pandas as pd
import numpy as np

data = pd.read_csv(r"C:\Users\HPR\Desktop\ML Syllabus\2.csv")

concepts = np.array(data)[:, :-1]
target = np.array(data)[:, -1]

def train(con, tar):
    # Step 1: Find the first positive example
    for i, val in enumerate(tar):
        if val == 'yes':
            specific_h = con[i].copy()
            break

    # Step 2: Update specific hypothesis for all positive examples
    for i, val in enumerate(con):
        if tar[i] == 'yes':
            for x in range(len(specific_h)):
                if val[x] != specific_h[x]:
                    specific_h[x] = '?'

    return specific_h


print(train(concepts, target))
