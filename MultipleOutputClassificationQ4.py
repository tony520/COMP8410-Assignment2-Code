import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

df = pd.read_csv("preprocessing_data.csv", usecols=["Q1", "Q5c", "Q5d", "Q5e", "Q5f", "StateMap", "p_age_group_sdc", "p_education_sdc", "Q4"])

df = df[["Q1", "Q5c", "Q5d", "Q5e", "Q5f", "StateMap", "p_age_group_sdc", "p_education_sdc", "Q4"]]

dftrain = df.head(1505)
dftest = df.tail(645)

training = dftrain.values
test = dftest.values
trainSet = []
testSet = []

evaluation = []

for i in range(len(training)):
  temp = []
  testtemp = []
  if training[i][8] != -98 and training[i][8] != -99 and training[i][8] != 97:
    for j in training[i]:
      if j == ' ':
        temp.append(0)
      else:
        temp.append(int(j))
    trainSet.append(temp)
  else:
    for k in range(len(training[i])-1):
      if training[i][k] == ' ':
        testtemp.append(0)
      else:
        testtemp.append(int(training[i][k]))
    testSet.append(testtemp)

for i in range(len(test)):
  temp = []
  if test[i][8] != -98 and test[i][8] != -99 and training[i][8] != 97:
    for j in test[i]:
      if j == ' ':
        temp.append(0)
      else:
        temp.append(int(j))

    evaluation.append(temp)

evalX = []

actual = []
predict = []

for i in evaluation:
  actual.append(i[8])
  temp = []
  for j in range(len(i)-1):
    temp.append(i[j])
  evalX.append(temp)




# The data from your screenshot
#  Q1 Q5c Q5d Q5e Q5f StateMap p_age_group_sdc p_education_sdc, Q4
train_data = np.array(trainSet)
# These I just made up
test_data_x = np.array(testSet)

eval_data_x = np.array(evalX)

x = train_data[:, :8]
y = train_data[:, 8:]
forest = RandomForestClassifier(n_estimators=100, random_state=1)
classifier = MultiOutputClassifier(forest, n_jobs=-1)
classifier.fit(x, y)
pr = classifier.predict(evalX)

print(pr)

for i in pr:
  predict.append(i[0])

error_count = 0

for i in range(len(actual)):
  if actual[i] != predict[i] :
    error_count += 1

print("Precision: ", (593 - error_count) / 593)