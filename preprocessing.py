import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def readFile(filePath):
	df = pd.read_csv(filePath, usecols=["srcid", "Mode", "Q1", "Q2", "Q3", "Q4", "Q5a", "Q5b", "Q5c", "Q5d", "Q5e", "Q5f", "Q6a", "Q6b", "Q6c", "Q6d", "Q7a", "Q7b", "Q7c", "Q7d", "Q7e", "Q8a", "Q8b", "Q8c", "Q8d", "Q8e", "Q8f", "Q8g", "Q8h", "Q10a", "Q10b", "Q10c", "Q10d", "Q12", "Q13a", "Q13b", "Q13c", "Q13d", "Q13e", "Q13f", "Q14", "Q15", "Q16", "Q17a", "Q17b", "Q17c", "Q17d", "Q17e", "Q17f", "Q18", "Q19a", "Q19b", "Q19c", "Q19d", "Q19e", "Q19f", "Q19g", "Sect_0_time", "Sect_0_time", "Sect_1_time", "Sect_2_time", "Sect_3_time", "Sect_4_time", "Sect_5_time", "Sect_6_time", "Sect_7_time", "Sect_8_time", "Sect_9_time", "Sect_10_time", "Sect_11_time", "Sect_12_time", "Sect_13_time", "Sect_14_time", "Sect_15_time", "Sect_16_time", "StateMap", "p_gender_sdc", "p_age_group_sdc", "p_education_sdc"], index_col=0)
	return df

def preprocessing(filePath):
	df = readFile(filePath)

	# Get total time
	df["total_time_taken"] = df[["Sect_0_time", "Sect_1_time", "Sect_2_time", "Sect_3_time", "Sect_4_time", "Sect_5_time", "Sect_6_time", "Sect_7_time", "Sect_8_time", "Sect_9_time", "Sect_10_time", "Sect_11_time", "Sect_12_time", "Sect_13_time", "Sect_14_time", "Sect_15_time", "Sect_16_time"]].apply(lambda x: float(x["Sect_0_time"].replace(' ', '0')) + float(x["Sect_1_time"].replace(' ', '0')) + float(x["Sect_2_time"].replace(' ', '0')) + float(x["Sect_3_time"].replace(' ', '0')) + float(x["Sect_4_time"].replace(' ', '0')) + float(x["Sect_5_time"].replace(' ', '0')) + float(x["Sect_6_time"].replace(' ', '0')) + float(x["Sect_7_time"].replace(' ', '0')) + float(x["Sect_8_time"].replace(' ', '0')) + float(x["Sect_9_time"].replace(' ', '0')) + float(x["Sect_10_time"].replace(' ', '0')) + float(x["Sect_11_time"].replace(' ', '0')) + float(x["Sect_12_time"].replace(' ', '0')) + float(x["Sect_13_time"].replace(' ', '0')) + float(x["Sect_14_time"].replace(' ', '0')) + float(x["Sect_15_time"].replace(' ', '0')) + float(x["Sect_16_time"].replace(' ', '0')), axis=1)
	# Get average trust point of Q8
	df["avg_trust_Q8"] = df[["Q8a", "Q8b", "Q8c", "Q8d", "Q8e", "Q8f", "Q8g", "Q8h"]].apply(lambda x: (float(x["Q8a"].replace(' ', '0')) + float(x["Q8b"].replace(' ', '0')) + float(x["Q8c"].replace(' ', '0')) + float(x["Q8d"].replace(' ', '0')) + float(x["Q8e"].replace(' ', '0')) + float(x["Q8f"].replace(' ', '0')) + float(x["Q8g"].replace(' ', '0')) + float(x["Q8h"].replace(' ', '0'))) / 8, axis=1)
	df = df.drop(["Sect_0_time", "Sect_1_time", "Sect_2_time", "Sect_3_time", "Sect_4_time", "Sect_5_time", "Sect_6_time", "Sect_7_time", "Sect_7_time", "Sect_8_time", "Sect_9_time", "Sect_10_time", "Sect_11_time", "Sect_12_time", "Sect_13_time", "Sect_14_time", "Sect_15_time", "Sect_16_time"], axis=1)
	# Fill blank cells
	#df = df.replace(r'^\s*$', "NaN", regex=True)
	# Get top 30 common answers of Q2
	Q2AnswerCount = df["Q2"].value_counts().index
	Q2Top30Answers = Q2AnswerCount[0:30]
	for i in df["Q2"]:
		if i not in Q2Top30Answers and i != "100":
			df["Q2"] = df["Q2"].replace(i, "100")
	# Get top 30 common answers of Q3
	Q3AnswerCount = df["Q3"].value_counts().index
	Q3Top30Answers = Q3AnswerCount[0:31]
	for i in df["Q3"]:
		if i not in Q3Top30Answers and i != "100":
			df["Q3"] = df["Q3"].replace(i, "100")
	# Get top 30 common answers for Q4
	Q4AnswerCount = df["Q4"].value_counts().index
	Q4Top30Answers = Q4AnswerCount[0:30]
	for i in df["Q4"]:
		if i not in Q4Top30Answers and i != "100":
			df["Q4"] = df["Q4"].replace(i, "100")
	# Divide answers of Q4
	Q4MainAnswers = [1, 3, 4]
	for i in df["Q4"]:
		if i not in Q4MainAnswers and i != -99 and i != -98 and i != 97:
			df["Q4"] = df["Q4"].replace(i, "101")
		
	

	# Count agree, disagree, strongly agree, strongly disagree in [1, 5, 6, 7, 10, 14, 16, 17, 18, 19]
	df["Vdisagree_count"] = df[["Q1", "Q5a", "Q5b", "Q5c", "Q5d", "Q5e", "Q5f", "Q6a", "Q6b", "Q6c", "Q6d", "Q7a", "Q7b", "Q7c", "Q7d", "Q7e", "Q10a", "Q10b", "Q10c", "Q10d", "Q14", "Q16", "Q17a", "Q17b", "Q17c", "Q17d", "Q17e", "Q17f", "Q18", "Q19a", "Q19b", "Q19c", "Q19d", "Q19e", "Q19f", "Q19g"]].apply(lambda x: pollFormat1(x["Q1"], "vd") + pollFormat2(x["Q5a"], "vd") + pollFormat2(x["Q5b"], "vd") + pollFormat2(x["Q5c"], "vd") + pollFormat2(x["Q5d"], "vd") + pollFormat2(x["Q5e"], "vd") + pollFormat2(x["Q5f"], "vd") + pollFormat3(x["Q6a"], "vd") + pollFormat3(x["Q6b"], "vd") + pollFormat3(x["Q6c"], "vd") + pollFormat3(x["Q6d"], "vd") + pollFormat2(x["Q7a"], "vd") + pollFormat2(x["Q7b"], "vd") + pollFormat2(x["Q7c"], "vd") + pollFormat2(x["Q7d"], "vd") + pollFormat2(x["Q7e"], "vd") + pollFormat1(x["Q10a"], "vd") + pollFormat1(x["Q10b"], "vd") + pollFormat1(x["Q10c"], "vd") + pollFormat1(x["Q10d"], "vd") + pollFormat4(x["Q14"], "vd") + pollFormat5(x["Q16"], "vd") + pollFormat5(x["Q17a"], "vd") + pollFormat5(x["Q17b"], "vd") + pollFormat5(x["Q17c"], "vd") + pollFormat5(x["Q17d"], "vd") + pollFormat5(x["Q17e"], "vd") + pollFormat5(x["Q17f"], "vd") + pollFormat5(x["Q18"], "vd") + pollFormat5(x["Q19a"], "vd") + pollFormat5(x["Q19b"], "vd") + pollFormat5(x["Q19c"], "vd") + pollFormat5(x["Q19d"], "vd") + pollFormat5(x["Q19e"], "vd") + pollFormat5(x["Q19f"], "vd") + pollFormat5(x["Q19g"], "vd"), axis=1)
	df["disagree_count"] = df[["Q1", "Q5a", "Q5b", "Q5c", "Q5d", "Q5e", "Q5f", "Q6a", "Q6b", "Q6c", "Q6d", "Q7a", "Q7b", "Q7c", "Q7d", "Q7e", "Q10a", "Q10b", "Q10c", "Q10d", "Q14", "Q16", "Q17a", "Q17b", "Q17c", "Q17d", "Q17e", "Q17f", "Q18", "Q19a", "Q19b", "Q19c", "Q19d", "Q19e", "Q19f", "Q19g"]].apply(lambda x: pollFormat1(x["Q1"], "d") + pollFormat2(x["Q5a"], "d") + pollFormat2(x["Q5b"], "d") + pollFormat2(x["Q5c"], "d") + pollFormat2(x["Q5d"], "d") + pollFormat2(x["Q5e"], "d") + pollFormat2(x["Q5f"], "d") + pollFormat3(x["Q6a"], "d") + pollFormat3(x["Q6b"], "d") + pollFormat3(x["Q6c"], "d") + pollFormat3(x["Q6d"], "d") + pollFormat2(x["Q7a"], "d") + pollFormat2(x["Q7b"], "d") + pollFormat2(x["Q7c"], "d") + pollFormat2(x["Q7d"], "d") + pollFormat2(x["Q7e"], "d") + pollFormat1(x["Q10a"], "d") + pollFormat1(x["Q10b"], "d") + pollFormat1(x["Q10c"], "d") + pollFormat1(x["Q10d"], "d") + pollFormat4(x["Q14"], "d") + pollFormat5(x["Q16"], "d") + pollFormat5(x["Q17a"], "d") + pollFormat5(x["Q17b"], "d") + pollFormat5(x["Q17c"], "d") + pollFormat5(x["Q17d"], "d") + pollFormat5(x["Q17e"], "d") + pollFormat5(x["Q17f"], "d") + pollFormat5(x["Q18"], "d") + pollFormat5(x["Q19a"], "d") + pollFormat5(x["Q19b"], "d") + pollFormat5(x["Q19c"], "d") + pollFormat5(x["Q19d"], "d") + pollFormat5(x["Q19e"], "d") + pollFormat5(x["Q19f"], "d") + pollFormat5(x["Q19g"], "d"), axis=1)
	df["neutral_count"] = df[["Q1", "Q5a", "Q5b", "Q5c", "Q5d", "Q5e", "Q5f", "Q6a", "Q6b", "Q6c", "Q6d", "Q7a", "Q7b", "Q7c", "Q7d", "Q7e", "Q10a", "Q10b", "Q10c", "Q10d", "Q14", "Q16", "Q17a", "Q17b", "Q17c", "Q17d", "Q17e", "Q17f", "Q18", "Q19a", "Q19b", "Q19c", "Q19d", "Q19e", "Q19f", "Q19g"]].apply(lambda x: pollFormat1(x["Q1"], "n") + pollFormat2(x["Q5a"], "n") + pollFormat2(x["Q5b"], "n") + pollFormat2(x["Q5c"], "n") + pollFormat2(x["Q5d"], "n") + pollFormat2(x["Q5e"], "n") + pollFormat2(x["Q5f"], "n") + pollFormat3(x["Q6a"], "n") + pollFormat3(x["Q6b"], "n") + pollFormat3(x["Q6c"], "n") + pollFormat3(x["Q6d"], "n") + pollFormat2(x["Q7a"], "n") + pollFormat2(x["Q7b"], "n") + pollFormat2(x["Q7c"], "n") + pollFormat2(x["Q7d"], "n") + pollFormat2(x["Q7e"], "n") + pollFormat1(x["Q10a"], "n") + pollFormat1(x["Q10b"], "n") + pollFormat1(x["Q10c"], "n") + pollFormat1(x["Q10d"], "n") + pollFormat4(x["Q14"], "n") + pollFormat5(x["Q16"], "n") + pollFormat5(x["Q17a"], "n") + pollFormat5(x["Q17b"], "n") + pollFormat5(x["Q17c"], "n") + pollFormat5(x["Q17d"], "n") + pollFormat5(x["Q17e"], "n") + pollFormat5(x["Q17f"], "n") + pollFormat5(x["Q18"], "n") + pollFormat5(x["Q19a"], "n") + pollFormat5(x["Q19b"], "n") + pollFormat5(x["Q19c"], "n") + pollFormat5(x["Q19d"], "n") + pollFormat5(x["Q19e"], "n") + pollFormat5(x["Q19f"], "n") + pollFormat5(x["Q19g"], "n"), axis=1)
	df["agree_count"] = df[["Q1", "Q5a", "Q5b", "Q5c", "Q5d", "Q5e", "Q5f", "Q6a", "Q6b", "Q6c", "Q6d", "Q7a", "Q7b", "Q7c", "Q7d", "Q7e", "Q10a", "Q10b", "Q10c", "Q10d", "Q14", "Q16", "Q17a", "Q17b", "Q17c", "Q17d", "Q17e", "Q17f", "Q18", "Q19a", "Q19b", "Q19c", "Q19d", "Q19e", "Q19f", "Q19g"]].apply(lambda x: pollFormat1(x["Q1"], "a") + pollFormat2(x["Q5a"], "a") + pollFormat2(x["Q5b"], "a") + pollFormat2(x["Q5c"], "a") + pollFormat2(x["Q5d"], "a") + pollFormat2(x["Q5e"], "a") + pollFormat2(x["Q5f"], "a") + pollFormat3(x["Q6a"], "a") + pollFormat3(x["Q6b"], "a") + pollFormat3(x["Q6c"], "a") + pollFormat3(x["Q6d"], "a") + pollFormat2(x["Q7a"], "a") + pollFormat2(x["Q7b"], "a") + pollFormat2(x["Q7c"], "a") + pollFormat2(x["Q7d"], "a") + pollFormat2(x["Q7e"], "a") + pollFormat1(x["Q10a"], "a") + pollFormat1(x["Q10b"], "a") + pollFormat1(x["Q10c"], "a") + pollFormat1(x["Q10d"], "a") + pollFormat4(x["Q14"], "a") + pollFormat5(x["Q16"], "a") + pollFormat5(x["Q17a"], "a") + pollFormat5(x["Q17b"], "a") + pollFormat5(x["Q17c"], "a") + pollFormat5(x["Q17d"], "a") + pollFormat5(x["Q17e"], "a") + pollFormat5(x["Q17f"], "a") + pollFormat5(x["Q18"], "a") + pollFormat5(x["Q19a"], "a") + pollFormat5(x["Q19b"], "a") + pollFormat5(x["Q19c"], "a") + pollFormat5(x["Q19d"], "a") + pollFormat5(x["Q19e"], "a") + pollFormat5(x["Q19f"], "a") + pollFormat5(x["Q19g"], "a"), axis=1)
	df["Vagree_count"] = df[["Q1", "Q5a", "Q5b", "Q5c", "Q5d", "Q5e", "Q5f", "Q6a", "Q6b", "Q6c", "Q6d", "Q7a", "Q7b", "Q7c", "Q7d", "Q7e", "Q10a", "Q10b", "Q10c", "Q10d", "Q14", "Q16", "Q17a", "Q17b", "Q17c", "Q17d", "Q17e", "Q17f", "Q18", "Q19a", "Q19b", "Q19c", "Q19d", "Q19e", "Q19f", "Q19g"]].apply(lambda x: pollFormat1(x["Q1"], "va") + pollFormat2(x["Q5a"], "va") + pollFormat2(x["Q5b"], "va") + pollFormat2(x["Q5c"], "va") + pollFormat2(x["Q5d"], "va") + pollFormat2(x["Q5e"], "va") + pollFormat2(x["Q5f"], "va") + pollFormat3(x["Q6a"], "va") + pollFormat3(x["Q6b"], "va") + pollFormat3(x["Q6c"], "va") + pollFormat3(x["Q6d"], "va") + pollFormat2(x["Q7a"], "va") + pollFormat2(x["Q7b"], "va") + pollFormat2(x["Q7c"], "va") + pollFormat2(x["Q7d"], "va") + pollFormat2(x["Q7e"], "va") + pollFormat1(x["Q10a"], "va") + pollFormat1(x["Q10b"], "va") + pollFormat1(x["Q10c"], "va") + pollFormat1(x["Q10d"], "va") + pollFormat4(x["Q14"], "va") + pollFormat5(x["Q16"], "va") + pollFormat5(x["Q17a"], "va") + pollFormat5(x["Q17b"], "va") + pollFormat5(x["Q17c"], "va") + pollFormat5(x["Q17d"], "va") + pollFormat5(x["Q17e"], "va") + pollFormat5(x["Q17f"], "va") + pollFormat5(x["Q18"], "va") + pollFormat5(x["Q19a"], "va") + pollFormat5(x["Q19b"], "va") + pollFormat5(x["Q19c"], "va") + pollFormat5(x["Q19d"], "va") + pollFormat5(x["Q19e"], "va") + pollFormat5(x["Q19f"], "va") + pollFormat5(x["Q19g"], "va"), axis=1)

	# Set opinionated, F -> False, T -> True
	df["opinionated"] = df[["Vdisagree_count", "disagree_count", "agree_count", "Vagree_count"]].apply(lambda x: "F" if (x["Vdisagree_count"]+x["Vagree_count"]) < (x["disagree_count"]+x["agree_count"]) else "T", axis=1)

	return df

# Poll functions to deal with different poll questions
def pollFormat1(q1, mode):
	if (mode == "vd" and q1 == 5) or (mode == "d" and q1 == 4) or (mode == "n" and q1 == 3) or (mode == "a" and q1 == 2) or (mode == "va" and q1 == 1):
		return 1
	return 0

def pollFormat2(q5, mode):
	if (mode == "vd" and q5 == 4) or (mode == "d" and q5 == 3) or (mode == "n" and q5 == -98) or (mode == "a" and q5 == 2) or (mode == "va" and q5 == 1):
		return 1
	return 0

def pollFormat3(q6, mode):
	if (mode == "vd" and q6 == 1) or (mode == "d" and q6 == 2) or (mode == "n" and q6 == 3) or (mode == "a" and q6 == 4) or (mode == "va" and q6 == 5):
		return 1
	return 0

def pollFormat4(q14, mode):
	if (mode == "va" and q14 == 1) or (mode == "a" and q14 == 2) or (mode == "n" and q14 == 3) or (mode == "d" and q14 == 4) or (mode == "vd" and q14 == 4):
		return 1
	return 0

def pollFormat5(q16, mode):
	if (mode == "vd" and q16 == 1) or (mode == "d" and q16 == 2) or (mode == "n" and q16 == -98) or (mode == "a" and q16 == 3) or (mode == "va" and q16 == 4):
		return 1
	return 0


def writeFile(filePath, targetPath):
	df = preprocessing(filePath)
	df.to_csv(targetPath)



writeFile("ANUPoll2018Data_CSV_01428.csv", "preprocessing_data.csv")
