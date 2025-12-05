import numpy as np
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination


heartDisease = pd.read_csv("heart.csv")
heartDisease = heartDisease.replace('?', np.nan)


print('Sample instances from the dataset are:')
print(heartDisease.head())


model = DiscreteBayesianNetwork([
    ('age', 'heartdisease'),
    ('sex', 'heartdisease'),
    ('exang', 'heartdisease'),
    ('cp', 'heartdisease'),
    ('heartdisease', 'restecg'),
    ('heartdisease', 'chol')
])


model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)


print('Inferencing with Bayesian Network')
infer = VariableElimination(model)


print('1. Probability of HeartDisease given evidence = restecg')
q1 = infer.query(variables=['heartdisease'], evidence={'restecg': 1})
print(q1)


print('2. Probability of HeartDisease given evidence = cp')
q2 = infer.query(variables=['heartdisease'], evidence={'cp': 2})
print(q2)
