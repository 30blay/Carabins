from auto_ml import Predictor
from data_extraction.analysis import get_subject_metrics
from sklearn.model_selection import train_test_split

df = get_subject_metrics()[[
    't0',
    'D1',
    'mu1',
    'ss1',
    'D2',
    'mu2',
    'ss2',
    'SNR',
    'avg_fatigue',
]]
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

column_descriptions = {
    'avg_fatigue': 'output',
}

ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

ml_predictor.train(train_set)

ml_predictor.score(test_set, test_set['avg_fatigue'])
