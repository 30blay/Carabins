from sqlalchemy import MetaData, create_engine
from sqlalchemy.orm import sessionmaker
from SQLModels import metadata, Subject, Fatigue, Medical
import pandas as pd
import os
import math


def value_or_null(df, row, col):
    value = df.iat[row, col]
    if isinstance(value, float) and math.isnan(value):
        return None
    else:
        return value


dbName = 'data.db'

os.remove(dbName)
meta = MetaData()
engine = create_engine('sqlite:///' + dbName)
metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# import fatigue data
fatigueFile = 'data/Questionnaires_IMF.xlsx'
xl = pd.ExcelFile(fatigueFile)

for sheet in xl.sheet_names:
    id = sheet
    df = xl.parse(sheet)
    subject = Subject(id=id)
    fatigue = Fatigue(
        id=id,
        date=value_or_null(df, 0, 3),
        injury=value_or_null(df, 2, 3),
        pain=value_or_null(df, 3, 3),
        gen=value_or_null(df, 26, 2),
        phys=value_or_null(df, 26, 3),
        men=value_or_null(df, 26, 4),
        act=value_or_null(df, 26, 5),
        mot=value_or_null(df, 26, 6),
    )
    session.add(subject)
    session.add(fatigue)

session.commit()

# import medical data
medicalFile = 'data/#_test médicaux carabins fév 2019.xlsx'
xl = pd.ExcelFile(medicalFile)
df = xl.parse('Feuil1', header=3)
for index, row in df.iterrows():
    id = row['#']
    subject = Subject(id=id)
    medical = Medical(
        id=row['#'],
        position=row['POSITION'],
        status=row['STATU'],
        height=row['Taille (cm)'],
        weight=row['Poids (kg)'],
        fat=row['% gras'],
        arm_length=row['bras (cm)'],
        asym_drop_box=(row['Asym drop box (X)'] == 'x'),
        asym_tuck_jump=(row['Asym tuck jump (X)'] == 'x'),
        hop_g1=row['HOP G - 1'],
    )
    session.merge(subject)  # no error if already exists
    session.add(medical)

session.commit()
