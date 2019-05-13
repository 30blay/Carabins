from sqlalchemy import MetaData, create_engine
from sqlalchemy.orm import sessionmaker
from SQLModels import metadata, Subject
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
    df = xl.parse(sheet)
    newSubject = Subject(
        id=int(sheet),
        date=value_or_null(df, 0, 3),
        injury=value_or_null(df, 2, 3),
        pain=value_or_null(df, 3, 3),
        gen=value_or_null(df, 26, 2),
        phys=value_or_null(df, 26, 3),
        men=value_or_null(df, 26, 4),
        act=value_or_null(df, 26, 5),
        mot=value_or_null(df, 26, 6),
    )
    session.add(newSubject)

session.commit()


# import medical data

