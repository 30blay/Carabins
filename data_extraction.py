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
        hop_g2=row['HOP G - 2'],
        hop_d1=row['HOP D - 1'],
        hop_d2=row['HOP D - 2'],
        hop3_g1=row['3 HOP G - 1'],
        hop3_g2=row['3 HOP G - 2'],
        hop3_d1=row['3 HOP D - 1'],
        hop3_d2=row['3 HOP D - 2'],
        hop3_cr_g1=row['3 HOP Cr G - 1'],
        hop3_cr_g2=row['3 HOP Cr G - 2'],
        hop3_cr_d1=row['3 HOP Cr D - 1'],
        hop3_cr_d2=row['3 HOP Cr D - 2'],
        re_gh_g1=row['RE GH G - 1'],
        re_gh_g2=row['RE GH G - 2'],
        re_gh_d1=row['RE GH D - 1'],
        re_gh_d2=row['RE GH D - 2'],
        flex_g1=row['FLEX G -1'],
        flex_g2=row['FLEX G - 2'],
        flex_d1=row['FLEX D - 1'],
        flex_d2=row['FLEX D - 2'],
        scap_g1=row['SCAP G - 1'],
        scap_g2=row['SCAP G - 2'],
        scap_d1=row['SCAP D - 1'],
        scap_d2=row['SCAP D - 2'],
    )
    session.merge(subject)  # no error if already exists
    session.add(medical)

session.commit()

# validate and filter
