from sqlalchemy import MetaData, create_engine, or_
from sqlalchemy.orm import sessionmaker
from SQLModels import metadata, Subject, Fatigue, Medical, Handwriting
import pandas as pd
import os
import re
import math


def value_or_null(df, row, col):
    value = df.iat[row, col]
    if isinstance(value, float) and math.isnan(value):
        return None
    else:
        return value


# import fatigue data
def extract_fatigue(session, filename='data/Questionnaires_IMF.xlsx'):
    xl = pd.ExcelFile(filename)

    for sheet in xl.sheet_names:
        subject_id = sheet
        df = xl.parse(sheet, header=None)
        subject = Subject(id=subject_id)
        fatigue = Fatigue(
            subject_id=subject_id,
            date=value_or_null(df, 0, 3),
            injury=value_or_null(df, 2, 3),
            pain=value_or_null(df, 3, 3),
            gen=value_or_null(df, 26, 2),
            phys=value_or_null(df, 26, 3),
            men=value_or_null(df, 26, 4),
            act=value_or_null(df, 26, 5),
            mot=value_or_null(df, 26, 6),
        )
        session.merge(subject)  # no error if already exists
        session.add(fatigue)

    session.commit()


# import medical data
def extract_medical(session, filename='data/#_test médicaux carabins fév 2019.xlsx'):
    xl = pd.ExcelFile(filename)
    df = xl.parse('Feuil1', header=3)
    for index, row in df.iterrows():
        subject_id = row['#']
        if not isinstance(subject_id, int):
            break
        subject = Subject(id=subject_id)
        medical = Medical(
            subject_id=row['#'],
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


def extract_handwriting(session, path='data/Delta_carabins'):
    delta_dir = os.path.join(path, 'Delta')
    filenames = os.listdir(delta_dir)  # get all files' and folders' names
    for name in filenames:  # loop through all the files and folders
        m = re.search('(?<=_)(\d+)(?=_)', name)
        subject_id = int(m.group(0))
        subject_dir = os.path.join(delta_dir, name)
        if not os.path.isdir(subject_dir):
            break
        excel_path = os.path.join(subject_dir, "xlsx/Traits_rapides_reaction_visuelle_simple.xlsx")
        subject = Subject(id=subject_id)
        session.merge(subject)  # no error if already exists
        xl = pd.ExcelFile(excel_path)
        df = xl.parse()
        for index, row in df.iterrows():
            handwriting = Handwriting(
                subject_id=subject_id,
                test_id=index,
                t0=row['t0'],
                D1=row['D1'],
                mu1=row['mu1'],
                ss1=row['ss1'],
                D2=row['D2'],
                mu2=row['mu2'],
                ss2=row['ss2'],
                SNR=row['SNR'],
            )
            session.add(handwriting)
    session.commit()


def apply_filters(session, null_fatigue=True, null_medical=True, null_handwriting=True, low_snr=True, bad_distance=True,
                  d1_d2=True, min_num_tests=True):
    fatigue_table = metadata.tables['fatigue']

    if null_fatigue:
        d = fatigue_table.delete().where(or_(
            fatigue_table.c.gen == None,
            fatigue_table.c.phys == None,
            fatigue_table.c.men == None,
            fatigue_table.c.act == None,
            fatigue_table.c.mot == None,
        ))  # is None won't work
        session.execute(d)

    if null_medical:
        session.execute("DELETE FROM medical WHERE height is NULL")

    if null_handwriting:
        session.execute("DELETE FROM handwriting WHERE t0 is NULL")

    if low_snr:
        session.execute("DELETE FROM handwriting WHERE SNR < 15")

    if bad_distance:
        session.execute("DELETE FROM handwriting WHERE (D1-D2) < 125 OR (D1-D2) > 250")

    if d1_d2:
        session.execute("DELETE FROM handwriting WHERE D1>500 OR D2>500")

    if min_num_tests:
        session.execute("""DELETE FROM handwriting WHERE subject_id IN 
                        (SELECT subject_id FROM handwriting GROUP BY subject_id HAVING count(*)<15)""")

    session.commit()


def create_db(db_name='data/data.db'):
    # delete existing db
    if os.path.exists(db_name):
        os.remove(db_name)

    engine = create_engine('sqlite:///' + db_name)
    metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    extract_fatigue(session)
    extract_medical(session)
    extract_handwriting(session)
    apply_filters(session)

    handwriting_rows = session.query(Handwriting).group_by('subject_id').count()
    print("Extracted " + str(handwriting_rows) + " valid handwriting tests")
    medical_rows = session.query(Medical).count()
    print("Extracted " + str(medical_rows) + " valid medical tests")
    fatigue_rows = session.query(Fatigue).count()
    print("Extracted " + str(fatigue_rows) + " valid fatigue tests")

    session.close()

    return engine


if __name__ == '__main__':
    create_db()
