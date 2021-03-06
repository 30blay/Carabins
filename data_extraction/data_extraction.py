from sqlalchemy import create_engine, or_
from sqlalchemy.orm import sessionmaker
from data_extraction.SQLModels import metadata, Subject, Fatigue, Medical, DeltaLog, SigmaLog, Lognormal
import pandas as pd
import os
import re
import math


medical_traits = ['arm_length', 'asym_drop_box', 'asym_tuck_jump', 'hop_g1', 'hop_g2', 'hop_d1', 'hop_d2', 'hop3_g1', 'hop3_g2', 'hop3_d1', 'hop3_d2', 'hop3_cr_g1', 'hop3_cr_g2', 'hop3_cr_d1', 'hop3_cr_d2', 're_gh_g1', 're_gh_g2', 're_gh_d1', 're_gh_d2', 'flex_g1', 'flex_g2', 'flex_d1', 'flex_d2', 'scap_g1', 'scap_g2', 'scap_d1', 'scap_d2']
delta_log_params = ['t0', 'D1', 'mu1', 'ss1', 'D2', 'mu2', 'ss2', 'SNR']


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
        subject = Subject(subject_id=subject_id)
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
        subject = Subject(subject_id=subject_id)
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


def subject_dir_gen(path):
    filenames = os.listdir(path)  # get all files' and folders' names
    for name in filenames:  # loop through all the files and folders
        m = re.search('(?<=_)(\d+)(?=_)', name)
        subject_id = int(m.group(0))
        subject_dir = os.path.join(path, name)
        yield(subject_id, subject_dir)


def extract_deltalog(session, path='data/Baseline', post_exercise=False):
    for (subject_id, subject_dir) in subject_dir_gen(path=path + '/Delta'):
        subject = Subject(subject_id=subject_id)
        session.merge(subject)  # no error if already exists
        for test_name in [
            'Traits_rapides_reaction_visuelle_simple',
            'Compromis_vitesse_precision_A',
            'Compromis_vitesse_precision_B',
            'Compromis_vitesse_precision_C',
            'Compromis_vitesse_precision_D',
        ]:
            excel_path = os.path.join(subject_dir, "xlsx/" + test_name + ".xlsx")
            if not os.path.exists(excel_path):
                continue

            xl = pd.ExcelFile(excel_path)
            df = xl.parse()
            for stroke_id, row in df.iterrows():

                # Because certain files have min/max rows
                if row[0] != 'C':
                    break

                handwriting = DeltaLog(
                    subject_id=subject_id,
                    test_name=test_name,
                    stroke_id=stroke_id,
                    post_exercise=post_exercise,
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


def extract_sigmalog(session, path='data/Baseline', post_exercise=False):
    for (subject_id, subject_dir) in subject_dir_gen(path=path + '/Sigma'):
        subject = Subject(subject_id=subject_id)
        session.merge(subject)  # no error if already exists
        for test_name in [
            'Traits_rapides_reaction_visuelle_simple',
            'Compromis_vitesse_precision_A',
            'Compromis_vitesse_precision_B',
            'Compromis_vitesse_precision_C',
            'Compromis_vitesse_precision_D',
        ]:
            test_path = os.path.join(subject_dir, test_name + '_hws')
            if not os.path.exists(test_path):
                continue

            filenames = os.listdir(test_path)  # get all files names
            for filename in [f for f in filenames if 'ana' in f]:
                m = re.search('(\d+)', filename)
                stroke_id = int(m.group(0))
                ana_path = os.path.join(test_path, filename)
                header = pd.read_csv(ana_path, sep=' ', skipinitialspace=True, nrows=3)
                df = pd.read_csv(ana_path, sep=' ', skipinitialspace=True, header=None, skiprows=4)

                sigmalog = SigmaLog(
                    subject_id=subject_id,
                    test_name=test_name,
                    stroke_id=stroke_id,
                    post_exercise=post_exercise,
                    version=header.iat[0, 1],
                    nb_lognorm=header.iat[1, 1],
                    SNR=header.iat[2, 1],
                )
                for _, row in df.iterrows():
                    lognormal = Lognormal(
                        t0=row.iat[0],
                        D=row.iat[1],
                        mu=row.iat[2],
                        ss=row.iat[3],
                        theta_start=row.iat[4],
                        theta_end=row.iat[5],
                    )
                    sigmalog.lognormals.append(lognormal)

                session.add(sigmalog)
                session.bulk_save_objects(sigmalog.lognormals)
    session.commit()


def apply_filters(session, null_fatigue=False, null_medical=False, null_handwriting=False, low_snr=False, mov_amplitude=False,
                  d1_d2_max=True, min_num_tests=True):
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
        session.execute("DELETE FROM deltalog WHERE t0 is NULL")

    if low_snr:
        session.execute("DELETE FROM deltalog WHERE SNR < 15")

    if mov_amplitude:
        session.execute("""DELETE FROM deltalog WHERE test_name='Traits_rapides_reaction_visuelle_simple'
         AND ((D1-D2) < 125 OR (D1-D2) > 250)""")

    if d1_d2_max:
        session.execute("DELETE FROM deltalog WHERE D1>500 OR D2>500")

    if min_num_tests:
        session.execute("""DELETE FROM deltalog WHERE subject_id IN 
                        (SELECT subject_id FROM deltalog GROUP BY subject_id HAVING count(*)<15)""")

    session.commit()


def create_db(db_name='data/carabins_data.db'):
    # delete existing db
    if os.path.exists(db_name):
        os.remove(db_name)

    engine = create_engine('sqlite:///' + db_name)
    metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    extract_fatigue(session)
    extract_medical(session)
    extract_deltalog(session, path='data/Baseline', post_exercise=False)
    extract_deltalog(session, path='data/Post_pratique', post_exercise=True)
    extract_sigmalog(session, path='data/Baseline', post_exercise=False)
    extract_sigmalog(session, path='data/Post_pratique', post_exercise=True)
    apply_filters(session,
                  null_fatigue=True,
                  null_handwriting=True,
                  null_medical=True,
                  low_snr=True,
                  d1_d2_max=True,
                  mov_amplitude=True,
                  min_num_tests=True,
                  )

    deltalog_rows = session.query(DeltaLog).group_by('subject_id').count()
    medical_rows = session.query(Medical).count()
    fatigue_rows = session.query(Fatigue).count()
    print("Extracted " + str(deltalog_rows) + " valid handwriting tests")
    print("Extracted " + str(medical_rows) + " valid medical tests")
    print("Extracted " + str(fatigue_rows) + " valid fatigue tests")

    session.close()

    return engine
