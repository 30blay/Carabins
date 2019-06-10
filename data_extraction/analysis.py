import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from sqlalchemy import create_engine
import numpy as np
from scipy import stats
from scipy.stats import norm, lognorm, kstest

default_db_name = 'data/data_carabins.db'


def get_subject_metrics(db_name=default_db_name):
    engine = create_engine('sqlite:///' + db_name)

    df = pd.read_sql_query("""SELECT
        subject_id as id,
        medical.*,
        fatigue.*
        FROM subject 
        LEFT JOIN medical USING (subject_id)
        LEFT JOIN fatigue USING (subject_id)
        """, con=engine.connect())
    df['avg_fatigue'] = df[['gen', 'phys', 'men', 'act', 'mot']].mean(axis=1)
    df.drop(columns="subject_id", inplace=True)

    return df


def get_traits_rapides_params(db_name=default_db_name):
    engine = create_engine('sqlite:///' + db_name)
    df = pd.read_sql_query("""SELECT
            subject_id,
            AVG(t0) as t0,
            AVG(D1) as D1,
            AVG(mu1) as mu1,
            AVG(ss1) as ss1,
            AVG(D2) as D2,
            AVG(mu2) as mu2,
            AVG(ss2) as ss2,
            AVG(SNR) as SNR,
            post_exercise
            FROM deltalog 
            WHERE test_name is 'Traits_rapides_reaction_visuelle_simple' 
            GROUP BY subject_id, post_exercise
            """, con=engine.connect())

    return df


def get_trait_rapide_stddev(db_name=default_db_name):
    engine = create_engine('sqlite:///' + db_name)

    df = pd.read_sql_query("""SELECT
        subject_id AS id,
        t0 ,
        D1 ,
        mu1,
        ss1,
        D2 ,
        mu2,
        ss2,
        SNR
        FROM deltalog
        """, con=engine.connect())

    df = df.groupby('id').agg(
        {
            't0': 'std',
            'D1': 'std',
            'mu1': 'std',
            'ss1': 'std',
            'D2': 'std',
            'mu2': 'std',
            'ss2': 'std',
            'SNR': 'std',
        }
    ).rename(columns={
        't0': 't0_std',
        'D1': 'D1_std',
        'mu1': 'mu1_std',
        'ss1': 'ss1_std',
        'D2': 'D2_std',
        'mu2': 'mu2_std',
        'ss2': 'ss2_std',
        'SNR': 'SNR_std',
    })

    return df


def get_sigmalog_params(db_name=default_db_name):
    engine = create_engine('sqlite:///' + db_name)
    sigmalog = pd.read_sql_query('''SELECT 
        subject_id,
         AVG(nb_lognorm)    as nb_lognorm,
         AVG(SNR)           as snr
        from sigmalog
        where test_name is 'Traits_rapides_reaction_visuelle_simple'
        group by subject_id
        ''', con=engine.connect())
    return sigmalog


def height_weight():
    df = get_subject_metrics()

    uniq = list(set(df['position']))

    # Set the color map to match the number of positions
    cNorm = colors.Normalize(vmin=0, vmax=len(uniq))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='Paired')

    # Plot each position
    for i in range(len(uniq)):
        indx = df['position'] == uniq[i]
        plt.scatter(df['weight'][indx], df['height'][indx], color=scalarMap.to_rgba(i), label=uniq[i])

    plt.xlabel('Weight (kg)')
    plt.ylabel('Height (cm)')
    plt.title('Height-Weight correlation')
    plt.legend(loc='lower right')
    plt.show()


def fatigue_handwriting_relationship():
    df = pd.merge(get_subject_metrics(), get_traits_rapides_params(), left_on='id', right_on='subject_id')
    sns.pairplot(df[[
        'gen',
        'phys',
        'mot',
        't0',
        'SNR',
    ]].dropna())
    plt.show()


def delta_log_params_relationship():
    df = get_traits_rapides_params()
    sns.pairplot(df[['t0', 'D1', 'mu1', 'ss1', 'D2', 'mu2', 'ss2', 'SNR']], dropna=True)
    plt.show()


def delta_log_params_relationship_all_tries(db_name=default_db_name):
    engine = create_engine('sqlite:///' + db_name)

    df = pd.read_sql_query("""SELECT
        t0, D1, mu1, ss1, D2, mu2, ss2, SNR
        FROM deltalog
        """, con=engine.connect())
    sns.pairplot(df[['t0', 'D1', 'mu1', 'ss1', 'D2', 'mu2', 'ss2', 'SNR']], dropna=True)
    plt.show()


def delta_log_params_distribution_all_tries(db_name=default_db_name):
    engine = create_engine('sqlite:///' + db_name)

    df = pd.read_sql_query("""SELECT
        subject_id, t0, D1, mu1, ss1, D2, mu2, ss2, SNR
        FROM deltalog
        """, con=engine.connect())
    sns.violinplot(y=df['t0'], x=df['subject_id'])
    plt.show()


def handwriting_test_count_dist(db_name=default_db_name):
    engine = create_engine('sqlite:///' + db_name)
    df = pd.read_sql_query("""
        SELECT subject_id, count(stroke_id) FROM deltalog GROUP BY subject_id
            """, con=engine.connect())
    sns.distplot(df['count(stroke_id)']).set_title('Distribution of the number of handwriting tests')
    plt.show()


def injury_analysis():
    df = pd.merge(get_subject_metrics(), get_traits_rapides_params(), left_on='id', right_on='subject_id')
    df['injured'] = np.where(np.equal(df['injury'], None), 'sain', 'blessé')
    for i, variable in enumerate(['t0', 'SNR', 'avg_fatigue', 'D1']):
        plt.subplot(2, 2, i+1)
        ax = sns.violinplot(x='injured', y=variable, data=df, inner='point', cut=0, bw='silverman')
        ax.set_xlabel("")
    plt.show()


def movement_amplitude():
    df = pd.merge(get_subject_metrics(), get_traits_rapides_params(), left_on='id', right_on='subject_id')
    df['amplitude'] = df['D1'] - df['D2']
    sns.distplot(df['amplitude'].dropna()).set_title('Distribution of the movement amplitude')
    plt.show()


def delta_log_linear_regressions(db_name=default_db_name):
    df = get_traits_rapides_params()
    corr = df[['t0', 'D1', 'mu1', 'ss1', 'D2', 'mu2', 'ss2', 'SNR']].corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    ax1 = sns.heatmap(corr, annot=True, cbar=False, square=True, mask=mask)
    ax1.set_title("R correlation coefficient on subject averages")
    plt.show()

    engine = create_engine('sqlite:///' + db_name)
    df = pd.read_sql_query("SELECT t0, D1, mu1, ss1, D2, mu2, ss2, SNR FROM deltalog", con=engine.connect())
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    ax2 = sns.heatmap(corr, annot=True, cbar=False, square=True, mask=mask)
    ax2.set_title("R correlation coefficient on individual tries")
    plt.show()


def handwriting_stddev_analysis():
    metrics = get_subject_metrics()
    hw_std = get_trait_rapide_stddev()
    df = pd.merge(hw_std, metrics, on='id')
    corr = df[['t0_std', 'D1_std', 'mu1_std', 'ss1_std', 'D2_std', 'mu2_std', 'ss2_std', 'SNR_std']].corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, annot=True, cbar=False, square=True, mask=mask)
    plt.show()
    sns.pairplot(df[['t0_std', 'D1_std', 'mu1_std', 'ss1_std', 'D2_std', 'mu2_std', 'ss2_std', 'SNR_std']])
    plt.show()


def normality_test():
    df = pd.merge(get_subject_metrics(), get_traits_rapides_params(), left_on='id', right_on='subject_id')
    results = pd.DataFrame(columns=['variable', 'model', 'p-value'])

    for tested_var_name in ['t0', 'D1', 'mu1', 'ss1', 'D2', 'mu2', 'ss2', 'SNR']:
        tested_data = df[tested_var_name].dropna()
        for model in [norm, lognorm]:
            statistic, p = kstest(tested_data, model.name, model.fit(tested_data))
            results = results.append({'variable': tested_var_name, 'model': model.name, 'p-value': p}, ignore_index=True)
    results = results.pivot(index='variable', columns='model', values='p-value')
    ax = sns.heatmap(results, annot=True, cbar=False)
    ax.set_title('P-value using Kolmogorov-Smirnov')
    plt.show()


def find_outliers(threshold=3):
    from data_extraction.data_extraction import medical_traits, delta_log_params
    result = pd.DataFrame(columns=['subject_id', 'variable', 'z-score'])
    all_data = pd.merge(get_subject_metrics(), get_traits_rapides_params(), left_on='id', right_on='subject_id')
    for var in medical_traits + delta_log_params:
        df = all_data[['id'] + [var]].dropna()
        search_data = df[var]
        z = np.abs(stats.zscore(search_data))
        indexes = np.where(z > threshold)
        for index in indexes[0]:
            result = result.append({
                'subject_id': df['id'].iloc[index],
                'variable': var,
                'z-score': z[index]
            }, ignore_index=True)
            result = result.sort_values(by=['z-score'], ascending=False)
    print(result.to_string(index=False))


def test_id_corr(db_name=default_db_name):
    engine = create_engine('sqlite:///' + db_name)
    df = pd.read_sql_query("SELECT * FROM deltalog", con=engine.connect())
    corr = df[['t0', 'D1', 'mu1', 'ss1', 'D2', 'mu2', 'ss2', 'SNR']].corrwith(df['stroke_id'])
    ax = corr.plot(kind='bar')
    ax.set_title('Correlation to stroke_id')
    ax.set_ylabel('R')
    ax.axhline(color='black', linewidth=0.5)
    plt.ylim(top=1, bottom=-1)
    plt.show()


def post_exercise_deltalog(db_name=default_db_name):
    engine = create_engine('sqlite:///' + db_name)
    df = pd.read_sql_query('''SELECT 
        subject_id,
        AVG(t0) as t0,
        AVG(D1) as D1,
        AVG(mu1) as mu1,
        AVG(ss1) as ss1,
        AVG(D2) as D2,
        AVG(mu2) as mu2,
        AVG(ss2) as ss2,
        AVG(SNR) as SNR,
        post_exercise
        from deltalog
     where subject_id in (SELECT DISTINCT subject_id from deltalog where post_exercise is 1)
     AND test_name='Traits_rapides_reaction_visuelle_simple'
     GROUP BY subject_id, post_exercise
    ''', con=engine.connect())
    df['post_exercise'].replace(inplace=True, to_replace=0, value='pre')
    df['post_exercise'].replace(inplace=True, to_replace=1, value='post')

    fig = plt.figure()
    fig.suptitle("Effect of exercice on Delta-lognormal parameters")
    for i, variable in enumerate(['t0', 'SNR', 'D1', 'D2', 'mu1', 'mu2', 'ss1', 'ss2']):
        ax = fig.add_subplot(2, 4, i+1)
        sns.violinplot(x='post_exercise', y=variable, data=df, inner='point', cut=0, bw='silverman')
        ax.set_xlabel('')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def sigmalog_dist(db_name=default_db_name):
    engine = create_engine('sqlite:///' + db_name)
    sigmalog = pd.read_sql_query('''SELECT 
        subject_id, AVG(nb_lognorm), AVG(SNR)
        from sigmalog
        where test_name is 'Traits_rapides_reaction_visuelle_simple'
        group by subject_id
        ''', con=engine.connect())
    deltalog = get_traits_rapides_params()
    df = pd.merge(sigmalog, deltalog, on='subject_id')
    df = pd.merge(df, get_subject_metrics(), left_on='subject_id', right_on='id')
    sns.distplot(df['AVG(nb_lognorm)'])
    plt.show()
    sns.scatterplot(data=df, x='AVG(nb_lognorm)', y='avg_fatigue')
    plt.show()
    return df


def medical_asymetry(db_name=default_db_name):
    engine = create_engine('sqlite:///' + db_name)
    df = pd.read_sql_query('''SELECT subject_id,
     ((hop_d1+hop_d2)/2 - (hop_g1+hop_g2)/2) as hop_diff,
     ((hop3_d1+hop3_d2)/2 - (hop3_g1+hop3_g2)/2) as hop3_diff,
     ((hop3_cr_d1+hop3_cr_d2)/2 - (hop3_cr_g1+hop3_cr_g2)/2) as hop3_cr_diff,
     ((re_gh_d1+re_gh_d2)/2 - (re_gh_g1+re_gh_g2)/2) as re_gh_diff,
     ((flex_d1+flex_d2)/2 - (flex_g1+flex_g2)/2) as flex_diff,
     ((scap_d1+scap_d2)/2 - (scap_g1+scap_g2)/2) as scap_diff
     from medical
     ''', con=engine.connect()).dropna()
    variables = ['hop_diff', 'hop3_diff', 'hop3_cr_diff', 're_gh_diff', 'flex_diff', 'scap_diff']

    z_scores = stats.zscore(df[['re_gh_diff', 'flex_diff', 'scap_diff']])
    df['upper_limb_avg_z_score'] = np.mean(z_scores, axis=1)

    fig = plt.figure()
    fig.suptitle("Asymétrie droite - gauche")
    for i, variable in enumerate(variables):
        plt.subplot(2, 3, i + 1)
        sns.distplot(df[variable])
        plt.axvline(0, 0, 1, color='black', linewidth=0.5)  # vertical line at 0
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    ax = sns.distplot(df['upper_limb_avg_z_score'])
    ax.text(2.5, 0.5, 'Fort droite', fontsize=15, horizontalalignment='center')  # add text
    ax.text(-2.5, 0.5, 'Fort gauche', fontsize=15, horizontalalignment='center')  # add text
    plt.show()

    return df
