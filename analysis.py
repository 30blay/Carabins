import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from sqlalchemy import create_engine
import numpy as np
from scipy import stats
from scipy.stats import shapiro, normaltest, norm, lognorm, kstest


def get_subject_metrics(db_name='data/data_carabins.db'):
    engine = create_engine('sqlite:///' + db_name)

    df = pd.read_sql_query("""SELECT
        subject_id as id,
        medical.*,
        fatigue.*,
        AVG(deltalog.t0) as t0,
        AVG(deltalog.D1) as D1,
        AVG(deltalog.mu1) as mu1,
        AVG(deltalog.ss1) as ss1,
        AVG(deltalog.D2) as D2,
        AVG(deltalog.mu2) as mu2,
        AVG(deltalog.ss2) as ss2,
        AVG(deltalog.SNR) as SNR

        FROM subject 
        LEFT JOIN medical USING (subject_id)
        LEFT JOIN deltalog USING (subject_id)
        LEFT JOIN fatigue USING (subject_id)
        GROUP BY subject_id
        """, con=engine.connect())
    df['avg_fatigue'] = df[['gen', 'phys', 'men', 'act', 'mot']].mean(axis=1)
    df.drop(columns="subject_id", inplace=True)

    return df


def get_handwriting_stddev():
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
        FROM handwriting
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
    df = get_subject_metrics()
    sns.pairplot(df[[
        'gen',
        'phys',
        'mot',
        't0',
        'SNR',
    ]].dropna())
    plt.show()


def delta_log_params_relationship():
    df = get_subject_metrics()
    sns.pairplot(df[['t0', 'D1', 'mu1', 'ss1', 'D2', 'mu2', 'ss2', 'SNR']], dropna=True)
    plt.show()


def delta_log_params_relationship_all_tries():
    engine = create_engine('sqlite:///' + db_name)

    df = pd.read_sql_query("""SELECT
        t0, D1, mu1, ss1, D2, mu2, ss2, SNR
        FROM handwriting
        """, con=engine.connect())
    sns.pairplot(df[['t0', 'D1', 'mu1', 'ss1', 'D2', 'mu2', 'ss2', 'SNR']], dropna=True)
    plt.show()


def delta_log_params_distribution_all_tries():
    engine = create_engine('sqlite:///' + db_name)

    df = pd.read_sql_query("""SELECT
        subject_id, t0, D1, mu1, ss1, D2, mu2, ss2, SNR
        FROM handwriting
        """, con=engine.connect())
    sns.violinplot(y=df['t0'], x=df['subject_id'])
    plt.show()


def handwriting_test_count_dist():
    engine = create_engine('sqlite:///' + db_name)
    df = pd.read_sql_query("""
        SELECT subject_id, count(test_id) FROM handwriting GROUP BY subject_id
            """, con=engine.connect())
    sns.distplot(df['count(test_id)']).set_title('Distribution of the number of handwriting tests')
    plt.show()


def injury_analysis():
    df = get_subject_metrics()
    df['injured'] = np.where(np.equal(df['injury'], None), 'sain', 'blessé')
    for i, variable in enumerate(['t0', 'SNR', 'avg_fatigue', 'D1']):
        plt.subplot(2, 2, i+1)
        ax = sns.violinplot(x='injured', y=variable, data=df, inner='point', cut=0, bw='silverman')
        ax.set_xlabel("")
    plt.show()


def movement_amplitude():
    df = get_subject_metrics()
    df['amplitude'] = df['D1'] - df['D2']
    sns.distplot(df['amplitude'].dropna()).set_title('Distribution of the movement amplitude')
    plt.show()


def delta_log_linear_regressions():
    df = get_subject_metrics()
    corr = df[['t0', 'D1', 'mu1', 'ss1', 'D2', 'mu2', 'ss2', 'SNR']].corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    ax1 = sns.heatmap(corr, annot=True, cbar=False, square=True, mask=mask)
    ax1.set_title("R correlation coefficient on subject averages")
    plt.show()

    engine = create_engine('sqlite:///' + db_name)
    df = pd.read_sql_query("SELECT t0, D1, mu1, ss1, D2, mu2, ss2, SNR FROM handwriting", con=engine.connect())
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    ax2 = sns.heatmap(corr, annot=True, cbar=False, square=True, mask=mask)
    ax2.set_title("R correlation coefficient on individual tries")
    plt.show()


def handwriting_stddev_analysis():
    metrics = get_subject_metrics()
    hw_std = get_handwriting_stddev()
    df = pd.merge(hw_std, metrics, on='id')
    corr = df[['t0_std', 'D1_std', 'mu1_std', 'ss1_std', 'D2_std', 'mu2_std', 'ss2_std', 'SNR_std']].corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, annot=True, cbar=False, square=True, mask=mask)
    plt.show()
    sns.pairplot(df[['t0_std', 'D1_std', 'mu1_std', 'ss1_std', 'D2_std', 'mu2_std', 'ss2_std', 'SNR_std']])
    plt.show()


def normality_test():
    alpha = 0.1
    df = get_subject_metrics()
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


def medical_outliers(threshold=3):
    from data_extraction import medical_traits, delta_log_params
    result = pd.DataFrame(columns=['subject_id', 'variable', 'z-score'])
    for var in medical_traits + delta_log_params:
        df = get_subject_metrics()[['id'] + [var]].dropna()
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


def test_id_corr():
    engine = create_engine('sqlite:///' + db_name)
    df = pd.read_sql_query("SELECT * FROM handwriting", con=engine.connect())
    corr = df[['t0', 'D1', 'mu1', 'ss1', 'D2', 'mu2', 'ss2', 'SNR']].corrwith(df['test_id'])
    ax = corr.plot(kind='bar')
    ax.set_title('Correlation to test_id')
    ax.set_ylabel('R')
    ax.axhline(color='black', linewidth=0.5)
    plt.ylim(top=1, bottom=-1)
    plt.show()


if __name__ == '__main__':
    # height_weight()
    # fatigue_handwriting_relationship()
    # delta_log_params_relationship()
    # delta_log_params_relationship_all_tries()
    # handwriting_test_count_dist()
    # injury_analysis()
    # movement_amplitude()
    # delta_log_linear_regressions()
    # delta_log_params_distribution_all_tries()
    handwriting_stddev_analysis()
    # normality_test()
    # medical_outliers()
    # test_id_corr()
