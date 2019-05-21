import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from sqlalchemy import create_engine
import numpy as np

db_name = 'data/data.db'


def get_subject_metrics():
    engine = create_engine('sqlite:///' + db_name)

    df = pd.read_sql_query("""SELECT
        id,
        medical.*,
        fatigue.*,
        AVG(handwriting.t0) as t0,
        AVG(handwriting.D1) as D1,
        AVG(handwriting.mu1) as mu1,
        AVG(handwriting.ss1) as ss1,
        AVG(handwriting.D2) as D2,
        AVG(handwriting.mu2) as mu2,
        AVG(handwriting.ss2) as ss2,
        AVG(handwriting.SNR) as SNR

        FROM subject 
        LEFT JOIN medical on id=medical.subject_id 
        LEFT JOIN handwriting on id=handwriting.subject_id 
        LEFT JOIN fatigue on id=fatigue.subject_id
        GROUP BY handwriting.subject_id
        """, con=engine.connect())
    df['avg_fatigue'] = df[['gen', 'phys', 'men', 'act', 'mot']].mean(axis=1)

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
    ax1 = sns.heatmap(corr, annot=True, cbar=False, square=True)
    ax1.set_title("R correlation coefficient on subject averages")
    plt.show()

    engine = create_engine('sqlite:///' + db_name)
    df = pd.read_sql_query("SELECT t0, D1, mu1, ss1, D2, mu2, ss2, SNR FROM handwriting", con=engine.connect())
    ax2 = sns.heatmap(df.corr(), annot=True, cbar=False, square=True)
    ax2.set_title("R correlation coefficient on individual tries")
    plt.show()


if __name__ == '__main__':
    # height_weight()
    # fatigue_handwriting_relationship()
    # delta_log_params_relationship()
    # delta_log_params_relationship_all_tries()
    # handwriting_test_count_dist()
    # injury_analysis()
    # movement_amplitude()
    delta_log_linear_regressions()
