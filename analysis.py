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

        FROM fatigue NATURAL JOIN medical NATURAL JOIN handwriting 
        GROUP BY handwriting.subject_id
        """, con=engine.connect())
    df['avg_fatigue'] = df[['t0', 'D1', 'mu1', 'ss1', 'D2', 'mu2', 'ss2', 'SNR']].mean(axis=1)

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
    ]])
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
        ax = sns.violinplot(x='injured', y=variable, data=df)
        ax.set_xlabel("")
    plt.show()


if __name__ == '__main__':
    height_weight()
    fatigue_handwriting_relationship()
    handwriting_test_count_dist()
    injury_analysis()
