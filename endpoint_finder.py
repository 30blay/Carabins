import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os


class Endpoint:
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(self.path, skiprows=7, sep=' ', skipinitialspace=True)
        self.df['x_speed'] = self.df['x_position'].diff()

    def plot(self):
        sns.lineplot(x='serial_number', y='x_position', data=self.df.where(self.df['pressure'] > 0))
        plt.show()

    def endpoint(self):
        return self.df['x_position'].max()

    def x_disp(self):
        move = self.df['x_position'].where(self.df['pressure'] > 0)
        disp = move.max() - move.min()
        return disp


def get_delta_x(path):
    df = pd.DataFrame(columns=['subject_id', 'test_name', 'stroke_id', 'x_disp'])

    delta_dir = os.path.join(path, 'original_data')
    filenames = os.listdir(delta_dir)  # get all files' and folders' names
    for name in filenames:  # loop through all the files and folders
        m = re.search('(?<=_)(\d+)(?=_)', name)
        subject_id = int(m.group(0))
        subject_dir = os.path.join(delta_dir, name)
        for test_name in [
            'Traits_rapides_reaction_visuelle_simple',
            'Compromis_vitesse_precision_A',
            'Compromis_vitesse_precision_B',
            'Compromis_vitesse_precision_C',
            'Compromis_vitesse_precision_D',
        ]:
            test_path = os.path.join(subject_dir, test_name)
            for trial_name in os.listdir(test_path):  # loop through all the files and folders
                # skip files
                if not os.path.isdir(os.path.join(test_path, trial_name)):
                    continue
                m = re.search('(\d+)', name)
                stroke_id = int(m.group(0))
                dat_path = os.path.join(test_path, trial_name, '1.dat')
                x_disp = Endpoint(dat_path).x_disp()
                df = df.append({'subject_id': subject_id,
                                 'test_name': test_name,
                                 'stroke_id': stroke_id,
                                 'x_disp': x_disp,
                                 }, ignore_index=True)
    return df


if __name__ == '__main__':
    invalids = get_delta_x('data/Baseline')
    sns.violinplot(x='test_name', y='x_disp', data=invalids)
    plt.show()
