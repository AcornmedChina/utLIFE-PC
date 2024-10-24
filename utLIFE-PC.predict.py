import numpy as np
import pandas as pd
import joblib,os
import click


class Predict_samples:
    def __init__(self, input_file, model_file, outpath, outid):
        self.input_file = input_file
        self.model_file = model_file
        self.outpath = outpath
        self.outid = outid
        self.outpath = f'{outpath}/{self.outid}'
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)

    def predict_prob(self, indata):
        trained_model = joblib.load(self.model_file)
        validation_df = pd.read_csv(indata, index_col=0, sep='\t')
        y_validation_predict = trained_model.predict_proba(validation_df)[:, 1]
        validation_df['score_predict'] = y_validation_predict
        return validation_df.loc[:, ['score_predict']]

    def run(self):
        input_file_df_label = self.predict_prob(self.input_file)
        input_file_df_label.to_csv('{}/{}_score.xls'.format(self.outpath, self.outid), sep='\t')


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--input_file', metavar='FILE', help='File that need to be predicted', required=True)
@click.option('--model_file', metavar='FILE', help='your well-trained model file', required=True)
@click.option('--outid', metavar='FILE', help='The name of the output dir/file', required=True)
@click.option('--outpath', metavar='FILE', help='Output path', required=True)

def main(**kwargs):
    obj = Predict_samples(**kwargs)
    obj.run()

if __name__ == '__main__':
    main()
