"""
"""


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
plt.ioff() # turn off interactive mode

def plot_loss(lossDf, outPath, cv=False):
    """
    :param lossDf: dataframe with columns=['epoch', 'train Loss', 'valid Loss']
    :param outPath: string representing output path
    :return .png of line plot

    Note:
    dataframe is indexed, epoch is the index
    """
    # create figure
    fig, ax = plt.subplots(1, figsize=(8,6))
    
    if cv == False:
        # make a plot
        lossDf[['train loss', 'valid loss']].plot(ax=ax)
        # add annotation
        ax.set_title('Loss Plot', fontsize=14)
    else:
        lossDf.groupby('epoch').mean()[['train loss', 'valid loss']].plot(ax=ax)
    # save figure
    fig.savefig(outPath+'.LossPlot.png', dpi=200, bbox_inches='tight')
    # return
    return fig


if __name__ == "__main__":
    import pandas as pd
    fn = '/repo4/ytang4/PHD/DeepDSC/GDSC.CHEM-TPM.DeepDSC.FNN.cv_10.Loss.txt'
    df = pd.read_csv(fn, header=0, sep="\t")
    plot_loss(df, '/repo4/ytang4/PHD/DeepDSC/GDSC.CHEM-TPM.DeepDSC.FNN', cv=True)
