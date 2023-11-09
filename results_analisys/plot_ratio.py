import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
cm = mpl.colormaps['tab10'].resampled(20)
df = pd.read_csv('./all_mae_data.csv',index_col=0,usecols=[0,1],nrows=7,skiprows=8)
print(df)
print(cm(0.5))
f,ax = plt.subplots()
for idx,col in enumerate(df.columns):
    print(col,df[col])
    linecolor = (*cm(idx/len(df.columns))[:3],0.3)
    dotcolor = (*cm(idx/len(df.columns))[:3],0.7)
    ax.plot(df[col],'^-',label=col,c= linecolor, mfc=dotcolor)
ax.set_xticks(ax.get_xticks(), df.index, rotation=45, ha='right')
ax.legend()
ax.grid(True)
ax.set_ylim(0,ax.get_ylim()[1])
f.tight_layout()
f.savefig('fig_ratio.png',dpi=500)
# p=df.plot()
# p.figure.savefig('fig.png')