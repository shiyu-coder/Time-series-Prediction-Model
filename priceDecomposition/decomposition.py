import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('testData/BTC_1min.csv')

means = []
stds = []
opens = []
difs = []
mdifs = []
window = 60

# for i in range(df.shape[0]-window):
for i in range(3000):
    difs.append(np.log(df.loc[i+window/2, 'Open'].item()) - np.log(df.loc[i+window/2-1, 'Open'].item()))
    opens.append(df.loc[i+window/2, 'Open'].item())
    means.append(df.loc[i:i+window, :].mean()['Open'])
    if i == 0:
        mdifs.append(np.log(means[0]) - np.log(df.loc[i:i+window-1, :].mean()['Open']))
    else:
        mdifs.append(np.log(means[i]) - np.log(means[i-1]))
    stds.append(df.loc[i:i + window, 'Open'].diff().std())
x = np.arange(len(opens))
ups = [means[i] - 3*stds[i] for i in range(len(means))]
downs = [means[i] + 3*stds[i] for i in range(len(means))]
plt.subplot(411)
plt.plot(x, opens, label='open')
plt.plot(x, means, label='mean')
plt.plot(x, ups, label='up')
plt.plot(x, downs, label='down')
plt.legend()
plt.subplot(412)
plt.plot(x, stds, label='std')
plt.legend()
plt.subplot(413)
plt.plot(x, mdifs, label='mean_price_log_return')
plt.legend()
plt.subplot(414)
plt.plot(x, difs, label='price_log_return')
plt.legend()
plt.show()


