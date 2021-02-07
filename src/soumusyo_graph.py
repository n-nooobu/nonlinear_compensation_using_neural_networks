import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import ArrowStyle

"""
https://www.soumu.go.jp/menu_news/s-news/01kiban04_02000171.html
https://www.soumu.go.jp/main_content/000699741.pdf
"""

rcParams['font.family'] = 'IPAexGothic'

n = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
data = np.array([2889, 3560, 4448, 5467, 6840, 8232, 8027, 8903, 10289, 10976, 12086, 12650, 19025])

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(n, data, '-o')
ax.annotate(s='', xy=[10, 19175], xytext=[10, 12086],
            arrowprops=dict(arrowstyle=ArrowStyle('simple', head_length=0.1, head_width=0.1, tail_width=0.1),
                            connectionstyle='arc3', facecolor='red', edgecolor='red'))
ax.annotate(s='', xy=[12, 19025], xytext=[9.93, 19025],
            arrowprops=dict(arrowstyle=ArrowStyle('simple', head_length=1, head_width=1, tail_width=0.1),
                            connectionstyle='arc3',facecolor='red', edgecolor='red'))
ax.text(11.3, 19625, '19,025', color='red', fontsize=15)
ax.text(9.7, 10676, '12,086', color='red', fontsize=15)
ax.text(6.3, 16625, '57%増加', color='red', fontsize=21)
plt.ylabel('Download traffic[Gbps]', fontsize=15)
ax.set_xlim((-0.7, 13.2))
ax.set_ylim((1500, 22000))
ax.set_xticks(np.linspace(0, 12, 13))
ax.set_xticklabels(['2014年5月', '2014年11月', '2015年5月', '2015年11月', '2016年5月', '2016年11月', '2017年5月',
                    '2017年11月', '2018年5月', '2018年11月', '2019年5月', '2019年11月', '2020年5月'])
plt.gcf().autofmt_xdate()
plt.grid(which='major', axis='both', color='#999999', linestyle='--')
ax.xaxis.set_tick_params(labelsize=13, direction='in')
ax.yaxis.set_tick_params(labelsize=13, direction='in')
plt.subplots_adjust(left=0.13, bottom=0.17, right=0.97, top=0.97)
plt.show()
