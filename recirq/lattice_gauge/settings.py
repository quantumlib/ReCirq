import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np

#Notebook Parameters

reps = 1_000

number_of_gauge_qubits = 17

plt.rcParams['figure.dpi'] = 72
plt.rcParams['font.size'] = 12
plt.rcParams['lines.markersize'] = 8
plt.rcParams['lines.markeredgecolor'] = 'k'
plt.rcParams['lines.markeredgewidth'] = 0.7
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.bottom'] = True
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelleft'] = True

WALA_INITIAL = np.array([0, 0.35746251, 0.14352941, 1])
TORIC_INITIAL = 'steelblue'
POLARIZED_INITIAL = "#fbbc04"
BREAKING_BOTTOM = '#67cd85ff'
BREAKING_TOP = '#e6a304ff'
BREAKING_VAC = 'k'

cmap1 = LinearSegmentedColormap.from_list("1", ['darkred', 'salmon'],gamma=1.0)
cmap2 = LinearSegmentedColormap.from_list("2", ['salmon','lightgrey'],gamma=1.0)
cmap3 = LinearSegmentedColormap.from_list("3", ['lightgrey','lightsteelblue'],gamma=1.0)
cmap4 = LinearSegmentedColormap.from_list("4", ['lightsteelblue','steelblue'],gamma=1.0)
color_list = [cmap1(i) for i in np.arange(0,1,1/375)]+ [cmap2(i) for i in np.arange(0,1,1/250)]  + [cmap3(i) for i in np.arange(0,1,1/30)] + [cmap4(i) for i in np.arange(0,1,1/95)]
charge_cmap = LinearSegmentedColormap.from_list("Charge", color_list,gamma=1)
charge_cmap_r = LinearSegmentedColormap.from_list("Charge_r", color_list[::-1],gamma=1)

he_list = [0,0.3,0.6,0.8,2.0]
blues_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "Blues", [(0.0,'k'),(0.5,'steelblue'), (1.0,(0.9*0.6901960784313725, 0.9*0.7686274509803922, 1.0*0.8705882352941177, 1.0))]
)
blues_cmap_r = matplotlib.colors.LinearSegmentedColormap.from_list(
    "Blues_r", [(1.0,'k'),(0.5,'steelblue'), (0.0,(0.9*0.6901960784313725, 0.9*0.7686274509803922, 1.0*0.8705882352941177, 1.0))][::-1]
)
blues_color_list = [blues_cmap(i) for i in np.arange(1,-0.01,-1/(max(len(he_list)-1,1)))]

colors_greens1 = ['white', '#009468ff']
colors_greens2 = ["#bdc1c6",'#009468ff']
colors_greens3 = ['#009468ff','black']
cmap_greens1 = LinearSegmentedColormap.from_list("1", colors_greens1,gamma=3.5)
cmap_greens2 = LinearSegmentedColormap.from_list("mycmap", colors_greens3,gamma=0.8)
color_list = ([cmap_greens1(i) for i in np.arange(0,1,1/10240)] + [cmap_greens2(i) for i in np.arange(0,1,1/12500)])
cmap_green = LinearSegmentedColormap.from_list("mycmap", color_list,gamma=1)
cmap_green_r = LinearSegmentedColormap.from_list("mycmap", color_list[::-1],gamma=1)

colors = ['darkred', 'salmon']
cmap1 = LinearSegmentedColormap.from_list("mycmap", colors,gamma=4.0)
colors = ['salmon','lightgrey']
cmap2 = LinearSegmentedColormap.from_list("mycmap", colors,gamma=1.0)
colors = ['lightgrey','lightsteelblue']
cmap3 = LinearSegmentedColormap.from_list("mycmap", colors,gamma=1.0)
colors = ['lightsteelblue','steelblue']
cmap4 = LinearSegmentedColormap.from_list("mycmap", colors,gamma=0.5)
color_list = [cmap1(i) for i in np.arange(0,1,1/1500)]+ [cmap2(i) for i in np.arange(0,1,1/300)]  + [cmap3(i) for i in np.arange(0,1,1/300)] + [cmap4(i) for i in np.arange(0,1,1/1500)]
diff_cmap = LinearSegmentedColormap.from_list("mycmap", color_list,gamma=1)
diff_cmap_r = LinearSegmentedColormap.from_list("mycmap", color_list[::-1],gamma=1)