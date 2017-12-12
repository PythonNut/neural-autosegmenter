from neural_autosegment import *
import shelve

m = keras.models.load_model('hall_of_fame/model.1906-0.0358.hdf5')
CACHEFILE = 'char_cache.dat'
if os.path.exists(CACHEFILE):
    s = shelve.open(CACHEFILE)
    text, chars, char_index_map, index_char_map = s['data']
    s.close()
else:
    text, chars, char_index_map, index_char_map = get_texts()
    s = shelve.open(CACHEFILE)
    s['data'] = text, chars, char_index_map, index_char_map
    s.close()

T = "This is a test of the ability of my neural network to automatically remove the words from a sequence."

sequences = []
for index in range(len(T) * 3 + 2):
    sequence = prepare_sequences(T+' '+T+' '+T, index, 40)
    if not sequence: continue
    sequences.append(sequence)

V = vectorize_sequences(sequences, char_index_map, 40)
Y = m.predict(V[0])
YY = [[] for x in range(len(Y) + len(Y[-1]))]
for i, x in enumerate(Y):
    for idx in np.arange(len(Y[i])): YY[idx+i].append(Y[i][idx])
YY = list(filter(lambda x:x, YY))
VV = np.zeros(len(YY))
# for i, x in enumerate(Y):
#     for idx in range(len(V[1][i])):
#         if V[1][i][idx]:VV = [0] * len(YY)
for i, x in enumerate(Y):
    for idx in np.arange(len(Y[i])):
        if V[1][i][idx]:
            VV[idx+i] = 1
fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    sharex=True,
    sharey=True
)

T_ns = T.replace(' ', '')


import matplotlib
matplotlib.use('TkAgg')

from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler


from matplotlib.figure import Figure

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

root = Tk.Tk()
root.wm_title("Embedding in TK")


f = Figure(figsize=(5, 4), dpi=100)
a = f.add_subplot(211)
a.plot(np.arange(len(T_ns))+1,VV[len(T_ns):len(T_ns)*2])
b = f.add_subplot(212)
b.violinplot(YY[len(T_ns):len(T_ns)*2],showmedians=True,showextrema=False)

# a tk.DrawingArea
canvas = FigureCanvasTkAgg(f, master=root)
canvas.show()
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

toolbar = NavigationToolbar2TkAgg(canvas, root)
toolbar.update()
canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)


def on_key_event(event):
    print('you pressed %s' % event.key)
    key_press_handler(event, canvas, toolbar)

canvas.mpl_connect('key_press_event', on_key_event)



def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
    # Fatal Python Error: PyEval_RestoreThread: NULL tstate

button = Tk.Button(master=root, text='Quit', command=_quit)
button.pack(side=Tk.BOTTOM)

e = Tk.Entry(root)
e.pack()

Tk.mainloop()
