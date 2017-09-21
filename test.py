from neural_autosegment import *
m = keras.models.load_model('model.hdf5')
text, chars, char_index_map, index_char_map = get_texts()
T = "This is a test of the ability of my neural network to automatically remove the words from a sequence."
V = vectorize_sequences([prepare_sequences(T+T+T,len(T)+i,40) for i in range(len(T))], char_index_map, 40)
Y = m.predict(V[0])
YY = [[] for x in range(200)]
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
axes[0].plot(np.arange(len(VV))+1,VV)
axes[1].violinplot(YY,showmeans=True,showextrema=False)
plt.show()
