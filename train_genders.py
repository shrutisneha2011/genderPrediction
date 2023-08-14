from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from keras import backend as K
K.tensorflow_backend._get_available_gpus()


import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras import __version__ as keras_version
import numpy as np




nEpochs = 200


print "loading data"

with open("male.txt") as f:
	m_names = f.readlines()
	m_names = [m_name.rstrip() for m_name in m_names]	#### this line added it will remove '\n' attached to each name

		
with open("female.txt") as f:
	f_names = f.readlines()
	f_names = [f_name.rstrip() for f_name in f_names]	#### this line added it will remove '\n' attached to each name

mf_names = []

for f_name in f_names:
	if f_name in m_names:
		mf_names.append(f_name)

m_names = [m_name.lower() for m_name in m_names if not m_name in mf_names]
f_names = [f_name.lower() for f_name in f_names if not f_name in mf_names]


#print "max len for m_names:",len(max( m_names , key=len))
#print "max_len for f_names:", len(max( f_names , key=len))


totalEntries = len(m_names) + len(f_names)
maxlen = len(max( m_names , key=len)) + len(max( f_names , key=len))

chars = set(  "".join(m_names) + "".join(f_names)  )
chars_new = sorted(chars)			#########################-------------> sorted order now :D

#char_indices = dict((c, i) for i, c in enumerate(chars))	#######-----> wrong sequence :'(

char_indices = {}
for integer_val , char_var in enumerate(chars_new):
	char_list = {char_var: integer_val}
	char_indices.update(char_list)
print "char_indices updated one",char_indices

print "char_indices: ", char_indices

#indices_char = dict((i, c) for i, c in enumerate(chars))


print "total endtries " , totalEntries
print "max len " , maxlen
print('total chars:', len(chars))



X = np.zeros((totalEntries , maxlen, len(chars) ), dtype=np.bool)
y = np.zeros((totalEntries , 2 ), dtype=np.bool)


for i, name in enumerate(m_names):
    for t, char in enumerate(name):
        X[i, t, char_indices[char]] = 1
    y[i, 0 ] = 1

for i, name in enumerate(f_names):
    for t, char in enumerate(name):
        X[i + len(m_names), t, char_indices[char]] = 1
    y[i + len(m_names) , 1 ] = 1


print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop')


json_string = model.to_json()

with open("model-200iter.json", "w") as text_file:
    text_file.write(json_string)


if keras_version[0] == '1':
	model.fit(X, y, batch_size=16, nb_epoch=nEpochs)
else:
	model.fit(X, y, batch_size=16, epochs=nEpochs)

model.save_weights('my_model_weights-200iter.h5')

print "done and weights saved"
score = model.evaluate(X, y, batch_size=16)
print "score " , score

