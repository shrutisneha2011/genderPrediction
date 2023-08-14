from keras.models import model_from_json
import numpy as np
import sys
import re

def predict(model, name, maxlen, chars, char_indices):
	x = np.zeros((1, maxlen, len(chars)))
	#print "enumerate name:", enumerate(name)
	for t, char in enumerate(name):
		x[0, t, char_indices[char]] = 1
		print "char_indices[char]:", char_indices[char]

	preds = model.predict(x, verbose=0)[0]
	#print "type of pred:",type(preds)
	return preds


maleFile = "male.txt"
femaleFile = "female.txt"

    

def load_model():
	mf_names = []
	with open(maleFile) as f:
		m_names = f.readlines()
		m_names = [m_name.rstrip() for m_name in m_names]	#### this line added it will remove '\n' attached to each name

	with open(femaleFile) as f:
		f_names = f.readlines()
		f_names = [f_name.rstrip() for f_name in f_names]	#### this line added it will remove '\n' attached to each name
		

	for f_name in f_names:
		if f_name in m_names:
			mf_names.append(f_name)

	m_names = [m_name.lower() for m_name in m_names if not m_name in mf_names]
	f_names = [f_name.lower() for f_name in f_names if not f_name in mf_names]
	
	#print "max len for m_names:",len(max( m_names , key=len))
	#print "max_len for f_names:", len(max( f_names , key=len))

	
	totalEntries = len(m_names) + len(f_names)
	
	
	maxlen_male = 0
	maxlen_female = 0
	for m_name in m_names :
		if(len(m_name)>maxlen_male):
			maxlen_male = len(m_name)
		
	#print "max male name:", maxlen_male
	
	for f_name in f_names :
		if(len(f_name)>maxlen_female):
			maxlen_female = (len(f_name))
	
	
	
	#print "max male name:", maxlen_female		
	#print "max female name:", maxlen_female	
	
	
	maxlen = maxlen_male + maxlen_female
	#print "maxlen:", maxlen
	
	#maxlen = len(max( m_names , key=len)) + len(max( f_names , key=len))
	
	#print (max( m_names, key=len))
	
	chars = set( "".join(m_names) + "".join(f_names) )
	chars_new = sorted(chars)
	#print "chars new:", chars_new
	
	
	
	
	#char_indices = dict((c, i) for i, c in enumerate(chars_new))
	
	char_indices = {}
	for integer_val , char_var in enumerate(chars_new):
		char_list = {char_var: integer_val}
		char_indices.update(char_list)
	print "char_indices updated one",char_indices	##### correct
	
	
	#indices_char = dict((i, c) for i, c in enumerate(chars_new))
    
	with open("model-200iter.json", 'r') as content_file:
		json_string = content_file.read()

	model = model_from_json(json_string)
	
	model.load_weights('my_model_weights-200iter.h5')
	return model, maxlen ,chars ,char_indices


if __name__ == "__main__":
	model, maxlen, chars, char_indices = load_model()
	while True:
		print "Enter any name:"
		n = raw_input()
		#print "type of name:", type(n)
		v = predict(model, n, maxlen, chars, char_indices)
		print "v:", v
		if v[0] > v[1]:
			print "Male"
		else:
			print "Female"
