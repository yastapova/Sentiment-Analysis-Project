import csv
import redis

TRAINING_FILE_NAME = "train.tsv"
TEST_FILE_NAME = "test.tsv"

r = redis.StrictRedis(host = 'localhost', port = 6379, db = 0)

class node(object):
	def __init__(self, value, parent = None, children = []):
		self.value = value
		self.parent = parent
		self.children = children
	def __repr__(self, level = 0):
		ret = '\t' * level + repr(self.value) + '\n'
		for child in self.children:
			ret += child.__repr__(level + 1)
		return ret
	def set_parent(self, parent):
		self.parent = parent
	def add_child(self, child):
		new_list = list(self.children)
		new_list.append(child)
		self.children = new_list
	def db_load(self):
		child_strings = []
		for child in self.children:
			child_strings.append(r.get(str(hash(child.value))))
		if self.parent is not None:
			m = {'parent': r.get(str(hash(self.parent.value))), 'children': ','.join(child_strings)}
		else:
			m = {'children': ','.join(child_strings)}
		r.hmset(r.get(str(hash(self.value))), m)
		for child in self.children:
			child.db_load()

trees = []

with open(TRAINING_FILE_NAME, 'rb') as training:
	reader = csv.reader(training, delimiter='\t')
	header = reader.next()
	for row in reader:
		if int(row[1]) > 96: ## MAX SAMPLING OF TRAINING DATA
			break
		if int(row[1]) > len(trees):
			tree = node(row[2])
			trees.append(tree)
		else:
			while tree is not None and row[2] not in tree.value:
				tree = tree.parent
			if tree is not None:
				s = node(row[2])
				s.set_parent(tree)
				tree.add_child(s)
				tree = s
			else:
				tree = node(row[2])
		r.set(str(hash(row[2])), 'p:'+row[1]+':'+row[0])
		r.hmset('p:'+row[1]+':'+row[0], {'id': str(hash(row[2])), 'value': row[3]})

	for tree in trees:
		tree.db_load()

### We now have a database structure for the values. Now, we can try to load the test data.

trees = []

with open(TEST_FILE_NAME, 'rb') as testing:
	reader = csv.reader(testing, delimiter='\t')
	header = reader.next()
	for row in reader:
		if int(row[1]) > 77: ## MAX SAMPLING OF TEST DATA
			break
		if int(row[1]) > len(trees):
			tree = node(row[2])
			trees.append(tree)
		else:
			while tree is not None and row[2] not in tree.value:
				tree = tree.parent
			if tree is not None:
				s = node(row[2])
				s.set_parent(tree)
				tree.add_child(s)
				tree = s
			else:
				tree = node(row[2])

## This now contains the list of all trees in the test data (up to 77)


## Strategy from here on forward:
		
		### Find all leaves of each parse tree and see if they appear in the redis database.
		### To do this we can use r.get(hash("string")) to see if a key exists.
		### If not, create such a key and set it to two (neutral).

		### Then, going up the parse tree, we now have a set of 2-tuples, on which we can run KNN, and find the value for the composite strings.
		### When we continue to run KNN going up, we produce tree structures with weights. When two parse trees have the same structure, run KNN.
		### Use the most specific structure you can acquire to calculate the sentiment of the reviews.




	### We will store all lines in the dictionary, given the phrase and sentence
	### Store list of all K/V pairs.

	### For each composite vector we will include the dimensionality of the total vector.
	### We now have composite dimensionality for each element in the vector.

	### Let phrase x be the vector of (x_0, x_1, x_2, x_3, ...). From this we can estimate the dimensionality of the remaning vectors. 

	### We must always normalize each sentence so that the number of dimensions is properly sized.
	### We will store the maximum dimensionality of each vector but only select certain parts to analyze at a time.

	