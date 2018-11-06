# Calculate the Gini index for a split dataset
def gini_index(groups, class_values):
	gini = 0.0
	for class_value in class_values:
		for group in groups:
			size = len(group)
			if size == 0:
				continue
			t1=[row[-1] for row in group]
			#t1=[row[0] for row in group]
			t2=t1.count(class_value)
			proportion =t2 / float(size)
			gini += (proportion * (1.0 - proportion))
	return gini

# test Gini values
print(gini_index([[[0]],
				  [[1]]],
				 [0, 1]))

print(gini_index([[[1, 1], [1, 0]], [[1, 1], [1, 0]]], [0, 1]))
print(gini_index([[[1, 0], [1, 0]], [[1, 1], [1, 1]]], [0, 1]))