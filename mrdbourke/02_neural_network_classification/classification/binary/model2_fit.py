from model2_init import *

# 3. Fit the model
model2.fit(X, y, epochs=100, verbose=0) # set verbose=0 to make the output print less

model2.save('data/model2.h5')

