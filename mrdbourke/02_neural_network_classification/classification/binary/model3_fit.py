from model3_init import *

# 3. Fit the model
model3.fit(X, y, epochs=100, verbose=1) # fit for 100 passes of the data

model3.save('data/model3.keras')

