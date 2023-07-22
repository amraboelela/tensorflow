from model1_init import *

# 3. Fit the model
model1.fit(X, y, epochs=5)

model1.save('data/model1.keras')

