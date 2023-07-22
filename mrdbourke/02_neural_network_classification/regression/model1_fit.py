from model1_init import *

# Fit the recompiled model
model1.fit(tf.expand_dims(X_reg_train, axis=-1),
            y_reg_train,
            epochs=100)

model1.save('data/model1.keras')

