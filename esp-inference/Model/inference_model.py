import tensorflow as tf 


model=tf.keras.models.load_model('single_step_model.keras')


model.predict("")