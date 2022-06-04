import tensorflow as tf
import numpy as np

dominio = np.array([2,4,6], dtype=int)
recorrido = np.array([5,9,13], dtype=int)

capa = tf.keras.layers.Dense(units = 1, input_shape = [1])
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(1),
    loss = 'mean_squared_error'
)
print("Comenzando entrenamiento")
modelo.fit(dominio, recorrido, epochs = 500, verbose = False)
print("Entrenamiento finalizado")
print(modelo.predict([1]))
