from xml import dom
import tensorflow as tf
import numpy as np

dominio = np.array([2,3,4,5],dtype=float)
recorrido = np.array([4,9,16,25], dtype=float)

oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units =3)
terminal = tf.keras.layers.Dense(units =1)
modelo = tf.keras.Sequential([oculta1, oculta2, terminal])

modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss = 'mean_squared_error'
)

print("Comenzando entrenamiento")
modelo.fit(dominio, recorrido, epochs = 1000, verbose = False)
print("Modelo terminado")

print(modelo.predict([2]))
