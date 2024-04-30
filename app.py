import streamlit as st
import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("ejalisco2016.csv", index_col= "CLAVE DE INMUEBLE")
df_train = pd.read_csv("train.csv", index_col= "CLAVE DE INMUEBLE")
df_test = pd.read_csv("test.csv", index_col= "CLAVE DE INMUEBLE")

st.set_page_config( page_title="Jalisco Connection Points")


st.title("Jalisco Connection Points :globe_with_meridians:")

columna_texto, columna_imagen = st.columns([2, 1])

with columna_texto:
    st.markdown("""
    <span style='line-height: 0.8;'>
    Esteban Javier Berumen Nieto\n
    ITESO  (Instituto Tecnológico y de Estudios Superiores de Occidente)\n
    10 de abril del 2024
    </span>
    """, unsafe_allow_html=True)
with columna_imagen:
    st.image("ITESO.png", use_column_width=True)

st.markdown("""
#### Introducción

En este proyecto, se emplea el algoritmo KNN para predecir el ancho de banda 
contratado en 2014, por un listado de instituciones públicas de Jalisco en el año 2016. El 
objetivo principal es evaluar la precisión del modelo en función de las características de 
estas instituciones. 

Dado que una predicción precisa del ancho de banda contratado para las instituciones públicas no solo es importante para la gestión eficiente de la infraestructura de TI y la asignación de recursos financieros,
sino que también tiene un impacto significativo en la calidad y disponibilidad de los servicios públicos digitales ofrecidos a los ciudadanos.
            
Se tomó la decisión de utilizar el algoritmo de KNN dado que como veremos más adelante la 
distrubución de los datos no es normal. Ademas de que con KNN no es nesesario encontrar una correlacion entre las variables y como tambien veremos precticamente no la hay 

Los datos fueron obtenidos del sitio [Datos Abiertos Jalisco](https://datos.jalisco.gob.mx/dataset/puntos-de-conexion-ejalisco) en donde encontramos un dataset con las siguientes variables incluidas:
**[clave de inmueble, tecnologia instalada, ancho de banda contratado 2014, institución, 
nombre del centro, turno/horario, nivel, región, municipio, localidad, domicilio, 
código postal, longitud, latitud]** en donde encontramos 6716 registros.
""")

st.write(df)

st.markdown("""
#### Desarrollo

Dentro del preprocesamiento de datos se realizó una limpieza en donde todos los 
datos faltantes y N/D del dataset fueron remplazados por la primera moda de la columna 
correspondiente, esto ya que todas las columnas del dataset son categóricas, esto a su vez 
hizo que fuera necesario usar el ***OrdinalEncoder*** de la librería ***sklearn***, para poder realizar la 
codificación de las variables, además se dividió el dataset en un train y un test en donde el train 
es el **20%** del dataset original y el test es el **80%**.

""")

st.markdown("""
##### Train data :bar_chart:
""")
st.write(df_train)

st.markdown("""
##### Test data :bar_chart:
""")
st.write(df_test)

st.markdown( """
A continuación, se eliminaron algunas columnas debido a diferentes problemas u 
objetivos; tales son las columnas de: **clave de inmueble, ancho de banda contratado 
2014, nombre del centro, longitud, latitud**. En el caso del ancho de banda, esta columna 
se convierte en el target. También se realizaron los histogramas de las diferentes columnas, 
así como un mapa de correlación de las variables. 
""")

st.markdown("""
En los histogramas veremos el eje y como la frecuciencia y el eje x como la variable ya codificada 
""")

st.image("histograms_x_train.png")
st.image("histograms_x_test.png")
st.image("histograms_y_train.png")
st.image("histograms_y_test.png")

st.markdown("""
Como ya mencionó anteriormente en ninguno de los features encontramos una distribución normal,
gracias a esto, tambien podemos ver que en entre el train y el test hay una gran simetría.

Algunas otras cosas que podemos ver en algunos features específicos son:
La mayoría de los regitros de horario corresponde a solo un horario (Matutino)

""")

st.image("Corr_X_train.png")
st.image("Corr_X_test.png")

st.markdown("""
Se implementó el algoritmo KNN en el cual se probó con diferentes k para poder 
establecer cuál sería el número de vecinos mediante la prueba del codo.
""")

#!---------------------------------------------------------------
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

@st.cache_data
def train_knn_model(X_train, y_train, X_test, y_test, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return knn, accuracy

# Barra lateral para ajustar el número de vecinos
n_neighbors = st.slider('Número de vecinos (k)', min_value=1, max_value=30, value=5, step=1)

# Reentrenar el modelo con los nuevos parámetros
knn_model, accuracy = train_knn_model(X_train, y_train, X_test, y_test, n_neighbors)

# Mostrar precisión del modelo
st.write(f'Precisión del modelo KNN con **{n_neighbors}** vecinos: **{accuracy:.5f}**')

if st.button('Generar gráfica'):
    precisions = [] 
    for k in range(1, n_neighbors + 1):
        knn_model, accuracy = train_knn_model(X_train, y_train, X_test, y_test, k)
        precisions.append(accuracy)

    # Graficar los resultados
    plt.plot(range(1, n_neighbors + 1), precisions, marker='o')
    plt.title('Prueba de Codo para KNN')
    plt.xlabel('Número de vecinos (k)')
    plt.ylabel('Precisión')
    st.pyplot(plt)

#!----------------------------------------------------------------

st.markdown("""
#### Resultados  
Visto lo anterior, podemos decir que el algoritmo llega a su mejor precisión con 23 
vecinos en donde nos un ofrece 84.8% de precisión
""")

code = """
from sklearn.neighbors  import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors= 23 )
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
"""
st.code(code, language='python')
st.markdown("0.846441947565543")
           
st.markdown("""
#### Conclusiones:  
Nuestro análisis de predicción del ancho de banda para instituciones públicas en Jalisco 
utilizando el algoritmo KNN ha arrojado resultados prometedores. Logramos obtener una 
precisión del 84.8% al utilizar 23 vecinos en el modelo. Esta precisión nos brinda confianza 
en la capacidad del modelo para predecir de manera efectiva el ancho de banda necesario en base 
a las características proporcionadas.

Al explorar las relaciones entre las diferentes variables, observamos que la mayoría de ellas 
tienen una correlación baja entre sí. Sin embargo, identificamos que las variables de tecnología 
instalada y estatus muestran una correlación significativa con el ancho de banda contratado. 
Esto sugiere que estas dos características pueden ser factores determinantes en la cantidad de 
ancho de banda requerido por una institución.

En resumen, aunque nuestras características no muestran una correlación fuerte entre sí, hemos 
encontrado que la tecnología instalada y el estatus son variables importantes a considerar al 
predecir el ancho de banda necesario. Esto destaca la importancia de tener en cuenta no solo la 
cantidad de ancho de banda disponible, sino también el tipo de tecnología utilizada y el estatus 
de la institución al planificar la infraestructura de red.
""")

st.markdown("""
#### Referencias  
Secretaría de Educación Pública de Jalisco. (s/f). Puntos de Conexión eJalisco [Conjunto de 
datos]. Recuperado de https://datos.jalisco.gob.mx/dataset/puntos-de-conexion-ejalisco  
scikit-learn. (s/f). Nearest Neighbors. Recuperado de https://scikit-learn.org/stable/modules/neighbors.html 
""")




