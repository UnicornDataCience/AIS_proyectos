import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title='Home', 
    page_icon="🎓",
    layout='wide',
    initial_sidebar_state='expanded'
    )


st.title('AlimentIA: Análisis de negocio y predicción de ventas')

# Barra lateral con Logo, selección de modelo y nubes de palabras
with st.sidebar:
    st.sidebar.success("Selecciona una demostración de la lista de arriba")
    st.write("---")
    st.image('logo.png', width=200)
    st.write("---")
    st.write('Andrés Peñafiel Rodas \n andres.pennafiel@gmail.com')
    st.write('David Monroy \n monroygonzalezdavid@gmail.com')
    st.write('Pablo Tomás \n 93pablotr@gmail.com')
    st.write('Mayra \n andres.pennafiel@gmail.com')
    st.write('Sergio Serna \n sgsernac@gmail.com')
    

st.write("## AI Saturdays Madrid - Proyecto de ML")

# añade una línea de separación
st.write('---')

st.markdown(
    """
    Este proyecto es una demostración de la aplicación de técnicas de predicción de ventas en un entorno de negocio real.
        
    **👈 Selecciona una demostración de la barra lateral** para ver gráficos, modelos y sus 
    métricas, calcular predicciones e interactuar con la interfaz.

    ### ¿Qué fuentes de datos hemos usado?

    ### ¿Qué artículos hemos consultado?

    ### ¿Qué modelos de aprendizaje automático hemos usado?
    
    """
)

