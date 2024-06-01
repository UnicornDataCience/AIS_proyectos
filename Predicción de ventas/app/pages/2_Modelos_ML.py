import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Configuración de la página
st.set_page_config(
    page_title='Machine Learning Demo', 
    page_icon=":robot_face:",
    initial_sidebar_state="expanded"
    )

df = pd.read_csv('metricas_modelos.csv')
df_español = df[['Accuracy_es', 'Recall_es', 'Precision_es', 'F1_es', 'Time_es']]
df_catalan = df[['Accuracy_ca', 'Recall_ca', 'Precision_ca', 'F1_ca', 'Time_ca']]
# mostrar los valores de la primera fila del índice
nb_es = df_español.loc[0]
nb_ca = df_catalan.loc[0]
# mostrar como una tabla nb
nb_es = pd.DataFrame(nb_es)
nb_ca = pd.DataFrame(nb_ca)
# cambiar el nombre de la columna
nb_es.columns = ['Naive Bayes']
nb_ca.columns = ['Naive Bayes']
svm_es = df_español.loc[1]
svm_ca = df_catalan.loc[1]
svm_es = pd.DataFrame(svm_es)
svm_ca = pd.DataFrame(svm_ca)
svm_es.columns = ['SVM']
svm_ca.columns = ['SVM']
rf_1_ca = df_español.loc[2]
rf_2_es = df_catalan.loc[3]
rf_1_ca = pd.DataFrame(rf_1_ca)
rf_2_es = pd.DataFrame(rf_2_es)
rf_1_ca.columns = ['Random Forest']
rf_2_es.columns = ['Random Forest']

with st.sidebar:
    st.header('Aprendizaje automático')
    # Desplegable para nubes de palabras (esto se manejará en otro fragmento de código)
    options = [
        'Naive Bayes',
        'SVM',
        'Random Forest',
        'Comparativas'
    ]
    selected_ml_model = st.selectbox('Selecciona una arquitectura de ML', options)
    st.write("---")
    st.image('Logo_UOC.jpg', width=200)
    st.write("---")
    st.write('Andrés Peñafiel Rodas \n apenafielro@uoc.edu ')

# Títulos y métricas en el cuerpo principal
st.header(selected_ml_model)
st.write("---")
# si se escoge Nube de palabras
if selected_ml_model == 'Naive Bayes':
    st.title('Catalán')

    col1, col2, col3 = st.columns(3)
    with col1:
        col1.caption("<div style='text-align: center;'>Matriz de Confusión</div>", unsafe_allow_html=True)
        col1.image('confusion_matrix_nb_ca.png')
        
    with col2:
        col2.caption("<div style='text-align: center;'>Métricas</div>", unsafe_allow_html=True)
        st.write(nb_ca)

    with col3:
        col3.caption("<div style='text-align: center;'>Curva ROC</div>", unsafe_allow_html=True)
        col3.image('roc_nb_ca.png')
    
    text = st.text_area("Escribe tu opinión aquí:", key="query")
    if st.button('Clasificar', key = 'classify', use_container_width=True):
        independencia_df_ca_balanced = pd.read_csv('independencia_df_ca_balanced.csv')
        textos_ca = independencia_df_ca_balanced['Texto_preprocesado'].apply(lambda x: ''.join(x))
        labels_ca = independencia_df_ca_balanced['Sentimiento']
        encoder = LabelEncoder()
        etiquetas_ca_encoded = encoder.fit_transform(labels_ca)
        # crear conjunto de train, test y valid
        train_sentences_ca, test_sentences_ca, train_labels_ca, test_labels_ca = train_test_split(textos_ca, etiquetas_ca_encoded, stratify = etiquetas_ca_encoded, test_size = 0.2, random_state = 1335)
        # crear conjunto de validación
        test_sentences_ca, val_sentences_ca, test_labels_ca, val_labels_ca = train_test_split(test_sentences_ca, test_labels_ca, stratify = test_labels_ca, test_size = 0.25, random_state = 1335)
        
        def predict_text_with_score(text):
            model = joblib.load('naivebayes_es.joblib')
            prediction = model.predict([text])
            prediction_proba = model.predict_proba([text])
            # Getting the class labels
            class_labels = model.classes_
            # Formatting the probability scores along with the class labels
            proba_scores = {class_labels[i]: prediction_proba[0][i] for i in range(len(class_labels))}
            if prediction[0] == 0:
                return 'Sentimiento positivo', proba_scores
            elif prediction[0] == 1:
                return 'Sentimiento neutral', proba_scores
            else:
                return 'Sentimiento negativo', proba_scores
            
        st.write(predict_text_with_score(text))

    st.write("---")
    st.title('Español')

    col4, col5, col6 = st.columns(3)
    with col4:
        col4.caption("<div style='text-align: center;'>Matriz de Confusión</div>", unsafe_allow_html=True)
        col4.image('confusion_matrix_nb_es.png')

    with col5:
        col5.caption("<div style='text-align: center;'>Métricas</div>", unsafe_allow_html=True)
        st.write(nb_es)
    
    with col6:
        col6.caption("<div style='text-align: center;'>Curva ROC</div>", unsafe_allow_html=True)
        col6.image('roc_nb_es.png')
        
    text = st.text_area("Escribe tu opinión aquí:", key="query2")
    # poner un boton a la derecha
    
    if st.button('Clasificar', key = 'classify2', use_container_width=True):
        independencia_df_es_balanced = pd.read_csv('independencia_df_es_balanced.csv')
        textos_es = independencia_df_es_balanced['Texto_preprocesado'].apply(lambda x: ''.join(x))
        labels_es = independencia_df_es_balanced['Sentimiento']
        encoder_es = LabelEncoder()
        etiquetas_es_encoded = encoder_es.fit_transform(labels_es)
        # crear conjunto de train, test y valid en español
        train_sentences_es, test_sentences_es, train_labels_es, test_labels_es = train_test_split(textos_es, etiquetas_es_encoded, test_size = 0.2, random_state = 1335)
        # crear conjunto de validación
        test_sentences_es, val_sentences_es, test_labels_es, val_labels_es = train_test_split(test_sentences_es, test_labels_es, stratify = test_labels_es, test_size = 0.25, random_state = 1335)
        def predict_text_with_score(text):
            model = joblib.load('naivebayes_es.joblib')
            prediction = model.predict([text])
            prediction_proba = model.predict_proba([text])
            # Getting the class labels
            class_labels = model.classes_
            # Formatting the probability scores along with the class labels
            proba_scores = {class_labels[i]: prediction_proba[0][i] for i in range(len(class_labels))}
            if prediction[0] == 0:
                return 'Sentimiento positivo', proba_scores
            elif prediction[0] == 1:
                return 'Sentimiento neutral', proba_scores
            else:
                return 'Sentimiento negativo', proba_scores
            
        st.write(predict_text_with_score(text))
    
    st.write("---")
    st.title('Palabras clasificadas')

    col7, col8 = st.columns(2)
    with col7:
        col7.markdown("<div style='text-align: center;'>Nubes de palabras procesadas del catalán</div>", unsafe_allow_html=True)
        col7.image('wordcloud_most_relevant_words_nb_ca.png')            
    with col8:
        col8.markdown("<div style='text-align: center;'>Nubes de palabras procesadas del español</div>", unsafe_allow_html=True)
        col8.image('wordcloud_most_relevant_words_nb_es.png')
        
if selected_ml_model == 'SVM':
    st.title('Catalán')

    col1, col2, col3 = st.columns(3)
    with col1:
        col1.caption("<div style='text-align: center;'>Matriz de Confusión</div>", unsafe_allow_html=True)
        col1.image('confusion_matrix_svm_ca.png')
        
    with col2:
        col2.caption("<div style='text-align: center;'>Métricas</div>", unsafe_allow_html=True)
        st.write(svm_ca)
        
    with col3:
        col3.caption("<div style='text-align: center;'>Curva ROC</div>", unsafe_allow_html=True)
        col3.image('roc_svm_ca.png')

    text = st.text_area("Escribe tu opinión aquí:", key="query3")
    if st.button('Clasificar', key = 'classify3', use_container_width=True):
        independencia_df_ca_balanced = pd.read_csv('independencia_df_ca_balanced.csv')
        textos_ca = independencia_df_ca_balanced['Texto_preprocesado'].apply(lambda x: ''.join(x))
        labels_ca = independencia_df_ca_balanced['Sentimiento']
        encoder = LabelEncoder()
        etiquetas_ca_encoded = encoder.fit_transform(labels_ca)
        # crear conjunto de train, test y valid
        train_sentences_ca, test_sentences_ca, train_labels_ca, test_labels_ca = train_test_split(textos_ca, etiquetas_ca_encoded, stratify = etiquetas_ca_encoded, test_size = 0.2, random_state = 1335)
        # crear conjunto de validación
        test_sentences_ca, val_sentences_ca, test_labels_ca, val_labels_ca = train_test_split(test_sentences_ca, test_labels_ca, stratify = test_labels_ca, test_size = 0.25, random_state = 1335)
        
        def predict_text_with_score(text):
            model = joblib.load('svm_ca.joblib')
            prediction = model.predict([text])
            decision_scores = model.decision_function([text])
            class_labels = model.classes_
            proba_scores = {class_labels[i]: decision_scores[0][i] for i in range(len(class_labels))}
            if prediction[0] == 0:
                return 'Sentimiento positivo', proba_scores
            elif prediction[0] == 1:
                return 'Sentimiento neutral', proba_scores
            else:
                return 'Sentimiento negativo', proba_scores
            
        st.write(predict_text_with_score(text))

    st.write("---")
    st.title('Español')

    col4, col5, col6 = st.columns(3)
    with col4:
        col4.caption("<div style='text-align: center;'>Matriz de Confusión</div>", unsafe_allow_html=True)
        col4.image('confusion_matrix_svm_es.png')
        
    with col5:
        col5.caption("<div style='text-align: center;'>Métricas</div>", unsafe_allow_html=True)
        st.write(svm_es)
        
    with col6:
        col6.caption("<div style='text-align: center;'>Curva ROC</div>", unsafe_allow_html=True)
        col6.image('roc_svm_es.png')

    text = st.text_area("Escribe tu opinión aquí", key="query4")
    if st.button('Clasificar', key = 'classify4', use_container_width=True):
        independencia_df_es_balanced = pd.read_csv('independencia_df_es_balanced.csv')
        textos_es = independencia_df_es_balanced['Texto_preprocesado'].apply(lambda x: ''.join(x))
        labels_es = independencia_df_es_balanced['Sentimiento']
        encoder_es = LabelEncoder()
        etiquetas_es_encoded = encoder_es.fit_transform(labels_es)
        # crear conjunto de train, test y valid en español
        train_sentences_es, test_sentences_es, train_labels_es, test_labels_es = train_test_split(textos_es, etiquetas_es_encoded, test_size = 0.2, random_state = 1335)

        def predict_text_with_score(text):
            model = joblib.load(open('svm_es.joblib', 'rb'))
            prediction = model.predict([text])
            decision_scores = model.decision_function([text])
            class_labels = model.classes_
            proba_scores = {class_labels[i]: decision_scores[0][i] for i in range(len(class_labels))}
            if prediction[0] == 0:
                return 'Sentimiento positivo', proba_scores
            elif prediction[0] == 1:
                return 'Sentimiento neutral', proba_scores
            else:
                return 'Sentimiento negativo', proba_scores
            
        st.write(predict_text_with_score(text))


if selected_ml_model == 'Random Forest':
    st.title('Catalán')

    col1, col2, col3 = st.columns(3)
    with col1:
        col1.caption("<div style='text-align: center;'>Matriz de Confusión</div>", unsafe_allow_html=True)
        col1.image('confusion_matrix_rf_ca_v1.png')
        
    with col2:
        col2.caption("<div style='text-align: center;'>Métricas</div>", unsafe_allow_html=True)
        st.write(rf_1_ca)
        
    with col3:
        col3.caption("<div style='text-align: center;'>Curva ROC</div>", unsafe_allow_html=True)
        col3.image('roc_rf_ca.png')
    
    st.write("---")
    st.title('Español')

    col4, col5, col6 = st.columns(3)
    with col4:
        col4.caption("<div style='text-align: center;'>Matriz de Confusión</div>", unsafe_allow_html=True)
        col4.image('confusion_matrix_rf_es_v2.png')
    with col5:
        col5.caption("<div style='text-align: center;'>Métricas</div>", unsafe_allow_html=True)
        st.write(rf_2_es)
        
    with col6:
        col6.caption("<div style='text-align: center;'>Curva ROC</div>", unsafe_allow_html=True)
        col6.image('roc_rf_es.png')

if selected_ml_model == 'Comparativas':
    cols = st.columns(3)
    for i, col in enumerate(cols):
        if i == 0:
            col.caption("<div style='text-align: center;'>Métricas/modelos en catalán</div>", unsafe_allow_html=True)
            col.image('comparativa_metricas_ca.png')
        if i == 1:
            col.caption("<div style='text-align: center;'>Métricas/modelos en español</div>", unsafe_allow_html=True)
            col.image('comparativa_metricas_es.png')
        if i == 2:
            col.caption("<div style='text-align: center;'>Tiempos de ejecución</div>", unsafe_allow_html=True)
            col.image('comparativa_tiempos_ejecución.png')
