import streamlit as st
import transformers

def load_models():
    ru_en_model = 'Helsinki-NLP/opus-mt-ru-en'
    en_fr_model = 'Helsinki-NLP/opus-mt-en-fr'
    miltilingual_model = 'Helsinki-NLP/opus-mt-en-fr'

    ru_en_pipeline = transformers.pipeline('translation_ru_to_en', model=ru_en_model)
    en_fr_pipeline = transformers.pipeline('translation_en_to_fr', model=en_fr_model)
    multilingual_pipeline = transformers.pipeline('translation_xx_to_yy', model=miltilingual_model)
    st.session_state['ru_en_pipeline'] = ru_en_pipeline
    st.session_state['en_fr_pipeline'] = en_fr_pipeline
    st.session_state['multilingual_pipeline'] = multilingual_pipeline


def direct():
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.header('Source')
            source_lang = st.radio('Source Language', ('English', 'French', 'Russian'))
        with col2:
            st.header('End')
            if source_lang == 'English':
                target_lang = st.radio('Target Language', ('French', 'Russian'))
            if source_lang == 'French':
                target_lang = st.radio('Target Language', ('English', 'Russian'))
            if source_lang == 'Russian':
                target_lang = st.radio('Target Language', ('French', 'English'))
    user_input = st.text_input("Input Text to Translate: ")
    if st.button('Translate'):
        prefix = f'translate from {source_lang} to {target_lang}: '
        to_model_for_translation = prefix + user_input
        translation = st.session_state.ru_en_pipeline(to_model_for_translation)

        if translation:
            st.markdown(f"**Translation:** {translation[0]['translation_text']}")


def daisy():
    user_input = st.text_input("Russian Input")
    
    if st.button('Translate!'):
        st.markdown(f"**Russian:** {user_input}")
        english_rep = st.session_state.ru_en_pipeline(user_input)
        st.markdown(f"**Intermediate language:** {english_rep[0]['translation_text']}")
        french_translation = st.session_state.en_fr_pipeline(english_rep[0]['translation_text'])
        st.markdown(f"**French translation:** {french_translation[0]['translation_text']}")


if __name__ == '__main__':
    st.title('Multilingual Machine Translation')
    st.write("Welcome to Aria and Tina's Machine Translation Project! Please select an option below to begin:")

    if 'ru_en_pipeline' not in st.session_state:
        with st.spinner(text="Loading models..."):
            load_models()
        st.success('All models are loaded and ready to translate!')

    model_type = st.radio('Select Translation Method', ('Daisy', 'Direct'))
    if model_type == 'Daisy':
        daisy()
    if model_type == 'Direct':
        direct()
