import os

import transformers
import streamlit as st


def load_pipeline(model_path):
    tokenizer = transformers.AutoTokenizer.from_pretrained('google/mt5-base')
    print("REPLACE тче токенизер тина")
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_path)
    def translate(input_string):
        encode_obj = tokenizer(input_string, return_tensors='pt').input_ids
        outputs = model.generate(encode_obj)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translate


def load_models():
    ru_en_model = os.path.abspath('models/ru-en')
    en_fr_model = os.path.abspath('models/fr-en')
    multilingual_model = os.path.abspath('models/all-langs')

    ru_en_pipeline = load_pipeline(ru_en_model)

    en_fr_pipeline = load_pipeline(en_fr_model)
    multilingual_pipeline = load_pipeline(multilingual_model)
    
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
            st.markdown(f"**Translation:** {translation}")
            st.balloons()


def daisy():
    user_input = st.text_input("Russian Input")
    
    if st.button('Translate!'):
        st.markdown(f"**Russian:** {user_input}")
        english_rep = st.session_state.ru_en_pipeline(user_input)
        st.markdown(f"**Intermediate language:** {english_rep}")
        french_translation = st.session_state.en_fr_pipeline(english_rep)
        st.markdown(f"**French translation:** {french_translation}")
        st.balloons()


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
