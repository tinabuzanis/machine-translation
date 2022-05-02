import multiprocessing
import streamlit as st
import transformers


# @st.cache(suppress_st_warning=True)
def part_one():
    st.title('Multilingual Machine Translation')

    st.write("Welcome to Aria and Tina's Machine Translation Project! Please select an option below to begin:")
    translation_method = st.radio('Choose a translation method:', ('Direct', 'Daisy-Chain'))

    return translation_method #, pipelines 

# comment below line for demo
# @st.cache(suppress_st_warning=True)
def part_two(translation_method):
    choice_dict = {'000' : None, '100' : 'en', '010' : 'fr', '001' : 'ru'}
    with st.container():
        lang_choices = []
        col1, col2, col3 = st.columns(3)
        with col1:
            st.header('Source')
            en = st.checkbox('English')
            fr =  st.checkbox('French')
            ru = st.checkbox('Russian') 
            opt = [en, fr, ru]
            lang_choices.append(''.join([str(1) if el is True else str(0) for el in opt]))
        if translation_method == 'Daisy-Chain':    
            with col2:
                st.header('Middle')
                en = st.checkbox('English ')
                fr =  st.checkbox('French ')
                ru = st.checkbox('Russian ') 
                opt = [en, fr, ru]
                lang_choices.append(''.join([str(1) if el is True else str(0) for el in opt]))
        else:
            lang_choices.append('000')
        with col3:
            st.header('Target')
            en = st.checkbox('English  ')
            fr =  st.checkbox('French  ')
            ru = st.checkbox('Russian  ') 
            opt = [en, fr, ru]
            lang_choices.append(''.join([str(1) if el is True else str(0) for el in opt]))

        src, mid, tgt = choice_dict[lang_choices[0]], choice_dict[lang_choices[1]], choice_dict[lang_choices[2]]

        return [src, mid, tgt]



def part_three(langs):
    ru_en_model = 'Helsinki-NLP/opus-mt-ru-en'
    en_fr_model = 'Helsinki-NLP/opus-mt-en-fr'
    ru_fr_model = 'Helsinki-NLP/opus-mt-ru-fr'

    ru_en_pipeline = transformers.pipeline('translation_ru_to_en', model=ru_en_model)
    en_fr_pipeline = transformers.pipeline('translation_en_to_fr', model=en_fr_model)
    ru_fr_pipeline = transformers.pipeline('translation_ru_to_fr', model=ru_fr_model)
    st.write('\n')

    input = st.text_area('Enter input text')
    output = en_fr_pipeline(input)
    if st.button('Translate'):
        st.write(output[0]['translation_text'])


#TODO: IMPLEMENT DAISY CHAIN
#TODO: can we use huggingface pipeline? (not without having the models on huggingface, so we'll have to use with the tokenizer etc
#TODO: prepend {src-lang} {tgt-lang} to each example 
#TODO: SWAP OUT DEFAULT MODELS FOR OURS 














translation_method = part_one()
langs = part_two(translation_method)
part_three(langs=langs)





# if demo_type == 'Direct Translation':
    # language_pair = st.multiselect('Please select your source and target language', ['Russian', 'French', 'English'])
    # lp = [language_pair[0], language_pair[1]]
    # if ['Russian', 'French'] in lp:
        # in_text =  st.text_area(f'Enter {lp[0]} text here:')
        # out_text =  ru_fr_pipeline(in_text)
        # st.write(out_text)

    # if ['French', 'English'] in language_pair:
        # # fr -> en direct translation
        # pass
    # if ['English', 'Russian'] in language_pair:
        # pass




