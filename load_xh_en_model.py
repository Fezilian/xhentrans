import streamlit as st
import pickle as pkl
import numpy as np
from keras.models import Model,load_model, model_from_json
from AttentionLayer import AttentionLayer


@st.cache(allow_output_mutation=True)
def load_xh_en_model():
    with open('tokenizer/xh_en/NMT_Entokenizer.pkl','rb') as f:
        vocab_size_source, Eword2index, englishTokenizer, max_length_english = pkl.load(f)

    with open('tokenizer/xh_en/NMT_Xhtokenizer.pkl', 'rb') as f:
        vocab_size_target, Xword2index, xhosaTokenizer, max_length_xhosa = pkl.load(f)

    with open('tokenizer/xh_en/NMT_data.pkl','rb') as f:
        X_train, y_train, X_test, y_test = pkl.load(f)

    # loading the model architecture and asigning the weights
    json_file = open('models/xh_en/NMT_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model_loaded = model_from_json(loaded_model_json, custom_objects={'AttentionLayer': AttentionLayer})
    # load weights into new model
    model_loaded.load_weights("models/xh_en/NMT_model_weight.h5")

    return [vocab_size_source, Eword2index, englishTokenizer, max_length_english, vocab_size_target, Xword2index, xhosaTokenizer, 
    max_length_xhosa, model_loaded, X_train, y_train, X_test, y_test];

    


