import streamlit as st
import pickle as pkl
import numpy as np
from keras.models import Model,load_model, model_from_json
from AttentionLayer import AttentionLayer
from urllib.request import urlopen
from keras.utils.data_utils import get_file
import urllib.request

@st.cache(allow_output_mutation=True)
def load_xh_en_model():

    with urllib.request.urlopen("https://xhentrans.s3.amazonaws.com/tokenizer/xh_en/xh_en_Entokenizer.pkl") as f:
        vocab_size_source, Eword2index, englishTokenizer, max_length_english = pkl.load(f)

    with urllib.request.urlopen("https://xhentrans.s3.amazonaws.com/tokenizer/xh_en/xh_en_Xhtokenizer.pkl") as f:
        vocab_size_target, Xword2index, xhosaTokenizer, max_length_xhosa = pkl.load(f)

    # loading the model architecture and asigning the weights
    json_file = urlopen("https://xhentrans.s3.amazonaws.com/models/xh_en/xh_en_model.json")
    loaded_model_json = json_file.read()
    json_file.close()
    model_loaded = model_from_json(loaded_model_json, custom_objects={'AttentionLayer': AttentionLayer})
    # load weights into new model
    weights_path = get_file(
            'weights',
            'https://xhentrans.s3.amazonaws.com/models/xh_en/xh_en_model_weight.h5')
    model_loaded.load_weights(weights_path)

    return [vocab_size_source, Eword2index, englishTokenizer, max_length_english, vocab_size_target, Xword2index, xhosaTokenizer, 
    max_length_xhosa, model_loaded, X_train, y_train, X_test, y_test];
    
@st.cache(allow_output_mutation=True)
def load_en_xh_model():

    with urllib.request.urlopen("https://xhentrans.s3.amazonaws.com/tokenizer/en_xh/en_xh_Entokenizer.pkl") as f:
        vocab_size_source, Eword2index, englishTokenizer, max_length_english = pkl.load(f)

    with urllib.request.urlopen("https://xhentrans.s3.amazonaws.com/tokenizer/en_xh/en_xh_Entokenizer.pkl") as f:
        vocab_size_target, Xword2index, xhosaTokenizer, max_length_xhosa = pkl.load(f)

    # loading the model architecture and asigning the weights
    json_file = urlopen("https://xhentrans.s3.amazonaws.com/models/en_xh/en_xh_model.json")
    loaded_model_json = json_file.read()
    json_file.close()
    model_loaded = model_from_json(loaded_model_json, custom_objects={'AttentionLayer': AttentionLayer})
    # load weights into new model
    weights_path = get_file(
            'weights',
            'https://xhentrans.s3.amazonaws.com/models/en_xh/en_xh_model_weight.h5')
    model_loaded.load_weights(weights_path)

    return [vocab_size_source, Eword2index, englishTokenizer, max_length_english, vocab_size_target, Xword2index, xhosaTokenizer, 
    max_length_xhosa, model_loaded, X_train, y_train, X_test, y_test];