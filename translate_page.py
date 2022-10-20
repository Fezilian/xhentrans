import streamlit as st
from PIL import Image
from load_xh_en_model import load_xh_en_model
from keras.models import Model,load_model, model_from_json
from keras.layers import LSTM, Input, Dense,Embedding, Concatenate, TimeDistributed
from keras import backend as K 
import tensorflow as tf
import numpy as np
import random
from keras.utils import pad_sequences
import pandas



def show_trans_page():

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        image = Image.open("icon/icon2.png")
        st.image(image, width = 100)
    with col3:
        st.write(' ')

    st.title("English IsiXhosa Translator")

    pairs = (
        "IsiXhosa to English",
        "English to IsiXhosa",)

    st.write("##### Select Translation:")
    pair = st.selectbox("Select Translation:", pairs, label_visibility="collapsed")

    st.write("##### Text to Translate:")
    #source_text = st.empty()
    source_text = st.text_area('Source text', '''''', label_visibility="collapsed")

    ok = st.button("Translate")

    st.write("##### Translated Text:")

    target_text = st.empty()
    target_text.text_area("Target text", disabled = True, label_visibility="collapsed")

    [vocab_size_source, Eword2index, englishTokenizer, max_length_english, vocab_size_target, 
    Xword2index, xhosaTokenizer, max_length_xhosa, model_loaded, X_train, y_train, X_test, y_test] = load_xh_en_model()

    Eindex2word = englishTokenizer.index_word
    Xindex2word = xhosaTokenizer.index_word
    max_length_xhosa = max_length_xhosa
    max_length_english = max_length_english

    latent_dim=512
    # encoder inference
    encoder_inputs = model_loaded.input[0]  #loading encoder_inputs
    encoder_outputs, state_h, state_c = model_loaded.layers[6].output #loading encoder_outputs

    encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])

    # decoder inference
    # Below tensors will hold the states of the previous time step
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_hidden_state_input = Input(shape=(max(max_length_english,max_length_xhosa),latent_dim))

    # Get the embeddings of the decoder sequence
    decoder_inputs = model_loaded.layers[3].output

    dec_emb_layer = model_loaded.layers[5]

    dec_emb2= dec_emb_layer(decoder_inputs)

    # To predict the next word in the sequence, set the initial states to the states from the previous time step
    decoder_lstm = model_loaded.layers[7]
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

    #attention inference
    attn_layer = model_loaded.layers[8]
    attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])

    concate = model_loaded.layers[9]
    decoder_inf_concat = concate([decoder_outputs2, attn_out_inf])

    # A dense softmax layer to generate prob dist. over the target vocabulary
    decoder_dense = model_loaded.layers[10]
    decoder_outputs2 = decoder_dense(decoder_inf_concat)

    tf.keras.backend.reset_uids()

    # Final decoder model
    decoder_model = Model(
    [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2])

    if ok:
        try:
            x = random.randint(0,500)
            mystr = "ucolisiso olululo lwalufuneka"

            print("\n")
            print(source_text)
            list_mystr = source_text.split()
            list_mystr_t = xhosaTokenizer.texts_to_sequences(list_mystr)
            print(list_mystr_t)  
            df = pandas.DataFrame(list_mystr_t,columns =['tokens'])
            list_mystr_t_panda = df['tokens'].tolist()
            list_mystr_t_array = np.array(list_mystr_t_panda)
            list_mystr_t_array01 = list_mystr_t_array.reshape(1,list_mystr_t_array.shape[0])
            list_mystr_t_pad = pad_sequences(list(list_mystr_t_array01), maxlen = max(max_length_english,max_length_xhosa), padding='post')

            #target_text.text_area('Source text', f"{seq2text(X_test[x],Xindex2word)}", label_visibility="collapsed")
            #source_text.text_area('Source text', f"{seq2text(X_test[x],Xindex2word)}", label_visibility="collapsed")
            target_text.text_area("Target text",f"{decode_sequence(list_mystr_t_pad.reshape(1,max(max_length_english,max_length_xhosa)),encoder_model, decoder_model, Eword2index, Eindex2word)}", disabled = True, label_visibility="collapsed")
            #print(max_length_xhosa)
            #print(max_length_english)
        except:
            target_text.text_area("Target text","Oops! text cannot be translated", disabled = True, label_visibility="collapsed")


def decode_sequence(input_seq, encoder_model, decoder_model, Eword2index, Eindex2word):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))

    # Chose the 'start' word as the first word of the target sequence
    target_seq[0, 0] = Eword2index['start']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index == 0:
          break
        else:
          sampled_token = Eindex2word[sampled_token_index]

          if(sampled_token!='end'):
              decoded_sentence += ' '+sampled_token

              # Exit condition: either hit max length or find stop word.
              if (sampled_token == 'end' or len(decoded_sentence.split()) >= (26-1)):
                  stop_condition = True

          # Update the target sequence (of length 1).
          target_seq = np.zeros((1,1))
          target_seq[0, 0] = sampled_token_index

          # Update internal states
          e_h, e_c = h, c

    return decoded_sentence

def seq2summary(input_seq,Eword2index, Eindex2word):
    newString=''
    for i in input_seq:
      if((i!=0 and i!=Eword2index['start']) and i!=Eword2index['end']):
        newString=newString+Eindex2word[i]+' '
    return newString

def seq2text(input_seq, Xindex2word):
    newString=''
    for i in input_seq:
      if(i!=0):
        newString=newString+Xindex2word[i]+' '
    return newString


def input2token(input_seq, Xindex2word):
    newString=''
    for i in input_seq:
      if(i!=0):
        newString=newString+Xindex2word[i]+' '
    return newString