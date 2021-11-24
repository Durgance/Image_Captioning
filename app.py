import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences



def getImage(file_path):
    test_img=Image.open(file_path)
    test_img=test_img.resize((224,224))
    #test_img=tf.keras.preprocessing.image.img_to_array(test_img)/255
    test_img=np.expand_dims(test_img,axis=0)
    return test_img

def get_sent(model,tokenizer,test_feature):
    text_inp=["start"]
    count=0
    caption=" "
    while count<25:
        count+=1
        encoded=[]
        for i in text_inp:
            encoded.append(tokenizer.word_docs[i])
        encoded=[encoded]
        encoded=pad_sequences(encoded,padding="post",truncating="post",maxlen=40)
        
        prediction=np.argmax(model.predict([test_feature,encoded]))
        
        sampled_word=tokenizer.index_word[prediction]
        caption=caption+" "+ sampled_word
        if sampled_word=="end":
            break
        text_inp.append(sampled_word)
    
    return text_inp

def main():
    # Creating the title of the page. 
    #st.image("./img.jpg",use_column_width=True)


    st.title("Image Captioning")
    st.subheader("Please upload the Image to be captioned : ")
    image_file=st.file_uploader("Upload Image",
                                type=["png","jpg","jpeg"])
    if st.button("Process"):
        img=getImage(image_file)
        st.image(img,caption="Uploaded Image")
        
        #img=tf.keras.preprocessing.image.img_to_array(img)
        
        # loading the tokenizer

        with open('./results/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        # loading the First model and predicting.

        model_vgg=tf.keras.models.load_model("model_vgg16.h5")
        test_feature=model_vgg.predict(img)
        st.write(test_feature)
        
        model=tf.keras.models.load_model("./results/best_model_acc.h5")
        
        get_sent(model,tokenizer,test_feature)
        
        
        # Loading the second model ot predict the sentence part.
        


        #print(img)
        #img=prepare(img)
        #print(img)
        #st.subheader(prediction_cls(model.predict(img)))

    pass

if __name__=="__main__":
    main()