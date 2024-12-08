import streamlit as st
import pickle

# Load the trained model and vectorizer
try:
    model = pickle.load(open('spam.pkl', 'rb'))
    cv = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please make sure 'spam.pkl' and 'vectorizer.pkl' are present.")
    st.stop()

def main():
    st.title("Email Spam Classification Application")
    st.write("This is a Machine Learning application to classify emails as spam or ham.")
    st.subheader("Classification")
    
    # User input for the email text
    user_input = st.text_area("Enter an email to classify", height=150)
    
    if st.button("Classify"):
        if user_input:
            # Transform input text using the vectorizer
            vec = cv.transform([user_input]).toarray()
            # Predict using the model
            result = model.predict(vec)
            
            # Display result
            if result[0] == 0:
                st.success("This is Not A Spam Email")
            else:
                st.error("This is A Spam Email")
        else:
            st.warning("Please enter an email to classify.")

# Run the application
if __name__ == "__main__":
    main()
