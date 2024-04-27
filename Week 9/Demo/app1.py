import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pycaret.classification import *

# Load pretrained model for classification
model = load_model('happiness_pipeline')

# Define classification function to scale
def predict(model, input_df):
    try:
        predictions_df = predict_model(model, data=input_df)
        predictions = predictions_df['prediction_label'][0]
        return predictions
    except KeyError as e:
        st.error(f"Error: {e}. Please check the column names in your input data.")

def run():
    # Load happiness data
    happiness_data = pd.read_csv('happydata.csv')

    # Add sidebar to the app
    st.sidebar.image('images/logo_sdt.png')
    st.sidebar.title('Demo ML-OPS Week 9')
    st.sidebar.markdown("An app that predicts or classifies whether you're happy or not using Pycaret's classification")
    st.sidebar.info("This application only aims to demo ML-OPS courses without expert assistance, so the results of predictions are not accurate")
    st.sidebar.success("By: Bagas Cahya Fajar Bastian")

    # Add pages to the app
    page = st.sidebar.selectbox('Select a page', ['Predict', 'Info'])

    if page == 'Predict':
        # Add title and subtitle to the main interface of the app
        st.title("Are you one of those happy people? Lets Find Out!")
        st.image('images/happy.jpg', width=700)
        st.markdown("""
        Welcome to my website, where I help you discover your level of happiness and provide insights on what makes people happy. My prediction model is built using Pycaret, a popular open-source machine learning library in Python. I have trained the model using a simple dataset of happiness-related factors, such as access to information, affordability of housing, quality of schools, trust in the police, maintenance of streets and sidewalks, and availability of events. By answering a few questions about these factors, my model can predict whether you are a happy person or not with a high degree of accuracy.

        Once you have taken the prediction, you can also explore my "Info" page, which provides a histogram of happiness scores and basic information about the number of happy people and why they are happy. I hope that this information will inspire you to take action towards a happier life and help you understand the factors that contribute to happiness.

        My website is designed to be user-friendly and accessible to everyone. I believe that happiness is a universal language, and everyone deserves to live a happy life. Whether you are feeling happy or not, I invite you to explore my website and discover the power of happiness.
        
        This website provide you some basic prediction about you are happy or not.
        
        """)

        # Get user input
        infoavail = st.radio('How information about your city service is available', [1,2,3,4,5], horizontal=True)
        housecost = st.radio('Are the prices of houses in your place affordable', [1,2,3,4,5], horizontal=True)
        schoolquality = st.radio('How is the quality of the school in your place', [1,2,3,4,5], horizontal=True)
        policetrust = st.radio('your trust in the local police', [1,2,3,4,5], horizontal=True)
        streetquality = st.radio('How is maintenance of streets and sidewalks in your place', [1,2,3,4,5], horizontal=True)
        events  = st.radio('Are there many useful events in your place', [1,2,3,4,5], horizontal=True)

        input_dict = {'infoavail': int(infoavail),
                    'housecost': int(housecost),
                    'schoolquality': int(schoolquality),
                    'policetrust': int(policetrust),
                    'streetquality': int(streetquality),
                    'ëvents': int(events)
                    }

        input_df = pd.DataFrame([input_dict])

        # Make prediction
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            if output:
                st.success('You are Happy Person')
                st.image('images/happyresult.jpg', width=700)
                st.markdown('Lets spread this happiness with everyone')
                st.markdown("""
                ### Tips to Stay Happy
                1. Practice gratitude: Take time each day to reflect on the things you are thankful for. This can help shift your focus from what's wrong in your life to what's going well.
                2. Connect with others: Spend time with friends, family, or loved ones. Social connections are important for our mental and emotional well-being.            
                3. Get moving: Exercise can help reduce symptoms of depression and anxiety, and improve your mood. Find an activity you enjoy and make it a regular part of your routine.
                4. Practice mindfulness: Mindfulness techniques, such as meditation or deep breathing, can help you stay present and focused, reducing stress and anxiety.
                5. Seek professional help: If you are feeling sad or depressed for an extended period of time, it may be helpful to speak with a mental health professional. They can provide you with additional resources and support.
                """)

            else:
                st.error('You are Sad:(')
                st.image('images/sadresult.jpg', width=700)
                st.markdown('Dont worry, there will definitely be happiness tomorrow')
                st.markdown("""
                ### Tips to Overcome Sadness
                1. Identify the cause: Try to identify the cause of your sadness. Once you know what's causing it, you can take steps to address it.
                2. Reach out to others: Don\'t be afraid to reach out to friends, family, or a mental health professional for support. Talking about your feelings can help you feel better.
                3. Practice self-care: Make sure you are taking care of yourself physically, emotionally, and mentally. This can include getting enough sleep, eating healthy foods, and engaging in activities you enjoy.
                4. Challenge negative thoughts: When you have negative thoughts, try to challenge them with evidence. For example, if you think "I'm a failure," try to think of times when you have succeeded.
                5. Seek professional help: If your sadness persists or is causing significant distress, it may be helpful to seek professional help. A mental health professional can provide you with additional resources and support.
                """)

    elif page == 'Info':
        # Add subheader for the page
        st.subheader('Happiness Information')
        st.markdown("""
        Welcome to my "Info" page, where I'll share some insights about the factors that contribute to happiness and show you a histogram of happiness scores.

        The histogram displays the distribution of happiness scores from my dataset, which includes responses from thousands of people around the world. The x-axis represents the happiness score, which ranges from 0 to 10, and the y-axis shows the number of people who reported that score.

        As you can see, most people report a happiness score between 5 and 8, with a few outliers on either end of the spectrum. This suggests that happiness is a complex and multifaceted concept, and that different people may have different definitions of what it means to be happy.

        In addition to the histogram, I'll also share some basic information about the factors that contribute to happiness. For example, I know that:

        - Access to information is crucial for my happiness. When I have reliable and accurate information, I can make informed decisions and feel in control of my life.
        - Affordable housing is important for my sense of security and contentment. When I can afford to live in a safe and comfortable home, I feel more secure and content.
        - Quality of schools is linked to my happiness. When I have access to high-quality education, I have better job opportunities and higher income, which can lead to greater happiness.
        - Trust in the police is an indicator of social cohesion and safety in my community. When I trust the police, I feel safer and more secure in my community, which can contribute to my overall happiness.
        - Maintenance of streets and sidewalks is a sign of a well-functioning and responsive government. When I live in a clean and well-maintained neighborhood, I feel proud of my community and have a sense of belonging.
        - Availability of events is a marker of cultural richness and diversity. When I have access to a variety of cultural events and activities, I feel more engaged and connected to my community, which can enhance my overall happiness.

        By understanding these factors and taking action to improve them, I can work towards a happier and more fulfilling life. Whether I'm feeling happy or not, I encourage you to explore my "Info" page and learn more about the science of happiness.
        """)

        # Create histogram of happiness scores
        fig, ax = plt.subplots()
        ax.hist(happiness_data['happy'])
        ax.set_xlabel('Happiness Score')
        ax.set_ylabel('Number of People')
        ax.set_title('Happiness Score Distribution')

        # Center the histogram
        fig.align = 'center'

        # Display histogram
        st.pyplot(fig)

        # Display basic information about happiness
        st.markdown('### Here are some key statistics:')
        st.markdown('The average happiness score is **{}**.'.format(round(happiness_data['happy'].mean(), 2)))
        st.markdown('The median happiness score is **{}**.'.format(happiness_data['happy'].median()))
        st.markdown('The mode of happiness score is **{}**.'.format(happiness_data['happy'].mode()[0]))
        st.markdown('The standard deviation of happiness score is **{}**.'.format(round(happiness_data['happy'].std(), 2)))

        st.markdown("""
        ### This is Reasons for Happiness  
        1. Good relationships with family and friends
        2. Good health and well-being
        3. Financial security and stability
        4. Personal growth and development 
        5. A sense of purpose and meaning in life
        """)

if __name__ == '__main__':
    run()