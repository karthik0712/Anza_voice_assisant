import json
import os
import streamlit as st
import datetime
import random
import string
import re
import smtplib
import hashlib
from twilio.rest import Client
from openai import OpenAI
import streamlit as st
import speech_recognition as sr
import Levenshtein
from beepy import beep
from playsound import playsound
from dotenv import load_dotenv
from gtts import gTTS
from pathlib import Path
import tensorflow as tf
import numpy as np
import geocoder
import cv2
import mediapipe as mp
import requests
import datetime


load_dotenv()
session_state = st.session_state
if "user_index" not in st.session_state:
    st.session_state["user_index"] = 0

st.set_page_config(
    page_title="Voice Assistant",
    page_icon="favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

preferred_contact = {
    "Name": "Ashlin",
    "Email": "ashlindivinesha2003@gmail.com",
    "Phone": "+917904432084"}

contacts = {
            "Mom": ["+917904432084", "ashlindivinesha2003@gmail.com"],
            "Dad": ["+917904432084", "dad@mail.com"],
            "Ashlin": ["+917904432084", "ashlindivinesha2003@gmail.com"],
        }

def detectSign(image):
    image = cv2.imread(image)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    model = tf.keras.models.load_model("model.h5")
    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    ) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return None
        coordinates = np.array(
            [[res.x, res.y] for res in results.multi_hand_landmarks[0].landmark]
        ).flatten()
        X = coordinates.reshape(1, 42)
        y = model.predict(X)
        classes = ["Zero","One", "Two", "Three", "Four", "Five"]
        y_pred = [classes[i] for i in np.argmax(y, axis=1)]
        return y_pred[0]
    
def perform_task(sign):
    if sign == "Zero":
        message = f"""
            1. Increased flexibility
            2. Increased muscle strength and tone
            3. Improved respiration, energy and vitality
            4. Maintaining a balanced metabolism
            5. Weight reduction
            6. Cardio and circulatory health
            7. Improved athletic performance
            8. Protection from injury
        """
        st.markdown(f"# Benefits of doing Yoga:")
        st.markdown(message)
        play("Benefits of doing Yoga are as follows:")
        play(message)
    elif sign == "One":
        g = geocoder.ip('me')
        if g.latlng is not None:
            latitude, longitude = g.latlng
            play(f"Your current latitude is {latitude} and longitude is {longitude}")
            play(f"Sending email to {preferred_contact['Name']} with your current location")
            message =f"Help me! I am in danger. My current location is Latitude: {latitude}, Longitude: {longitude}"
            send_email(preferred_contact["Email"], message)
            sendSMS(preferred_contact["Phone"], message)
            st.markdown(f"Your current latitude is {latitude} and longitude is {longitude}")
            play("Email and SMS sent successfully!")
        else:
            play("Could not retrieve your location. Please try again.")
        
    elif sign == "Two":
        play("Booking Doctor's appointment")
        play("Doctor's appointment booked successfully!")
    elif sign == "Three":
        daily_remaineders = f"""
            1. Attend the meeting at 10:00 AM
            2. Pay electricity bill
            3. Call Mom
            4. Take medicine at 5:00 PM
            5. Have dinner at 8:00 PM
            6. Go to bed at 10:00 PM
        """
        st.markdown(f"# Your daily reminders are as follows:")
        st.markdown(daily_remaineders)
        play("Your daily reminders are as follows:")
        play(daily_remaineders)

    elif sign == "Four":
        play("Getting current News")
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        url = (f'https://newsapi.org/v2/everything?'
       'q=India&'
       'from={current_date}&'
       'sortBy=popularity&'
       'apiKey=d74f7a3fd21a4ecb8bbdb98f7126a07f')

        response = requests.get(url)

        news = f""
        for article in response.json()['articles']:
            news += f"{article['title']}\n{article['description']}\n\n"
        
        if len(news) > 0:
            st.markdown(f"# News Headlines for {current_date}")
            st.markdown(news)
            play("Playing news headlines")
            play(news)
        else:
            play("No news found")

    elif sign == "Five":
        play("Opening Maps")
        st.markdown("Opening Maps")
        g = geocoder.ip('me')
        if g.latlng is not None:
            latitude, longitude = g.latlng
            dataframe = {
                "LATITUDE": [latitude],
                "LONGITUDE": [longitude],
            }
            st.map(dataframe)
    else:
        play("No task found for the detected sign")

def play(text):
    try:
        speech = gTTS(text = text,lang='en', slow = False)
        speech.save('audio.mp3')
        audio_file = Path().cwd() /   'audio.mp3'
        playsound(audio_file)
        if os.path.exists('audio.mp3'):
            os.remove('audio.mp3')
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def voice_assistant(voice_command):
    try:
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        
        tasks = ['Email', 'SMS']
        message = ""
        receiver = ""


        prompt = f"""
            You're a voice assistant that can help with various tasks. Your task is to identify the intent of the user's voice command and segment the command into the following categories:
            
            1. Task: {tasks}
            2. Message
            3. Receiver
            
            Voice Command: "{voice_command}"
            
            You should return the segmented categories separated by commas.
            
            For example:
            - "Send an email to John saying hello" should return: Email,hello,John
            - "Send a message to +917904432084 saying hi" should return: SMS, hi,+917904432084
            
            If the voice command is not clear or does not match any of the categories, return an empty string.
        """

        messages = [{"role": "system", "content": prompt}]
        response = client.chat.completions.create(
            messages=messages,
            model="gpt-3.5-turbo-0125",
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def checkSimilarity(phrase, wakeup_phrase):
    distance = Levenshtein.distance(phrase, wakeup_phrase)
    max_length = max(len(phrase), len(wakeup_phrase))
    similarity_ratio = 1 - (distance / max_length)
    return similarity_ratio > 0.3


# Function to listen for a wakeup phrase
def listen(wakeup_phrase):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        st.text("Listening for wakeup phrase...")
        audio = recognizer.listen(source)
        
    try:
        phrase = recognizer.recognize_google(audio)
        print(phrase)
        # check if the output phrase is the more than 50% similar to the wakeup phrase
        
        if phrase.lower() == wakeup_phrase.lower() or checkSimilarity(phrase, wakeup_phrase):
            st.text("Wakeup phrase recognized!")
            return True
        else:
            st.text("Wakeup phrase not recognized. Listening again...")
            return False
    except sr.UnknownValueError:
        st.text("Could not understand audio. Listening again...")
        return False
    except sr.RequestError as e:
        st.text(f"Error: {e}")
        return False
    
def send_email(receiver, message):
    SENDER_MAIL_ID = os.getenv("SENDER_MAIL_ID")
    APP_PASSWORD = os.getenv("APP_PASSWORD")
    RECEIVER = receiver
    for contact in contacts:
        if checkSimilarity(receiver, contact):
            RECEIVER = contacts[contact][1]
            break
    server = smtplib.SMTP_SSL("smtp.googlemail.com", 465)
    server.login(SENDER_MAIL_ID, APP_PASSWORD)
    message = f"Subject: Voice Assistant Email\n\n{message}"
    server.sendmail(SENDER_MAIL_ID, RECEIVER, message)
    server.quit()
    st.success("Email sent successfully!")
    play("Email sent successfully!")
    return True

def sendSMS(receiver, message):
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")
    to_number = receiver
    for contact in contacts:
        if checkSimilarity(receiver, contact):
            to_number = contacts[contact][0]
            break
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body=message,
        from_=from_number,
        to=to_number
    )
    st.success("SMS sent successfully!")
    play("SMS sent successfully!")
    return True


def user_exists(email, json_file_path):
    # Function to check if user with the given email exists
    with open(json_file_path, "r") as file:
        users = json.load(file)
        for user in users["users"]:
            if user["email"] == email:
                return True
    return False

def send_verification_code(email, code):
    SENDER_MAIL_ID = os.getenv("SENDER_MAIL_ID")
    APP_PASSWORD = os.getenv("APP_PASSWORD")
    RECEIVER = email
    server = smtplib.SMTP_SSL("smtp.googlemail.com", 465)
    server.login(SENDER_MAIL_ID, APP_PASSWORD)
    message = f"Subject: Your Verification Code\n\nYour verification code is: {code}"
    server.sendmail(SENDER_MAIL_ID, RECEIVER, message)
    server.quit()
    st.success("Email sent successfully!")
    return True


def generate_verification_code(length=6):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))

def signup(json_file_path="data.json"):
    st.title("Signup Page")
    with st.form("signup_form"):
        st.write("Fill in the details below to create an account:")
        name = st.text_input("Name:")
        email = st.text_input("Email:")
        age = st.number_input("Age:", min_value=0, max_value=120)
        sex = st.radio("Sex:", ("Male", "Female", "Other"))
        password = st.text_input("Password:", type="password")
        confirm_password = st.text_input("Confirm Password:", type="password")
        if (
            session_state.get("verification_code_eval") is None
            or session_state.get("verification_time_eval") is None
            or datetime.datetime.now() - session_state.get("verification_time_eval")
            > datetime.timedelta(minutes=5)
        ):
            verification_code = generate_verification_code()
            session_state["verification_code_eval"] = verification_code
            session_state["verification_time_eval"] = datetime.datetime.now()
        if st.form_submit_button("Signup"):
            if not name:
                st.error("Name field cannot be empty.")
            elif not email:
                st.error("Email field cannot be empty.")
            elif not re.match(r"^[\w\.-]+@[\w\.-]+$", email):
                st.error("Invalid email format. Please enter a valid email address.")
            elif user_exists(email, json_file_path):
                st.error(
                    "User with this email already exists. Please choose a different email."
                )
            elif not age:
                st.error("Age field cannot be empty.")
            elif not password or len(password) < 6:  # Minimum password length of 6
                st.error("Password must be at least 6 characters long.")
            elif password != confirm_password:
                st.error("Passwords do not match. Please try again.")
            else:
                verification_code = session_state["verification_code_eval"]
                send_verification_code(email, verification_code)
                entered_code = st.text_input(
                    "Enter the verification code sent to your email:"
                )
                if entered_code == verification_code:
                    user = create_account(
                        name, email, age, sex, password, json_file_path
                    )
                    session_state["logged_in"] = True
                    session_state["user_info"] = user
                    st.success("Signup successful. You are now logged in!")
                    
                elif len(entered_code) == 6 and entered_code != verification_code:
                    st.error("Incorrect verification code. Please try again.")

def check_login(username, password, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)


        for user in data["users"]:
            if user["email"] == username and user["password"] == password:
                session_state["logged_in"] = True
                session_state["user_info"] = user
                st.success("Login successful!")
    
                return user
        return None
    except Exception as e:
        st.error(f"Error checking login: {e}")
        return None

def initialize_database(
    json_file_path="data.json"
):
    try:
        if not os.path.exists(json_file_path):
            data = {"users": []}
            with open(json_file_path, "w") as json_file:
                json.dump(data, json_file)

        
    except Exception as e:
        print(f"Error initializing database: {e}")

def create_account(name, email, age, sex, password, json_file_path="data.json"):
    try:
        if not os.path.exists(json_file_path) or os.stat(json_file_path).st_size == 0:
            data = {"users": []}
        else:
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)

        # Append new user data to the JSON structure
        email = email.lower()
        password = hashlib.md5(password.encode()).hexdigest()
        user_info = {
            "name": name,
            "email": email,
            "age": age,
            "sex": sex,
            "password": password,
        }

        data["users"].append(user_info)

        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        st.success("Account created successfully! You can now login.")
        return user_info
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        st.error(f"Error creating account: {e}")
        return None

def login(json_file_path="data.json"):
    st.title("Login Page")
    username = st.text_input("Email:")
    password = st.text_input("Password:", type="password")
    password = hashlib.md5(password.encode()).hexdigest()
    username = username.lower()

    login_button = st.button("Login")

    if login_button:
        user = check_login(username, password, json_file_path)
        if user is not None:
            session_state["logged_in"] = True
            session_state["user_info"] = user
        else:
            st.error("Invalid credentials. Please try again.")

def get_user_info(email, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            for user in data["users"]:
                if user["email"] == email:
                    return user
        return None
    except Exception as e:
        st.error(f"Error getting user information: {e}")
        return None

def render_dashboard(user_info, json_file_path="data.json"):
    try:
        st.title(f"Welcome to the Dashboard, {user_info['name']}!")
        st.subheader("User Information:")
        st.write(f"Name: {user_info['name']}")
        st.write(f"Sex: {user_info['sex']}")
        st.write(f"Age: {user_info['age']}")
        
        st.image("https://www.stevenvanbelleghem.com/content/uploads/2023/11/19IcqVZ48A0tQba1-F_yIpg-820x540.jpeg", use_column_width=True)
        
    except Exception as e:
        st.error(f"Error rendering dashboard: {e}")
    
    
def main():
    st.title("Voice Assistant")
    page = st.sidebar.radio(
        "Go to",
        (
            "Signup",
            "Login",
            "Dashboard",
            "Voice Assistant",
            "Sign Detection",
        ),
        key="page",
    )

    if page == "Signup":
        signup()
    if page == "Login":
        login()

    elif page == "Dashboard":
        if session_state.get("logged_in"):
            render_dashboard(session_state["user_info"])
        else:
            st.warning("Please login/signup to view the dashboard.")

    elif page == "Voice Assistant":
        if session_state.get("logged_in"):
            st.title("Give a voice command:")
            wakeup_phrase = "Hey Anza"
            while True:
                if listen(wakeup_phrase):
                    
                    beep(sound='ping')
                    play("  Recording...")
                    break
            while True:
                recognizer = sr.Recognizer()
                microphone = sr.Microphone()
                with microphone as source:
                    recognizer.adjust_for_ambient_noise(source)
                    audio = recognizer.listen(source)
                    
                try:
                    voice_command = recognizer.recognize_google(audio)
                    print(voice_command)
                    with st.spinner("Processing..."):
                        task = voice_assistant(voice_command).split(",")
                    if task[0] == "Email":
                        play("Sending email to " + task[2] + " saying " + task[1])
                        send_email(task[2], task[1])
                        break
                    elif task[0] == "SMS":
                        play("Sending SMS to " + task[2] + " saying " + task[1])
                        sendSMS(task[2], task[1])
                        break
                except sr.UnknownValueError:
                    play("Could not understand audio. Listening again...")
                
        else:
            st.warning("Please login/signup to access this page.")
    elif page == "Sign Detection":
        st.title("Sign Detection")
        st.subheader("SIGNS:")
        st.image("https://media.gettyimages.com/id/1338456617/vector/fingers-for-teaching-early-counting-in-children-education-stock-illustration-hands-body.jpg?s=612x612&w=0&k=20&c=9QaO2NDs1BChM_zY4iFrL_Rh4OpxIE615NR0vQNY9ds=")
        st.markdown("0. Zero: Benefits of doing Yoga")
        st.markdown("1. One: Send your current location to a preferred contact")
        st.markdown("2. Two: Book a Doctor's appointment")
        st.markdown("3. Three: View your daily reminders")
        st.markdown("4. Four: Get current News")
        st.markdown("5. Five: View your current location on Maps")
        
        image = st.camera_input("Capture the sign:")
        if image is not None:
            # save the image
            with open("image.jpg", "wb") as file:
                file.write(image.read())
            sign = detectSign("image.jpg")
            if os.path.exists("image.jpg"):
                os.remove("image.jpg")
            if sign is not None:
                st.success(f"Sign detected: {sign}")
                play(f"Sign detected: {sign}")
                perform_task(sign)
            else:
                st.error("No sign detected.")
        else:
            st.warning("Please capture an image to detect the sign.")


if __name__ == "__main__":
    initialize_database()
    main()
