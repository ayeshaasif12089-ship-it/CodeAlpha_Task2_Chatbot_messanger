import customtkinter as ctk
import json
import os
import threading
import pygame
from gtts import gTTS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime

# --- CONFIGURATION ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class SmartChatbot(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("CodeAlpha AI Support Agent")
        self.geometry("500x700")
        self.resizable(False, True)

        # Logic Setup
        self.load_knowledge_base()
        self.train_ai()
        pygame.mixer.init()

        # UI Setup
        self.create_ui()
        self.welcome_user()

    # --- AI ENGINE ---
    def load_knowledge_base(self):
        try:
            with open('knowledge_base.json', 'r') as f:
                data = json.load(f)
                self.qa_pairs = data['questions']
        except FileNotFoundError:
            self.qa_pairs = [{"q": "hello", "a": "Error: knowledge_base.json not found."}]
        
        # Extract questions for training
        self.questions = [item['q'] for item in self.qa_pairs]

    def train_ai(self):
        # TF-IDF Vectorization (Converts text to numbers)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.questions)

    def get_best_response(self, user_input):
        # Transform user input
        user_vec = self.vectorizer.transform([user_input])
        
        # Calculate similarity
        similarities = cosine_similarity(user_vec, self.tfidf_matrix)
        
        # Get best match
        best_index = similarities.argmax()
        confidence = similarities[0][best_index]

        # Confidence Threshold (0.3 means 30% sure)
        if confidence > 0.3:
            return self.qa_pairs[best_index]['a'], round(confidence * 100)
        else:
            return "I'm not sure about that. Could you please rephrase or email support?", 0

    # --- GUI CONSTRUCTION ---
    def create_ui(self):
        # Header
        self.header = ctk.CTkFrame(self, height=60, fg_color="#1F6AA5")
        self.header.pack(fill="x")
        
        self.label_title = ctk.CTkLabel(
            self.header, 
            text="🤖 AI Support Agent", 
            font=("Roboto", 20, "bold"),
            text_color="white"
        )
        self.label_title.pack(pady=15)

        # Chat Area (Scrollable)
        self.chat_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.chat_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Input Area
        self.input_frame = ctk.CTkFrame(self, height=80, fg_color="transparent")
        self.input_frame.pack(fill="x", padx=10, pady=10)

        self.entry_msg = ctk.CTkEntry(
            self.input_frame, 
            placeholder_text="Type your question here...", 
            height=50,
            font=("Arial", 14)
        )
        self.entry_msg.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.entry_msg.bind("<Return>", self.send_message)

        self.btn_send = ctk.CTkButton(
            self.input_frame, 
            text="➤", 
            width=50, 
            height=50,
            font=("Arial", 20),
            command=self.send_message
        )
        self.btn_send.pack(side="right")

        # Voice Toggle
        self.voice_var = ctk.BooleanVar(value=False)
        self.chk_voice = ctk.CTkCheckBox(self, text="Enable Voice Response", variable=self.voice_var)
        self.chk_voice.pack(pady=5)

    def welcome_user(self):
        self.add_message("Bot", "Hello! I am your virtual assistant. Ask me anything about our services!")

    # --- CHAT LOGIC ---
    def send_message(self, event=None):
        text = self.entry_msg.get().strip()
        if not text:
            return
        
        self.entry_msg.delete(0, "end")
        self.add_message("User", text)

        # Get AI Response (Simulate slight delay for realism)
        self.after(500, lambda: self.process_bot_response(text))

    def process_bot_response(self, text):
        response, confidence = self.get_best_response(text)
        
        if confidence > 0:
            full_response = f"{response}\n(Confidence: {confidence}%)"
        else:
            full_response = response

        self.add_message("Bot", full_response)
        
        if self.voice_var.get():
            self.speak(response)

    def add_message(self, sender, text):
        # Create a bubble for the message
        bubble_frame = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        bubble_frame.pack(fill="x", pady=5)

        if sender == "User":
            # User Bubble (Right, Blue)
            bubble = ctk.CTkLabel(
                bubble_frame, 
                text=text, 
                fg_color="#3B8ED0", 
                text_color="white", 
                corner_radius=15,
                padx=15, pady=10,
                wraplength=300,
                justify="left",
                font=("Arial", 14)
            )
            bubble.pack(side="right", padx=10)
        else:
            # Bot Bubble (Left, Dark Gray)
            bubble = ctk.CTkLabel(
                bubble_frame, 
                text=text, 
                fg_color="#444444", 
                text_color="white", 
                corner_radius=15,
                padx=15, pady=10,
                wraplength=300,
                justify="left",
                font=("Arial", 14)
            )
            bubble.pack(side="left", padx=10)

        # Auto-scroll to bottom
        self.update_idletasks() # Force update
        self.chat_frame._parent_canvas.yview_moveto(1.0)

    # --- AUDIO FEATURE ---
    def speak(self, text):
        threading.Thread(target=self._speak_thread, args=(text,)).start()

    def _speak_thread(self, text):
        try:
            tts = gTTS(text=text, lang='en')
            filename = "response.mp3"
            tts.save(filename)
            
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
            # Cleanup later
        except Exception as e:
            print("Audio Error:", e)

if __name__ == "__main__":
    app = SmartChatbot()
    app.mainloop()