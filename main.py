# ============================================================
#   FAQ Chatbot
#   CodeAlpha AI Internship — Task 2
#   Built with Python, Tkinter, NLTK, and Cosine Similarity
# ============================================================

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import sys
import subprocess
import json
import re
import math
from datetime import datetime

# Auto-install dependencies
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
except ImportError:
    install("nltk")
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)

# ── FAQ Dataset ───────────────────────────────────────────────
FAQS = [
    {"question": "What is CodeAlpha?",
     "answer": "CodeAlpha is a leading software development company dedicated to driving innovation and excellence across emerging technologies."},
    {"question": "What internship programs does CodeAlpha offer?",
     "answer": "CodeAlpha offers internship programs in Artificial Intelligence, Web Development, Data Science, Cybersecurity, and more."},
    {"question": "How do I apply for a CodeAlpha internship?",
     "answer": "You can apply for a CodeAlpha internship by visiting www.codealpha.tech and filling out the application form on the website."},
    {"question": "What are the benefits of the CodeAlpha internship?",
     "answer": "Benefits include an Offer Letter, Completion Certificate (QR Verified), Unique ID Certificate, Letter of Recommendation, Job Placement Support, and Resume Building Support."},
    {"question": "How long is the internship?",
     "answer": "The CodeAlpha internship typically lasts 4 to 6 weeks depending on the domain and project requirements."},
    {"question": "Is the internship paid?",
     "answer": "CodeAlpha internships are primarily learning-focused. Compensation details are shared during the selection process."},
    {"question": "How many tasks do I need to complete?",
     "answer": "You need to complete a minimum of 2 or 3 tasks out of the 4 assigned. Submitting only 1 task will not qualify for the certificate."},
    {"question": "How do I submit my project?",
     "answer": "Upload your source code to GitHub in a repository named CodeAlpha_ProjectName, post a video on LinkedIn with the GitHub link, and submit via the form shared in your WhatsApp group."},
    {"question": "What is the GitHub repository naming convention?",
     "answer": "Your GitHub repository must be named: CodeAlpha_ProjectName (e.g., CodeAlpha_LanguageTranslationTool)."},
    {"question": "Do I need to post on LinkedIn?",
     "answer": "Yes! You must share your internship status on LinkedIn tagging @CodeAlpha, and also post a video explanation of your project with the GitHub repo link."},
    {"question": "Will I get a certificate?",
     "answer": "Yes, upon successful completion of minimum 2-3 tasks, you will receive a QR-verified Completion Certificate and a Unique ID Certificate."},
    {"question": "Can I get a letter of recommendation?",
     "answer": "Yes, a Letter of Recommendation is provided based on your performance during the internship."},
    {"question": "What programming languages can I use?",
     "answer": "You can use any programming language suitable for the task. Python is most commonly used for AI tasks at CodeAlpha."},
    {"question": "How do I contact CodeAlpha?",
     "answer": "You can contact CodeAlpha via: Website: www.codealpha.tech | WhatsApp: +91 9336576683 | Email: services@codealpha.tech"},
    {"question": "What AI tasks are available?",
     "answer": "The AI internship tasks are: Task 1 - Language Translation Tool, Task 2 - Chatbot for FAQs, Task 3 - Music Generation with AI, Task 4 - Object Detection and Tracking."},
    {"question": "What is the deadline for submission?",
     "answer": "The deadline is mentioned in your WhatsApp group. Make sure to submit on time to be eligible for the certificate."},
    {"question": "What tools are used for AI tasks?",
     "answer": "Common tools include Python, TensorFlow, PyTorch, OpenCV, NLTK, SpaCy, YOLO, and music21 depending on the task."},
    {"question": "Is prior experience required?",
     "answer": "Basic knowledge of Python and fundamentals of the chosen domain is helpful, but CodeAlpha provides mentorship and guidance throughout."},
    {"question": "What is object detection?",
     "answer": "Object detection is a computer vision task that identifies and locates objects in images or video frames using models like YOLO or Faster R-CNN."},
    {"question": "What is NLP?",
     "answer": "NLP (Natural Language Processing) is a branch of AI that helps computers understand, interpret, and generate human language. It is used in chatbots, translation, and sentiment analysis."},
]

# ── Colours ───────────────────────────────────────────────────
BG_DARK   = "#1a1a2e"
BG_CARD   = "#16213e"
BG_ACCENT = "#0f3460"
RED       = "#e94560"
TEAL      = "#00d4aa"
TEXT      = "#e2e2e2"
MUTED     = "#a8a8b3"
USER_CLR  = "#e94560"
BOT_CLR   = "#00d4aa"


# ─────────────────────────────────────────────────────────────
class FAQChatbot:
    """NLP engine: tokenize → stem → TF-IDF cosine similarity"""

    def __init__(self, faqs):
        self.faqs = faqs
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words("english"))
        except Exception:
            self.stop_words = set()
        self.faq_vectors = [self._vectorize(f["question"]) for f in faqs]

    def _preprocess(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
        tokens = [self.stemmer.stem(t) for t in tokens if t not in self.stop_words]
        return tokens

    def _vectorize(self, text):
        tokens = self._preprocess(text)
        vec = {}
        for t in tokens:
            vec[t] = vec.get(t, 0) + 1
        return vec

    def _cosine(self, v1, v2):
        common = set(v1) & set(v2)
        if not common:
            return 0.0
        dot = sum(v1[k] * v2[k] for k in common)
        mag1 = math.sqrt(sum(x ** 2 for x in v1.values()))
        mag2 = math.sqrt(sum(x ** 2 for x in v2.values()))
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot / (mag1 * mag2)

    def get_response(self, user_input):
        vec = self._vectorize(user_input)
        scores = [self._cosine(vec, fv) for fv in self.faq_vectors]
        best_idx = scores.index(max(scores))
        best_score = scores[best_idx]

        if best_score < 0.1:
            return (
                "I'm sorry, I couldn't find a relevant answer to your question.\n"
                "Please try rephrasing, or contact CodeAlpha directly:\n"
                "📧 services@codealpha.tech\n"
                "📱 +91 9336576683"
            )
        return self.faqs[best_idx]["answer"]


# ─────────────────────────────────────────────────────────────
class ChatApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🤖 FAQ Chatbot  |  CodeAlpha")
        self.root.geometry("820x680")
        self.root.minsize(600, 500)
        self.root.configure(bg=BG_DARK)

        self.bot = FAQChatbot(FAQS)
        self._setup_styles()
        self._build_header()
        self._build_chat()
        self._build_suggestions()
        self._build_input()
        self._build_statusbar()
        self._bot_message("👋 Hello! I'm the CodeAlpha FAQ Bot.\nAsk me anything about the internship program, tasks, submission, or certificates!")

    def _setup_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("App.TFrame",     background=BG_DARK)
        s.configure("H1.TLabel",      background=BG_DARK, foreground=RED,
                                       font=("Helvetica", 20, "bold"))
        s.configure("Sub.TLabel",     background=BG_DARK, foreground=MUTED,
                                       font=("Helvetica", 10))
        s.configure("Send.TButton",   background=RED, foreground="white",
                                       font=("Helvetica", 12, "bold"), padding=(18, 8))
        s.map("Send.TButton", background=[("active", "#c73652")])
        s.configure("Chip.TButton",   background=BG_ACCENT, foreground=MUTED,
                                       font=("Helvetica", 9), padding=(8, 4))
        s.map("Chip.TButton", background=[("active", "#1a4a7a")])

    def _build_header(self):
        f = ttk.Frame(self.root, style="App.TFrame")
        f.pack(fill="x", padx=24, pady=(18, 4))
        ttk.Label(f, text="🤖  CodeAlpha FAQ Chatbot",
                  style="H1.TLabel").pack(side="left")
        ttk.Label(f, text="NLP  •  Cosine Similarity  •  NLTK",
                  style="Sub.TLabel").pack(side="right", pady=(8, 0))
        tk.Frame(self.root, bg=RED, height=2).pack(fill="x", padx=24, pady=4)

    def _build_chat(self):
        f = ttk.Frame(self.root, style="App.TFrame")
        f.pack(fill="both", expand=True, padx=24, pady=(4, 0))

        self.chat_area = tk.Text(
            f, font=("Helvetica", 12),
            bg=BG_CARD, fg=TEXT,
            relief="flat", wrap="word",
            padx=14, pady=10,
            state="disabled", cursor="arrow",
            highlightthickness=1,
            highlightbackground=BG_ACCENT,
        )
        self.chat_area.pack(side="left", fill="both", expand=True)

        sb = ttk.Scrollbar(f, command=self.chat_area.yview)
        sb.pack(side="right", fill="y")
        self.chat_area.configure(yscrollcommand=sb.set)

        # Text tags
        self.chat_area.tag_configure("user_tag",
            foreground=USER_CLR, font=("Helvetica", 11, "bold"))
        self.chat_area.tag_configure("bot_tag",
            foreground=BOT_CLR, font=("Helvetica", 11, "bold"))
        self.chat_area.tag_configure("user_msg",
            foreground=TEXT, font=("Helvetica", 12),
            lmargin1=20, lmargin2=20)
        self.chat_area.tag_configure("bot_msg",
            foreground=TEXT, font=("Helvetica", 12),
            lmargin1=20, lmargin2=20)
        self.chat_area.tag_configure("time_tag",
            foreground=MUTED, font=("Helvetica", 8))
        self.chat_area.tag_configure("divider",
            foreground=BG_ACCENT)

    def _build_suggestions(self):
        f = ttk.Frame(self.root, style="App.TFrame")
        f.pack(fill="x", padx=24, pady=(6, 2))

        tk.Label(f, text="Quick questions:", bg=BG_DARK,
                 fg=MUTED, font=("Helvetica", 9)).pack(side="left", padx=(0, 6))

        suggestions = [
            "How to submit?",
            "What tasks are available?",
            "Will I get a certificate?",
            "How to contact CodeAlpha?",
        ]
        for s in suggestions:
            ttk.Button(f, text=s, style="Chip.TButton",
                       command=lambda q=s: self._send(q)).pack(side="left", padx=3)

    def _build_input(self):
        f = ttk.Frame(self.root, style="App.TFrame")
        f.pack(fill="x", padx=24, pady=(6, 14))

        self.input_var = tk.StringVar()
        self.input_field = tk.Entry(
            f, textvariable=self.input_var,
            font=("Helvetica", 13),
            bg=BG_CARD, fg=TEXT,
            insertbackground="white",
            relief="flat",
            highlightthickness=1,
            highlightbackground=BG_ACCENT,
            highlightcolor=RED
        )
        self.input_field.pack(side="left", fill="x", expand=True,
                               ipady=10, padx=(0, 10))
        self.input_field.bind("<Return>", lambda e: self._send())
        self.input_field.focus()

        ttk.Button(f, text="Send ➤",
                   style="Send.TButton",
                   command=self._send).pack(side="left")

    def _build_statusbar(self):
        self.status = tk.StringVar(value="  Ready — Type a question or click a suggestion")
        tk.Label(self.root, textvariable=self.status,
                 bg=BG_ACCENT, fg=MUTED,
                 font=("Helvetica", 9), anchor="w",
                 padx=12, pady=5).pack(fill="x", side="bottom")

    # ── Chat Logic ────────────────────────────────────────────
    def _send(self, text=None):
        msg = text or self.input_var.get().strip()
        if not msg:
            return
        self.input_var.set("")
        self._user_message(msg)
        self.status.set("  🤔 Thinking...")
        self.root.update_idletasks()
        threading.Thread(target=self._respond, args=(msg,), daemon=True).start()

    def _respond(self, msg):
        import time
        time.sleep(0.4)
        response = self.bot.get_response(msg)
        self.root.after(0, lambda: self._bot_message(response))
        self.root.after(0, lambda: self.status.set("  ✅ Ready — Ask another question"))

    def _user_message(self, msg):
        self.chat_area.configure(state="normal")
        t = datetime.now().strftime("%I:%M %p")
        self.chat_area.insert("end", "\n")
        self.chat_area.insert("end", f"  You  ", "user_tag")
        self.chat_area.insert("end", f"{t}\n", "time_tag")
        self.chat_area.insert("end", f"  {msg}\n", "user_msg")
        self.chat_area.configure(state="disabled")
        self.chat_area.see("end")

    def _bot_message(self, msg):
        self.chat_area.configure(state="normal")
        t = datetime.now().strftime("%I:%M %p")
        self.chat_area.insert("end", "\n")
        self.chat_area.insert("end", f"  🤖 Bot  ", "bot_tag")
        self.chat_area.insert("end", f"{t}\n", "time_tag")
        self.chat_area.insert("end", f"  {msg}\n", "bot_msg")
        self.chat_area.insert("end", "  " + "─" * 60 + "\n", "divider")
        self.chat_area.configure(state="disabled")
        self.chat_area.see("end")


# ── Entry Point ───────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()
