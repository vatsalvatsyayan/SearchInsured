# 🏥 SearchInsured — AI-Powered Health Insurance Assistant

🎥 **[Watch the Demo Presentation](https://drive.google.com/file/d/1YJSw9R9odhGJLk5V6SV51_B49kiOUUQ9/view?usp=sharing)**

---

**SearchInsured** is an intelligent, conversational platform that simplifies how users understand their health insurance coverage and find in-network healthcare providers.  
It turns complex insurance data and provider directories into an easy-to-use chat experience powered by FastAPI and a lightweight frontend.

---

## 🚀 Overview

Health insurance portals are often confusing — users must log in, navigate multiple menus, and read through dense plan documents to find which doctors are covered.

**SearchInsured** makes this process as easy as chatting.

You can ask:
- “Find cardiologists in Los Angeles”
- “Does my plan cover emergency care?”
- “What’s my deductible?”

The system understands your query, analyzes intent, calls the right backend tools (like provider search or plan coverage analysis), and responds instantly in plain English.

---

## 🧠 Key Features

✅ **Natural Language Interaction**  
Ask insurance-related questions conversationally — no complex menus.  

✅ **Provider Search**  
Find doctors by specialty and location with structured, readable results.  

✅ **Smart Reasoning (Agent Mode)**  
The `/ask/agent` endpoint uses reasoning steps and maintains session context across questions.  

✅ **Lightweight Frontend**  
A clean HTML/CSS/JS interface that runs directly in the browser — no frameworks required.  

✅ **FastAPI Backend**  
Handles user requests, invokes tools, and structures responses in JSON.  

✅ **Future Partnerships**  
Currently, most insurance provider APIs require registration or paid access.  
Our next step is to **partner with insurance companies** to access official APIs and deliver even more relevant, personalized results.

---

## 🧩 Architecture

![High Level Diagram](high%20level%20diagram.png)

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-------------|
| **Frontend** | HTML, CSS, JavaScript |
| **Backend** | Python (FastAPI) |
| **Embeddings / RAG** | FAISS, Sentence Transformers |
| **APIs / Data** | FHIR-based provider directories, insurance datasets |
| **Hosting** | Local / deployable via Render, AWS, or GCP |

---

## 📈 Future Enhancements

- 🤝 Partner with insurers for official API access  
- 🧾 Integrate real plan PDF embeddings (via FAISS)  
- 🧍 Add user accounts and saved searches  
- 🌗 Improve UI with dark mode and chat enhancements  

---

## 👥 Contributors

| Name | Contact |
|------|----------|
| **Vatsal Vatsyayan** | vatsyaya@usc.edu |
| **Sam Koog** | koog@usc.edu |

---

## 🏁 Acknowledgements

Built as part of a **Hackathon project at the University of Southern California (USC)**.  
Special thanks to open-source libraries and APIs that made this possible.

---

### 📜 License
This project is released for educational and hackathon demonstration purposes.  
