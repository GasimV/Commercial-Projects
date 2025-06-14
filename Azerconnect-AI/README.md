# 🤖 Azerconnect-AI — Natural Language SQL Assistant (PoC)

This Proof-of-Concept (PoC) web app allows users to ask questions in **Azerbaijani** and receive summarized answers and charts based on structured company data — powered by **OpenAI GPT-4o** and built with **Django**.

---

## 📌 Features

- 🧠 Natural language to SQL translation (Azerbaijani)
- 📈 Results are summarized and visualized in charts
- ⚡ Real-time querying of internal data (customers, orders, reviews)
- 🧾 UI in native language for internal demos
- ✅ Lightweight, fast, no frontend frameworks required

---

## 💡 Example Queries (AZ)

- `Ən çox sifariş alan ilk 3 şəhər hansıdır?`
- `Hər bir şəhərdə sifarişlər üçün rəy balı neçədir?`
- `Hansı ödəniş tipi daha çox istifadə olunub?`

---

## 🛠 Tech Stack

| Layer     | Tech                          |
|-----------|-------------------------------|
| Backend   | Django 5.2.3                  |
| AI Model  | OpenAI GPT-4o (`openai` SDK) |
| Env Mgmt  | `python-dotenv`               |
| Frontend  | HTML/CSS + Vanilla JavaScript |
| Database  | SQLite3 (for simplicity)      |

---

## 📁 Project Structure

```
├── manage.py
├── core/
│   ├── views.py         # Handles AI logic
│   ├── models.py        # Customer, Order, Review tables
│   └── templates/
│       └── home.html    # Main UI
├── .env                 # Secret keys
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔐 Setup & Run

### 1. Clone and enter the project:
```bash
git clone https://github.com/GasimV/Commercial-Projects/Azerconnect-AI.git
cd Azerconnect-AI
```

### 2. Create and activate virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate      # On Windows
source .venv/bin/activate  # On Mac/Linux
```

### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file and add your OpenAI key:
```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

### 5. Run migrations and the dev server:
```bash
python manage.py migrate
python manage.py runserver
```

Visit 👉 `http://127.0.0.1:8000/`

---

## 🧠 How It Works

1. **User** types a question in Azerbaijani via the UI.
2. **GPT-4o** generates a SQL query based on a fixed schema.
3. Django runs the SQL against the local database.
4. The app returns:
   - Raw data
   - A short summary (in AZ)
   - A simple bar chart (HTML-based)

---

## 📦 Requirements

```txt
Django==5.2.3
openai==1.86.0
python-dotenv==1.1.0
```

---

## 📝 License

This project is for internal PoC/demo purposes at **Azerconnect**. Not intended for public release or production use.

---

## 🙋‍♀️ Author

Developed by **Gasym A. Valiyev**.
