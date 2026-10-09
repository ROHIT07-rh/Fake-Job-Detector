🛡️ JobShield – Fake Job Detection System
JobShield is an AI-based project that helps users identify fake job postings using Machine Learning. It analyzes job descriptions and predicts whether a job posting is real or potentially fake.

🚀 Features
- Detect potentially fake job postings
- Analyze job descriptions using Machine Learning
- Display prediction results and confidence scores
- User login and registration
- View previous prediction history
- Dashboard with statistics and analytics
- Admin functionality

🛠️ Technologies Used
Frontend
- React.js
- JavaScript
- Tailwind CSS
Backend
- Python
- FastAPI
Machine Learning
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
Database & Security
- SQLite
- SQLAlchemy
- JWT Authentication
- bcrypt password hashing

⚙️ How It Works
1. The user enters a job description.
2. The frontend sends the text to the backend.
3. TF-IDF converts the text into numerical features.
4. The Machine Learning model analyzes the features.
5. The system predicts whether the job is real or potentially fake.
6. The result is displayed on the dashboard.

📂 Project Structure
JobShield/
├── data/
├── models/
├── src/
├── frontend-react/
├── api.py
├── auth.py
├── database.py
└── README.md
Update this structure to match your actual project files.

▶️ Getting Started
1. Clone the repository
git clone YOUR_GITHUB_REPOSITORY_URL
cd JobShield
2. Set up the Python environment
python -m venv venv
Activate the environment:
Windows
venv\Scripts\activate
Linux / macOS
source venv/bin/activate
3. Install backend dependencies
pip install -r requirements.txt
4. Run the backend
uvicorn api:app --reload
The API will be available at http://127.0.0.1:8000.
FastAPI documentation: http://127.0.0.1:8000/docs
5. Run the frontend
Open a new terminal:
cd frontend-react
npm install
npm run dev
Open the local URL displayed in your terminal.
Make sure the file paths, dependencies, and startup commands match your actual project setup.

🎯 Objective
The main goal of JobShield is to help job seekers recognize potentially fraudulent job advertisements and make safer decisions while searching for jobs.
🔮 Future Improvements
- Improve prediction performance using better datasets and models
- Add more advanced text analysis
- Integrate company verification
- Improve the user interface and reporting features

⚠️ Disclaimer
JobShield provides predictions based on machine learning. The results are not a guarantee that a job posting is genuine or fraudulent. Users should independently verify employers before sharing personal information or making payments.
