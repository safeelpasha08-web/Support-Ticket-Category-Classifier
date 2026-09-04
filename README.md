\# 🎫 Support Ticket Category Classifier



\### 🚀 Intelligent NLP-powered customer support ticket routing



> \*\*Turn unstructured customer messages into actionable support categories — automatically.\*\*



A machine-learning application that reads customer support tickets, understands their text, predicts the most appropriate support category, and identifies the urgency level.



Built with \*\*Python • NLP • TF-IDF • Logistic Regression • Streamlit\*\*



\---



<div align="center">



\### 📊 Model Performance



|  Accuracy  | Weighted F1 |  Macro F1  | Categories |

| :--------: | :---------: | :--------: | :--------: |

| \*\*95.49%\*\* |  \*\*95.61%\*\* | \*\*87.33%\*\* |   \*\*14\*\*   |



\*\*Dataset:\*\* 78,215 customer support tickets



</div>



\---



\## 💡 Why This Project?



Imagine a company receiving \*\*thousands of support tickets every day\*\*.



A customer writes:



> \*"Someone used my card without permission and I need help immediately."\*



Instead of an employee manually reading and routing the ticket, this system automatically determines:



```text

🎫 Ticket

&#x20;     ↓

🧹 Text Cleaning

&#x20;     ↓

🔢 TF-IDF Features

&#x20;     ↓

🧠 Logistic Regression

&#x20;     ↓

📂 Category Prediction

&#x20;     ↓

🚨 Urgency Detection

&#x20;     ↓

✅ Actionable Result

```



\### Example



```text

Input:

"My credit card payment was charged twice."



Prediction:

📂 Category  → Credit card

🚨 Urgency   → Medium

📊 Confidence → High

```



\---



\# ✨ Key Features



\### 🧠 Intelligent Ticket Classification



Automatically categorizes customer support tickets into \*\*14 different financial-service categories\*\*.



\### 🚨 Smart Urgency Detection



Uses lightweight keyword-based rules to identify:



\* 🔴 \*\*HIGH\*\* — urgent or critical issues

\* 🟠 \*\*MEDIUM\*\* — problems requiring attention

\* 🟢 \*\*LOW\*\* — general requests



\### 📊 Confidence Analysis



The application displays the model's prediction confidence to provide an indication of how strongly the classifier supports its prediction.



\### 🖥️ Interactive Web Application



Built with \*\*Streamlit\*\*, allowing users to paste a ticket and receive an instant prediction.



\### ⚡ Lightweight \& Fast



Uses TF-IDF and Logistic Regression instead of computationally expensive deep-learning models, making the application easy to run locally.



\---



\# 🧠 Machine Learning Pipeline



```text

&#x20;                CUSTOMER TICKET

&#x20;                      │

&#x20;                      ▼

&#x20;             ┌─────────────────┐

&#x20;             │  Text Cleaning  │

&#x20;             │                 │

&#x20;             │ • Lowercase     │

&#x20;             │ • Remove URLs   │

&#x20;             │ • Clean symbols │

&#x20;             └────────┬────────┘

&#x20;                      │

&#x20;                      ▼

&#x20;             ┌─────────────────┐

&#x20;             │  TF-IDF         │

&#x20;             │  Vectorization  │

&#x20;             │                 │

&#x20;             │  Unigrams       │

&#x20;             │  + Bigrams      │

&#x20;             └────────┬────────┘

&#x20;                      │

&#x20;                      ▼

&#x20;             ┌─────────────────┐

&#x20;             │ Logistic        │

&#x20;             │ Regression      │

&#x20;             └────────┬────────┘

&#x20;                      │

&#x20;                      ▼

&#x20;             ┌─────────────────┐

&#x20;             │ Category        │

&#x20;             │ Prediction      │

&#x20;             └────────┬────────┘

&#x20;                      │

&#x20;                      ▼

&#x20;             ┌─────────────────┐

&#x20;             │ Urgency Rules   │

&#x20;             └────────┬────────┘

&#x20;                      │

&#x20;                      ▼

&#x20;             🎯 FINAL RESULT

```



\---



\# 📚 Dataset



The model was trained using a customer support ticket dataset containing:



| Dataset Property     |      Value |

| -------------------- | ---------: |

| Total tickets        | \*\*78,215\*\* |

| Training records     | \*\*62,572\*\* |

| Testing records      | \*\*15,643\*\* |

| Number of categories |     \*\*14\*\* |

| Train/Test split     |  \*\*80/20\*\* |



The split was performed using \*\*stratification\*\* to maintain category distribution.



\---



\# 🗂️ Supported Categories



The classifier supports the following categories:



|  # | Category                                                                     |

| -: | ---------------------------------------------------------------------------- |

| 01 | Bank account or service                                                      |

| 02 | Checking or savings account                                                  |

| 03 | Consumer Loan                                                                |

| 04 | Credit card                                                                  |

| 05 | Credit card or prepaid card                                                  |

| 06 | Credit reporting                                                             |

| 07 | Credit reporting, credit repair services, or other personal consumer reports |

| 08 | Debt collection                                                              |

| 09 | Money transfer, virtual currency, or money service                           |

| 10 | Money transfers                                                              |

| 11 | Mortgage                                                                     |

| 12 | Payday loan, title loan, or personal loan                                    |

| 13 | Student loan                                                                 |

| 14 | Vehicle loan or lease                                                        |



\---



\# 📈 Evaluation Results



The classifier was evaluated on \*\*15,643 unseen test records\*\*.



| Metric               |     Result |

| -------------------- | ---------: |

| 🎯 Accuracy          | \*\*95.49%\*\* |

| ⚖️ Weighted F1-score | \*\*95.61%\*\* |

| 📊 Macro F1-score    | \*\*87.33%\*\* |



\### What do these numbers mean?



\*\*Accuracy — 95.49%\*\*



Approximately 95 out of every 100 test tickets were assigned to the correct category.



\*\*Weighted F1 — 95.61%\*\*



Provides a balanced view of precision and recall while accounting for the different sizes of the categories.



\*\*Macro F1 — 87.33%\*\*



Calculates F1 equally across all categories and therefore provides a better indication of performance on smaller categories.



\---



\# 🚨 Urgency Intelligence



The classifier combines machine learning with a simple rule-based urgency engine.



\### 🔴 HIGH



Triggered by terms such as:



```text

urgent

immediately

emergency

critical

blocked

account locked

fraud

hacked

payment failed

cannot access

```



\### 🟠 MEDIUM



Triggered by terms such as:



```text

problem

issue

error

failed

delay

trouble

unable

please help

```



\### 🟢 LOW



Tickets without high- or medium-priority indicators are classified as \*\*LOW\*\*.



\---



\# 🖥️ Streamlit Application



The web interface provides:



```text

┌──────────────────────────────────────────┐

│       🎫 CUSTOMER SUPPORT CLASSIFIER      │

├──────────────────────────────────────────┤

│                                          │

│  Enter your support ticket:              │

│                                          │

│  "I cannot access my bank account..."    │

│                                          │

│             \[ Predict Ticket ]            │

│                                          │

├──────────────────────────────────────────┤

│                                          │

│  📂 Category        🟠 Urgency           │

│  Bank account      MEDIUM                │

│                                          │

│  📊 Confidence: 94%                      │

│                                          │

└──────────────────────────────────────────┘

```



\---



\# 🛠️ Technology Stack



| Technology             | Purpose                 |

| ---------------------- | ----------------------- |

| 🐍 Python              | Core development        |

| 🐼 Pandas              | Data processing         |

| 🔢 NumPy               | Numerical operations    |

| 🧠 Scikit-learn        | Machine learning        |

| 📝 TF-IDF              | Text feature extraction |

| 📈 Logistic Regression | Ticket classification   |

| 💾 Joblib              | Model serialization     |

| 🖥️ Streamlit          | Web application         |

| 🌱 Git                 | Version control         |

| 🐙 GitHub              | Project hosting         |



\---



\# 📁 Project Architecture



```text

Support-Ticket-Category-Classifier/

│

├── 🎫 app.py

│   └── Streamlit application

│

├── 🧠 train\_model.py

│   └── Model training \& evaluation

│

├── 📋 requirements.txt

│   └── Python dependencies

│

├── 📖 README.md

│   └── Project documentation

│

├── 🚫 .gitignore

│   └── Ignored development files

│

├── 📂 data/

│   └── tickets.csv

│

└── 🤖 model/

&#x20;   ├── customer\_support\_ticket\_classifier.pkl

&#x20;   └── tfidf\_vectorizer.pkl

```



\---



\# 🚀 Run Locally



\## 1️⃣ Clone the repository



```bash

git clone https://github.com/safeelpasha08-web/Support-Ticket-Category-Classifier.git

cd Support-Ticket-Category-Classifier

```



\## 2️⃣ Create a virtual environment



\### Windows



```bash

python -m venv venv

```



Activate it:



```bash

venv\\Scripts\\activate

```



\## 3️⃣ Install dependencies



```bash

pip install -r requirements.txt

```



\## 4️⃣ Launch the application



```bash

streamlit run app.py

```



The Streamlit application will open in your browser.



\---



\# 🧪 Example Predictions



\### Example 1 — Credit Card



```text

Input:

"My credit card payment was charged twice."



Prediction:

📂 Credit card

🚨 Medium

```



\### Example 2 — Bank Account



```text

Input:

"My bank account has an unauthorized transaction."



Prediction:

📂 Bank account or service

🚨 High

```



\### Example 3 — Mortgage



```text

Input:

"I need help with my mortgage payment."



Prediction:

📂 Mortgage

🚨 Medium

```



\---



\# 🔍 Design Decisions



\### Why TF-IDF?



TF-IDF is simple, fast, interpretable, and highly effective for traditional text classification tasks.



\### Why Logistic Regression?



Logistic Regression provides a strong baseline for high-dimensional sparse text features and works efficiently with TF-IDF representations.



\### Why rule-based urgency?



Urgency detection does not necessarily require a separate machine-learning model. A lightweight keyword-based system is easy to understand, fast, and suitable for a demonstration project.



\### Why Streamlit?



Streamlit makes it possible to convert a Python machine-learning model into an interactive web application with minimal frontend code.



\---



\# 🔮 Future Roadmap



\### Version 2.0



\* \[ ] Add transformer-based NLP models

\* \[ ] Compare Logistic Regression with SVM and Naive Bayes

\* \[ ] Add confusion matrix visualization

\* \[ ] Add per-category F1 scores

\* \[ ] Add multilingual ticket support

\* \[ ] Extract order IDs and product names

\* \[ ] Add spaCy Named Entity Recognition

\* \[ ] Store predictions in a database

\* \[ ] Add authentication

\* \[ ] Deploy the application to the cloud



\---



\# 🎓 What This Project Demonstrates



This project demonstrates practical experience with:



\*\*NLP\*\*

→ Text preprocessing

→ TF-IDF

→ Text classification



\*\*Machine Learning\*\*

→ Train/test split

→ Logistic Regression

→ Accuracy

→ F1-score

→ Model serialization



\*\*Application Development\*\*

→ Streamlit

→ Interactive prediction

→ Confidence visualization

→ Rule-based urgency detection



\*\*Software Engineering\*\*

→ Project structure

→ Requirements management

→ Git

→ GitHub

→ Documentation



\---



\# 🌟 Project Highlights



> \*\*78,215\*\* tickets processed



> \*\*14\*\* support categories



> \*\*95.49%\*\* accuracy



> \*\*95.61%\*\* weighted F1-score



> \*\*TF-IDF + Logistic Regression\*\*



> \*\*Interactive Streamlit application\*\*



> \*\*Automated urgency detection\*\*



\---



\## 👨‍💻 Author



\### Safeel Pasha



\*\*B.Tech | Machine Learning | NLP | Python\*\*



\---



<div align="center">



\### ⭐ If you find this project interesting, consider giving it a star!



\*\*Built with Python \& Machine Learning\*\*



</div>



