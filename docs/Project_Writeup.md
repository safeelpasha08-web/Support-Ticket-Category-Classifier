\# Support-Ticket Category Classifier



\## 1. Introduction



Customer support teams receive a large number of support tickets every day. Manually reading and assigning each ticket to the correct category can be time-consuming and may lead to inconsistent classification. The \*\*Support-Ticket Category Classifier\*\* project was developed to automatically classify customer support tickets into the appropriate category using Natural Language Processing (NLP) and Machine Learning.



The system takes a customer's support ticket as input and predicts its most relevant category. It also determines the urgency level of the ticket as \*\*Low, Medium, or High\*\*. A Streamlit-based web application was developed to provide an easy-to-use interface for testing new support tickets.



\## 2. Objective



The main objectives of this project are:



\* To automatically classify customer support tickets into predefined categories.

\* To apply NLP techniques for processing text-based customer queries.

\* To train and evaluate a machine learning classification model.

\* To identify the urgency level of a support ticket.

\* To develop a simple interactive web application for real-time predictions.



\## 3. Dataset



The project uses a customer support ticket dataset containing \*\*78,215 records\*\* and \*\*14 different categories\*\*.



The dataset was divided into:



\* \*\*Training data:\*\* 62,572 records

\* \*\*Testing data:\*\* 15,643 records



The categories include areas such as bank accounts, credit cards, mortgages, debt collection, student loans, vehicle loans, money transfers, and credit reporting.



Before training, the ticket text was cleaned by converting text to lowercase, removing URLs and unnecessary characters, and normalizing extra spaces.



\## 4. Methodology



The project follows a standard Natural Language Processing and Machine Learning pipeline:



\*\*Customer Ticket → Text Cleaning → TF-IDF Feature Extraction → Logistic Regression → Category Prediction\*\*



\### Text Preprocessing



The input text is cleaned before being passed to the model. The preprocessing includes:



\* Converting text to lowercase.

\* Removing URLs.

\* Removing unnecessary non-alphabetic characters.

\* Removing extra spaces.



\### TF-IDF



The cleaned ticket text is converted into numerical features using \*\*Term Frequency-Inverse Document Frequency (TF-IDF)\*\*.



TF-IDF helps the model identify words and phrases that are important for distinguishing between different support ticket categories.



\### Classification Model



A \*\*Logistic Regression\*\* classifier was used for the final classification task. The model was trained using the TF-IDF features and evaluated on a separate test dataset.



\## 5. Model Evaluation



The trained model achieved the following results:



| Metric            |     Result |

| ----------------- | ---------: |

| Accuracy          | \*\*95.49%\*\* |

| Weighted F1-score | \*\*95.61%\*\* |

| Macro F1-score    | \*\*87.33%\*\* |



The accuracy of \*\*95.49%\*\* indicates that the classifier correctly predicted the category for the majority of unseen test tickets.



The weighted F1-score of \*\*95.61%\*\* indicates strong overall classification performance while taking the number of samples in each category into account.



The macro F1-score of \*\*87.33%\*\* provides a more balanced view of performance across all categories, including categories with fewer training examples.



Overall, the results show that the combination of TF-IDF and Logistic Regression performs effectively for this customer support ticket classification problem.



\## 6. Urgency Detection



In addition to category classification, the application determines the urgency of a ticket.



Three urgency levels are used:



\* \*\*High:\*\* Used when the ticket contains critical terms such as "urgent", "fraud", "hacked", "account locked", "lost money", or "cannot access".

\* \*\*Medium:\*\* Used for terms such as "problem", "issue", "error", "delay", "trouble", or "failed".

\* \*\*Low:\*\* Assigned when no high- or medium-priority keywords are detected.



This provides support teams with an additional indication of which tickets may require faster attention.



\## 7. Demo Application



A web-based demonstration was developed using \*\*Streamlit\*\*.



The application allows a user to:



1\. Enter a customer support ticket.

2\. Submit the ticket for prediction.

3\. View the predicted support category.

4\. View the detected urgency level.

5\. View the model's prediction confidence.

6\. Review a summary of the submitted ticket and prediction.



The application makes the machine learning model accessible without requiring users to interact directly with Python code.



\## 8. Technologies Used



The main technologies used in the project are:



\* \*\*Python\*\*

\* \*\*Pandas\*\*

\* \*\*Scikit-learn\*\*

\* \*\*TF-IDF\*\*

\* \*\*Logistic Regression\*\*

\* \*\*Joblib\*\*

\* \*\*Streamlit\*\*

\* \*\*Git and GitHub\*\*



\## 9. Conclusion



The Support-Ticket Category Classifier successfully demonstrates how Natural Language Processing and Machine Learning can be used to automate customer support ticket classification.



The trained model achieved \*\*95.49% accuracy\*\* and a \*\*95.61% weighted F1-score\*\* across 14 support categories. The Streamlit application provides an interactive interface for predicting the category and urgency of new customer support tickets.



The project demonstrates practical skills in text preprocessing, feature extraction, supervised machine learning, model evaluation, and deployment of a machine learning application.



\## 10. Future Improvements



Future versions of the project could include:



\* Training with a larger and more balanced dataset.

\* Using advanced NLP models such as BERT or other transformer-based models.

\* Improving urgency detection using a trained classification model instead of keyword rules.

\* Adding ticket priority and sentiment analysis.

\* Deploying the application to a cloud platform.

\* Adding a database to store and track classified support tickets.



