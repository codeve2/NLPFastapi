
# NLP Text Classification API

This project is a simple natural language processing (NLP) system for classifying text messages into three categories: normal, spam, and question. The model is built using scikit-learn and deployed باستخدام FastAPI and Docker to provide a ready-to-use API.

---

## Features

* Text classification using TF-IDF and Logistic Regression
* Detects whether a message is normal, spam, or a question
* Lightweight and fast inference
* REST API built with FastAPI
* Docker support for easy deployment

---

## Model Details

The model is implemented using a scikit-learn pipeline:

* TF-IDF Vectorizer

  * ngram range: (1, 2)
  * max features: 5000
  * English stop words removal

* Logistic Regression

  * max iterations: 1000
  * class weight: balanced

---

## Project Structure

```
.
├── model_training.py
├── nlp_model_pipeline.joblib
├── main.py
├── Dockerfile
└── README.md
```

---

## Training the Model

Run the following command to train the model:

```bash
python model_training.py
```

The trained model will be saved as:

```
nlp_model_pipeline.joblib
```

---

## Running the API

### Run locally

```bash
uvicorn main:app --reload
```

Open the following URL in your browser:

```
http://127.0.0.1:8000/docs
```

---

### Run with Docker

```bash
docker build -t nlp-api .
docker run -p 8000:8000 nlp-api
```

---

## API Usage

### Endpoint

```
POST /predict
```

### Request

```json
{
  "text": "hello how are you"
}
```

### Response

```json
{
  "prediction": "normal"
}
```

---

## Notes

* The dataset used in this project is small and intended for demonstration purposes
* Model accuracy is limited and can be improved with a larger dataset
* Additional preprocessing and feature engineering can improve performance

---

## Future Improvements

* Support for Arabic language
* Use of advanced models such as transformers (e.g., BERT)
* Add logging and monitoring
* Build a simple user interface

---

If you want, I can also help you make this README stronger for recruiters or turn the project into a more production-ready system.
