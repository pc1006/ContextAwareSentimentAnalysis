
# **Aspect-Based Sentiment Analysis using DCASAM + HAN**

This repository contains the implementation of a novel approach to aspect-based sentiment analysis, combining the **Deep Context-Aware Sentiment Analysis Model (DCASAM)** with **Hierarchical Attention Networks (HAN)**. The model is applied to the **IMDB dataset** to analyze fine-grained sentiments associated with specific aspects of movie reviews.

---

## **Introduction**

Aspect-based sentiment analysis (ABSA) plays a crucial role in extracting fine-grained sentiments tied to specific aspects within text. Unlike traditional sentiment analysis that determines the overall sentiment, ABSA provides detailed insights, such as identifying positive or negative sentiment for specific attributes of a review.

This project builds upon the DCASAM framework, integrating **Hierarchical Attention Networks (HAN)** to enhance the contextual understanding of sentiment polarities. By leveraging the **IMDB dataset**, this model captures complex contextual relationships in movie reviews, addressing limitations in traditional approaches.

---

## **Features**

- **BERT embeddings**: Contextual word representations for rich semantic understanding.
- **Bidirectional LSTM**: Captures sequential dependencies in text.
- **Hierarchical Attention Networks (HAN)**: Enhances attention to critical segments of text across multiple levels.
- **IMDB Dataset**: A large-scale collection of movie reviews, ideal for sentiment analysis benchmarking.

---

## **How to Run**

### **Prerequisites**
1. Python 3.8+ installed on your system.
2. Clone this repository:
   ```bash
   git clone https://github.com/konasanijanardh/ABSA-DCASAM-HAN.git
   cd ABSA-DCASAM-HAN
   ```
### **Steps**
1. **Prepare the Dataset**: 
   - Download the IMDB dataset using the `tensorflow.keras.datasets.imdb` module or manually.
   - Place the data in the `data/` directory.
2. **Run the Model**:
   - Execute the training script:
     ```bash
     python train.py
     ```
   - Evaluate the model:
     ```bash
     python evaluate.py
     ```

---

## **Experiments and Observations**

### **Experiment 1: LSTM on Laptop Dataset**
- **Objective**: Train and test a basic LSTM model on the Laptop14 dataset.
- **Observation**: 
  - The model achieved relatively low accuracy due to the smaller size of the dataset, which limited its learning capability.
  - The domain-specific challenges in the Laptop14 dataset made it difficult for the LSTM model to capture complex sentiment patterns effectively.

### **Experiment 2: DCASAM on IMDB Dataset**
- **Objective**: Evaluate the performance of the DCASAM model on the IMDB dataset.
- **Observation**: 
  - The accuracy was lower compared to the LSTM model on the IMDB dataset.
  - This indicates that DCASAM alone faced challenges in capturing aspect-specific sentiment nuances within the IMDB dataset.

### **Experiment 3: DCASAM + HAN on IMDB Dataset**
- **Objective**: Enhance the DCASAM model by integrating Hierarchical Attention Networks (HAN) and test the combined model on the IMDB dataset.
- **Observation**: 
  - The integration of HAN significantly improved the model's accuracy.
  - HAN enabled the model to focus on important words and sentences, enhancing its ability to perform aspect-level sentiment classification effectively.

---

## **Results**

- Achieved significant accuracy and F1 improvements on the IMDB dataset.
- Demonstrated the model's ability to capture aspect-level sentiments effectively.
- Example output for the review:
  *"I loved the movie's soundtrack, but the plot was terrible."*
  - **Sentiment for "soundtrack"**: Positive.
  - **Sentiment for "plot"**: Negative.

---

## **Technologies Used**

- Python 3.8+
- TensorFlow / Keras
- BERT
- NumPy
- Pandas
- Matplotlib

---

## **Future Enhancements**

- Extending the model to other datasets like Restaurant14, Laptop14, or Twitter.
- Experimenting with additional attention mechanisms for improved context capture.
- Deploying the model as a web application for real-time sentiment analysis.

---

## **Contributors**

- **Venkata Naga Janardhan Konasani ** - [GitHub Profile](https://github.com/konasanijanardh)

---

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
