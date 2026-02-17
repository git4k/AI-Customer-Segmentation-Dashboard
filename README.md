# AI-Customer-Segmentation-Dashboardn using K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## Project Overview

This project focuses on analyzing online retail transaction data to identify distinct customer segments. By implementing Unsupervised Machine Learning (K-Means Clustering) on an RFM (Recency, Frequency, Monetary) framework, the system groups customers based on their purchasing behavior rather than arbitrary rules.

To extend the utility of these insights, the project integrates the Google Gemini API. This Generative AI component interprets the statistical profile of each cluster and drafts targeted marketing communications, demonstrating a complete pipeline from raw data analysis to actionable business strategy.

## Key Features

* **Data Processing Pipeline:** Ingests and cleans large-scale transaction datasets (500,000+ rows), handling missing values and cancellations.
* **RFM Analysis:** Calculates Recency, Frequency, and Monetary value for each unique customer ID.
* **Unsupervised Learning:** Utilizes K-Means Clustering to mathematically determine the optimal number of customer segments.
* **AI-Driven Insights:** Connects cluster statistics to a Large Language Model (LLM) to generate automated, context-aware marketing campaigns.
* **Visualization:** Provides detailed scatter plots and distribution charts to visualize cluster separation and density.

## Technical Architecture

The project follows a linear ETL (Extract, Transform, Load) workflow:

1.  **Data Ingestion:** Loads the "Online Retail II" dataset from the UCI Machine Learning Repository.
2.  **Preprocessing:**
    * Removes records with null Customer IDs.
    * Filters out cancelled transactions (Invoice numbers starting with 'C').
    * Computes total spend per transaction.
3.  **Feature Engineering:**
    * Aggregates data to the customer level.
    * Applies Log Transformation to handle skewed distributions in monetary and frequency data.
    * Scales features using `StandardScaler` to ensure equal weighting during clustering.
4.  **Modeling:**
    * Fits a K-Means model (k=4) to the scaled data.
    * Assigns cluster labels to the original dataset.
5.  **Reporting:**
    * Generates a statistical summary of each cluster.
    * Feeds summary data into the Gemini API for qualitative analysis.

## Technologies Used

* **Language:** Python 3.x
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Visualization:** Matplotlib, Seaborn
* **GenAI Integration:** Google Generative AI SDK

## Installation and Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Retail-Customer-Clustering.git](https://github.com/YOUR_USERNAME/Retail-Customer-Clustering.git)
    cd Retail-Customer-Clustering
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn google-generativeai openpyxl
    ```

3.  **Set up API Key (Optional):**
    To use the GenAI features, you must obtain an API key from Google AI Studio.
    * Create a file named `.env` or export the variable in your terminal:
    ```bash
    export GEMINI_API_KEY="your_api_key_here"
    ```

4.  **Run the analysis:**
    Open the Jupyter Notebook to run the pipeline step-by-step:
    ```bash
    jupyter notebook analysis.ipynb
    ```

## Results

The analysis identified four distinct customer segments based on the 2010-2011 dataset:

| Cluster | Label | Characteristics | Strategy |
| :--- | :--- | :--- | :--- |
| **0** | **Loyal Regulars** | Moderate spend, high frequency, low recency. | Focus on increasing basket size via upsells. |
| **1** | **Champions** | Very high spend, very high frequency. | VIP treatment; no discounts required. |
| **2** | **At Risk** | High historical spend, but high recency (>3 months). | Aggressive win-back campaigns. |
| **3** | **Lost/Low Value** | Low spend, low frequency, high recency. | Low-priority automated re-engagement. |

## Future Improvements

* **Elbow Method Visualization:** Implement a script to visually plot the WCSS (Within-Cluster Sum of Square) to dynamically determine the optimal 'k' value.
* **Streamlit Dashboard:** Develop a frontend interface to allow non-technical stakeholders to upload new datasets and view updated clusters in real-time.
* **Production Deployment:** Containerize the application using Docker for cloud deployment.

## License

This project is open-source and available under the MIT License.
