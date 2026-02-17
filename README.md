# AI-Customer-Segmentation-Dashboardn using K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📌 Project Overview

This project applies **RFM Analysis** (Recency, Frequency, Monetary) combined with **K-Means Clustering** to segment customers from a real-world online retail dataset. An AI marketing agent then analyzes each cluster and generates a recommended campaign strategy and draft email — automatically.

The goal is to move beyond "one-size-fits-all" marketing and give businesses a data-driven way to treat each customer group differently based on their actual purchasing behavior.

---

## 🧠 What We Did — Step by Step

### 1. Environment Setup
- Installed and configured the **Google Gemini API** (`google-generativeai`) for potential LLM-powered enhancements.
- Imported core libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.

### 2. Data Acquisition
- Downloaded the **UCI Online Retail Dataset** directly from the UCI Machine Learning Repository.
- The raw dataset contains **541,909 transactions** across 8 columns including `CustomerID`, `InvoiceNo`, `InvoiceDate`, `Quantity`, `UnitPrice`, and more.

### 3. Data Cleaning & Feature Engineering
- **Removed rows with missing `CustomerID`** — we can only segment customers we can identify.
- **Filtered out cancelled orders** — invoices starting with `'C'` represent returns/cancellations and would distort spending metrics.
- **Created a `TotalSum` column** by multiplying `Quantity × UnitPrice` to represent revenue per transaction.
- After cleaning, **397,924 valid transaction rows** remained.

### 4. RFM Metric Calculation
Each customer was summarized into three behavioral scores:

| Metric | Description |
|---|---|
| **Recency (R)** | Days since the customer's last purchase (lower = more recent) |
| **Frequency (F)** | Total number of purchases made |
| **Monetary (M)** | Total amount spent across all purchases |

A "snapshot date" (one day after the last transaction in the dataset) was used as the reference point for Recency calculation.

### 5. Preprocessing for Clustering
- Applied **log transformation** (`log(x + 1)`) to RFM values to reduce the effect of extreme outliers and right-skewed distributions.
- Applied **StandardScaler** to normalize all three features to zero mean and unit variance — ensuring no single metric dominates the clustering.

### 6. K-Means Clustering
- Used **K-Means++ initialization** with `k=4` clusters for stability and reproducibility (`random_state=42`).
- Each customer was assigned to one of 4 clusters based on their normalized RFM profile.

### 7. AI Marketing Agent
A rule-based AI marketing agent analyzed the **mean RFM statistics** of each cluster and automatically assigned:
- A **segment name** (e.g., "Champions", "Loyal Regulars")
- A **campaign strategy**
- A **draft marketing email**

**Clusters Identified:**

| Cluster | Avg Recency | Avg Frequency | Avg Spend | Segment | Strategy |
|---|---|---|---|---|---|
| 0 | 20 days | 39 orders | $612 | **Loyal Regulars** | Upsell — increase basket size |
| 1 | 13 days | 283 orders | $7,043 | **Champions (VIPs)** | Exclusive early access, no discounts needed |
| 2 | 185 days | 15 orders | $298 | **Low Value / Lost** | Generic re-engagement newsletter |
| 3 | 96 days | 80 orders | $1,522 | **Low Value / Lost** | Generic re-engagement newsletter |

### 8. Visualization
Two charts were produced to explore and communicate the segmentation results:

- **`customer_segments.png`** — Scatter plot of Recency vs. Total Spend, color-coded by cluster. Shows the clear behavioral separation between groups.
- **`cluster_boxplots.png`** — Three side-by-side box plots showing the distribution of Monetary, Frequency, and Recency across all four clusters, revealing intra-cluster spread and outliers.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail) |
| File | `Online Retail.xlsx` |
| Raw Size | 541,909 rows × 8 columns |
| After Cleaning | 397,924 rows |
| Time Period | Dec 2010 – Dec 2011 |
| Region | UK-based online retailer |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| pandas | Data loading, cleaning, aggregation |
| NumPy | Log transformations, numerical ops |
| scikit-learn | StandardScaler, KMeans clustering |
| Matplotlib & Seaborn | Visualizations |
| Google Gemini API | AI integration (configured, extensible) |
| Google Colab | Runtime environment |

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl google-generativeai
```

### Run the Notebook
1. Clone this repository:
```bash
   git clone https://github.com/your-username/ai-customer-segmentation.git
   cd ai-customer-segmentation
```
2. Open `AI_Customer_Segmentation.ipynb` in Jupyter or Google Colab.
3. If using Colab, add your `GEMINI_API_KEY` to the **Secrets** tab (🔑 icon on the left sidebar).
4. Run all cells top to bottom.

The dataset is downloaded automatically in Cell 2 — no manual data download required.

---

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
