
# 📊 Market Research Analysis - App Store Insights

An **ETL-based data analysis project** focused on mobile app market research. This project extracts, cleans, transforms, and analyzes app data to help identify top-performing applications in specific categories, along with user sentiment insights.

---

## 🚀 Features

* 📥 **Data Extraction** from CSV files
* 🧹 **Data Cleaning & Transformation**
* 🔍 **Exploratory Data Analysis (EDA)**
* 🌿 **Correlation Analysis & Visualization**
* 📆 **SQLite Database Integration**

---

## 📂 Project Structure

```
📆 market-research-project
🕛

📄 apps_data.csv
📄 review_data.csv

📄 result.csv                # Processed results
📄 etl_pipeline.py           # ETL process (Extract, Transform, Load)
📄 market_research.db        # SQLite database
📄 requirements.txt          # Python dependencies
📄 README.md                 # Project overview
📄 marketresearch.ipymb      # Implementation on colab
```

---

## ⚙️ Tech Stack

| Tool/Library               | Purpose                          |
| -------------------------- | -------------------------------- |
| **Python (pandas, numpy)** | Data processing & transformation |
| **Seaborn, Matplotlib**    | Data visualization               |
| **SQLite**                 | Data storage                     |
| **Jupyter / Colab**        | EDA & rapid experimentation      |

---

## 🛠️ How to Run Locally

### 1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/market-research-project.git
cd market-research-project
```

### 2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 3. **Run the ETL Pipeline**

```bash
python etl_pipeline.py
```

---

## 📊 Example Visualizations

* 📈 Rating distributions
* 🔥 Top apps by reviews
* 🔗 Correlation heatmaps
* 🎝 Sentiment polarity insights

---

## 📌 Customization

* Change the **category filter** in the ETL pipeline to analyze different app categories.

---

## 🌟 Future Enhancements

* Integrate with **cloud databases** (e.g., PostgreSQL)
* Add **real-time data sources via APIs**
* Deploy as an interactive dashboard using tools like **Streamlit** or **Dash**

---

## 🤝 Contributing

Feel free to fork the repository and submit pull requests to:

* Improve visualizations
* Add machine learning models
* Optimize data pipeline

---

## 📄 License

This project is licensed under the **MIT License**.


