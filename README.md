# 🎌 Anime Recommender System ⭐

![Anime Banner](https://placehold.co/850x200/1a1a2e/ffffff?text=Welcome+to+the+Anime+Recommender+System&font=raleway)

## 🛠 Tech Stack  

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white) ![Surprise](https://img.shields.io/badge/Surprise-Recommender-lightgrey) ![Pandas](https://img.shields.io/badge/Pandas-Data--Handling-yellow?logo=pandas&logoColor=black) ![NumPy](https://img.shields.io/badge/NumPy-Math-blue?logo=numpy&logoColor=white)



**Otaku Alert!** 🚀 This notebook is your ultimate weapon for discovering hidden anime gems, powered by AI! Like a Sharingan scanning for chakra signatures, we'll use:
- **Content-Based Filtering:** Matches anime by genres (e.g., if you love 'Dragon Ball', you’ll get action-packed titles!).
- **Collaborative Filtering:** Learns from user ratings (what would L recommend to Light?).
- **Hybrid Approach:** Combines both for ultimate power-ups! 💥

Rate some anime, choose your favorite and let the recommendations flow like chakra!

**Kon'nichiwa, User!** Rate at least 5 anime to unlock personalized recs. Demo mode uses sampled data for speed. Let's go! 🎉

</br>

## 📑 Table of Contents  

- ✨ [Demo](#-demo)  
- 📊 [About the Dataset](#-about-the-dataset)  
- 🌳 [Directory Tree](#-directory-tree)
- 📂 [Repository Structure](#-repository-structure)
- ⚙️ [Installation](#️-installation--local-setup)  
- 🧑‍💻 [Code Walkthrough](#-code-walkthrough)  
- 🐞 [Bug / ✨ Feature Request](#-bug-feature-request)  
- 🔮 [Future Work](#-future-work)  

</br>

## ✨ Demo  
**Click to try the Live App** 👉 <a href="https://anime-recommender-system-iambatman.streamlit.app/">Anime Recommendation System Live</a>

</br>

## 📊 About the Dataset

This project is powered by the comprehensive **Anime Recommendations Database** from Kaggle, which was originally scraped from MyAnimeList.net. It provides the two critical scrolls of data needed to train a sophisticated hybrid model.

### 📜 `anime.csv` — The Encyclopedia of Anime
This file is the metadata catalog, containing information on **12,294 unique anime**. It's the primary fuel for the **Content-Based model**, allowing it to understand *what an anime is about*.

-   **Key Columns:** `anime_id`, `name`, `genre`, `type`, `rating`

### 📈 `rating.csv` — The Collective Will of the Fans
This file contains the raw power of crowd wisdom: **7.8 million ratings** from **73,516 users**. This massive user-item interaction matrix is the essential fuel for the **Collaborative Filtering (SVD) engine**, allowing it to learn *what fans actually think*.

-   **Key Columns:** `user_id`, `anime_id`, `rating`

<br>

➡️ **Source:** [Anime Recommendations Database on Kaggle](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)

</br>

## 🌳 Directory Tree  
```bash
anime-recommender-system/
├── .streamlit/
│   └── config.toml
├── app.py
├── anime.csv
├── rating.csv
├── packages.txt
└── requirements.txt
```

</br>

## 📂 Repository Structure

Here’s a breakdown of the key files and what they do:

| File | Purpose |
| :--- | :--- |
| **`app.py`** | 🚀 The heart of the operation! This single script contains all the Streamlit UI code, data processing functions, and the recommendation algorithms. Run this file to launch the app. |
| **`anime.csv`** | 📜 The Anime Encyclopedia. This dataset holds all the metadata for thousands of anime titles, including their genres, type, and community ratings. Powers the **Content-Based model**. |
| **`rating.csv`** | 📊 The Scroll of User Jutsus. Contains millions of user ratings, forming the massive user-item interaction matrix. This is the fuel for the **Collaborative Filtering (SVD) engine**. |
| **`requirements.txt`** <br> **`packages.txt`** <br> **`.streamlit/config.toml`** | ⚙️ The Deployment Spellbook. This collection of files provides the critical instructions for Streamlit Community Cloud, defining the exact Python version, system-level packages (like C++ compilers), and Python libraries needed to build and run the application successfully. | 

</br>


## ⚙️ Installation & Local Setup

Want to run the Anime Recommender on your own machine? No problem! Here are two ways to get the project set up on **Windows**.

---

### Method 1: Using Git Method

This is the recommended method if you're comfortable with the command line. It makes getting updates super easy.

**1. Install Git:**
If you don't have Git, download and install it from the official website. The default settings are fine.
➡️ [git-scm.com/downloads](https://git-scm.com/downloads)

**2. Clone the Repository:**
Open your command prompt (`cmd`) or terminal and run the following commands:
```bash
# Clone the repository to your computer
git clone https://github.com/Finnwindhoek/anime-recommender-system.git
```
```bash
# Navigate into the newly created project folder
cd anime-recommender-system
```
</br>

### Method 2: Manual Download

No command line needed for this.

**1. Go to the GitHub Repository:**
➡️ [github.com/Finnwindhoek/anime-recommender-system](https://github.com/Finnwindhoek/anime-recommender-system)

**2. Download the ZIP file:**
Click the green < > Code button, then click "Download ZIP".

**3. Unzip the folder:**
Find the downloaded ```anime-recommender-system-main.zip``` file in your Downloads folder and unzip/extract it to a location you'll remember (like your Desktop).

</br>

### 💻 Running the Application (For Both Methods)
- Click WinKey and search for "*cmd* " and enter ```python --version```
Note: Install python - https://www.python.org/downloads/windows/

<div style="margin-top:20px;">

- Navigate to https://www.anaconda.com/download/success and install **miniconda installer** 
</div>

</br>

#### Step 2: After installation Completed.
- Open anaconda prompt.
- Create enviroment with Python 3.10.11
```
conda create -n ai_env python=3.10.11
```
- Activate the python enviroment
```
conda activate ai_env
```
- Navigate to **Anime Recommender System (Streamlit)** 
```
cd C:\Users\File Directory\anime recommender system
```
</br>

##### Install command for packages:

```
conda install -c conda-forge scikit-learn scikit-surprise -y
```

```
pip install -r requirements.txt
```
After package installations, run ```streamlit run app.py``` to run the code and start the model.

</br>

## 🧑‍💻 Code Walkthrough

The entire application is powered by a single, comprehensive script: `app.py`. Here’s a high-level breakdown of its architecture and logic.

### 1. The Ninja's Arsenal: Imports & Configuration 🧰
The script begins by importing all the necessary libraries—our "ninja tools"—including `Streamlit` for the UI, `Pandas` for data handling, `Scikit-learn` for content-based logic, and `Surprise` for the collaborative engine. Key global variables like `REQUIRED_RATINGS` and `DEMO_MODE` are also set here, allowing for easy configuration.

### 2. Summoning the Scrolls: Data Loading & Caching 📜
To ensure the app is fast and responsive, all data loading and heavy computations are cached using Streamlit's powerful decorators:
-   **`@st.cache_data`:** Used for loading the CSV files (`anime.csv`, `rating.csv`). This ensures the data is loaded from disk only once, making subsequent runs instantaneous.
-   **`@st.cache_resource`:** Used for building the resource-intensive similarity matrix and training the SVD models. This keeps these heavy objects in memory, preventing them from being recalculated on every user interaction.

### 3. The Three Great Jutsus: The Core Algorithms 💥
This is the heart of the recommender system, where the three distinct recommendation methods are defined as functions:
-   **`get_content_recs()`:** Implements the **Content-Based** model. It takes a user's favorite anime, finds its vector in the pre-computed TF-IDF Cosine Similarity matrix, and returns the most similar anime.
-   **`get_collab_recs()`:** Powers the **Collaborative Filtering** model. It takes the user's current ratings, adds them to the main dataset, trains a personalized SVD model on the fly, and predicts ratings for all unseen anime to find the top recommendations.
-   **`get_hybrid_recs()`:** The "Sage Mode" of the app. It calls the other two functions to get two separate lists of candidates, then merges and re-ranks them using a custom `hybrid_score` to produce a final list that is the best of both worlds.

### 4. The Power Level Scanner: Performance Evaluation ⚡
The `calculate_evaluation_metrics()` function is a standout feature. It takes a user's ratings, performs a train/test split, and trains a temporary SVD model to evaluate its predictive accuracy. This function calculates the **RMSE, Precision, Recall, and F1-Score**, powering the real-time "System Evaluation" tab and providing direct insight into the model's performance.

### 5. The Mission Control Center: The Streamlit UI 🎮
The final part of the script contains the main application logic. It uses Streamlit commands (`st.selectbox`, `st.button`, `st.tabs`, etc.) to draw the entire user interface. A key feature here is the use of **`st.session_state`**, which acts as the app's memory. It's used to store the user's ratings across multiple interactions, allowing for a stateful and personalized experience without needing a database.


</br>


## 🐞Bug / ✨ Feature Request

Spotted a glitch in the matrix or have an idea for a Super Saiyan-level upgrade? I'd love to hear from you! The best way to get in touch is by opening an issue on this repository's GitHub page.

<p align="center">
  <a href="https://github.com/Finnwindhoek/anime-recommender-system/issues/new/choose">
    <img src="https://img.shields.io/badge/-Open%20a%20New%20Issue-brightgreen?style=for-the-badge&logo=github" alt="Open a New Issue">
  </a>
</p>

### Reporting a Bug 🐛
If you find a bug, please include the following to help me squash it:
- A clear and descriptive title.
- Steps to reproduce the bug.
- What you expected to happen.
- What actually happened (screenshots are a huge help!).

### Suggesting a Feature ✨
If you have an idea for a new feature:
- A clear and descriptive title.
- A detailed description of the proposed feature.
- Explain why this feature would be a great addition to the recommender!

</br>

## 🔮 Future Work

The current system is a powerful Shounen protagonist, but there's always a new form to unlock! Here are some of the potential power-ups for the next season:

-   **🧠 Deeper Content Analysis with NLP**
    Go beyond simple genre tags. The next step is to implement advanced NLP models (like BERT or Sentence Transformers) to analyze anime synopses, reviews, and fan tags. This would allow the system to understand nuanced themes like "dystopian world-building" or "morally grey protagonist," leading to incredibly insightful content-based recommendations.

-   **🌐 Real-Time API Integration**
    Keep the database fresh! Integrate with APIs like AniList or MyAnimeList (Jikan) to automatically fetch new anime releases, updated ratings, and seasonal charts. This would transform the app from a static snapshot into a living, breathing recommendation platform.

-   **👤 Personalized User Profiles**
    Allow users to create accounts to save their ratings, view their watch history, and create custom watchlists. This is the foundation for a truly long-term personalized experience where the model can learn and adapt to a user's evolving tastes over time.

-   **📈 Advanced Visualizations & Explainability**
    Show the user *why* they're getting a recommendation. Add interactive charts (using Plotly or Altair) to visualize a user's genre preferences, rating distribution, and how a recommendation connects to their taste profile. This builds trust and makes the system's "thinking" more transparent.

-   **📱 Mobile App Deployment**
    Take the recommender on the go! Package the system into a mobile app using a framework like React Native or Flutter, providing a seamless and accessible experience for users on any device.

