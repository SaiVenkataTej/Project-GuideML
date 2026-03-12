# 📘 Project GuideML: The User Guide

**Welcome, friend.**

We—the team of engineers who built this—are thrilled you’re here. We created **GuideML** because we remember what it was like starting out with Machine Learning. It felt like a black box of complex math and confusing code. 

We wanted to build something that strips away that complexity, letting you see the *power* of ML without getting bogged down in the *setup*. This tool is our way of handing you the keys to a powerful engine, with us in the passenger seat guiding you.

---

## 🚀 Getting Started (The "Hello World")

Let's get this running on your machine. We've designed it to be as simple as possible—like installing a game.

### prerequisites
You'll need **Python** installed. Think of Python as the engine that runs our car. If you don't have it, grab it from python.org.

### Step 1: Installation
Open your terminal (Command Prompt or PowerShell). Don't worry if you've strictly used a mouse before; typing these commands makes you look like a pro.

1.  **Clone the Repository** (Download the code):
    ```bash
    git clone https://github.com/SaiVenkataTej/Project-GuideML.git
    cd Project-GuideML
    ```

2.  **Create an Environment** (A clean workspace):
    *We recommend this so GuideML doesn't mess with your other projects.*
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies** (Get the tools we need):
    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Launching the App
Now for the magic moment. Run this command:

```bash
python interface/app.py
```

You'll see a message like `Running on http://127.0.0.1:5000`. 
Open your web browser (Chrome, Edge, etc.) and type that address in. 

**🎉 You're in!**

---

## 🕹️ How to Use GuideML

Imagine you have a spreadsheet (CSV file) with data—maybe it's house prices, or patient records. You want to predict something (like the price, or if a patient is sick), but you don't know which ML model is best.

**GuideML does that hard work for you.**

1.  **Upload Your Data**: Click the "Choose File" button and pick your CSV.
2.  **Pick Your Target**: Tell us which column you want to predict (e.g., "Price").
3.  **Select Models (Optional)**: You can leave this blank to let us run *everything*, or pick specific ones like "Random Forest" if you're curious.
4.  **Explore the DNA**: Once training is complete, dive into the **Explainability** tab to see your model's "DNA"—the precise settings it chose—and the **SHAP Global Influence** plot.
5.  **Run Analysis**: Click "Start Training".

**What happens next?**
While you wait, our specific "worker bees" (the backend code) are:
*   Cleaning your data (fixing missing values).
*   Training multiple AI brains simultaneously.
*   Testing them against each other.
*   **Decoding the Brain**: Calculating SHAP values to explain *why* it made its decisions.

When it's done, you'll see a **Dashboard** showing you the winner!

---

## 🧠 Key Concepts (Simplified)

### What is "Preprocessing"?
Data in the real world is messy. It has holes, typos, and weird formats. **Preprocessing** is like washing and chopping vegetables before cooking. We handle this automatically so the AI doesn't get "food poisoning" from bad data.

### What is "Model Selection"?
There are many ways to predict the future. Some are simple (like drawing a line), some are complex (like a decision tree). We run a tournament: we train *all* of them on your data and see which one gets the highest score.

---

**Ready to dive deeper?** 
If you want to understand the *magic* behind the models, check out `ML_THEORY.md`.
If you want to see how we built the website part, check out `FRONTEND_GUIDE.md`.
