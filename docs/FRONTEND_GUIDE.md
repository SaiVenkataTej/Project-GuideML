# 🎨 The Senior Frontend Dev's Guide

**"So, you want to build a website? Let's talk about how the web *actually* works."**

We often see students get lost in React, Angular, or Vue before understanding the basics. To really own the web, you need to understand the **core trio**: HTML, CSS, and JavaScript.

Let's use our favorite analogy: **The Restaurant**.

---

## 1. The Core Concepts (The Restaurant Analogy)

### HTML (The Skeleton / The Building)
Imagine walking into a restaurant. You see walls, tables, chairs, and a menu. They aren't painted yet, and they don't move. They just *exist*.
*   **HTML (HyperText Markup Language)** is the structure. It tells the browser: "Put a button here," "Put a table there," "This is a heading."
*   In our project, `dashboard.html` is the building.

### CSS (The Style / The Decor)
Now imagine painting the walls blue, putting fancy tablecloths on the tables, and making the menu look elegant.
*   **CSS (Cascading Style Sheets)** is the style. It tells the browser: "Make that button blue," "Make the text bold," "Put this box in the center."
*   In our project, `style.css` is the decorator.

### JavaScript (The Interaction / The Staff)
A restaurant with just a building and decor is boring. You need waiters, chefs, and action! If you wave your hand, someone should come over.
*   **JavaScript (JS)** is the logic. It tells the browser: "When the user clicks this button, do X," "If the data is loaded, show a graph."
*   In our project, `script.js` is the staff.

### AJAX (The Communication / The Waiter)
This is the magic part. In old websites, if you wanted new info, you had to reload the whole page (like leaving the restaurant and coming back in).
*   **AJAX (Asynchronous JavaScript and XML)** allows the page to talk to the server (the kitchen) strictly in the background. It sends a note to the kitchen ("Cook model A") and gets a result back *without you ever leaving your seat*.

---

## 2. A Walkthrough of Our Interface

Let's look at `index.html` and `script.js` using this analogy.

### The HTML Structure (`templates/index.html`)
You'll see tags like `<div>`, `<h1>`, and `<form>`.
*   `<form id="uploadForm" action="/process" method="POST">`: This is the order form. When submitted, it sends the CSV file and settings directly to the `/process` route on the server.
*   `<div id="loadingOverlay">`: This is a waiting room. It's hidden by default. When JS detects the form is submitted, it shows this overlay to give the user *visual* feedback while the server is working.
*   `<div id="storyContainer">`: A table inside the overlay that animates through a sequence of processing steps to simulate progress.

### The JavaScript Logic (`static/js/script.js`)

#### The "Event Listener" (The Greeter)
```javascript
document.addEventListener('DOMContentLoaded', () => { ... });
```
This means: "Wait until the page is loaded before doing anything."

#### The "Submission" (Taking the Order)
```javascript
form.addEventListener('submit', function() {
    loadingOverlay.classList.remove('d-none');
    startEngineOrchestration();
});
```
*   **Note:** There is no `e.preventDefault()` here. The form performs a standard, synchronous browser POST to `/process`. The loading overlay and its animations play purely on the client side while the browser waits for the server response.
*   The animation sequence in `startEngineOrchestration()` is **cosmetic only** — it runs on a timer and is not connected to actual backend progress.

#### The CSV Header Parser
```javascript
function parseCSVLine(text) { ... }
```
*   When a file is selected, `script.js` reads just the first 8KB of the file to parse column names and populate the "Target Column" dropdown — without uploading anything yet.

### The Python Connection (`app.py`)
This is the Kitchen.
*   It receives the `POST` request with the uploaded file.
*   It runs the **entire ML pipeline synchronously** on the request thread — training all models, generating plots, and saving the result.
*   Once complete, it stores results in a global `LAST_RESULTS` variable and issues an HTTP `302` redirect to `/dashboard`.

---

## 3. Why This Architecture Matters

We built it this way for simplicity as a **local, single-user tool**:
*   **Without AJAX:** The user submits the form and the browser waits. The loading overlay provides visual feedback, but it is purely cosmetic — the browser is blocked until the server responds.
*   **The Limitation:** If the ML training takes a long time (e.g., large dataset, many models), the browser may display a "page unresponsive" warning or the connection may time out. This is a known current limitation.
*   **Transparency First:** By exposing the **Model DNA** and **SHAP influence** on the dashboard, we turn a \"Black Box\" into a \"Glass Box.\"

**The difference between a local tool and a production application** is async processing (Celery/Redis + WebSockets). That is the natural next step for this project.
