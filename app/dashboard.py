# Import Flask and the tools we need
from flask import Flask
# render_template -> lets us show HTML files from the "templates" folder
from flask import render_template, request
# Import our custom prediction function from src/predict.py
from src.predict import predict_message


# Create the Flask application
app = Flask(__name__)


# ---------------------------
# ROUTE: Home Page
# ---------------------------
@app.route("/")
@app.route("/home")
def home():
    # Simply show the index.html page when someone visits "/" or "/home"
    return render_template("index.html")


# ---------------------------
# ROUTE: About Page
# ---------------------------
@app.route("/about")
def about():
    # Simply show the about.html page when someone visits "/about"
    return render_template("about.html")


# ---------------------------
# ROUTE: Predict Message
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    # Get the message text submitted from the form on index.html
    # "message" must match the "name" attribute of the <textarea> in index.html
    message = request.form["message"]

    # Call our machine learning function to get a prediction
    # It returns two values: the label (Spam/Ham) and the confidence score
    label, confidence = predict_message(message)

    # Send the results back to index.html so they can be displayed
    return render_template(
        "index.html",
        prediction=label,
        confidence=confidence,
        message=message
    )


# ---------------------------
# RUN THE APP
# ---------------------------
if __name__ == "__main__":
    # debug=True gives helpful error messages while developing
    app.run(debug=True)