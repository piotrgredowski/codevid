"""Simple Flask login demo application for Codevid tutorials."""

from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = "demo-secret-key"

# Demo credentials
VALID_USERS = {
    "demo@example.com": "password123",
    "admin@example.com": "admin123",
}


@app.route("/")
def index():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "")
        password = request.form.get("password", "")

        if email in VALID_USERS and VALID_USERS[email] == password:
            return redirect(url_for("dashboard", user=email.split("@")[0].title()))
        else:
            flash("Invalid email or password", "error")

    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    user = request.args.get("user", "Guest")
    return render_template("dashboard.html", user=user)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
