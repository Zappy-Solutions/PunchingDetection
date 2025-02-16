import os
import sqlite3
import logging
from flask import Flask, render_template, send_from_directory, abort, g, flash, url_for

# -------------------------------
# 🔹 Configuration
# -------------------------------
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev_key")  # Use a secure secret key in production

# SQLite Database Path
DATABASE = os.path.join(os.getcwd(), "violations.db")

# Directory where violation images are stored
VIOLATION_DIR = os.path.join(os.getcwd(), "violation_frames")
os.makedirs(VIOLATION_DIR, exist_ok=True)  # Ensure directory exists

# -------------------------------
# 🔹 Custom Template Filters
# -------------------------------
@app.template_filter('basename')
def basename_filter(path):
    """Returns the base filename from a path."""
    return os.path.basename(path) if path else ""


# -------------------------------
# 🔹 Database Helper Functions
# -------------------------------
def get_db_connection():
    """Connect to SQLite database and return the connection object."""
    try:
        db = getattr(g, '_database', None)
        if db is None:
            db = g._database = sqlite3.connect(DATABASE)
            db.row_factory = sqlite3.Row  # Enables access via column name
        return db
    except sqlite3.Error as e:
        app.logger.error("Database connection error: %s", e)
        abort(500)


@app.teardown_appcontext
def close_connection(exception):
    """Close the database connection when the request ends."""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


# -------------------------------
# 🔹 Routes
# -------------------------------
@app.route("/")
def index():
    """Fetch all violations and display them on the UI."""
    try:
        conn = get_db_connection()
        cur = conn.execute("SELECT * FROM Violations ORDER BY Time DESC")
        violations = cur.fetchall()
    except Exception as e:
        app.logger.error("Error querying database: %s", e)
        flash("An error occurred while retrieving violation records.", "danger")
        violations = []
    return render_template("violations.html", violations=violations)


@app.route("/violation_frames/<path:filename>")
def violation_frames(filename):
    """Serve violation images securely."""
    try:
        return send_from_directory(VIOLATION_DIR, filename)
    except Exception as e:
        app.logger.error("Error serving file %s: %s", filename, e)
        abort(404)


if __name__ == "__main__":
    # Check that the database file exists
    if not os.path.exists(DATABASE):
        app.logger.error("Database file %s does not exist. Please create it first.", DATABASE)
        exit(1)

    # Run the Flask development server
    app.run(host="0.0.0.0", port=5000, debug=True)