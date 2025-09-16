#!/usr/bin/env python3
"""
Simple test to check if the app can start
"""

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return "DriveGuard AI is running!"

@app.route('/login')
def login():
    return "Login page"

if __name__ == '__main__':
    print("Starting simple test server...")
    app.run(debug=True, host='0.0.0.0', port=3001)
