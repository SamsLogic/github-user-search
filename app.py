from flask import Flask
from routes.repo_routes import repo

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

app.register_blueprint(repo, url_prefix='/api/v0.1/repo')

@app.route('/')
def hello_world():
    return 'repo search api'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

