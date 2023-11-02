from flask import Blueprint
from controllers.repo_controller import repo_controller
from controllers.upload_controller import upload_controller

repo = Blueprint('repo', 'repo')

repo.route('repo', methods=['POST'])(repo_controller)
repo.route('upload', methods=['POST'])(upload_controller)