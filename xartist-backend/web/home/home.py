from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound

home_page = Blueprint('home', __name__, template_folder='templates', static_folder='static')

@home_page.route('/home')
def home():
    try:
        return render_template('home.html')
    except TemplateNotFound:
        abort(404)


@home_page.route('/')
def index():
    try:
        return render_template('home.html')
    except TemplateNotFound:
        abort(404)