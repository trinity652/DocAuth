import os
from flask import Flask, request, url_for, redirect, flash, send_from_directory, render_template, session
from wtforms import validators, StringField, PasswordField, BooleanField
from passlib.hash import sha256_crypt
import requests
import sys
from wtforms import Form
from werkzeug.utils import secure_filename
from jinja2 import evalcontextfilter, Markup, escape
from MySQLdb import escape_string as thwart
import MySQLdb
import gc
import CopyMoveDetection
import cv2
from PIL import Image



def connection():
    conn = MySQLdb.connect(host="localhost",
                           user="root",
                           passwd="Mahakali",
                           db="test")
    c = conn.cursor()
    return c,conn



UPLOAD_FOLDER = '/home/abhilasha/app/static/assets/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
app.secret_key = "Abhi"

@app.route('/')
def homepage():
    return render_template('lp.html')

@app.route('/login', methods = ['GET','POST'])
def loginpage():
    try:
            c, conn = connection()
            if request.method == "POST":
                if request.form['username'] == 'admin' and request.form['password'] == 'admin@620':
                    return redirect(url_for('userpage'))
                #data = c.execute("select * from users where uname = (%s)",(thwart(request.form['username']),))
                #data = c.fetchone ()[2]
                #if request.form['password'] == str(data):
                #    session['logged_in'] = True
                #    session['username'] = request.form['username']
                #    flash("You are now logged in!")
                #    redirect(url_for('userpage'))
                else:
                    error = "Invalid credentials, Try Again!"
                    return str(error)

            gc.collect()
            return render_template('login.html')

    except Exception as e:
        flash("Invalid Credentials!")
        return render_template('login.html')


class RegistrationForm(Form):
    username = StringField('Username', [validators.Length(min=4, max=20)])
    email = StringField('Email Address', [validators.Length(min=6, max=30)])
    password = PasswordField('Password', [validators.DataRequired(),validators.EqualTo('confirm',message='Password mismatch')])
    confirm = PasswordField('Repeat Password')
    accept_tos = BooleanField('I accept the <a href="/tos/">Terms of Service</a> and Privacy Notice',
                              [validators.DataRequired()])

@app.route('/userregister/', methods=['GET','POST'])
def registerpage():
    try:
        form = RegistrationForm(request.form)
        if request.method == "POST" and form.validate():
            username = form.username.data
            email = form.email.data
            password = str(form.password.data)
            c, conn = connection()

            x = c.execute("select * from users where uname = (%s)",
                              (thwart(username),))
            if int(x) > 0:
                flash("That uname is already taken, please try another")
                return render_template('register.html', form=form)
            else:
                c.execute("insert into users(uname, password, name) values(%s,%s,%s)",
                            (thwart(username), thwart(password), thwart(email)))
                flash("Thanks for registering")
                conn.commit()
                c.close()
                conn.close()
                gc.collect()

                session['logged_in'] = True
                session['username'] = username

                return redirect(url_for('userpage'))
        return render_template("register.html", form=form)
    except Exception as e:
        return str(e)



@app.route('/upload-file', methods=['GET', 'POST'])
def upload_file():
    try:
        c, conn = connection()
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                return redirect(url_for('uploaded_file',
                                    filename=filename))
        return render_template('fileupload.html')
    except Exception as e:
        return str(e)

from flask import send_from_directory


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    try:
        CopyMoveDetection.detect('/home/abhilasha/app/static/assets/', filename, '/home/abhilasha/app/images/', blockSize=32)
        return '''<html><head></head><body><h1>Details successfuly added to database!</h1></body></html>'''
    except Exception as e:
        return str(e)
    #return send_from_directory(app.config['UPLOAD_FOLDER'],
    #                           filename)


def get_values():
    c, conn = connection()
    name = c.execute("select name from final_ana")
    name = c.fetchall()
    date = c.execute("select dateof from final_ana")
    date = c.fetchall()
    forgery = c.execute("select forgery from final_ana")
    forgery = c.fetchall()
    forged = c.execute("select forged from final_ana")
    forged = c.fetchall()
    return name,date,forgery,forged


@app.route('/userpage')
def userpage():
    try:
        name, date, forgery, forged = get_values()
        return render_template('userpage.html',data=zip(name,date,forgery,forged))
    except Exception as e:
        return str(e)

@app.route('/analysispage')
def analysepage():
    return render_template('analysispage.html')

if __name__ == "__main__":
    app.run(debug=True)
