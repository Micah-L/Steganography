import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

from steg import *
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return True #allow all for now
    #return '.' in filename and \
    #       filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)           

@app.route('/steg', methods=['GET', 'POST'])
def steg():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'cover_img' not in request.files or 'msg_img' not in request.files:
            flash('No file part')
            return redirect(request.url)
        cover_img = request.files['cover_img']
        msg_img = request.files['msg_img']

        # if user does not select file, browser also
        # submit an empty part without filename
        if cover_img.filename == '' or msg_img.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if cover_img and allowed_file(cover_img.filename) and msg_img and allowed_file(msg_img.filename):
            cover_img_name = secure_filename(cover_img.filename)
            msg_img_name = secure_filename(msg_img.filename)
            cover_img.save(os.path.join(app.config['UPLOAD_FOLDER'], cover_img_name))
            msg_img.save(os.path.join(app.config['UPLOAD_FOLDER'], msg_img_name))

            cover_img = cv.imread( os.path.join(app.config['UPLOAD_FOLDER'], cover_img_name) )
            with open( os.path.join(app.config['UPLOAD_FOLDER'], msg_img_name), 'rb' ) as f:
                msg_img = f.read()
            img = Image(cover_img)
            num_bytes_written = len(msg_img)
            img.write_bytes(msg_img, num_writeable_bits=2, stride=1)
            saved_name = f"secret_embdedded_{num_bytes_written}_bytes_{cover_img_name}"
            saved_name = ".".join(saved_name.split('.')[:-1]) + ".png"
            cv.imwrite( os.path.join(app.config['UPLOAD_FOLDER'], saved_name), img.img_arr)
            return redirect(url_for('uploaded_file',
                                    filename=saved_name))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Choose an image and a file to embed.</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=cover_img>
      <input type=file name=msg_img>
      <input type=submit value=Upload>
    </form>
    '''
@app.route('/unsteg', methods=['GET', 'POST'])
def unsteg():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'secret_msg_img' not in request.files:
            flash('No file part')
            return redirect(request.url)
        secret_msg_img = request.files['secret_msg_img']
        num_bytes_stored = int(request.form['num_bytes_stored'])
        # if user does not select file, browser also
        # submit an empty part without filename
        if secret_msg_img.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if secret_msg_img and allowed_file(secret_msg_img.filename):
            secret_msg_img_name = secure_filename(secret_msg_img.filename)
            secret_msg_img.save(os.path.join(app.config['UPLOAD_FOLDER'], secret_msg_img_name))

            img = cv.imread( os.path.join(app.config['UPLOAD_FOLDER'], secret_msg_img_name) )
            img = Image(img)
            inner_img = img.read_bytes(num_bytes_to_read = num_bytes_stored, num_writeable_bits = 2, stride = 1)
            saved_name = "hidden_message"
            with open(os.path.join(app.config['UPLOAD_FOLDER'], saved_name), 'wb') as f:
                f.write(inner_img)
            return redirect(url_for('uploaded_file',
                                    filename=saved_name))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload an image that contains a secret embedded file</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=secret_msg_img>
      Enter the number of bytes that the embedded file takes up: 
      <input type=number name=num_bytes_stored>
      <input type=submit value=Upload>
    </form>
    '''
