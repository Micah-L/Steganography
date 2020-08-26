import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template
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
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)           

@app.route('/', methods=['GET', 'POST'])
@app.route('/steg', methods=['GET', 'POST'])
def steg():
    embedding_stride = 1
    embedding_bits_used_per_byte = 2
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
            img.write_bytes(msg_img, num_writeable_bits=embedding_bits_used_per_byte, stride=embedding_stride)
            saved_name = f"secret_embdedded_{num_bytes_written}_bytes_{cover_img_name}"
            saved_name = ".".join(saved_name.split('.')[:-1]) + ".png"
            cv.imwrite( os.path.join(app.config['UPLOAD_FOLDER'], saved_name), img.img_arr)
            return render_template("image_embedding_processed.html", file_path = url_for('uploaded_file', filename=saved_name), embedding_num_bytes = num_bytes_written, embedding_stride = embedding_stride, embedding_bits_used_per_byte = embedding_bits_used_per_byte)
            # return redirect(url_for('uploaded_file', filename=saved_name))
    return render_template("steg.html")

@app.route('/unsteg', methods=['GET', 'POST'])
def unsteg():
    embedding_stride = 1
    embedding_bits_used_per_byte = 2    
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
            inner_img = img.read_bytes(num_bytes_to_read = num_bytes_stored, num_writeable_bits = embedding_bits_used_per_byte, stride = embedding_stride)
            saved_name = "hidden_message"
            with open(os.path.join(app.config['UPLOAD_FOLDER'], saved_name), 'wb') as f:
                f.write(inner_img)
            return redirect(url_for('uploaded_file',
                                    filename=saved_name))
    return render_template("unsteg.html")