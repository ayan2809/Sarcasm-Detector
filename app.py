from urllib import response
from sarcasmDetector import *;
from flask import *;
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route('/', methods=["POST", "GET"])
def home():
    return render_template('home.html')

@app.route('/result', methods=['POST'])
def convert():
    text= request.form['text']
    text = text.lower()
    print (text)
    result =sarcasmChecker(text)
    print(result[0][0])
    response={
        'result':str(result[0][0])
    }
    return response,200


@app.route('/resultPage', methods=["POST", "GET"])
def result():
    return render_template('result.html')

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5000,
                        type=int, help="port to listen to")
    args = parser.parse_args()
    port = args.port

    app.run(host='127.0.0.1', port=port, debug=True)