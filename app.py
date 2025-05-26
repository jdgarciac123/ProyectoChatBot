from flask import Flask, request, session, render_template
from inference import load_models, generate_response

app = Flask(__name__)
app.secret_key = 'your-secret-key'

# Carga modelo/tokenizer una vez
tokenizer, enc_model, dec_model = load_models()

@app.route('/', methods=['GET', 'POST'])
def chat():
    if 'history' not in session:
        session['history'] = []
    history = session['history']
    if request.method == 'POST':
        user_msg = request.form['message']
        bot_msg = generate_response(user_msg, history, tokenizer, enc_model, dec_model)
        history.append((user_msg, bot_msg))
        session['history'] = history[-2:]
    return render_template('index.html', history=session['history'])

if __name__ == '__main__':
    app.run(port=5000, debug=True)
