from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/trading/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/api/trading/signal')
def signal():
    return jsonify({'signal': 'hold'})

if __name__ == '__main__':
    app.run()
