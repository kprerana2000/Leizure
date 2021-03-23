from flask import Flask,request,jsonify
from flask_cors import CORS
import recommendation

app = Flask(__name__)
CORS(app) 
        
@app.route('/book', methods=['GET'])
def recommend_movies():
        res = recommendation.results(request.args.get('title'))
        return jsonify(res)

if __name__=='__main__':
          port = int(os.environ.get("PORT", 5000))
          app.run(host='0.0.0.0', port=port)
