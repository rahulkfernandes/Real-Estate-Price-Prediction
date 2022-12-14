from flask import Flask, request, jsonify
import utils
app = Flask(__name__)

@app.route('/get_location_names')
def get_location_names():
    response = jsonify({
        'locations': utils.get_location_names()
    })
    response.headers.add("Access-Control-Allow-Origin", '*')
    return

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
   total_sqft = float(request.form['total_sqft'])
   location = request.form['location']
   bath = int(request.form['bath'])
   bhk = int(request.form['bhk'])
   
   response = jsonify({
    'estimated_price': utils.get_estimated_price(location,total_sqft,bhk,bath)
   })
   return response

if __name__ == "__main__":
    print("Starting Python Flask Server")
    app.run()