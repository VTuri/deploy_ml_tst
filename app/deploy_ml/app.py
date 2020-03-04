import pickle as pkl
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = pkl.load(open("notebook/token_predictor_model.pkl", 'rb'))


@app.route('/predict/')
def predict():
    """
    example:http://0.0.0.0:8000/predict/?forecast=3
    :return: [{"day":0,"prediction":129298.36155585825},{"day":1,"prediction":129471.13316899545},{"day":2,"prediction":129404.59708590152}]
    """
    data = request.args.get("forecast")
    data = int(data)

    predictions = model.forecast(steps=data)

    output = predictions[0]

    output_json = []
    for i in range(len(output)):
        pred_dict = {}
        pred_dict = {"day": i+1,"prediction":output[i]}
        output_json.append(pred_dict)
    return jsonify(output_json)


if __name__ == "__main__":
    app.run(debug=True)
