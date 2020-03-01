import pickle as pkl
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = pkl.load(open("notebook/token_predictor_model.pkl", 'rb'))


@app.route('/predict/')
def predict():
    data = request.args.get("forecast")
    data = int(data)

    predictions = model.forecast(steps=data)

    output = predictions[0]

    output_json = []
    for i in range(len(output)):
        pred_dict = {}
        pred_dict = {"day": i,"prediction":output[i]}
        output_json.append(pred_dict)
    return jsonify(output_json)


if __name__ == "__main__":
    app.run(debug=True)
