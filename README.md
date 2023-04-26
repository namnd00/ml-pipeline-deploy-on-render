# Guide in this project: `ML pipeline to expose API on` [Render](https://render.com/)
### Introduction:
This project to create a pipeline to train a model and publish it with a public API on [Render](https://render.com/).

### Development:
- Install nescessary packages: `pip install -r requirements.txt`
- Build Machine Learning pipeline and you can run full end to end pipeline on script:`python ml_pipeline.py` in the following help:
```bash
usage: ml_pipeline.py [-h] [--steps {basic_cleaning,train_model,evaluate_model,all}]

ML training pipeline

optional arguments:
  -h, --help            show this help message and exit
  --steps {basic_cleaning,train_model,evaluate_model,all}
                        Pipeline steps
```
- The prepared data saved to [path](./data/) and trained model, other artifacts saved to [path](./model/)

### Testing
- Running test case on script: `pip install -e . && pytest .` to install [starter](./starter/) module and execute pytest script.

### API
- Running script: `uvicorn main:app --host 0.0.0.0 --port 10000` and access to [link](http:localhost:10000/docs) to using swagger UI and try to use API.

### CI/CD
*Note*: I've used google drive to save data, model instead AWS S3.

**Pipeline**:
- Pulls data from **DVC** google drive, sync model to deploy directory and execute **Flake8** + **pytest** doing every test.
- Automatically deploy to production on **Render** cloud platform instead **Heroku**
- This deployed API link: https://mlops-app.onrender.com/

**Check Render deployed API**
- Running script: `python check_render_api.py`
- The stdout should be:
```bash
Response code: 200
Response body: {'prediction': '[0]', 'class_name': '<=50K', 'success': True}
```

### Screenshots
- You can access to [screenshots](./screenshots/) to see them.