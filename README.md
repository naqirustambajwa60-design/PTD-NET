# DermShift: PTD-Net skin lesion demo

Research demo of PTD-Net (EfficientNet-B0, two view network with cross attention fusion), trained on
HAM10000 and BCN20000 dermoscopy and adapted without target labels to PAD-UFES-20 smartphone photographs.
Held out PAD-UFES-20 AUC 0.802. Not a medical device.

## Files
- `app.py` Streamlit interface
- `model.py` network, preprocessing, prediction and Grad-CAM
- `features.py` the 48 ABCD descriptors
- `app_model_v13.pt` trained weights, feature scaling and decision threshold
- `app_results.json` held out results shown in the footer
- `requirements.txt` CPU packages for Streamlit Community Cloud

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free)
1. Create a public GitHub repository and upload all files of this folder, including `.streamlit/config.toml`.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose the repository and branch, and set the main file to `app.py`.
4. Open **Advanced settings** and choose Python 3.11.
5. Click **Deploy**. The first build takes about 5 to 10 minutes. The app link can then be shared.
