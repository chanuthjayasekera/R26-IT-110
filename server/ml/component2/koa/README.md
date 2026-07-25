# Component 2B KOA V14 Screening Dashboard

Final Streamlit screening dashboard built with the KOA V14 youtube-ataxia-once non-KOA model and predictor.

## Run

```powershell
cd component2_koa_v14_screening_dashboard
pip install -r requirements.txt
streamlit run app.py --server.port 8503
```

## Included

- Streamlit screening dashboard
- Video upload flow
- CSV upload flow
- MediaPipe pose extraction
- Manual walking direction override: Auto, L2R, R2L
- Manual FPS override
- KOA V14 youtube-ataxia-once non-KOA predictor
- Model artifacts:
  - component2_koa_v14_youtube_ataxia_once_holdout_eval_model.joblib
  - component2_koa_v14_youtube_ataxia_once_feature_list.json
  - component2_koa_v14_youtube_ataxia_once_settings.json

## Notes

This is a screening support dashboard, not a medical diagnosis tool.
