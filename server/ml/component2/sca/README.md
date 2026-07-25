# Component 2A SCA V28 Screening Dashboard

Final dashboard package for the **SCA V28 antalgic hard-negative model only**.

## Included model artifacts

- `06_models/component_2_sca/component2_sca_model_v28_antalgic_hard_negative_holdout_eval.joblib`
- `06_models/component_2_sca/component2_sca_feature_list_v28_antalgic_hard_negative.json`
- `06_models/component_2_sca/component2_sca_settings_v28_antalgic_hard_negative.json`

## Run

```powershell
cd component2_sca_v28_screening_dashboard
pip install -r requirements.txt
streamlit run app.py --server.port 8502
```

## Notes

- Upload a walking video or a training-safe landmark CSV.
- Manual direction override is available: `Auto`, `L2R`, `R2L`.
- Manual FPS override is available.
- The dashboard saves a window-level SCA V28 inference report CSV when prediction runs.
