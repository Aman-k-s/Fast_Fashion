# Save Model & Preprocessors
joblib.dump(log_reg, "logreg_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_cat, "labelencoder_category.pkl")
joblib.dump(le_subcat, "labelencoder_subcategory.pkl")
