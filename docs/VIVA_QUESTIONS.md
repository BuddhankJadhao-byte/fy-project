# Viva Questions and Short Answers

1. **What is load forecasting?** Predicting future electrical demand for a selected time horizon.
2. **Why is it important in a microgrid?** It supports scheduling, storage planning, peak management, reserve planning, and economical grid interaction.
3. **What is the project target?** Hourly active power load in kilowatts.
4. **Which dataset is used?** One year of hourly Godishala 33/11 kV substation data from Telangana, India.
5. **Why use Random Forest?** It captures non-linear relationships, resists overfitting through tree averaging, and provides feature importance.
6. **How many data samples are available?** 8,760 hourly observations.
7. **What are lag features?** Earlier load values used to predict the current/future hour, such as 1, 24, and 168 hours ago.
8. **Why use sine and cosine for time?** They represent the cyclic nature of hours, weekdays, and months without artificial discontinuities.
9. **What is data leakage?** Supplying information during training that would not be known at forecast time.
10. **Why remove voltage, current, and power factor?** They directly calculate active power and would leak the target.
11. **Why not randomly split the dataset?** Random splitting mixes future patterns into training; chronological splitting simulates real deployment.
12. **What is MAE?** The average absolute difference between actual and predicted load.
13. **What is RMSE?** The square root of mean squared error; it penalizes large mistakes more strongly.
14. **What does R² = 0.916 mean?** About 91.6% of the test-set load variance is explained by the model.
15. **Which model performed best?** Random Forest, with 130.97 kW MAE and 293.06 kW RMSE.
16. **What baselines are compared?** Linear Regression, ARIMA(2,1,0), and a 24-hour seasonal-naive forecast.
17. **Why is ARIMA weak here?** The simple non-seasonal ARIMA baseline lacks weather, calendar, and strong daily/weekly feature interactions.
18. **Is normalization necessary for Random Forest?** No, tree split decisions are not distance based; consistent numeric cleaning is still necessary.
19. **How is future weather handled?** The user may upload it; otherwise the last daily weather profile is repeated as a stated scenario assumption.
20. **What is recursive forecasting?** Each predicted hour is appended and used to create lag features for the next hour.
21. **Can the model directly control a grid?** No. It is decision-support software; operational control needs protection, validation, and operator safeguards.
22. **How can the work be improved?** Add multi-year data, holidays, real weather forecasts, probabilistic intervals, renewable/storage variables, and advanced models.
